"""Main FastAPI application for Vercel deployment."""

import asyncio
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
import xml.etree.ElementTree as ET

import httpx
import feedparser
import yaml
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, select_autoescape

# === Configuration ===

@dataclass
class ResearchArea:
    name: str
    terms: list[str]
    weight: float = 1.0

@dataclass
class Config:
    areas: dict[str, ResearchArea]
    lookback_days: int
    max_results: int
    title_multiplier: float
    pubmed_enabled: bool
    biorxiv_enabled: bool
    arxiv_enabled: bool
    arxiv_categories: list[str]
    high_impact_journals: list[str]
    high_impact_lookback_days: int
    other_lookback_options: list[int]

def load_config() -> Config:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    areas = {}
    for name, area_raw in raw.get("areas", {}).items():
        areas[name] = ResearchArea(
            name=name,
            terms=area_raw.get("terms", []),
            weight=area_raw.get("weight", 1.0),
        )

    sources = raw.get("sources", {})
    arxiv = sources.get("arxiv", {})

    return Config(
        areas=areas,
        lookback_days=raw.get("collection", {}).get("lookback_days", 7),
        max_results=raw.get("collection", {}).get("max_results", 200),
        title_multiplier=raw.get("scoring", {}).get("title_multiplier", 2.0),
        pubmed_enabled=sources.get("pubmed", True),
        biorxiv_enabled=sources.get("biorxiv", True),
        arxiv_enabled=arxiv.get("enabled", True) if isinstance(arxiv, dict) else arxiv,
        arxiv_categories=arxiv.get("categories", ["cs.AI", "q-bio.GN"]) if isinstance(arxiv, dict) else ["cs.AI", "q-bio.GN"],
        high_impact_journals=raw.get("high_impact_journals", []),
        high_impact_lookback_days=raw.get("high_impact_lookback_days", 90),
        other_lookback_options=raw.get("other_lookback_options", [7, 14, 30, 60, 90]),
    )

# === Simple Cache (in-memory, best-effort for serverless) ===

CACHE_TTL_SECONDS = 300
_CACHE = {"timestamp": 0.0, "key": None, "papers": None}
DISPLAY_LIMIT = 20

# === Paper Model ===

@dataclass
class Paper:
    title: str
    authors: str
    abstract: str
    url: str
    published_date: date
    source: str
    journal: str = ""
    matched_areas: list[str] = field(default_factory=list)
    score: float = 0.0

# === Collectors ===

async def fetch_pubmed(days: int, max_results: int, terms: list[str], client: httpx.AsyncClient) -> list[Paper]:
    """Fetch papers from PubMed matching keywords."""
    return await _fetch_pubmed_with_query(days, max_results, terms, None, client)


async def fetch_pubmed_high_impact(
    days: int, max_results: int, terms: list[str], journals: list[str], client: httpx.AsyncClient
) -> list[Paper]:
    """Fetch papers from high-impact journals matching keywords."""
    return await _fetch_pubmed_with_query(days, max_results, terms, journals, client)


def _fetch_pubmed_sync(
    days: int, max_results: int, terms: list[str], journals: Optional[list[str]]
) -> list[Paper]:
    """Core PubMed fetch using urllib (sync, more reliable on serverless)."""
    import urllib.request
    import urllib.parse
    import json

    papers = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    date_range = f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[edat]"

    try:
        # Search for IDs
        term_query = _build_pubmed_query(terms=terms, date_range=date_range, journals=journals)
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_data = urllib.parse.urlencode({
            "db": "pubmed",
            "term": term_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date",
        }).encode("utf-8")
        search_req = urllib.request.Request(search_url, data=search_data, method="POST")

        with urllib.request.urlopen(search_req, timeout=20) as resp:
            data = json.loads(resp.read())
            ids = data.get("esearchresult", {}).get("idlist", [])

        if not ids:
            return papers

        # Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_data = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(ids[:max_results]),
            "retmode": "xml",
        }).encode("utf-8")
        fetch_req = urllib.request.Request(fetch_url, data=fetch_data, method="POST")

        with urllib.request.urlopen(fetch_req, timeout=20) as resp:
            xml_text = resp.read().decode("utf-8")

        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            try:
                paper = _parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
            except Exception:
                continue
    except Exception as e:
        print(f"PubMed error: {e}")

    return papers


async def _fetch_pubmed_with_query(
    days: int, max_results: int, terms: list[str], journals: Optional[list[str]], client: httpx.AsyncClient
) -> list[Paper]:
    """Wrapper that runs sync PubMed fetch in thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_pubmed_sync, days, max_results, terms, journals)

def _build_pubmed_query(
    terms: Optional[list[str]], date_range: str, journals: Optional[list[str]] = None
) -> str:
    """Build PubMed query with optional journal filter."""
    parts = []

    # Keywords (Title/Abstract + MeSH)
    if terms:
        query_terms = []
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            if " " in cleaned:
                cleaned = f'"{cleaned}"'
            query_terms.append(f"({cleaned}[Title/Abstract] OR {cleaned}[MeSH Terms])")
        if query_terms:
            parts.append(f"({' OR '.join(query_terms)})")

    # Journal filter
    if journals:
        journal_terms = []
        for j in journals:
            j_clean = j.strip()
            if j_clean:
                journal_terms.append(f'"{j_clean}"[Journal]')
        if journal_terms:
            parts.append(f"({' OR '.join(journal_terms)})")

    # Date range
    parts.append(date_range)

    return " AND ".join(parts)

def _parse_pubmed_article(article: ET.Element) -> Optional[Paper]:
    medline = article.find(".//MedlineCitation")
    if medline is None:
        return None

    article_elem = medline.find(".//Article")
    if article_elem is None:
        return None

    title_elem = article_elem.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
    if not title:
        return None

    abstract_parts = []
    for abs_text in article_elem.findall(".//Abstract/AbstractText"):
        text = "".join(abs_text.itertext()).strip()
        abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    authors = []
    for author in article_elem.findall(".//AuthorList/Author"):
        last = author.find("LastName")
        if last is not None and last.text:
            authors.append(last.text)
    authors_str = ", ".join(authors[:5])
    if len(authors) > 5:
        authors_str += " et al."

    pmid_elem = medline.find(".//PMID")
    pmid = pmid_elem.text if pmid_elem is not None else ""
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    journal_title_elem = article_elem.find(".//Journal/Title")
    journal_title = "".join(journal_title_elem.itertext()).strip() if journal_title_elem is not None else ""

    published = _parse_pubmed_date(article_elem, medline)
    return Paper(
        title=title,
        authors=authors_str,
        abstract=abstract,
        url=url,
        published_date=published,
        source="pubmed",
        journal=journal_title,
    )

def _parse_pubmed_date(article_elem: ET.Element, medline: ET.Element) -> date:
    """Best-effort pub date parsing for PubMed articles."""
    # Prefer explicit ArticleDate
    article_date = article_elem.find(".//ArticleDate")
    if article_date is not None:
        year = article_date.findtext("Year")
        month = article_date.findtext("Month")
        day = article_date.findtext("Day")
        parsed = _date_from_parts(year, month, day)
        if parsed:
            return parsed

    # Fallback to Journal PubDate
    pub_date = medline.find(".//JournalIssue/PubDate")
    if pub_date is not None:
        year = pub_date.findtext("Year")
        month = pub_date.findtext("Month")
        day = pub_date.findtext("Day")
        parsed = _date_from_parts(year, month, day)
        if parsed:
            return parsed

        medline_date = pub_date.findtext("MedlineDate")
        parsed = _date_from_medline_text(medline_date)
        if parsed:
            return parsed

    return date.today()

def _date_from_medline_text(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    parts = value.replace("-", " ").split()
    year = next((part for part in parts if part.isdigit() and len(part) == 4), None)
    month = next((part for part in parts if part[:3].lower() in {
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    }), None)
    return _date_from_parts(year, month, None)

def _date_from_parts(year: Optional[str], month: Optional[str], day: Optional[str]) -> Optional[date]:
    if not year:
        return None
    try:
        y = int(year)
    except ValueError:
        return None

    if month:
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            m = int(month)
        except ValueError:
            m = month_map.get(month.strip().lower()[:3])
        if not m:
            m = 1
    else:
        m = 1

    if day:
        try:
            d = int(day)
        except ValueError:
            d = 1
    else:
        d = 1

    try:
        return date(y, m, d)
    except ValueError:
        return None

async def fetch_biorxiv(days: int, max_results: int, client: httpx.AsyncClient) -> list[Paper]:
    """Fetch papers from bioRxiv."""
    papers = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    try:
        url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/0"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("collection", [])[:max_results]:
            try:
                doi = item.get("doi", "")
                papers.append(Paper(
                    title=item.get("title", "").strip(),
                    authors=item.get("authors", ""),
                    abstract=item.get("abstract", ""),
                    url=f"https://www.biorxiv.org/content/{doi}" if doi else "",
                    published_date=date.fromisoformat(item.get("date", str(date.today()))),
                    source="biorxiv",
                    journal="bioRxiv",
                ))
            except Exception:
                continue
    except Exception as e:
        print(f"bioRxiv error: {e}")

    return papers

async def fetch_arxiv(days: int, max_results: int, categories: list[str], client: httpx.AsyncClient) -> list[Paper]:
    """Fetch papers from arXiv."""
    papers = []

    try:
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"({cat_query})",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        resp = await client.get(url, params=params)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        cutoff = date.today() - timedelta(days=days)

        for entry in feed.entries:
            try:
                published = entry.get("published", "")
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                pub_date = dt.date()

                if pub_date < cutoff:
                    continue

                authors = [a.get("name", "") for a in entry.get("authors", [])]
                authors_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_str += " et al."

                papers.append(Paper(
                    title=entry.get("title", "").replace("\n", " ").strip(),
                    authors=authors_str,
                    abstract=entry.get("summary", "").replace("\n", " ").strip(),
                    url=entry.get("link", ""),
                    published_date=pub_date,
                    source="arxiv",
                    journal="arXiv",
                ))
            except Exception:
                continue
    except Exception as e:
        print(f"arXiv error: {e}")

    return papers

# === Scoring ===

def score_paper(paper: Paper, config: Config) -> Paper:
    """Score a paper based on keyword matching."""
    title = paper.title.lower()
    abstract = paper.abstract.lower()

    matched_areas = []
    total_score = 0.0

    for area_name, area in config.areas.items():
        area_score = 0.0
        for term in area.terms:
            term_lower = term.lower()
            if term_lower in title:
                area_score += config.title_multiplier
            if term_lower in abstract:
                area_score += 1.0

        if area_score > 0:
            matched_areas.append(area_name)
            total_score += area_score * area.weight

    paper.matched_areas = matched_areas
    paper.score = round(total_score, 1)
    return paper

def _normalize_journal(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())

def is_high_impact(paper: Paper, config: Config) -> bool:
    if not paper.journal:
        return False
    paper_norm = _normalize_journal(paper.journal)
    if not paper_norm:
        return False
    alias_map = {
        "pnas": "procnatlacadsciusa",
    }
    for journal in config.high_impact_journals:
        journal_norm = _normalize_journal(journal)
        if not journal_norm:
            continue
        if paper_norm == journal_norm:
            return True
        if journal_norm in paper_norm or paper_norm in journal_norm:
            return True
        if alias_map.get(journal_norm) == paper_norm:
            return True
        if alias_map.get(paper_norm) == journal_norm:
            return True
    return False

def _collect_search_terms(config: Config) -> list[str]:
    terms = []
    seen = set()
    for area in config.areas.values():
        for term in area.terms:
            cleaned = term.strip()
            if not cleaned or cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            terms.append(cleaned)
    return terms

def _area_label(area_name: str) -> str:
    labels = {
        "ai_ml": "AI / ML",
        "ngs": "NGS",
    }
    return labels.get(area_name, area_name.replace("_", " ").title())

# === Main Fetch Function ===

async def fetch_all_papers(config: Config) -> list[Paper]:
    """Fetch papers from all sources and score them."""
    return await fetch_all_papers_for_days(config, config.lookback_days, config.high_impact_lookback_days)

async def fetch_all_papers_for_days(config: Config, lookback_days: int, high_impact_days: int) -> list[Paper]:
    """Fetch papers from all sources for a given lookback and score them."""
    search_terms = _collect_search_terms(config)
    cache_key = (
        lookback_days,
        high_impact_days,
        config.max_results,
        config.pubmed_enabled,
        config.biorxiv_enabled,
        config.arxiv_enabled,
        tuple(config.arxiv_categories),
        tuple((k, v.weight, tuple(v.terms)) for k, v in sorted(config.areas.items())),
        config.title_multiplier,
        tuple(config.high_impact_journals),
    )
    now = time.time()
    if _CACHE["key"] == cache_key and (now - _CACHE["timestamp"]) < CACHE_TTL_SECONDS:
        return list(_CACHE["papers"] or [])

    async with httpx.AsyncClient(timeout=25.0) as client:
        all_results = []

        # Fetch PubMed sequentially to avoid rate limits
        if config.pubmed_enabled:
            # Regular keyword search
            try:
                pubmed_papers = await fetch_pubmed(lookback_days, config.max_results, search_terms, client)
                all_results.append(pubmed_papers)
            except Exception as e:
                print(f"PubMed regular search failed: {e}")
                all_results.append([])

            # High-impact journal search
            if config.high_impact_journals:
                try:
                    hi_papers = await fetch_pubmed_high_impact(
                        high_impact_days, config.max_results, search_terms, config.high_impact_journals, client
                    )
                    all_results.append(hi_papers)
                except Exception as e:
                    print(f"PubMed high-impact search failed: {e}")
                    all_results.append([])

        # Fetch other sources in parallel
        other_tasks = []
        if config.biorxiv_enabled:
            other_tasks.append(fetch_biorxiv(lookback_days, config.max_results, client))
        if config.arxiv_enabled:
            other_tasks.append(fetch_arxiv(lookback_days, config.max_results, config.arxiv_categories, client))

        if other_tasks:
            other_results = await asyncio.gather(*other_tasks, return_exceptions=True)
            all_results.extend(other_results)

        results = all_results

    all_papers = []
    seen_urls = set()
    for result in results:
        if isinstance(result, list):
            for paper in result:
                if paper.url and paper.url not in seen_urls:
                    seen_urls.add(paper.url)
                    all_papers.append(paper)

    # Score and filter
    scored = [score_paper(p, config) for p in all_papers]
    relevant = [p for p in scored if p.score > 0]

    # Sort by score descending, then newest first.
    relevant.sort(key=lambda p: (p.score, p.published_date), reverse=True)

    _CACHE["timestamp"] = now
    _CACHE["key"] = cache_key
    _CACHE["papers"] = list(relevant)
    return relevant

# === FastAPI App ===

app = FastAPI(title="PaperFeed")

@app.get("/manifest.webmanifest")
async def manifest():
    return JSONResponse({
        "name": "PaperFeed",
        "short_name": "PaperFeed",
        "description": "A compact feed of high-impact research papers.",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#0d1117",
        "theme_color": "#0d1117",
        "icons": [
            {
                "src": "/icon.svg",
                "sizes": "any",
                "type": "image/svg+xml",
            }
        ],
    })

@app.get("/icon.svg")
async def icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<rect width="512" height="512" rx="96" fill="#0d1117"/>
<rect x="112" y="80" width="288" height="352" rx="28" fill="#161b22" stroke="#30363d" stroke-width="12"/>
<path d="M164 160h184M164 218h184M164 276h132" stroke="#58a6ff" stroke-width="28" stroke-linecap="round"/>
<circle cx="342" cy="344" r="34" fill="#3fb950"/>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#0d1117">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-title" content="PaperFeed">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>PaperFeed</title>
    <link rel="manifest" href="/manifest.webmanifest">
    <link rel="icon" href="/icon.svg" type="image/svg+xml">
    <link rel="apple-touch-icon" href="/icon.svg">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --card: #1c2128;
            --card-hover: #242c38;
            --border: #30363d;
            --text: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent: #58a6ff;
            --accent-hover: #79b8ff;

            /* Area colors */
            --area-metagenomics: #3fb950;
            --area-viruses_outbreaks: #f85149;
            --area-ngs: #a371f7;
            --area-ai_ml: #79c0ff;
            --area-human_genome: #ffa657;
            --area-bioinformatics: #ff7b72;

            /* Source colors */
            --pubmed: #4493f8;
            --biorxiv: #da3b01;
            --arxiv: #b31b1b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            font-size: 14px;
            min-height: 100vh;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 1.5rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent), #a371f7);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--text), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stats {
            font-size: 0.9rem;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            border: 1px solid var(--border);
        }

        .controls {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        select {
            padding: 0.6rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.9rem;
            background: var(--card);
            color: var(--text);
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
        }

        select:hover {
            border-color: var(--accent);
            background: var(--card-hover);
        }

        select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
        }

        button {
            padding: 0.6rem 1.25rem;
            border: none;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            background: var(--accent);
            color: var(--bg);
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }

        button:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(0);
        }

        .secondary-button {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }

        .secondary-button:hover {
            background: var(--card-hover);
            color: var(--text);
        }

        .papers {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .section-title {
            margin: 1.5rem 0 0.75rem;
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .section-title span {
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .paper {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
            position: relative;
            overflow: hidden;
            opacity: 0;
            transform: translateY(24px) rotateX(4deg);
            transition: opacity 0.5s ease, transform 0.5s ease, border-color 0.2s, box-shadow 0.2s;
            will-change: transform, opacity;
        }

        .paper::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: var(--area-color, var(--accent));
        }

        .paper:hover {
            border-color: var(--text-muted);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        .paper.in-view {
            opacity: 1;
            transform: translateY(0) rotateX(0);
        }

        .paper-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }

        .paper-title {
            font-size: 1.05rem;
            font-weight: 500;
            color: var(--text);
            text-decoration: none;
            line-height: 1.4;
            transition: color 0.2s;
        }

        .paper-title:hover {
            color: var(--accent);
        }

        .score {
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            white-space: nowrap;
            box-shadow: 0 2px 8px rgba(46, 160, 67, 0.3);
        }

        .paper-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        .journal {
            color: var(--text-secondary);
            max-width: 260px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .source-badge {
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .source-pubmed { background: rgba(68, 147, 248, 0.15); color: var(--pubmed); }
        .source-biorxiv { background: rgba(218, 59, 1, 0.15); color: var(--biorxiv); }
        .source-arxiv { background: rgba(179, 27, 27, 0.15); color: var(--arxiv); }

        .authors {
            color: var(--text-muted);
            max-width: 400px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .date {
            color: var(--text-muted);
        }

        .areas {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.75rem;
        }

        .area-tag {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .area-metagenomics { background: rgba(63, 185, 80, 0.15); color: var(--area-metagenomics); }
        .area-viruses_outbreaks { background: rgba(248, 81, 73, 0.15); color: var(--area-viruses_outbreaks); }
        .area-ngs { background: rgba(163, 113, 247, 0.15); color: var(--area-ngs); }
        .area-ai_ml { background: rgba(121, 192, 255, 0.15); color: var(--area-ai_ml); }
        .area-human_genome { background: rgba(255, 166, 87, 0.15); color: var(--area-human_genome); }
        .area-bioinformatics { background: rgba(255, 123, 114, 0.15); color: var(--area-bioinformatics); }

        .abstract {
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.7;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .empty {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px dashed var(--border);
        }

        .empty p:first-child {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }

        @media (max-width: 640px) {
            .container { padding: 1rem; }
            .controls {
                flex-direction: column;
                gap: 0.5rem;
            }
            select, button { width: 100%; }
            .paper-header { flex-direction: column; gap: 0.5rem; }
            .score { align-self: flex-start; }
            .authors { max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">P</div>
                <h1>PaperFeed</h1>
            </div>
            <div class="stats">{{ papers|length }} papers</div>
        </header>

        <form class="controls" method="get" id="filter-form">
            <select name="topic">
                <option value="">Any interest</option>
                {% for a in area_options %}
                <option value="{{ a.name }}" {% if topic == a.name %}selected{% endif %}>{{ a.label }}</option>
                {% endfor %}
            </select>
            <select name="method">
                <option value="">Any method or second interest</option>
                {% for a in area_options %}
                <option value="{{ a.name }}" {% if method == a.name %}selected{% endif %}>{{ a.label }}</option>
                {% endfor %}
            </select>
            <select name="scope">
                <option value="high_impact_first" {% if scope == 'high_impact_first' %}selected{% endif %}>High-impact first</option>
                <option value="high_impact_only" {% if scope == 'high_impact_only' %}selected{% endif %}>High-impact only</option>
                <option value="all" {% if scope == 'all' %}selected{% endif %}>All ranked papers</option>
            </select>
            <select name="source">
                <option value="">All sources</option>
                <option value="pubmed" {% if source == 'pubmed' %}selected{% endif %}>PubMed</option>
                <option value="biorxiv" {% if source == 'biorxiv' %}selected{% endif %}>bioRxiv</option>
                <option value="arxiv" {% if source == 'arxiv' %}selected{% endif %}>arXiv</option>
            </select>
            <select name="sort">
                <option value="score" {% if sort == 'score' %}selected{% endif %}>Sort by relevance</option>
                <option value="date" {% if sort == 'date' %}selected{% endif %}>Sort by date</option>
            </select>
            <select name="days">
                {% for d in other_lookback_options %}
                <option value="{{ d }}" {% if days == d %}selected{% endif %}>Last {{ d }} days</option>
                {% endfor %}
            </select>
            <button type="submit">Apply Filters</button>
            <button type="button" class="secondary-button" id="reset-preferences">Reset</button>
        </form>

        {% if papers %}
            <div class="section-title">High-impact journals <span>{{ high_impact_papers|length }} papers</span></div>
            <div class="papers">
                {% if high_impact_papers %}
                    {% for paper in high_impact_papers %}
                    <article class="paper" style="--area-color: var(--area-{{ paper.matched_areas[0] if paper.matched_areas else 'default' }}, var(--accent))">
                        <div class="paper-header">
                            <a href="{{ paper.url }}" target="_blank" rel="noopener" class="paper-title">
                                {{ paper.title }}
                            </a>
                            <span class="score">{{ paper.score }}</span>
                        </div>
                        <div class="paper-meta">
                            <span class="source-badge source-{{ paper.source }}">{{ paper.source }}</span>
                            {% if paper.journal %}
                            <span class="journal">{{ paper.journal }}</span>
                            {% endif %}
                            <span class="authors">{{ paper.authors }}</span>
                            <span class="date">{{ paper.published_date }}</span>
                        </div>
                        {% if paper.matched_areas %}
                        <div class="areas">
                            {% for a in paper.matched_areas %}
                            <span class="area-tag area-{{ a }}">{{ a.replace('_', ' ') }}</span>
                            {% endfor %}
                        </div>
                        {% endif %}
                        <p class="abstract">{{ paper.abstract }}</p>
                    </article>
                    {% endfor %}
                {% else %}
                    <div class="empty">
                        <p>No high-impact journal papers in this selection.</p>
                    </div>
                {% endif %}
            </div>

            <div class="section-title">Other papers <span>{{ other_papers|length }} papers</span></div>
            <div class="papers">
                {% for paper in other_papers %}
                <article class="paper" style="--area-color: var(--area-{{ paper.matched_areas[0] if paper.matched_areas else 'default' }}, var(--accent))">
                    <div class="paper-header">
                        <a href="{{ paper.url }}" target="_blank" rel="noopener" class="paper-title">
                            {{ paper.title }}
                        </a>
                        <span class="score">{{ paper.score }}</span>
                    </div>
                    <div class="paper-meta">
                        <span class="source-badge source-{{ paper.source }}">{{ paper.source }}</span>
                        {% if paper.journal %}
                        <span class="journal">{{ paper.journal }}</span>
                        {% endif %}
                        <span class="authors">{{ paper.authors }}</span>
                        <span class="date">{{ paper.published_date }}</span>
                    </div>
                    {% if paper.matched_areas %}
                    <div class="areas">
                        {% for a in paper.matched_areas %}
                        <span class="area-tag area-{{ a }}">{{ a.replace('_', ' ') }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                    <p class="abstract">{{ paper.abstract }}</p>
                </article>
                {% endfor %}
            </div>
        {% else %}
            <div class="empty">
                <p>No papers found matching your criteria.</p>
                <p>Try adjusting filters or check back later.</p>
            </div>
        {% endif %}
    </div>
    <script>
        const filterStorageKey = 'paperfeed.filters.v1';
        const filterForm = document.querySelector('#filter-form');
        const filterNames = ['topic', 'method', 'scope', 'source', 'sort', 'days'];

        function readFiltersFromForm() {
            const filters = {};
            for (const name of filterNames) {
                const field = filterForm.elements[name];
                if (field && field.value) {
                    filters[name] = field.value;
                }
            }
            return filters;
        }

        function buildFilterQuery(filters) {
            const params = new URLSearchParams();
            for (const name of filterNames) {
                if (filters[name]) {
                    params.set(name, filters[name]);
                }
            }
            return params.toString();
        }

        if (filterForm && !window.location.search) {
            const savedFilters = localStorage.getItem(filterStorageKey);
            if (savedFilters) {
                try {
                    const query = buildFilterQuery(JSON.parse(savedFilters));
                    if (query) {
                        window.location.replace(`${window.location.pathname}?${query}`);
                    }
                } catch {
                    localStorage.removeItem(filterStorageKey);
                }
            }
        }

        filterForm?.addEventListener('submit', () => {
            localStorage.setItem(filterStorageKey, JSON.stringify(readFiltersFromForm()));
        });

        document.querySelector('#reset-preferences')?.addEventListener('click', () => {
            localStorage.removeItem(filterStorageKey);
            for (const field of filterForm.elements) {
                if (field.tagName === 'SELECT') {
                    field.selectedIndex = 0;
                }
            }
            filterForm.submit();
        });

        const observer = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                    observer.unobserve(entry.target);
                }
            }
        }, { threshold: 0.15 });

        document.querySelectorAll('.paper').forEach((paper) => observer.observe(paper));
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    topic: str = Query(""),
    method: str = Query(""),
    scope: str = Query("high_impact_first"),
    source: str = Query(""),
    sort: str = Query("score"),
    days: int = Query(7),
):
    config = load_config()
    if days not in config.other_lookback_options:
        days = config.lookback_days
    if topic not in config.areas:
        topic = ""
    if method not in config.areas:
        method = ""
    if scope not in {"high_impact_first", "high_impact_only", "all"}:
        scope = "high_impact_first"

    papers = await fetch_all_papers_for_days(config, days, config.high_impact_lookback_days)

    # Filter by topic and method independently.
    if topic:
        papers = [p for p in papers if topic in p.matched_areas]
    if method:
        papers = [p for p in papers if method in p.matched_areas]

    # Filter by source
    if source:
        papers = [p for p in papers if p.source == source]

    # Sort
    if sort == "date":
        papers.sort(key=lambda p: p.published_date, reverse=True)
    else:
        papers.sort(key=lambda p: (p.score, p.published_date), reverse=True)

    today = date.today()
    high_cutoff = today - timedelta(days=config.high_impact_lookback_days)
    other_cutoff = today - timedelta(days=days)
    high_candidates = [
        paper for paper in papers
        if is_high_impact(paper, config) and paper.published_date >= high_cutoff
    ]
    other_candidates = [
        paper for paper in papers
        if (not is_high_impact(paper, config)) and paper.published_date >= other_cutoff
    ]

    if scope == "high_impact_only":
        high_impact_papers = high_candidates[:DISPLAY_LIMIT]
        other_papers = []
    elif scope == "all":
        displayed = []
        for paper in papers:
            if is_high_impact(paper, config) and paper.published_date >= high_cutoff:
                displayed.append(paper)
            elif (not is_high_impact(paper, config)) and paper.published_date >= other_cutoff:
                displayed.append(paper)
            if len(displayed) >= DISPLAY_LIMIT:
                break
        high_impact_papers = [paper for paper in displayed if is_high_impact(paper, config)]
        other_papers = [paper for paper in displayed if not is_high_impact(paper, config)]
    else:
        high_impact_papers = high_candidates[:DISPLAY_LIMIT]
        remaining_slots = DISPLAY_LIMIT - len(high_impact_papers)
        other_papers = other_candidates[:remaining_slots]

    displayed_papers = high_impact_papers + other_papers
    area_options = [
        {"name": name, "label": _area_label(name)}
        for name in config.areas
    ]

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(HTML_TEMPLATE)
    html = template.render(
        papers=displayed_papers,
        high_impact_papers=high_impact_papers,
        other_papers=other_papers,
        area_options=area_options,
        topic=topic,
        method=method,
        scope=scope,
        source=source,
        sort=sort,
        days=days,
        other_lookback_options=config.other_lookback_options,
    )
    return HTMLResponse(html)

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/debug")
async def debug():
    """Debug endpoint to test PubMed connectivity."""
    import urllib.request
    import urllib.parse

    config = load_config()
    search_terms = _collect_search_terms(config)

    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    date_range = f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[edat]"

    # Test regular query
    query = _build_pubmed_query(search_terms[:5], date_range, None)

    results = {
        "terms_count": len(search_terms),
        "query_preview": query[:200],
        "high_impact_journals": config.high_impact_journals[:5],
    }

    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax=10&retmode=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            import json
            data = json.loads(resp.read())
            results["pubmed_count"] = data.get("esearchresult", {}).get("count", "N/A")
            results["pubmed_ids"] = len(data.get("esearchresult", {}).get("idlist", []))
            results["pubmed_status"] = "OK"
    except Exception as e:
        results["pubmed_status"] = f"ERROR: {e}"

    return results

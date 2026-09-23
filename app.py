"""PaperFeed — a ranked feed of recent papers from PubMed, bioRxiv/medRxiv and arXiv.

Single-file FastAPI app for Vercel's zero-config Python deployment.
"""

import asyncio
import json
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import feedparser
import httpx
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, select_autoescape
from markupsafe import Markup, escape

# === Configuration ===

@dataclass
class ResearchArea:
    name: str
    terms: list[str]
    weight: float = 1.0


@dataclass(frozen=True)
class InterestFilter:
    """One required condition: a paper must match at least one of `terms`
    (or, for area filters, have been tagged with one of `area_names`)."""
    label: str
    terms: tuple[str, ...]
    area_names: tuple[str, ...] = ()
    custom: bool = False


@dataclass
class Config:
    areas: dict[str, ResearchArea]
    lookback_days: int
    max_results: int
    title_multiplier: float
    pubmed_enabled: bool
    biorxiv_enabled: bool
    medrxiv_enabled: bool
    arxiv_enabled: bool
    arxiv_categories: list[str]
    high_impact_journals: list[str]
    high_impact_lookback_days: int
    other_lookback_options: list[int]


@lru_cache(maxsize=1)
def load_config() -> Config:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    areas = {
        name: ResearchArea(name=name, terms=area_raw.get("terms", []), weight=area_raw.get("weight", 1.0))
        for name, area_raw in raw.get("areas", {}).items()
    }
    sources = raw.get("sources", {})
    arxiv = sources.get("arxiv", {})

    return Config(
        areas=areas,
        lookback_days=raw.get("collection", {}).get("lookback_days", 7),
        max_results=raw.get("collection", {}).get("max_results", 200),
        title_multiplier=raw.get("scoring", {}).get("title_multiplier", 2.0),
        pubmed_enabled=sources.get("pubmed", True),
        biorxiv_enabled=sources.get("biorxiv", True),
        medrxiv_enabled=sources.get("medrxiv", False),
        arxiv_enabled=arxiv.get("enabled", True) if isinstance(arxiv, dict) else bool(arxiv),
        arxiv_categories=arxiv.get("categories", ["cs.AI", "q-bio.GN"]) if isinstance(arxiv, dict) else ["cs.AI", "q-bio.GN"],
        high_impact_journals=raw.get("high_impact_journals", []),
        high_impact_lookback_days=raw.get("high_impact_lookback_days", 90),
        other_lookback_options=raw.get("other_lookback_options", [7, 14, 30, 60, 90]),
    )


# === Cache (in-memory, best-effort on serverless) ===

CACHE_TTL_SECONDS = 300
CACHE_ENTRIES = 16
_CACHE: "OrderedDict[tuple, tuple[float, list, dict]]" = OrderedDict()
MAX_DISPLAY = 200      # ranked papers sent to the page
PAGE_SIZE = 25         # shown per "Show more" step in each section
MAX_QUERY_LENGTH = 200
SOURCES = ("pubmed", "biorxiv", "medrxiv", "arxiv")
SOURCE_LABELS = {"pubmed": "PubMed", "biorxiv": "bioRxiv", "medrxiv": "medRxiv", "arxiv": "arXiv"}

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
    doi: str = ""
    pmid: str = ""
    matched_areas: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def key(self) -> str:
        return self.doi.lower() if self.doi else self.url

    def to_dict(self) -> dict:
        data = asdict(self)
        data["published_date"] = self.published_date.isoformat()
        return data


# === Search syntax ===

_TOKEN = re.compile(r'(-?)"([^"]*)"?|(\S+)')


def parse_query(q: str) -> tuple[list[tuple[str, ...]], list[str]]:
    """Parse the search box into AND-ed groups of OR-ed terms plus exclusions.

    ``"long read" nanopore OR illumina -plasmid`` →
    groups ``[("long read",), ("nanopore", "illumina")]`` and excludes ``["plasmid"]``.
    """
    groups: list[list[str]] = []
    excludes: list[str] = []
    pending_or = False
    for match in _TOKEN.finditer(q[:MAX_QUERY_LENGTH]):
        neg_phrase, phrase, bare = match.groups()
        if bare == "OR":
            pending_or = bool(groups)
            continue
        negate = bool(neg_phrase) or (bare is not None and bare.startswith("-") and len(bare) > 1)
        text = phrase if phrase is not None else (bare[1:] if negate else bare)
        text = " ".join(re.sub(r'[\[\]"()]', " ", text or "").split())
        if not text:
            pending_or = False
            continue
        if negate:
            excludes.append(text)
        elif pending_or and groups:
            groups[-1].append(text)
        else:
            groups.append([text])
        pending_or = False
    return [tuple(g) for g in groups], excludes


# === Collectors ===

class SourceError(Exception):
    pass


def _pubmed_term(term: str) -> str:
    cleaned = " ".join(term.replace('"', " ").replace("[", " ").replace("]", " ").split())
    if not cleaned:
        return ""
    if " " in cleaned or "-" in cleaned:
        cleaned = f'"{cleaned}"'
    return f"({cleaned}[Title/Abstract] OR {cleaned}[MeSH Terms])"


def _build_pubmed_query(
    term_groups: list[list[str]], date_range: str,
    journals: Optional[list[str]] = None, excludes: Optional[list[str]] = None,
) -> str:
    """AND of OR-groups, optional journal filter, date range and NOT terms."""
    parts = []
    for group in term_groups:
        ors = [t for t in (_pubmed_term(term) for term in group) if t]
        if ors:
            parts.append(f"({' OR '.join(ors)})")
    if journals:
        journal_terms = [f'"{j.strip()}"[Journal]' for j in journals if j.strip()]
        if journal_terms:
            parts.append(f"({' OR '.join(journal_terms)})")
    parts.append(date_range)
    query = " AND ".join(parts)
    for term in excludes or []:
        t = _pubmed_term(term)
        if t:
            query += f" NOT {t}"
    return query


def _fetch_pubmed_sync(
    days: int, max_results: int, term_groups: list[list[str]],
    journals: Optional[list[str]], excludes: Optional[list[str]] = None,
) -> list[Paper]:
    """PubMed E-utilities search + fetch (urllib is more reliable on serverless)."""
    import urllib.parse
    import urllib.request

    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    date_range = f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[edat]"
    try:
        term_query = _build_pubmed_query(term_groups, date_range, journals, excludes)
        search_req = urllib.request.Request(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            data=urllib.parse.urlencode({
                "db": "pubmed", "term": term_query, "retmax": max_results, "retmode": "json", "sort": "pub_date",
            }).encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(search_req, timeout=20) as resp:
            ids = json.loads(resp.read()).get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetch_req = urllib.request.Request(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            data=urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids[:max_results]), "retmode": "xml"}).encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(fetch_req, timeout=20) as resp:
            root = ET.fromstring(resp.read().decode("utf-8"))
    except Exception as e:  # network, HTTP or XML errors
        raise SourceError(f"PubMed: {e}") from e

    papers = []
    for article in root.findall(".//PubmedArticle"):
        try:
            paper = _parse_pubmed_article(article)
            if paper:
                papers.append(paper)
        except Exception:
            continue
    return papers


async def fetch_pubmed(
    days: int, max_results: int, term_groups: list[list[str]],
    journals: Optional[list[str]] = None, excludes: Optional[list[str]] = None,
) -> list[Paper]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_pubmed_sync, days, max_results, term_groups, journals, excludes)


def _parse_pubmed_article(article: ET.Element) -> Optional[Paper]:
    medline = article.find(".//MedlineCitation")
    if medline is None:
        return None
    article_elem = medline.find(".//Article")
    if article_elem is None:
        return None

    title_elem = article_elem.find(".//ArticleTitle")
    title = " ".join("".join(title_elem.itertext()).split()) if title_elem is not None else ""
    if not title:
        return None

    # Keep structured-abstract labels ("Background: …") so sections stay readable.
    abstract_parts = []
    for abs_text in article_elem.findall(".//Abstract/AbstractText"):
        text = " ".join("".join(abs_text.itertext()).split())
        if not text:
            continue
        label = (abs_text.get("Label") or "").strip()
        abstract_parts.append(f"{label.capitalize()}: {text}" if label and label.upper() != "UNLABELLED" else text)
    abstract = "\n".join(abstract_parts)

    authors = []
    for author in article_elem.findall(".//AuthorList/Author"):
        last = author.findtext("LastName")
        if last:
            initials = author.findtext("Initials") or ""
            authors.append(f"{last} {initials}".strip())
        elif author.findtext("CollectiveName"):
            authors.append(author.findtext("CollectiveName").strip())
    authors_str = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")

    pmid = medline.findtext(".//PMID") or ""
    doi = ""
    for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip()
            break
    if not doi:
        for eloc in article_elem.findall(".//ELocationID"):
            if eloc.get("EIdType") == "doi" and eloc.text:
                doi = eloc.text.strip()
                break
    journal_title_elem = article_elem.find(".//Journal/Title")
    journal_title = "".join(journal_title_elem.itertext()).strip() if journal_title_elem is not None else ""

    return Paper(
        title=title,
        authors=authors_str,
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        published_date=_parse_pubmed_date(article_elem, medline),
        source="pubmed",
        journal=journal_title,
        doi=doi,
        pmid=pmid,
    )


def _parse_pubmed_date(article_elem: ET.Element, medline: ET.Element) -> date:
    """Best-effort pub date parsing for PubMed articles."""
    article_date = article_elem.find(".//ArticleDate")
    if article_date is not None:
        parsed = _date_from_parts(article_date.findtext("Year"), article_date.findtext("Month"), article_date.findtext("Day"))
        if parsed:
            return parsed
    pub_date = medline.find(".//JournalIssue/PubDate")
    if pub_date is not None:
        parsed = _date_from_parts(pub_date.findtext("Year"), pub_date.findtext("Month"), pub_date.findtext("Day"))
        if parsed:
            return parsed
        parsed = _date_from_medline_text(pub_date.findtext("MedlineDate"))
        if parsed:
            return parsed
    return date.today()


def _date_from_medline_text(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    parts = value.replace("-", " ").split()
    year = next((part for part in parts if part.isdigit() and len(part) == 4), None)
    month = next((part for part in parts if part[:3].lower() in {
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    }), None)
    return _date_from_parts(year, month, None)


def _date_from_parts(year: Optional[str], month: Optional[str], day: Optional[str]) -> Optional[date]:
    if not year:
        return None
    try:
        y = int(year)
    except ValueError:
        return None
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    m = 1
    if month:
        try:
            m = int(month)
        except ValueError:
            m = month_map.get(month.strip().lower()[:3], 1)
    d = 1
    if day:
        try:
            d = int(day)
        except ValueError:
            d = 1
    try:
        return date(y, m, d)
    except ValueError:
        return None


RXIV_PAGE = 100  # bioRxiv/medRxiv API page size


async def fetch_rxiv(server: str, days: int, max_results: int, client: httpx.AsyncClient) -> list[Paper]:
    """Fetch bioRxiv or medRxiv preprints, several pages in parallel."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    pages = max(1, min(8, -(-max_results // RXIV_PAGE)))

    async def page(cursor: int) -> list[dict]:
        resp = await client.get(f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/{cursor}")
        resp.raise_for_status()
        return resp.json().get("collection", [])

    results = await asyncio.gather(*(page(i * RXIV_PAGE) for i in range(pages)), return_exceptions=True)
    if results and isinstance(results[0], Exception):
        raise SourceError(f"{SOURCE_LABELS[server]}: {results[0]}")
    label = SOURCE_LABELS[server]
    papers, seen = [], set()
    for items in results:
        if isinstance(items, Exception):
            continue
        for item in items:
            doi = item.get("doi", "")
            if not doi or doi in seen:  # later versions of the same preprint repeat the DOI
                continue
            seen.add(doi)
            try:
                papers.append(Paper(
                    title=" ".join(item.get("title", "").split()),
                    authors=_short_authors(item.get("authors", "")),
                    abstract=" ".join(item.get("abstract", "").split()),
                    url=f"https://www.{server}.org/content/{doi}",
                    published_date=date.fromisoformat(item.get("date", str(date.today()))),
                    source=server,
                    journal=label,
                    doi=doi,
                ))
            except Exception:
                continue
    return papers[:max_results]


async def fetch_biorxiv(days: int, max_results: int, client: httpx.AsyncClient) -> list[Paper]:
    return await fetch_rxiv("biorxiv", days, max_results, client)


def _short_authors(value: str) -> str:
    names = [n.strip() for n in value.split(";") if n.strip()]
    return ", ".join(names[:6]) + (" et al." if len(names) > 6 else "")


async def fetch_arxiv(
    days: int, max_results: int, categories: list[str], client: httpx.AsyncClient,
    term_groups: Optional[list[tuple[str, ...]]] = None,
) -> list[Paper]:
    """Fetch recent arXiv papers in the configured categories (narrowed by typed search terms)."""
    cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
    search_query = f"({cat_query})"
    for group in term_groups or []:
        ors = " OR ".join(f'all:"{t}"' if " " in t else f"all:{t}" for t in group)
        search_query += f" AND ({ors})"
    try:
        resp = await client.get("https://export.arxiv.org/api/query", params={
            "search_query": search_query, "start": 0, "max_results": max_results,
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        resp.raise_for_status()
    except Exception as e:
        raise SourceError(f"arXiv: {e}") from e

    feed = feedparser.parse(resp.text)
    cutoff = date.today() - timedelta(days=days)
    papers = []
    for entry in feed.entries:
        try:
            pub_date = datetime.fromisoformat(entry.get("published", "").replace("Z", "+00:00")).date()
            if pub_date < cutoff:
                continue
            authors = [a.get("name", "") for a in entry.get("authors", [])]
            papers.append(Paper(
                title=" ".join(entry.get("title", "").split()),
                authors=", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""),
                abstract=" ".join(entry.get("summary", "").split()),
                url=entry.get("link", ""),
                published_date=pub_date,
                source="arxiv",
                journal="arXiv",
                doi=entry.get("arxiv_doi", "") or "",
            ))
        except Exception:
            continue
    return papers


# === Scoring ===

@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> "re.Pattern[str]":
    """Case-insensitive, word-start-anchored matcher: ``virus`` matches ``viruses``,
    but ``NGS`` does not match inside ``amongst``."""
    return re.compile(r"\b" + re.escape(term.strip()), re.IGNORECASE)


def _text_contains_term(text: str, term: str) -> bool:
    if not term or not term.strip():
        return False
    return _term_pattern(term).search(text) is not None


def _score_terms(title: str, abstract: str, terms: tuple[str, ...], title_multiplier: float) -> float:
    score = 0.0
    for term in terms:
        if not term or not term.strip():
            continue
        if _text_contains_term(title, term):
            score += title_multiplier
        if _text_contains_term(abstract, term):
            score += 1.0
    return score


def score_paper(paper: Paper, config: Config, interest_filters: Optional[list[InterestFilter]] = None) -> Paper:
    """Score a paper against the configured areas plus any typed search terms."""
    matched_areas = []
    total_score = 0.0
    for area_name, area in config.areas.items():
        area_score = _score_terms(paper.title, paper.abstract, tuple(area.terms), config.title_multiplier)
        if area_score > 0:
            matched_areas.append(area_name)
            total_score += area_score * area.weight
    for interest in interest_filters or []:
        if interest.custom:
            total_score += _score_terms(paper.title, paper.abstract, interest.terms, config.title_multiplier)
    paper.matched_areas = matched_areas
    paper.score = round(total_score, 1)
    return paper


def _normalize_journal(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def is_high_impact(paper: Paper, config: Config) -> bool:
    if not paper.journal or paper.source != "pubmed":
        return False
    paper_norm = _normalize_journal(paper.journal)
    if not paper_norm:
        return False
    alias_map = {"pnas": "procnatlacadsciusa"}
    for journal in config.high_impact_journals:
        journal_norm = _normalize_journal(journal)
        if not journal_norm:
            continue
        if paper_norm == journal_norm or journal_norm in paper_norm or paper_norm in journal_norm:
            return True
        if alias_map.get(journal_norm) == paper_norm or alias_map.get(paper_norm) == journal_norm:
            return True
    return False


def _area_label(area_name: str) -> str:
    return {"ai_ml": "AI / ML", "ngs": "NGS", "viruses_outbreaks": "Viruses / Outbreaks"}.get(
        area_name, area_name.replace("_", " ").title())


def _term_groups_for_search(config: Config, interest_filters: list[InterestFilter]) -> list[list[str]]:
    """PubMed term groups: each filter is one OR-group; with no filters, any configured area term."""
    def dedupe(terms):
        seen, out = set(), []
        for t in terms:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out
    if interest_filters:
        return [dedupe(f.terms) for f in interest_filters]
    return [dedupe(t for area in config.areas.values() for t in area.terms)]


def _paper_matches_interest(paper: Paper, interest: InterestFilter) -> bool:
    if interest.area_names:
        return any(a in paper.matched_areas for a in interest.area_names)
    return any(
        _text_contains_term(paper.title, term) or _text_contains_term(paper.abstract, term)
        for term in interest.terms if term
    )


def _paper_excluded(paper: Paper, excludes: list[str]) -> bool:
    return any(_text_contains_term(paper.title, t) or _text_contains_term(paper.abstract, t) for t in excludes)


# === Main Fetch Function ===

async def fetch_all_papers_for_days(
    config: Config,
    lookback_days: int,
    high_impact_days: int,
    interest_filters: Optional[list[InterestFilter]] = None,
    excludes: Optional[list[str]] = None,
) -> tuple[list[Paper], dict[str, str]]:
    """Fetch from all enabled sources, score and rank. Returns (papers, source status)."""
    interest_filters = interest_filters or []
    excludes = excludes or []
    term_groups = _term_groups_for_search(config, interest_filters)
    custom_groups = [f.terms for f in interest_filters if f.custom]
    cache_key = (
        lookback_days, high_impact_days,
        tuple((f.label.lower(), f.area_names, f.custom, f.terms) for f in interest_filters),
        tuple(e.lower() for e in excludes),
        config.max_results, config.pubmed_enabled, config.biorxiv_enabled, config.medrxiv_enabled,
        config.arxiv_enabled, tuple(config.arxiv_categories),
        tuple((k, v.weight, tuple(v.terms)) for k, v in sorted(config.areas.items())),
        config.title_multiplier, tuple(config.high_impact_journals),
    )
    now = time.time()
    hit = _CACHE.get(cache_key)
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        _CACHE.move_to_end(cache_key)
        return list(hit[1]), dict(hit[2])

    status: dict[str, str] = {}
    tasks: dict[str, "asyncio.Future"] = {}
    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
        if config.biorxiv_enabled:
            tasks["biorxiv"] = asyncio.ensure_future(fetch_rxiv("biorxiv", lookback_days, config.max_results, client))
        if config.medrxiv_enabled:
            tasks["medrxiv"] = asyncio.ensure_future(fetch_rxiv("medrxiv", lookback_days, config.max_results, client))
        if config.arxiv_enabled:
            tasks["arxiv"] = asyncio.ensure_future(
                fetch_arxiv(lookback_days, config.max_results, config.arxiv_categories, client, custom_groups))

        results: list[list[Paper]] = []
        if config.pubmed_enabled:
            # Sequential PubMed calls avoid the E-utilities rate limit.
            pubmed: list[Paper] = []
            errors = []
            try:
                pubmed += await fetch_pubmed(lookback_days, config.max_results, term_groups, None, excludes)
            except Exception as e:
                errors.append(str(e))
            if config.high_impact_journals:
                try:
                    pubmed += await fetch_pubmed(high_impact_days, config.max_results, term_groups,
                                                 config.high_impact_journals, excludes)
                except Exception as e:
                    errors.append(str(e))
            status["pubmed"] = "error" if errors and not pubmed else "ok"
            results.append(pubmed)

        for name, task in tasks.items():
            try:
                results.append(await task)
                status[name] = "ok"
            except Exception as e:
                print(f"{name} failed: {e}")
                status[name] = "error"

    all_papers, seen = [], set()
    for batch in results:
        for paper in batch:
            if paper.url and paper.key not in seen:
                seen.add(paper.key)
                all_papers.append(paper)

    scored = [score_paper(p, config, interest_filters) for p in all_papers]
    relevant = [p for p in scored if p.score > 0 and not _paper_excluded(p, excludes)]
    relevant.sort(key=lambda p: (p.score, p.published_date), reverse=True)

    # Only cache when every source answered, so a transient outage isn't pinned for 5 minutes.
    if all(v == "ok" for v in status.values()):
        _CACHE[cache_key] = (now, list(relevant), dict(status))
        while len(_CACHE) > CACHE_ENTRIES:
            _CACHE.popitem(last=False)
    return relevant, status


# === Feed assembly (shared by HTML and JSON) ===

@dataclass
class FeedParams:
    q: str = ""
    areas: tuple[str, ...] = ()
    scope: str = "high_impact_first"
    source: str = ""
    sort: str = "score"
    days: int = 7


def normalize_params(config: Config, q: str, area: list[str], scope: str, source: str, sort: str,
                     days: int, topic: str = "", method: str = "") -> FeedParams:
    """Validate query params. Legacy ``topic``/``method`` params map to areas or search text."""
    aliases = {}
    for name in config.areas:
        for alias in (name, _area_label(name), name.replace("_", " ")):
            aliases[alias.lower()] = name
    areas = [aliases[a.lower()] for a in area if a and a.lower() in aliases]
    extra = []
    for legacy in (topic, method):
        legacy = " ".join(legacy.split())[:80]
        if not legacy:
            continue
        if legacy.lower() in aliases:
            areas.append(aliases[legacy.lower()])
        else:
            extra.append(f'"{legacy}"' if " " in legacy else legacy)
    q = " ".join([" ".join(q.split())] + extra).strip()[:MAX_QUERY_LENGTH]
    return FeedParams(
        q=q,
        areas=tuple(dict.fromkeys(areas)),
        scope=scope if scope in {"high_impact_first", "high_impact_only", "all"} else "high_impact_first",
        source=source if source in SOURCES else "",
        sort=sort if sort in {"score", "date"} else "score",
        days=days if days in config.other_lookback_options else config.lookback_days,
    )


async def build_feed(config: Config, params: FeedParams) -> dict:
    groups, excludes = parse_query(params.q)
    filters: list[InterestFilter] = []
    if params.areas:
        filters.append(InterestFilter(
            label=" / ".join(_area_label(a) for a in params.areas),
            terms=tuple(t for a in params.areas for t in config.areas[a].terms),
            area_names=params.areas,
        ))
    filters += [InterestFilter(label=" OR ".join(g), terms=g, custom=True) for g in groups]

    papers, status = await fetch_all_papers_for_days(
        config, params.days, config.high_impact_lookback_days, filters, excludes)
    for interest in filters:
        papers = [p for p in papers if _paper_matches_interest(p, interest)]
    if params.source:
        papers = [p for p in papers if p.source == params.source]
    if params.sort == "date":
        papers.sort(key=lambda p: (p.published_date, p.score), reverse=True)
    else:
        papers.sort(key=lambda p: (p.score, p.published_date), reverse=True)

    today = date.today()
    high_cutoff = today - timedelta(days=config.high_impact_lookback_days)
    other_cutoff = today - timedelta(days=params.days)
    high = [p for p in papers if is_high_impact(p, config) and p.published_date >= high_cutoff]
    other = [p for p in papers if not is_high_impact(p, config) and p.published_date >= other_cutoff]
    if params.scope == "high_impact_only":
        other = []
    elif params.scope == "all":
        shown = {id(p) for p in high} | {id(p) for p in other}
        merged = [p for p in papers if id(p) in shown][:MAX_DISPLAY]
        return _feed_dict(params, [], merged, status, groups, excludes, config, merged_view=True)
    return _feed_dict(params, high[:MAX_DISPLAY], other[:MAX_DISPLAY], status, groups, excludes, config)


def _feed_dict(params, high, other, status, groups, excludes, config, merged_view=False) -> dict:
    # Highlight typed terms; with only topic chips selected, show which topic terms matched.
    highlight = [t for g in groups for t in g] or [t for a in params.areas for t in config.areas[a].terms]
    return {
        "params": params, "high": high, "other": other, "merged_view": merged_view,
        "status": status, "groups": groups, "excludes": excludes, "highlight": highlight,
    }


# === Presentation helpers ===

def highlight_text(text: str, terms: list[str]) -> Markup:
    """HTML-escape ``text`` and wrap word-start matches of ``terms`` in <mark>."""
    terms = sorted({t.strip() for t in terms if t and len(t.strip()) > 1}, key=len, reverse=True)
    if not terms:
        return Markup(escape(text))
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(escape(text[last:m.start()]))
        out.append(Markup("<mark>") + escape(m.group(0)) + Markup("</mark>"))
        last = m.end()
    out.append(escape(text[last:]))
    return Markup("").join(out)


def abstract_html(text: str, terms: list[str]) -> Markup:
    """Structured abstracts: one paragraph per section with a bold label."""
    if not text:
        return Markup('<p class="muted">No abstract available.</p>')
    parts = []
    for line in text.split("\n"):
        m = re.match(r"^([A-Z][A-Za-z /&-]{1,40}): (.*)$", line)
        if m:
            parts.append(Markup("<p><b>") + escape(m.group(1)) + Markup("</b> ") + highlight_text(m.group(2), terms) + Markup("</p>"))
        else:
            parts.append(Markup("<p>") + highlight_text(line, terms) + Markup("</p>"))
    return Markup("").join(parts)


def paper_json(paper: Paper) -> str:
    """Compact data for client-side saving and citation (safe inside an HTML attribute)."""
    return json.dumps({
        "title": paper.title, "authors": paper.authors, "journal": paper.journal, "url": paper.url,
        "doi": paper.doi, "pmid": paper.pmid, "date": paper.published_date.isoformat(), "source": paper.source,
    }, ensure_ascii=False)


def _relative_date(d: date) -> str:
    delta = (date.today() - d).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if delta < 7:
        return f"{delta} days ago"
    return f"{d.day} {d:%b %Y}"


# === FastAPI App ===

app = FastAPI(title="PaperFeed")


@app.get("/manifest.webmanifest")
async def manifest():
    return JSONResponse({
        "name": "PaperFeed", "short_name": "PaperFeed",
        "description": "A ranked feed of recent research papers.",
        "start_url": "/", "scope": "/", "display": "standalone",
        "background_color": "#0d1117", "theme_color": "#0d1117",
        "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
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




HTML_TEMPLATE = r"""
{% macro paper_card(paper, index, sid) %}
{% set primary = paper.matched_areas[0] if paper.matched_areas else 'default' %}
<article class="paper{% if index >= page_size %} extra{% endif %}" {% if index >= page_size %}hidden{% endif %}
         style="--area-color: var(--area-{{ primary }}, var(--accent))" data-paper="{{ paper_json(paper) }}">
    <div class="kicker">
        <span class="source src-{{ paper.source }}">{{ source_labels.get(paper.source, paper.source) }}</span>
        {% if paper.journal and paper.source == 'pubmed' %}<span class="journal" title="{{ paper.journal }}">{{ paper.journal }}</span>{% endif %}
        <time datetime="{{ paper.published_date.isoformat() }}" title="{{ paper.published_date.isoformat() }}">{{ relative_date(paper.published_date) }}</time>
        <span class="score" title="Relevance {{ paper.score }}: each topic term counts {{ title_multiplier|round(1) }} in the title and 1 in the abstract, weighted by topic.">{{ paper.score }}</span>
    </div>
    <h3 class="title"><a href="{{ paper.url }}" target="_blank" rel="noopener">{{ hl(paper.title) }}</a></h3>
    {% if paper.authors %}<p class="authors">{{ paper.authors }}</p>{% endif %}
    {% if paper.matched_areas %}
    <div class="tags">{% for a in paper.matched_areas %}<a class="tag area-{{ a }}" href="{{ area_link(a) }}" title="Show only {{ area_label(a) }}">{{ area_label(a) }}</a>{% endfor %}</div>
    {% endif %}
    <div class="abstract" id="abs-{{ sid }}-{{ index }}">{{ abstract_html(paper.abstract, highlight) }}</div>
    <div class="actions">
        {% if paper.abstract and paper.abstract|length > 320 %}<button type="button" class="link-btn toggle-abstract" aria-expanded="false" aria-controls="abs-{{ sid }}-{{ index }}">Full abstract</button>{% endif %}
        <span class="spacer"></span>
        <button type="button" class="icon-btn save-btn" aria-pressed="false" title="Save (s)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4h12v17l-6-4-6 4z"/></svg><span>Save</span></button>
        <button type="button" class="icon-btn cite-btn" title="Copy citation"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg><span>Cite</span></button>
        {% if paper.doi %}<a class="icon-btn" href="https://doi.org/{{ paper.doi }}" target="_blank" rel="noopener" title="Open via DOI"><span>DOI</span></a>{% endif %}
    </div>
</article>
{% endmacro %}
{% macro section(title, papers, sid, note='') %}
<section class="section" aria-labelledby="h-{{ sid }}">
    <h2 id="h-{{ sid }}">{{ title }} <span class="count">{{ papers|length }}</span>{% if note %}<span class="note">{{ note }}</span>{% endif %}</h2>
    {% if papers %}
    <div class="papers">{% for paper in papers %}{{ paper_card(paper, loop.index0, sid) }}{% endfor %}</div>
    {% if papers|length > page_size %}<button type="button" class="more-btn" data-section="{{ sid }}">Show {{ [page_size, papers|length - page_size]|min }} more <span>of {{ papers|length - page_size }} remaining</span></button>{% endif %}
    {% else %}
    <p class="empty-inline">No papers here for this selection.</p>
    {% endif %}
</section>
{% endmacro %}
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#0d1117" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#f6f8fa" media="(prefers-color-scheme: light)">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="PaperFeed">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="description" content="A ranked feed of recent papers from PubMed, bioRxiv, medRxiv and arXiv.">
<title>{% if params.q %}{{ params.q }} · {% endif %}PaperFeed</title>
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0d1117; --bg-2: #161b22; --card: #161b22; --card-hover: #1c2330; --border: #30363d; --border-soft: #21262d;
  --text: #e6edf3; --text-2: #9da7b3; --muted: #7d8590;
  --accent: #58a6ff; --accent-strong: #1f6feb; --accent-soft: #58a6ff1f; --on-accent: #ffffff;
  --mark: #bb800966; --mark-text: #f2cc60; --good: #3fb950;
  --area-metagenomics: #3fb950; --area-viruses_outbreaks: #f85149; --area-ngs: #a371f7;
  --area-ai_ml: #79c0ff; --area-human_genome: #ffa657; --area-bioinformatics: #ff7b72; --area-default: #58a6ff;
  --pubmed: #4493f8; --biorxiv: #f0883e; --medrxiv: #3fb9a0; --arxiv: #f47067;
  --shadow: 0 8px 24px #01040966; color-scheme: dark;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f6f8fa; --bg-2: #ffffff; --card: #ffffff; --card-hover: #fbfcfd; --border: #d0d7de; --border-soft: #e6eaef;
    --text: #1f2328; --text-2: #4b5563; --muted: #656d76;
    --accent: #0969da; --accent-strong: #0969da; --accent-soft: #0969da14; --on-accent: #ffffff;
    --mark: #fff8c5; --mark-text: #1f2328; --good: #1a7f37;
    --area-metagenomics: #1a7f37; --area-viruses_outbreaks: #cf222e; --area-ngs: #8250df;
    --area-ai_ml: #0969da; --area-human_genome: #bc4c00; --area-bioinformatics: #c2410c; --area-default: #0969da;
    --pubmed: #0969da; --biorxiv: #bc4c00; --medrxiv: #0f766e; --arxiv: #b31b1b;
    --shadow: 0 8px 24px #8c959f33; color-scheme: light;
  }
}
* { box-sizing: border-box; }
html { scroll-padding-top: 90px; }
body { margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.55; -webkit-font-smoothing: antialiased; }
a { color: inherit; }
button, input, select { font: inherit; color: inherit; }
button { cursor: pointer; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
mark { background: var(--mark); color: var(--mark-text); border-radius: 3px; padding: 0 1px; }
kbd { font: 500 11px/1 Inter, sans-serif; padding: 3px 5px; border: 1px solid var(--border); border-bottom-width: 2px; border-radius: 4px; color: var(--muted); }

/* Top bar */
.topbar { position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border-soft); }
.topbar-inner { max-width: 1040px; margin: 0 auto; padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
.logo { display: flex; align-items: center; gap: 10px; text-decoration: none; font-weight: 700; font-size: 17px; letter-spacing: -.3px; flex-shrink: 0; }
.logo img { width: 30px; height: 30px; border-radius: 8px; }
.search { flex: 1; display: flex; align-items: center; min-width: 0; height: 42px; background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px; transition: border-color .15s, box-shadow .15s; }
.search:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.search svg { width: 18px; height: 18px; margin-left: 12px; flex-shrink: 0; fill: none; stroke: var(--muted); stroke-width: 2; stroke-linecap: round; }
.search input { flex: 1; width: 0; min-width: 0; height: 100%; padding: 0 10px; border: 0; background: none; outline: none; font-size: 15px; }
.search input::placeholder { color: var(--muted); }
.search input::-webkit-search-cancel-button { cursor: pointer; }
.search kbd { margin-right: 6px; }
.search:focus-within kbd { visibility: hidden; }
.search .go { height: 32px; margin-right: 5px; padding: 0 14px; border: 0; border-radius: 7px; background: var(--accent-strong); color: var(--on-accent); font-weight: 600; font-size: 13px; }
.help-btn, .saved-btn { display: inline-flex; align-items: center; gap: 6px; height: 36px; padding: 0 11px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-2); font-size: 13px; font-weight: 500; color: var(--text-2); flex-shrink: 0; }
.help-btn:hover, .saved-btn:hover { color: var(--text); border-color: var(--muted); }
.saved-btn b { min-width: 18px; padding: 0 5px; border-radius: 9px; background: var(--accent-strong); color: var(--on-accent); font-size: 11px; text-align: center; }
.saved-btn b:empty { display: none; }
.help { max-width: 1040px; margin: 0 auto; padding: 0 20px 12px; display: none; }
.help.open { display: block; }
.help-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 6px 18px; padding: 12px 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg-2); font-size: 13px; color: var(--text-2); }
.help-grid code { color: var(--accent); font-size: 12.5px; }

/* Layout */
.wrap { max-width: 1040px; margin: 0 auto; padding: 18px 20px 60px; }
.filters { display: flex; flex-direction: column; gap: 12px; margin-bottom: 18px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { position: relative; }
.chip input { position: absolute; opacity: 0; pointer-events: none; }
.chip span { display: inline-flex; align-items: center; gap: 7px; height: 30px; padding: 0 12px; border: 1px solid var(--border); border-radius: 15px; font-size: 13px; color: var(--text-2); cursor: pointer; user-select: none; transition: border-color .15s, background .15s, color .15s; }
.chip span::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
.chip:hover span { border-color: var(--c); color: var(--text); }
.chip input:checked + span { background: color-mix(in srgb, var(--c) 16%, transparent); border-color: var(--c); color: var(--text); font-weight: 600; }
.chip input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
.selects { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.selects select { height: 34px; padding: 0 30px 0 11px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-2) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%237d8590' stroke-width='2.5'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E") no-repeat right 10px center; appearance: none; font-size: 13px; cursor: pointer; }
.selects select:hover { border-color: var(--muted); }
.selects .apply { height: 34px; padding: 0 14px; border: 0; border-radius: 8px; background: var(--accent-strong); color: var(--on-accent); font-weight: 600; font-size: 13px; }
.js .selects .apply { display: none; }
.reset { margin-left: auto; font-size: 13px; color: var(--accent); text-decoration: none; }
.reset:hover { text-decoration: underline; }

.summary { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 10px 0 14px; border-bottom: 1px solid var(--border-soft); font-size: 13px; color: var(--text-2); }
.summary strong { color: var(--text); }
.qchip { display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 6px 0 10px; border-radius: 13px; background: var(--accent-soft); color: var(--accent); font-size: 12.5px; font-weight: 500; text-decoration: none; }
.qchip.neg { background: color-mix(in srgb, var(--area-viruses_outbreaks) 14%, transparent); color: var(--area-viruses_outbreaks); }
.qchip i { font-style: normal; opacity: .7; }
.qchip:hover i { opacity: 1; }
.notice { margin: 12px 0 0; padding: 10px 12px; border: 1px solid color-mix(in srgb, var(--biorxiv) 45%, transparent); border-radius: 8px; background: color-mix(in srgb, var(--biorxiv) 10%, transparent); font-size: 13px; }

/* Sections & cards */
.section { margin-top: 22px; }
.section h2 { display: flex; align-items: baseline; gap: 8px; margin: 0 0 10px; font-size: 15px; font-weight: 650; }
.section h2 .count { padding: 1px 8px; border-radius: 10px; background: var(--bg-2); border: 1px solid var(--border-soft); font-size: 12px; font-weight: 600; color: var(--text-2); }
.section h2 .note { font-size: 12px; font-weight: 400; color: var(--muted); }
.papers { display: flex; flex-direction: column; gap: 10px; }
.paper { position: relative; padding: 16px 18px 12px 20px; background: var(--card); border: 1px solid var(--border-soft); border-radius: 12px; transition: border-color .15s, background .15s; }
.paper::before { content: ''; position: absolute; left: 0; top: 14px; bottom: 14px; width: 3px; border-radius: 0 3px 3px 0; background: var(--area-color); }
.paper:hover { border-color: var(--border); background: var(--card-hover); }
.paper.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.kicker { display: flex; align-items: center; gap: 8px; min-width: 0; font-size: 12px; color: var(--muted); }
.source { flex-shrink: 0; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: .3px; }
.src-pubmed { background: color-mix(in srgb, var(--pubmed) 15%, transparent); color: var(--pubmed); }
.src-biorxiv { background: color-mix(in srgb, var(--biorxiv) 15%, transparent); color: var(--biorxiv); }
.src-medrxiv { background: color-mix(in srgb, var(--medrxiv) 15%, transparent); color: var(--medrxiv); }
.src-arxiv { background: color-mix(in srgb, var(--arxiv) 15%, transparent); color: var(--arxiv); }
.journal { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-style: italic; color: var(--text-2); }
.kicker time { flex-shrink: 0; }
.kicker time::before { content: '·'; margin-right: 8px; }
.score { margin-left: auto; flex-shrink: 0; padding: 1px 8px; border-radius: 10px; border: 1px solid color-mix(in srgb, var(--good) 45%, transparent); color: var(--good); font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; cursor: help; }
.title { margin: 6px 0 4px; font-size: 16.5px; font-weight: 600; line-height: 1.38; letter-spacing: -.1px; }
.title a { text-decoration: none; }
.title a:hover { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
.title a:visited { color: color-mix(in srgb, var(--text) 75%, var(--area-ngs)); }
.authors { margin: 0; font-size: 13px; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.tag { padding: 1px 8px; border-radius: 4px; font-size: 11.5px; font-weight: 500; text-decoration: none; background: color-mix(in srgb, var(--c) 14%, transparent); color: var(--c); }
.tag:hover { background: color-mix(in srgb, var(--c) 24%, transparent); }
{% for a in areas %}.area-{{ a }} { --c: var(--area-{{ a }}, var(--accent)); }
{% endfor %}
.abstract { margin-top: 9px; font-size: 13.5px; line-height: 1.65; color: var(--text-2); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.abstract p { margin: 0; display: inline; }
.abstract p + p::before { content: ' '; }
.abstract b { color: var(--text); font-weight: 600; }
.abstract.expanded { display: block; -webkit-line-clamp: unset; color: var(--text); }
.abstract.expanded p { display: block; }
.abstract.expanded p + p { margin-top: 8px; }
.abstract.expanded p + p::before { content: none; }
.muted { color: var(--muted); }
.actions { display: flex; align-items: center; gap: 4px; margin-top: 8px; }
.actions .spacer { flex: 1; }
.link-btn { padding: 4px 0; border: 0; background: none; color: var(--accent); font-size: 13px; font-weight: 500; }
.link-btn:hover { text-decoration: underline; }
.icon-btn { display: inline-flex; align-items: center; gap: 5px; height: 30px; padding: 0 9px; border: 1px solid transparent; border-radius: 7px; background: none; color: var(--muted); font-size: 12.5px; font-weight: 500; text-decoration: none; }
.icon-btn:hover { background: var(--bg); border-color: var(--border-soft); color: var(--text); }
.icon-btn svg { width: 15px; height: 15px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linejoin: round; }
.save-btn[aria-pressed="true"] { color: var(--accent); }
.save-btn[aria-pressed="true"] svg { fill: currentColor; }
.more-btn { display: block; width: 100%; margin-top: 10px; padding: 10px; border: 1px dashed var(--border); border-radius: 10px; background: none; color: var(--accent); font-weight: 600; font-size: 13px; }
.more-btn span { font-weight: 400; color: var(--muted); }
.more-btn:hover { border-style: solid; background: var(--accent-soft); }
.empty-inline { margin: 0; padding: 14px 16px; border: 1px dashed var(--border); border-radius: 10px; color: var(--muted); font-size: 13px; }
.empty { margin-top: 24px; padding: 40px 20px; text-align: center; border: 1px dashed var(--border); border-radius: 12px; color: var(--text-2); }
.empty h2 { margin: 0 0 6px; font-size: 17px; color: var(--text); }
.empty p { margin: 0 auto 14px; max-width: 46ch; }
.empty a { display: inline-block; margin: 3px; padding: 6px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 13px; text-decoration: none; color: var(--accent); }
.empty a:hover { border-color: var(--accent); }
footer { margin-top: 36px; padding-top: 14px; border-top: 1px solid var(--border-soft); display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; font-size: 12px; color: var(--muted); }
footer kbd { margin: 0 2px 0 8px; }

/* Loading state (server fetches can take several seconds) */
.progress { position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 20; overflow: hidden; opacity: 0; transition: opacity .2s; }
.progress::after { content: ''; position: absolute; inset: 0; width: 40%; background: var(--accent); animation: slide 1.1s ease-in-out infinite; }
@keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(260%); } }
.loading .progress { opacity: 1; }
.loading main { opacity: .45; pointer-events: none; transition: opacity .2s .15s; }
.loading-note { display: none; }
.loading .loading-note { display: inline; }

/* Saved drawer */
dialog.saved { width: min(560px, 100vw); max-width: 100vw; height: 100dvh; max-height: 100dvh; margin: 0 0 0 auto; padding: 0; border: 0; border-left: 1px solid var(--border); background: var(--bg); color: var(--text); box-shadow: var(--shadow); }
dialog.saved::backdrop { background: #01040999; }
.saved-head { position: sticky; top: 0; display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--border-soft); background: var(--bg); }
.saved-head h2 { flex: 1; margin: 0; font-size: 16px; }
.saved-list { list-style: none; margin: 0; padding: 8px 18px 24px; }
.saved-list li { padding: 12px 0; border-bottom: 1px solid var(--border-soft); }
.saved-list a { font-weight: 600; text-decoration: none; }
.saved-list a:hover { color: var(--accent); }
.saved-list small { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; }
.saved-list .row { display: flex; gap: 4px; margin-top: 4px; }
.saved-empty { padding: 30px 18px; color: var(--muted); }
.toast { position: fixed; left: 50%; bottom: 22px; transform: translate(-50%, 20px); padding: 9px 14px; border-radius: 8px; background: var(--text); color: var(--bg); font-size: 13px; font-weight: 500; opacity: 0; pointer-events: none; transition: opacity .2s, transform .2s; z-index: 30; }
.toast.show { opacity: 1; transform: translate(-50%, 0); }

@media (max-width: 720px) {
  .topbar-inner { flex-wrap: wrap; gap: 10px; padding: 10px 14px; }
  .logo { order: 1; }
  .top-actions { order: 2; margin-left: auto; display: flex; gap: 6px; }
  .search { order: 3; flex-basis: 100%; }
  .search kbd, .help-btn .label, .saved-btn .label { display: none; }
  .wrap { padding: 14px 14px 50px; }
  .chips { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; margin: 0 -14px; padding: 0 14px; }
  .chip span { white-space: nowrap; }
  .selects select { flex: 1 1 40%; min-width: 0; }
  .reset { margin-left: 0; }
  .paper { padding: 14px 14px 10px 16px; }
  .title { font-size: 15.5px; }
  .journal { display: none; }
  .icon-btn span { display: none; }
  .icon-btn { width: 34px; justify-content: center; padding: 0; }
  .icon-btn:not(:has(svg)) span { display: inline; }
  .icon-btn:not(:has(svg)) { width: auto; padding: 0 8px; }
  footer .keys { display: none; }
}
@media (min-width: 721px) { .top-actions { display: flex; gap: 6px; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
<script>document.documentElement.classList.add('js')</script>
</head>
<body>
<div class="progress" aria-hidden="true"></div>
<form method="get" action="/" id="feed-form" role="search">
<header class="topbar">
    <div class="topbar-inner">
        <a class="logo" href="/" aria-label="PaperFeed home"><img src="/icon.svg" alt="">PaperFeed</a>
        <label class="search">
            <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></svg>
            <span class="sr-only">Search papers</span>
            <input type="search" name="q" id="q" value="{{ params.q }}" maxlength="{{ max_query }}" autocomplete="off" spellcheck="false"
                   placeholder='Search, e.g. nanopore OR illumina "wastewater" -plasmid'>
            <kbd aria-hidden="true">/</kbd>
            <button class="go" type="submit">Search</button>
        </label>
        <div class="top-actions">
            <button type="button" class="help-btn" aria-expanded="false" aria-controls="help"><span aria-hidden="true">?</span><span class="label">Syntax</span></button>
            <button type="button" class="saved-btn" id="open-saved"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 4h12v17l-6-4-6 4z"/></svg><span class="label">Saved</span><b id="saved-count"></b></button>
        </div>
    </div>
    <div class="help" id="help">
        <div class="help-grid">
            <div><code>nanopore</code> word or prefix (matches <i>nanopores</i>)</div>
            <div><code>"long read"</code> exact phrase</div>
            <div><code>nanopore OR illumina</code> either term</div>
            <div><code>-plasmid</code> exclude a term</div>
            <div>Several terms must <b>all</b> match</div>
            <div>Topic chips narrow to any of the chosen topics</div>
        </div>
    </div>
</header>

<main class="wrap" id="main">
    <div class="filters">
        <div class="chips" role="group" aria-label="Topics">
            {% for a in areas %}
            <label class="chip area-{{ a }}"><input type="checkbox" name="area" value="{{ a }}" {% if a in params.areas %}checked{% endif %}><span>{{ area_label(a) }}</span></label>
            {% endfor %}
        </div>
        <div class="selects">
            <label class="sr-only" for="days">Date range</label>
            <select name="days" id="days">{% for d in lookback_options %}<option value="{{ d }}" {% if params.days == d %}selected{% endif %}>Last {{ d }} days</option>{% endfor %}</select>
            <label class="sr-only" for="sort">Sort</label>
            <select name="sort" id="sort">
                <option value="score" {% if params.sort == 'score' %}selected{% endif %}>Most relevant</option>
                <option value="date" {% if params.sort == 'date' %}selected{% endif %}>Newest first</option>
            </select>
            <label class="sr-only" for="scope">Journals</label>
            <select name="scope" id="scope">
                <option value="high_impact_first" {% if params.scope == 'high_impact_first' %}selected{% endif %}>High-impact first</option>
                <option value="high_impact_only" {% if params.scope == 'high_impact_only' %}selected{% endif %}>High-impact only</option>
                <option value="all" {% if params.scope == 'all' %}selected{% endif %}>One combined list</option>
            </select>
            <label class="sr-only" for="source">Source</label>
            <select name="source" id="source">
                <option value="">All sources</option>
                {% for s in enabled_sources %}<option value="{{ s }}" {% if params.source == s %}selected{% endif %}>{{ source_labels[s] }}</option>{% endfor %}
            </select>
            <button class="apply" type="submit">Apply</button>
            {% if modified %}<a class="reset" href="/?reset=1" id="reset">Reset all</a>{% endif %}
        </div>
    </div>

    <div class="summary" role="status">
        <span class="loading-note">Fetching from {{ enabled_sources|map('source_label')|join(', ') }}…</span>
        <span><strong>{{ total }}</strong> {{ 'paper' if total == 1 else 'papers' }}</span>
        {% for g in groups %}<a class="qchip" href="{{ without_term(g) }}" title="Remove">{{ g|join(' OR ') }} <i aria-hidden="true">×</i><span class="sr-only">remove</span></a>{% endfor %}
        {% for e in excludes %}<a class="qchip neg" href="{{ without_exclude(e) }}" title="Remove">−{{ e }} <i aria-hidden="true">×</i></a>{% endfor %}
        {% for a in params.areas %}<a class="qchip" href="{{ area_toggle(a) }}" title="Remove">{{ area_label(a) }} <i aria-hidden="true">×</i></a>{% endfor %}
        <span class="muted">Last {{ params.days }} days{% if params.scope != 'all' %} · high-impact journals: {{ high_impact_days }} days{% endif %}</span>
    </div>
    {% if failed %}<p class="notice" role="alert">{{ failed|join(', ') }} didn’t respond, so {{ 'its' if failed|length == 1 else 'their' }} papers are missing. Reload to try again.</p>{% endif %}

    {% if total %}
        {% if merged_view %}
            {{ section('All ranked papers', other, 'all') }}
        {% else %}
            {% if params.scope != 'all' %}{{ section('High-impact journals', high, 'high', 'last ' ~ high_impact_days ~ ' days') }}{% endif %}
            {% if params.scope == 'high_impact_first' %}{{ section('Other papers', other, 'other') }}{% endif %}
        {% endif %}
    {% else %}
        <div class="empty">
            <h2>No papers match</h2>
            <p>{% if groups or excludes or params.areas or params.source %}Try fewer terms, remove a topic, or widen the date range.{% else %}Nothing was retrieved for this period. Try a longer date range.{% endif %}</p>
            {% if params.days < lookback_options[-1] %}<a href="{{ with_days(lookback_options[-1]) }}">Search the last {{ lookback_options[-1] }} days</a>{% endif %}
            {% if params.areas or groups or excludes or params.source %}<a href="/?reset=1">Clear all filters</a>{% endif %}
        </div>
    {% endif %}

    <footer>
        <span>Sources: {{ enabled_sources|map('source_label')|join(' · ') }}. Relevance = topic-term matches (title ×{{ title_multiplier|round(1) }}, abstract ×1).</span>
        <span class="keys"><kbd>/</kbd> search <kbd>j</kbd><kbd>k</kbd> move <kbd>o</kbd> open <kbd>s</kbd> save</span>
    </footer>
</main>
</form>

<dialog class="saved" id="saved" aria-labelledby="saved-title">
    <div class="saved-head">
        <h2 id="saved-title">Saved papers</h2>
        <button type="button" class="icon-btn" id="export-saved" title="Copy all as citations"><span>Copy all</span></button>
        <button type="button" class="icon-btn" id="close-saved" aria-label="Close"><span aria-hidden="true" style="display:inline;font-size:18px">×</span></button>
    </div>
    <ul class="saved-list" id="saved-list"></ul>
</dialog>
<div class="toast" id="toast" role="status" aria-live="polite"></div>

<script>
(() => {
  const form = document.getElementById('feed-form');
  const q = document.getElementById('q');
  const KEY_FILTERS = 'paperfeed.filters.v2', KEY_SAVED = 'paperfeed.saved.v1';
  const store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d } catch { return d } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)) } catch {} },
    del(k) { try { localStorage.removeItem(k) } catch {} },
  };
  const toast = (msg) => { const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 1800) };
  const params = new URLSearchParams(location.search);

  // Remember filters; restore them on a bare visit. ?reset=1 clears them.
  if (params.has('reset')) { store.del(KEY_FILTERS); history.replaceState(null, '', '/') }
  else if (!location.search) {
    const saved = store.get(KEY_FILTERS, null);
    if (saved && saved.length) { location.replace('/?' + saved); return }
  } else store.set(KEY_FILTERS, location.search.slice(1));

  // Loading feedback: fetching the sources can take several seconds.
  const busy = () => document.body.classList.add('loading');
  window.addEventListener('pageshow', () => document.body.classList.remove('loading'));
  form.addEventListener('submit', () => {
    // Keep URLs tidy: drop empty fields.
    for (const el of form.elements) if (el.name && !el.value && el.tagName !== 'BUTTON') el.disabled = true;
    busy();
  });
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="/"]');
    if (a && !e.metaKey && !e.ctrlKey && !e.shiftKey && a.target !== '_blank') busy();
  });
  // Filters apply immediately.
  form.querySelectorAll('select, .chip input').forEach(el => el.addEventListener('change', () => form.requestSubmit()));

  // Syntax help
  const helpBtn = document.querySelector('.help-btn'), help = document.getElementById('help');
  helpBtn.addEventListener('click', () => { const open = help.classList.toggle('open'); helpBtn.setAttribute('aria-expanded', open) });

  // Abstract expand
  document.querySelectorAll('.toggle-abstract').forEach(btn => btn.addEventListener('click', () => {
    const abs = document.getElementById(btn.getAttribute('aria-controls'));
    const open = abs.classList.toggle('expanded');
    btn.setAttribute('aria-expanded', open); btn.textContent = open ? 'Show less' : 'Full abstract';
  }));

  // Show more
  document.querySelectorAll('.more-btn').forEach(btn => btn.addEventListener('click', () => {
    const hidden = [...btn.previousElementSibling.querySelectorAll('.paper[hidden]')];
    hidden.slice(0, {{ page_size }}).forEach(p => p.hidden = false);
    const left = hidden.length - {{ page_size }};
    if (left <= 0) btn.remove(); else btn.innerHTML = `Show ${Math.min({{ page_size }}, left)} more <span>of ${left} remaining</span>`;
    hidden[0]?.querySelector('.title a')?.focus({ preventScroll: true });
  }));

  // Citations
  const citation = (p) => {
    const year = (p.date || '').slice(0, 4);
    const link = p.doi ? `https://doi.org/${p.doi}` : p.url;
    return `${p.authors.replace(/\.$/, '')} (${year}). ${p.title.replace(/\.$/, '')}. ${p.journal}. ${link}`;
  };
  const copy = async (text) => {
    try { await navigator.clipboard.writeText(text); return true } catch {
      const t = Object.assign(document.createElement('textarea'), { value: text }); document.body.append(t); t.select();
      const ok = document.execCommand('copy'); t.remove(); return ok;
    }
  };

  // Saved papers (this browser only)
  let saved = store.get(KEY_SAVED, []);
  const keyOf = (p) => (p.doi || p.url || '').toLowerCase();
  const isSaved = (p) => saved.some(s => keyOf(s) === keyOf(p));
  const paint = () => {
    document.getElementById('saved-count').textContent = saved.length || '';
    document.querySelectorAll('.paper').forEach(card => {
      const on = isSaved(JSON.parse(card.dataset.paper));
      const b = card.querySelector('.save-btn'); b.setAttribute('aria-pressed', on); b.querySelector('span').textContent = on ? 'Saved' : 'Save';
    });
  };
  const toggleSave = (p) => {
    saved = isSaved(p) ? saved.filter(s => keyOf(s) !== keyOf(p)) : [p, ...saved].slice(0, 500);
    store.set(KEY_SAVED, saved); paint(); renderSaved();
    toast(isSaved(p) ? 'Saved' : 'Removed from saved');
  };
  document.querySelectorAll('.paper').forEach(card => {
    const p = JSON.parse(card.dataset.paper);
    card.querySelector('.save-btn').addEventListener('click', () => toggleSave(p));
    card.querySelector('.cite-btn').addEventListener('click', async () => { if (await copy(citation(p))) toast('Citation copied') });
  });
  const dlg = document.getElementById('saved'), list = document.getElementById('saved-list');
  const esc = (s) => String(s || '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function renderSaved() {
    list.innerHTML = saved.length ? saved.map((p, i) => `<li><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a>
      <small>${esc(p.authors)} · ${esc(p.journal)} · ${esc(p.date)}</small>
      <div class="row"><button type="button" class="icon-btn" data-cite="${i}"><span>Cite</span></button><button type="button" class="icon-btn" data-remove="${i}"><span>Remove</span></button></div></li>`).join('')
      : '<li class="saved-empty">Nothing saved yet. Use <b>Save</b> on any paper (or press <kbd>s</kbd>). Saved papers stay in this browser.</li>';
  }
  list.addEventListener('click', async (e) => {
    const c = e.target.closest('[data-cite]'), r = e.target.closest('[data-remove]');
    if (c && await copy(citation(saved[+c.dataset.cite]))) toast('Citation copied');
    if (r) toggleSave(saved[+r.dataset.remove]);
  });
  document.getElementById('open-saved').addEventListener('click', () => { renderSaved(); dlg.showModal() });
  document.getElementById('close-saved').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close() });
  document.getElementById('export-saved').addEventListener('click', async () => {
    if (saved.length && await copy(saved.map(citation).join('\n\n'))) toast(`${saved.length} citations copied`);
  });
  paint();

  // Keyboard: / search · j/k move · o open · s save · Esc
  let active = -1;
  const visible = () => [...document.querySelectorAll('.paper:not([hidden])')];
  document.addEventListener('keydown', (e) => {
    const typing = /INPUT|SELECT|TEXTAREA/.test(e.target.tagName);
    if (e.key === 'Escape' && typing) { e.target.blur(); return }
    if (typing || e.metaKey || e.ctrlKey || e.altKey || dlg.open) return;
    if (e.key === '/') { e.preventDefault(); q.focus(); q.select(); return }
    const cards = visible(); if (!cards.length) return;
    if (e.key === 'j' || e.key === 'k') {
      e.preventDefault();
      active = Math.max(0, Math.min(cards.length - 1, active + (e.key === 'j' ? 1 : -1)));
      cards.forEach((c, i) => c.classList.toggle('active', i === active));
      cards[active].scrollIntoView({ block: 'nearest' });
      cards[active].querySelector('.title a').focus({ preventScroll: true });
    } else if (e.key === 'o' && cards[active]) {
      window.open(cards[active].querySelector('.title a').href, '_blank', 'noopener');
    } else if (e.key === 's' && cards[active]) {
      toggleSave(JSON.parse(cards[active].dataset.paper));
    }
  });
})();
</script>
</body>
</html>
"""

_jinja_env = Environment(autoescape=select_autoescape(["html", "xml"]))
_jinja_env.filters["source_label"] = lambda s: SOURCE_LABELS.get(s, s)
_template = _jinja_env.from_string(HTML_TEMPLATE)


def _url(params: FeedParams, **changes) -> str:
    """Link to the feed with some parameters changed."""
    from urllib.parse import urlencode
    data = {"q": params.q, "area": list(params.areas), "days": params.days, "sort": params.sort,
            "scope": params.scope, "source": params.source}
    data.update(changes)
    pairs = []
    for k, v in data.items():
        for item in (v if isinstance(v, (list, tuple)) else [v]):
            if item not in ("", None):
                pairs.append((k, item))
    return "/?" + urlencode(pairs) if pairs else "/?reset=1"


def _requote(groups: list[tuple[str, ...]], excludes: list[str]) -> str:
    fmt = lambda t: f'"{t}"' if " " in t else t
    parts = [" OR ".join(fmt(t) for t in g) for g in groups] + [f"-{fmt(e)}" for e in excludes]
    return " ".join(parts)


def _enabled_sources(config: Config) -> list[str]:
    flags = {"pubmed": config.pubmed_enabled, "biorxiv": config.biorxiv_enabled,
             "medrxiv": config.medrxiv_enabled, "arxiv": config.arxiv_enabled}
    return [s for s in SOURCES if flags[s]]


def _query_args(request: Request) -> dict:
    qp = request.query_params
    try:
        days = int(qp.get("days", "0") or 0)
    except ValueError:
        days = 0
    return dict(q=qp.get("q", ""), area=qp.getlist("area"), scope=qp.get("scope", "high_impact_first"),
                source=qp.get("source", ""), sort=qp.get("sort", "score"), days=days,
                topic=qp.get("topic", ""), method=qp.get("method", ""))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    config = load_config()
    params = normalize_params(config, **_query_args(request))
    feed = await build_feed(config, params)
    groups, excludes = feed["groups"], feed["excludes"]
    terms = feed["highlight"]

    def area_toggle(a: str) -> str:
        areas = [x for x in params.areas if x != a] if a in params.areas else [*params.areas, a]
        return _url(params, area=areas)

    html = _template.render(
        params=params, high=feed["high"], other=feed["other"], merged_view=feed["merged_view"],
        total=len(feed["high"]) + len(feed["other"]), groups=groups, excludes=excludes, highlight=terms,
        areas=list(config.areas), area_label=_area_label,
        area_link=lambda a: _url(params, area=[a]), area_toggle=area_toggle,
        without_term=lambda g: _url(params, q=_requote([x for x in groups if x != g], excludes)),
        without_exclude=lambda e: _url(params, q=_requote(groups, [x for x in excludes if x != e])),
        with_days=lambda d: _url(params, days=d),
        hl=lambda text: highlight_text(text, terms), abstract_html=abstract_html, paper_json=paper_json,
        relative_date=_relative_date, source_labels=SOURCE_LABELS, enabled_sources=_enabled_sources(config),
        failed=[SOURCE_LABELS[s] for s, v in feed["status"].items() if v != "ok"],
        lookback_options=config.other_lookback_options, high_impact_days=config.high_impact_lookback_days,
        title_multiplier=config.title_multiplier, page_size=PAGE_SIZE, max_query=MAX_QUERY_LENGTH,
        modified=bool(params.q or params.areas or params.source or params.sort != "score"
                      or params.scope != "high_impact_first" or params.days != config.lookback_days),
    )
    return HTMLResponse(html)


@app.get("/api/papers")
async def api_papers(request: Request):
    """The same feed as JSON (same query parameters as ``/``)."""
    config = load_config()
    params = normalize_params(config, **_query_args(request))
    feed = await build_feed(config, params)
    return {
        "params": asdict(params), "sources": feed["status"],
        "high_impact": [p.to_dict() for p in feed["high"]],
        "other": [p.to_dict() for p in feed["other"]],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/debug")
async def debug():
    """Check PubMed connectivity and show the generated query."""
    import urllib.parse
    import urllib.request

    config = load_config()
    term_groups = _term_groups_for_search(config, [])
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    query = _build_pubmed_query([term_groups[0][:5]], f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[edat]")
    results = {"terms_count": len(term_groups[0]), "query_preview": query[:200],
               "high_impact_journals": config.high_impact_journals[:5]}
    try:
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
               f"{urllib.parse.quote(query)}&retmax=10&retmode=json")
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        results["pubmed_count"] = data.get("esearchresult", {}).get("count", "N/A")
        results["pubmed_ids"] = len(data.get("esearchresult", {}).get("idlist", []))
        results["pubmed_status"] = "OK"
    except Exception as e:
        results["pubmed_status"] = f"ERROR: {e}"
    return results

"""Offline tests: python -m pytest tests  (collectors are replaced by fixtures)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402
import fixtures  # noqa: E402


@pytest.fixture()
def client():
    fixtures.install()
    return TestClient(app.app)


def test_parse_query_phrases_or_and_exclusions():
    groups, excludes = app.parse_query('"long read" nanopore OR illumina -plasmid -"mouse model"')
    assert groups == [("long read",), ("nanopore", "illumina")]
    assert excludes == ["plasmid", "mouse model"]
    assert app.parse_query("OR virus") == ([("virus",)], [])
    assert app.parse_query('a[Title] (b)') == ([("a Title",), ("b",)], [])


def test_pubmed_query_ands_groups_and_excludes():
    q = app._build_pubmed_query([["long read"], ["nanopore", "illumina"]], "2026/09/01:2026/09/08[edat]",
                                ["Nature"], ["plasmid"])
    assert q.startswith('(("long read"[Title/Abstract] OR "long read"[MeSH Terms])) AND ((nanopore[Title/Abstract]')
    assert '("Nature"[Journal])' in q
    assert q.endswith("NOT (plasmid[Title/Abstract] OR plasmid[MeSH Terms])")


def test_feed_renders_sections_and_highlights(client):
    r = client.get("/?q=nanopore")
    assert r.status_code == 200
    assert "High-impact journals" in r.text and "Other papers" in r.text
    assert "<mark>nanopore</mark>" in r.text or "<mark>Nanopore</mark>" in r.text


def test_exclusions_and_areas_filter_results(client):
    everything = client.get("/api/papers").json()
    total = len(everything["high_impact"]) + len(everything["other"])
    no_influenza = client.get("/api/papers", params={"q": "-influenza"}).json()
    titles = [p["title"] for p in no_influenza["high_impact"] + no_influenza["other"]]
    assert titles and all("influenza" not in t.lower() for t in titles)
    ai = client.get("/api/papers", params={"area": "ai_ml", "scope": "all"}).json()
    assert ai["other"] and all("ai_ml" in p["matched_areas"] for p in ai["other"])
    assert len(ai["other"]) < total


def test_legacy_topic_and_method_params_still_work(client):
    r = client.get("/?topic=Metagenomics&method=nanopore")
    assert 'value="nanopore"' in r.text
    assert 'value="metagenomics" checked' in r.text


def test_source_failure_is_reported_not_silent(client):
    async def broken(*args, **kwargs):
        raise app.SourceError("arXiv: down")
    app.fetch_arxiv = broken
    app._CACHE.clear()
    r = client.get("/")
    assert "arXiv didn’t respond" in r.text


def test_highlight_escapes_html():
    out = app.highlight_text("<b>virus</b> & viruses", ["virus"])
    assert "<b>" not in out and "&lt;b&gt;" in out
    assert out.count("<mark>") == 2


def test_invalid_params_fall_back_to_defaults(client):
    r = client.get("/api/papers", params={"days": "abc", "scope": "x", "sort": "y", "source": "z", "area": "nope"})
    p = r.json()["params"]
    assert (p["days"], p["scope"], p["sort"], p["source"], p["areas"]) == (7, "high_impact_first", "score", "", [])


def test_pubmed_parser_keeps_labels_doi_and_initials():
    import xml.etree.ElementTree as ET
    xml = """<PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
      <Journal><Title>Nature Microbiology</Title></Journal><ArticleTitle>Nanopore <i>metagenomics</i></ArticleTitle>
      <Abstract><AbstractText Label="BACKGROUND">Why.</AbstractText><AbstractText Label="RESULTS">What.</AbstractText></Abstract>
      <AuthorList><Author><LastName>Silva</LastName><Initials>A</Initials></Author></AuthorList>
      <ArticleDate><Year>2026</Year><Month>09</Month><Day>01</Day></ArticleDate></Article></MedlineCitation>
      <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle>"""
    p = app._parse_pubmed_article(ET.fromstring(xml))
    assert p.title == "Nanopore metagenomics"
    assert p.abstract == "Background: Why.\nResults: What."
    assert (p.doi, p.pmid, p.authors, str(p.published_date)) == ("10.1/x", "123", "Silva A", "2026-09-01")
    html = app.abstract_html(p.abstract, [])
    assert "<b>Background</b>" in html

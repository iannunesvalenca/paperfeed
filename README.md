# PaperFeed

A compact, self-hosted feed of recent research papers, ranked by topics you care about.

PaperFeed pulls the latest papers from **PubMed**, **bioRxiv**, and **arXiv**, scores each
one against keyword "areas" you define, and renders a single dark-themed page that separates
**high-impact-journal** papers from everything else. You can filter by interest, source, and
date range; your filter choices are remembered in the browser.

It is a single FastAPI app (`app.py`) designed to deploy on Vercel with zero extra config.

## How it works

1. **Collect** — for each enabled source, fetch papers published in the lookback window:
   - **PubMed** via the NCBI E-utilities API (a keyword search plus a separate
     high-impact-journal search with a longer lookback).
   - **bioRxiv** via its details API.
   - **arXiv** via its Atom API, restricted to the configured categories.
2. **Score** — each paper is matched against the terms in every area in `config.yaml`.
   A title hit is worth `title_multiplier` points; an abstract hit is worth 1. Area scores
   are multiplied by the area's `weight`. Matching is **word-start anchored**, so `virus`
   still matches `viruses` but short acronyms like `NGS` don't match inside unrelated words.
3. **Rank & split** — papers with a score above zero are sorted by score (then recency) and
   split into a "High-impact journals" section and an "Other papers" section.
4. **Filter (in the browser)** — interest, source, sort, and date-range controls re-query the
   server; selections persist in `localStorage`.

Results are cached in memory for 5 minutes per filter combination (best-effort on serverless).

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | The entire application: collectors, scoring, FastAPI routes, and the HTML template. |
| `config.yaml` | Research areas/terms, sources, lookback windows, and the high-impact journal list. |
| `requirements.txt` | Python dependencies. |

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml):

- **`areas`** — named topic groups, each with a list of `terms` and an optional `weight`
  (default `1.0`). These drive both the PubMed search and the relevance score.
- **`sources`** — toggle `pubmed` / `biorxiv` / `arxiv`, and set arXiv `categories`.
- **`collection`** — `lookback_days` (default window) and `max_results` per source.
- **`high_impact_journals`** — journal names that get their own section and a longer lookback
  (`high_impact_lookback_days`).
- **`other_lookback_options`** — the date-range choices shown in the UI dropdown.
- **`scoring.title_multiplier`** — how much more a title hit counts than an abstract hit.

> If you add a new area, give it a matching CSS color in `app.py` (`--area-<name>` and the
> `.area-<name>` rule) so its tag renders with a distinct color; otherwise it falls back to the
> accent color.

## Local development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000.

Useful endpoints:

- `GET /` — the feed (accepts `topic`, `method`, `scope`, `source`, `sort`, `days` query params).
- `GET /api/health` — liveness check.
- `GET /api/debug` — checks PubMed connectivity and shows the generated query.

## Deployment (Vercel)

The repo uses Vercel's zero-config FastAPI support: it detects `requirements.txt` and the
`app` ASGI variable in `app.py`. Import the repo into Vercel and deploy — no `vercel.json`
needed. (`config.yaml` is read at runtime relative to `app.py`, so it ships with the function.)

## Limitations / ideas

- The in-memory cache resets on every cold start; a shared cache (e.g. Vercel KV) would make it
  more effective across invocations.
- bioRxiv is fetched a single page at a time, so very high `max_results` values won't all arrive.
- Scoring is keyword-based; an embedding-based relevance pass could improve ranking quality.

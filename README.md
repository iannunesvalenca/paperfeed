# PaperFeed

A compact feed of recent research papers, ranked by topics you care about.

**🔗 Live app: [paperfeed-app.vercel.app](https://paperfeed-app.vercel.app/)**

PaperFeed pulls the latest papers from **PubMed**, **bioRxiv**, **medRxiv** and **arXiv**, scores
each one against keyword "areas" you define, and renders a single page (dark or light, following
your system) that separates **high-impact-journal** papers from everything else.

## Using it

- **Search box** — plain words must all match (prefixes count: `virus` finds *viruses*).
  `"long read"` matches a phrase, `nanopore OR illumina` matches either, `-plasmid` excludes.
  Typed terms narrow the PubMed and arXiv queries and add to the relevance score.
- **Topic chips** — pick one or more research areas from `config.yaml`; papers must match any of them.
- **Filters** — date range, relevance or newest first, high-impact first / only / one combined list,
  and source. Filters apply as soon as you change them; active terms appear as removable chips.
- **Reading** — matched terms are highlighted, structured abstracts keep their sections
  (Background, Methods, …), and **Full abstract** expands in place. Up to 200 ranked papers load,
  25 at a time per section with **Show more**.
- **Save & cite** — **Save** keeps papers in this browser (the **Saved** panel lists them);
  **Cite** copies a reference, **Copy all** copies every saved reference. **DOI** opens the publisher.
- **Keyboard** — `/` search, `j`/`k` move between papers, `o` open, `s` save.
- If a source fails, a notice says which one instead of silently showing fewer papers.
- Your last filters are remembered in the browser; **Reset all** clears them. Every view has a
  shareable URL (old `?topic=`/`?method=` links still work).

It is a single FastAPI app (`app.py`) designed to deploy on Vercel with zero extra config.

## How it works

1. **Collect** — for each enabled source, fetch papers published in the lookback window:
   - **PubMed** via the NCBI E-utilities API (a keyword search plus a separate
     high-impact-journal search with a longer lookback).
   - **bioRxiv** and **medRxiv** via the details API (several pages fetched in parallel).
   - **arXiv** via its Atom API, restricted to the configured categories (and typed search terms).
2. **Score** — each paper is matched against the terms in every area in `config.yaml`.
   A title hit is worth `title_multiplier` points; an abstract hit is worth 1. Area scores
   are multiplied by the area's `weight`. Matching is **word-start anchored**, so `virus`
   still matches `viruses` but short acronyms like `NGS` don't match inside unrelated words.
3. **Rank & split** — papers with a score above zero are sorted by score (then recency) and
   split into a "High-impact journals" section and an "Other papers" section.
4. **Filter** — search terms, topics, source, sort and date range are applied on the server;
   selections persist in `localStorage`.

Results are cached in memory for 5 minutes for up to 16 filter combinations (best-effort on
serverless). Results are not cached when a source fails, so a retry can recover.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | The entire application: collectors, scoring, FastAPI routes, and the HTML template. |
| `config.yaml` | Research areas/terms, sources, lookback windows, and the high-impact journal list. |
| `requirements.txt` | Python dependencies. |
| `tests/` | Offline tests and a mock server with generated papers (no network needed). |

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml):

- **`areas`** — named topic groups, each with a list of `terms` and an optional `weight`
  (default `1.0`). These drive both the PubMed search and the relevance score.
- **`sources`** — toggle `pubmed` / `biorxiv` / `medrxiv` / `arxiv`, and set arXiv `categories`.
- **`collection`** — `lookback_days` (default window) and `max_results` per source.
- **`high_impact_journals`** — journal names that get their own section and a longer lookback
  (`high_impact_lookback_days`).
- **`other_lookback_options`** — the date-range choices shown in the UI dropdown.
- **`scoring.title_multiplier`** — how much more a title hit counts than an abstract hit.

> If you add a new area, give it a color in `app.py` (`--area-<name>` in both the dark and light
> `:root` blocks); otherwise it uses the accent color.

## Local development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000 to view your local instance.

Offline development and tests (no network access to the sources needed):

```bash
pip install -r requirements-dev.txt
python -m pytest tests
python tests/mock_server.py   # generated papers at http://127.0.0.1:8000
``` (The public site runs at
[paperfeed-app.vercel.app](https://paperfeed-app.vercel.app/).)

Useful endpoints:

- `GET /` — the feed (accepts `q`, repeated `area`, `scope`, `source`, `sort`, `days`).
- `GET /api/papers` — the same feed as JSON.
- `GET /api/health` — liveness check.
- `GET /api/debug` — checks PubMed connectivity and shows the generated query.

## Deployment (Vercel)

The repo uses Vercel's zero-config FastAPI support: it detects `requirements.txt` and the
`app` ASGI variable in `app.py`. Import the repo into Vercel and deploy — no `vercel.json`
needed. (`config.yaml` is read at runtime relative to `app.py`, so it ships with the function.)

The production deployment lives at
[paperfeed-app.vercel.app](https://paperfeed-app.vercel.app/).

## Limitations / ideas

- The in-memory cache resets on every cold start; a shared cache (e.g. Vercel KV) would make it
  more effective across invocations.
- bioRxiv is fetched a single page at a time, so very high `max_results` values won't all arrive.
- Scoring is keyword-based; an embedding-based relevance pass could improve ranking quality.

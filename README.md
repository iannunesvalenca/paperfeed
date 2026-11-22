# PaperFeed

A simple feed of recent research papers, ranked by topics you care about.

**Live app:** [paperfeed-app.vercel.app](https://paperfeed-app.vercel.app/)

PaperFeed pulls recent papers from **PubMed**, **bioRxiv**, **medRxiv**, and **arXiv**. It scores each paper against keyword areas you define in `config.yaml`, then renders a single page (dark or light, matching your system) with high-impact journals separated from the rest.

## Using it

- **Search box:** Plain words must all match, and prefixes count (`virus` matches *viruses*). Use quotes for phrases (`"long read"`), `OR` for alternatives (`nanopore OR illumina`), and `-` to exclude terms (`-plasmid`). Search terms also narrow the PubMed and arXiv queries and boost the score.
- **Topic chips:** Pick one or more areas from `config.yaml`. Papers must match at least one selected area.
- **Filters:** Change the date range, sort order (relevance or newest), journal grouping, or source. Filters apply right away, and active terms appear as removable chips.
- **Reading:** Matched terms are highlighted, structured abstracts keep their section labels, and **Full abstract** expands in place. Up to 200 ranked papers load, shown 25 at a time per section.
- **Save & cite:** **Save** stores papers in your browser (under the **Saved** panel). **Cite** copies a reference, **Copy all** copies every saved reference, and **DOI** opens the publisher page.
- **Keyboard shortcuts:** `/` focuses search, `j`/`k` move between papers, `o` opens the selected paper, and `s` saves it.
- **Error notices:** If a source fails to respond, the page shows which one instead of quietly dropping results.
- **Saved state & links:** Your last filters stay saved in the browser, and **Reset all** clears them. Every view has a shareable URL (older `?topic=` and `?method=` links still work).

Everything lives in one FastAPI file (`app.py`), ready to deploy on Vercel with no extra config.

## How it works

1. **Collect:** Fetch recent papers from each enabled source:
   - **PubMed** via NCBI E-utilities (a keyword query plus a separate high-impact journal query with a longer lookback).
   - **bioRxiv** and **medRxiv** via the details API, fetching several pages in parallel.
   - **arXiv** via its Atom API, limited to the configured categories and typed search terms.
2. **Score:** Match each paper against the terms in `config.yaml`. Title hits count for `title_multiplier` points and abstract hits count for 1, scaled by each area's `weight`. Matching is anchored at word starts so `virus` matches `viruses` without short acronyms like `NGS` matching inside unrelated words.
3. **Rank and split:** Keep papers with a positive score, sort by score and recency, and split them into "High-impact journals" and "Other papers".
4. **Filter:** Apply search terms, topics, source, sort order, and date range on the server. Selections are remembered in `localStorage`.

Results stay in memory for 5 minutes (up to 16 filter combinations, best-effort on serverless). Failed source responses are not cached, so reloading can recover cleanly.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Collectors, scoring, FastAPI routes, and the HTML template. |
| `config.yaml` | Research areas, sources, lookback windows, and high-impact journals. |
| `requirements.txt` | Python dependencies. |
| `tests/` | Offline tests and a mock server with generated papers (no network needed). |

## Configuration

All settings live in [`config.yaml`](config.yaml):

- **`areas`:** Named topic groups with a list of `terms` and an optional `weight` (default `1.0`). These drive both the PubMed search and the relevance score.
- **`sources`:** Enable or disable `pubmed`, `biorxiv`, `medrxiv`, and `arxiv`, and set arXiv `categories`.
- **`collection`:** Default `lookback_days` and `max_results` per source.
- **`high_impact_journals`:** Journals shown in their own section with a longer lookback (`high_impact_lookback_days`).
- **`other_lookback_options`:** Date-range choices shown in the UI dropdown.
- **`scoring.title_multiplier`:** Weight applied to title matches compared to abstract matches.

> If you add a new area, also add a color variable (`--area-<name>`) to both the dark and light `:root` blocks in `app.py`. Otherwise it falls back to the accent color.

## Local development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

To run the offline tests or mock server (no network access needed):

```bash
pip install -r requirements-dev.txt
python -m pytest tests
python tests/mock_server.py   # generated papers at http://127.0.0.1:8000
```

The live site runs at [paperfeed-app.vercel.app](https://paperfeed-app.vercel.app/).

Useful endpoints:

- `GET /`: The main feed (accepts `q`, repeated `area`, `scope`, `source`, `sort`, `days`).
- `GET /api/papers`: Same feed as JSON.
- `GET /api/health`: Health check.
- `GET /api/debug`: Checks PubMed connectivity and prints the generated query.

## Deployment (Vercel)

Vercel picks up `requirements.txt` and the `app` ASGI object in `app.py` automatically, so no `vercel.json` is needed. `config.yaml` is read relative to `app.py` at runtime and ships with the deployment.

The live instance is hosted at [paperfeed-app.vercel.app](https://paperfeed-app.vercel.app/).

## Limitations and ideas

- The in-memory cache resets on cold starts. A shared cache (such as Vercel KV) would hold results across invocations.
- bioRxiv and medRxiv fetching is capped at 8 pages (800 papers) per request to keep response times fast.
- Scoring is keyword-based. An embedding pass could improve ranking for looser topic matches.

## License

MIT. See [`LICENSE`](LICENSE).

# Search backends

`web_search` routes through a provider layer. A configured zero-key SearXNG instance is tried
first, followed by keyed providers and finally zero-key Bing, DuckDuckGo, Brave, and Wiby.
Marginalia remains available as an explicit provider. Results are normalized to one shape
regardless of backend. Every provider is relevance-filtered and deduplicated before
cross-provider merging; repeated URLs from the same provider do not receive an agreement bonus.

| Provider | Env vars | Notes |
|----------|----------|-------|
| SearXNG | `FOOTNOTE_SEARXNG_URL` (or `SEARXNG_URL`) | Zero-key JSON API; instance must enable JSON output. |
| Tavily | `TAVILY_API_KEY` | LLM-oriented search API. |
| Brave | `BRAVE_API_KEY` | Independent web index. |
| Google | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Programmable Search (Custom Search JSON API). |
| Bing + DuckDuckGo + Brave | none | Default fallback; scraped, no key. |
| Wiby | none | Public JSON endpoint; result metadata includes required Wiby attribution. |
| Marginalia | none | Shared public API; result metadata preserves its `CC-BY-NC-SA 4.0` license. |

`auto` (default) queries every configured provider plus the latency-bounded zero-key fallbacks
and merges the complete result set. Marginalia is excluded from `auto` because its shared public
endpoint can be slow; use `provider="auto+marginalia"` to include it in the merged search, or
`provider="marginalia"` to isolate it. Force one isolated backend with the `provider` argument
(`searxng`/`tavily`/`brave`/`google`/`wiby`/`marginalia`/`scrape`). Brave and DuckDuckGo enter a
temporary cooldown after rate limiting; override the defaults with
`FOOTNOTE_BRAVE_COOLDOWN_SECONDS` and `FOOTNOTE_DDG_COOLDOWN_SECONDS`.

## Specialized zero-key discovery

The public MCP surface is organized by user intent rather than by HTTP API:

| Intent tool | Backends | Routing notes |
|-------------|----------|---------------|
| `papers_search` | Crossref + arXiv | `source=auto` queries both; force either backend when needed. |
| `encyclopedia_search` | Wikipedia + Wikidata | Entity search by default; optional read-only SPARQL for structured facts. |
| `github_search` | GitHub REST search | Public zero-key requests work at GitHub's unauthenticated rate limit; `GITHUB_TOKEN` is optional. |
| `archive_search` | Wayback + Common Crawl | Accepts a URL/host pattern. `fetch_text=true` attempts archived-content extraction. |

All four return `title`, `url`, `snippet`, `published`, `authors`, `source`, and
`source_type` where those fields apply.

## Deep search

`web_deep_search` is a separate, slower research loop. It accepts an optional `sources`
array (`web`, `papers`, `encyclopedia`, `github`, `archive`); with an empty array it always
uses general web discovery and adds specialized sources when the query signals their intent.

Set `model` (or `FOOTNOTE_RESEARCH_MODEL`) to enable requirement decomposition, gap-specific
query planning, and strict fact extraction. It maintains a serializable research state and
evidence ledger, expands fetch/chunk budgets across iterations, parses HTML tables and linked
CSV/XLS/XLSX/PDF/JSON files, and verifies `subject`, `metric`, `period`, `value`, and `unit`
against an exact source quote before admitting an item. The result includes `answer_ready`,
unresolved requirements, per-iteration diagnostics, and the cumulative funnel
`candidates → deduplicated_documents → relevant_documents → successful_fetches → extracted_facts → verified_evidence`.

## Semantic reranking

Pass `semantic: true` to `web_search` to reorder by meaning rather than keyword overlap: it
over-fetches, embeds query and results with a local Ollama model, and sorts by cosine
similarity (each result gains `semantic_score`). Best-effort — if Ollama is unavailable the
original order is returned. Model: `FOOTNOTE_EMBED_MODEL` (default `bge-m3`).

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

## Free first, metered only if needed

Under the default strategy, `auto` does **not** call every configured provider on every query.
The free tier runs first — Bing, DuckDuckGo, Brave and Wiby, plus a self-hosted SearXNG, which
is keyed but unmetered. Only if that returns fewer than `FOOTNOTE_MIN_FREE_RESULTS` strong
matches (default `3`) is a metered provider called, and then just one: Tavily, the Brave API
and Google take turns, so one quota is not drained while the others sit unused. If the chosen
one answers empty, exactly one alternate is tried.

| Variable | Default | Effect |
|---|---|---|
| `FOOTNOTE_MIN_FREE_RESULTS` | `3` | Strong free results below which a metered call is worth spending |
| `FOOTNOTE_PROVIDER_STRATEGY` | `cost_aware` | Set to `merge` to call every configured provider on every query, as older versions did |

A keyed provider with no key set is skipped entirely rather than failing: Google needs
**both** `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`, and with either missing it never enters the
rotation. So the server searches perfectly well with no keys at all — it simply never has a
metered tier to fall back on when the free one comes up thin.

Marginalia is excluded from `auto` because its shared public endpoint can be slow; use
`provider="auto+marginalia"` to include it, or `provider="marginalia"` to isolate it. Force one
isolated backend with the `provider` argument
(`searxng`/`tavily`/`brave`/`google`/`wiby`/`marginalia`/`scrape`); forcing one whose key is
missing returns nothing rather than falling back. Brave and DuckDuckGo enter a temporary
cooldown after rate limiting; override the defaults with `FOOTNOTE_BRAVE_COOLDOWN_SECONDS` and
`FOOTNOTE_DDG_COOLDOWN_SECONDS`.

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
over-fetches, embeds query and results with bge-m3, and sorts by cosine similarity (each
result gains `semantic_score`). Best-effort — with no embedding runtime available the original
order is returned.

The weights are the same either way; only the runtime differs. `FOOTNOTE_EMBED_BACKEND=ollama`
talks to a running daemon, which costs nothing extra where one is already installed.
`local` loads `BAAI/bge-m3` through transformers inside the server process — install
`requirements-embed.txt` first. That path needs no daemon, which makes it the only one that
works in the Docker image or on a hosted instance. `auto` (the default) tries the daemon and
falls back to in-process. Model name: `FOOTNOTE_EMBED_MODEL` (default `bge-m3`, mapped to
`BAAI/bge-m3` for the in-process backend).

## Surviving a rate limit

The scraped engines are free, which means their budget is a rate limit rather
than a bill. Three things defend it.

**Requests look like a search, not a typed URL.** Queries carry `Referer`,
`Origin` and `Sec-Fetch-Site` for the engine's own domain. Measured against
live DuckDuckGo from a cool IP at one request per second: 2 accepted without
those headers, 7 with them.

**Repeat queries are answered from disk.** Results from Bing, DuckDuckGo and
Brave are cached for `FOOTNOTE_SEARCH_CACHE_TTL` seconds (default 86400) under
`FOOTNOTE_SEARCH_CACHE` (default `~/.footnote-mcp/search_cache/`). A refusal is
never cached, so a block does not turn into a day of empty answers.

**A refusal escalates before it gives up.** A `202`/`429` is retried through a
rotating proxy when `FOOTNOTE_PROXIES` is set, and then — for Bing and Brave —
rendered in headless Chromium. Only when every tier is refused does the
provider go into cooldown.

DuckDuckGo deliberately skips the browser tier: `html.duckduckgo.com` and
`lite.duckduckgo.com` answer Chromium with a 273-byte error stub, and the main
site returns a shell with no result nodes, so rendering would trade a clean
refusal for an empty page. Its two hosts also share one rate-limit budget — a
lite request sent right after html refuses is refused too — so `lite` is only
tried when html fails some other way.

Cooldowns are written to `provider_cooldowns.json` in the source cache, so a
restarted server does not walk straight back into a ban that is keyed to its IP.

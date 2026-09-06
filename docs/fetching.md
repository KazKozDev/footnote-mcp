# Fetching, anti-bot ladder, and politeness

`web_read` fetches through an escalation ladder
([`scraper.py`](../src/footnote_mcp/scraper.py)): the cheapest method runs first and escalates
only when a result looks blocked or empty. A block/quality detector decides when to escalate;
a per-domain rate limiter, circuit breaker, and negative cache keep it polite. The tier used
and the full attempt trace come back in `fetch_tier` / `scrape_tiers`.

| Tier | Method | Enabled by |
|------|--------|-----------|
| 1 | HTTP (curl_cffi TLS impersonation) | always |
| 2 | HTTP through a rotating proxy | `FOOTNOTE_PROXIES` set |
| 3 | Headless Chromium (runs JavaScript) | `FOOTNOTE_BROWSER_FALLBACK=1` (default on) |
| 4 | Chromium through a proxy | proxies + browser |
| 5 | Hosted scrape API (Firecrawl / ScrapingBee) | `FOOTNOTE_SCRAPE_API` set |

With nothing configured it is the plain HTTP path plus an automatic browser fallback for
JavaScript-rendered pages.

| Env var | Default | Purpose |
|---------|---------|---------|
| `FOOTNOTE_BROWSER_FALLBACK` | `1` | Escalate blocked/JS pages to headless Chromium. |
| `FOOTNOTE_PROXIES` | _(none)_ | Comma-separated proxy URLs; sticky per domain with health tracking. |
| `FOOTNOTE_SCRAPE_API` | _(none)_ | `firecrawl` or `scrapingbee` (needs the matching API key). |
| `FOOTNOTE_DOMAIN_RPS` / `_BURST` | `3` / `5` | Per-domain rate limit (token bucket). |
| `FOOTNOTE_BREAKER_THRESHOLD` / `_COOLDOWN` | `5` / `120` | Per-domain circuit breaker. |
| `FOOTNOTE_NEGCACHE_TTL` | `300` | Seconds to remember a blocked URL. |
| `FOOTNOTE_RETRY_AFTER_MAX_SECONDS` | `30` | Longest a request blocks waiting out a 429/503 before handing the refusal back. |
| `FOOTNOTE_HTTP_CACHE` | `1` | Store `ETag`/`Last-Modified` and revalidate with conditional requests. |
| `FOOTNOTE_HTTP_CACHE_MAX_BYTES` | `1000000` | Largest body kept for revalidation. |
| `FOOTNOTE_THIN_CONTENT_CHARS` | `200` | Below this extracted length, a script-heavy page counts as a JS shell. |

The rate limit, circuit breaker, and negative cache apply to **every** outbound request, not
only to pages fetched through the ladder: they live in
[`politeness.py`](../src/footnote_mcp/politeness.py) and are taken inside `fetch._get`, which
each tool's HTTP call funnels through. A `429` or `503` is waited out (honoring `Retry-After`)
rather than retried immediately; `web_crawl` stops at the first refusal; and parallel fetching
runs across hosts, never several workers at one host.

## Runtime data

```text
~/.footnote-mcp/source_cache/        # persistent page cache (with provenance)
~/.footnote-mcp/source_cache/http/   # ETag/Last-Modified bodies for conditional requests
~/.footnote-mcp/research_memory.json # persistent research memory
```

Override the cache location with `FOOTNOTE_SOURCE_CACHE=/path/to/cache footnote-mcp`.

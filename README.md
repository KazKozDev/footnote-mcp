# WebOperator MCP Server

MCP server and LangGraph research agent for source-grounded web research.

## Tool Surface

Discovery and reading:

| Tool | Description |
|------|-------------|
| `web_search` | Search via a keyed provider (Tavily/Brave/Google) when available, else Bing + DuckDuckGo. Snippets are discovery only. |
| `web_search_recent` | Web search restricted to a recency window (day/week/month/year) via DuckDuckGo's date filter. |
| `web_deep_search` | Search, fetch, extract, rerank, and return source context. |
| `web_read` | Fetch one URL, extract text, classify source quality, and persist cache metadata. |
| `scholarly_search` | Search specialized corpora missing from general web search: arXiv (papers) or Wikipedia (encyclopedic). |
| `web_archive_fetch` | Find the closest Wayback Machine snapshot for a dead or changed URL and optionally read its text. |
| `web_fetch_authenticated` | Fetch a page that needs cookies or custom headers (logged-in or gated pages). |
| `web_crawl` | Breadth-first crawl from a start URL, staying on the start host by default (capped at 50 pages). |
| `generate_search_queries` | Generate operator-style queries such as `site:`, `filetype:csv`, API, and data-table variants. |

Structured data:

| Tool | Description |
|------|-------------|
| `web_extract_tables` | Parse HTML tables into structured `columns` and `rows` with source URL provenance. |
| `web_detect_downloads` | Detect linked CSV, TSV, XLS, XLSX, PDF, JSON, and XML files. |
| `web_parse_file` | Download and parse CSV, TSV, XLS, XLSX, PDF, and JSON. |
| `web_fetch_json` | Fetch direct API/JSON endpoints into parsed JSON with source URL provenance. |
| `check_date_completeness` | Validate required date coverage for `day`, `week`, and `month` granularity. |
| `resolve_units` | Detect currencies, currency pairs, and measurement units. |
| `validate_unit_rows` | Reject structured rows with incompatible units or currency pairs. |
| `reconcile_time_series` | Align several series on a common key, compute deltas vs the first series, and flag missing keys and outliers. |
| `export_dataset` | Write consolidated rows to a `csv`, `xlsx`, or `json` file and return the path. |

Controlled extraction recipes:

| Tool | Description |
|------|-------------|
| `tool_spec_propose` | Propose a task-specific extraction recipe spec when generic tools fail. |
| `tool_code_generate` | Generate a starter `extract(source_text, input_payload)` recipe. |
| `tool_code_validate` | Validate recipe code against the static safety allowlist. |
| `tool_code_run_sandboxed` | Run validated recipe code in a limited subprocess with JSON output only. |
| `tool_promote` | Save a validated successful recipe as reusable memory without editing the MCP server. |
| `recipe_registry` | Manage promoted recipes: `list`, `get`, `run`, or `delete` saved recipes. |

Source quality and verification:

| Tool | Description |
|------|-------------|
| `classify_source` | Classify official, aggregator, blog, forum, interactive, blocked, and error sources. |
| `evidence_entailment` | Strict claim-vs-source checker with `heuristic`, `auto`, `ollama`, and optional `local_nli` backends. |
| `corroborate_claim` | Triangulate a claim across multiple source excerpts into a verdict (corroborated / conflicting / single_source / …). |
| `locate_claim_span` | Locate the supporting sentence(s) in a source with character offsets and a containment score (span-level provenance). |
| `source_cache_get` / `source_cache_put` | Inspect and write persistent source cache entries. |
| `build_research_debug_report` | Build a compact report of queries, URLs, source quality, and verification gaps. |
| `startup_health_check` | Check parser, OCR, browser, and cache dependencies. |

Browser fallback:

| Tool | Description |
|------|-------------|
| `web_navigate` | Navigate controlled Chromium to a URL. |
| `web_snapshot` | Capture visible interactive elements with stable refs. |
| `web_click` | Click a snapshot ref. |
| `web_type` | Type into a snapshot ref. |
| `web_extract` | Extract visible text or HTML. |
| `web_scroll` | Scroll the current page. |
| `browser_set_date_range` | Best-effort date range setter for inputs and common calendar cells. |
| `browser_extract_tables` | Extract visible HTML tables from the current browser page. |
| `browser_extract_tables_for_date_range` | Set a date range, submit, then extract visible tables. |
| `web_screenshot` | Capture a PNG of the current page, save it to disk, and optionally OCR text locked inside the image. |

## Install

### One command (pipx / uvx)

The server is a package with a `weboperator-mcp` entry point. Install it isolated:

```bash
pipx install /path/to/weboperator-mcp          # or:  pipx install git+<repo-url>
python -m playwright install chromium           # one-time: fetch the headless browser
```

Or run it ad hoc without installing, straight from the source dir:

```bash
uvx --from /path/to/weboperator-mcp weboperator-mcp
```

### Docker (no Python/browser setup)

```bash
docker build -t weboperator-mcp .
docker run -i --rm weboperator-mcp
```

The image bundles Chromium and tesseract, so there is nothing else to install.

### From source

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Pinned runtime versions are recorded in `requirements.lock`.

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Optional local NLI backend:

```bash
python -m pip install -r requirements-nli.txt
```

Then call `evidence_entailment` with `backend="local_nli"`. The default model is controlled by `WEBOPERATOR_NLI_MODEL`.

## OCR

PDF OCR uses `pytesseract` plus the system `tesseract` binary. On macOS:

```bash
brew install tesseract
```

Use `startup_health_check` to confirm whether OCR, parsers, browser support, and cache paths are available.

## Run

MCP server:

```bash
python server.py
python server.py --headed
```

The LangGraph research agent that drives this server lives in a separate project,
[weboperator-agent](../weboperator-agent) (it is an MCP *client* of this server).

## MCP Client Config

Drop one of these into your client's MCP settings (Claude Desktop:
`claude_desktop_config.json`; Cursor: `~/.cursor/mcp.json`). Add API keys under `env`
as needed (see Search Backends).

**Installed via pipx (entry point on PATH):**

```json
{
  "mcpServers": {
    "weboperator": {
      "command": "weboperator-mcp"
    }
  }
}
```

**Run via uvx (no install, from the source dir):**

```json
{
  "mcpServers": {
    "weboperator": {
      "command": "uvx",
      "args": ["--from", "/path/to/weboperator-mcp", "weboperator-mcp"],
      "env": { "TAVILY_API_KEY": "tvly-..." }
    }
  }
}
```

**Docker:**

```json
{
  "mcpServers": {
    "weboperator": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "weboperator-mcp"]
    }
  }
}
```

**From source (no packaging):**

```json
{
  "mcpServers": {
    "weboperator": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/weboperator-mcp"
    }
  }
}
```

Add `"--headed"` to the `args` to show the browser window.

## Search Backends

`web_search` (and everything built on it — `web_deep_search`, the agent) routes through
a provider layer. When an API key is set it uses that provider; otherwise it falls back to
scraping Bing + DuckDuckGo. Results are normalized to one shape regardless of backend.

| Provider | Env vars | Notes |
|----------|----------|-------|
| Tavily | `TAVILY_API_KEY` | LLM-oriented search API. |
| Brave | `BRAVE_API_KEY` | Independent web index. |
| Google | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Programmable Search (Custom Search JSON API). |
| Bing + DuckDuckGo | none | Default fallback; scraped, no key required. |

`auto` (default) tries every provider that has a key, in order Tavily → Brave → Google,
then falls back to scraping. Force one with the `provider` argument
(`tavily` | `brave` | `google` | `scrape`).

**Semantic reranking.** Pass `semantic: true` to `web_search` to reorder results by
*meaning* rather than keyword overlap: it over-fetches candidates, embeds the query and
each result with a local ollama embedding model, and sorts by cosine similarity (each
result gains a `semantic_score`). Best-effort — if ollama is unavailable the original
order is returned. The model is `WEBOPERATOR_EMBED_MODEL` (default `bge-m3`).

## Verification Benchmark

The server's differentiator is source-grounded verification — telling claims that a
source actually supports from ones it does not. [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py)
measures this on a labeled set of claim/source pairs and also demos `corroborate_claim`
and `locate_claim_span`.

| Backend | Set | Accuracy | Unsupported-claim catch rate | Precision on "supported" |
|---------|-----|----------|------------------------------|--------------------------|
| heuristic (offline) | data domain (numeric/factual) | **100%** | **100%** | **100%** |
| heuristic (offline) | overall (incl. semantic) | 83% | 78% | 80% |
| ollama LLM judge | overall (incl. semantic) | 78% | **100%** | **100%** |

On its design domain — numeric and factual data claims — the offline heuristic is flawless:
it never blesses an unsupported claim and always catches one. Its only blind spot is
purely-semantic negation/paraphrase. The optional LLM backend (`evidence_entailment`
`backend="ollama"`) closes that: across all cases it never lets an unsupported claim
through (100% catch rate, 100% precision on "supported"), erring conservatively by
flagging borderline claims for review rather than wrongly clearing them.

```bash
python benchmarks/run_benchmark.py                    # offline heuristic
python benchmarks/run_benchmark.py --backend ollama   # LLM judge (needs ollama)
```

## Fetching & Anti-Bot Ladder

`web_read` fetches pages through an escalation ladder ([scraper.py](scraper.py)): the
cheapest method runs first and it escalates only when a result looks blocked or empty.
A block/quality detector decides when to escalate; a per-domain rate limiter, circuit
breaker, and negative cache keep things polite. The fetched tier and the full attempt
trace are returned in `fetch_tier` / `scrape_tiers`.

| Tier | Method | Enabled by |
|------|--------|-----------|
| 1 | HTTP (curl_cffi TLS impersonation) | always (default) |
| 2 | HTTP through a rotating proxy | `WEBOPERATOR_PROXIES` set |
| 3 | Headless Chromium (runs JavaScript) | `WEBOPERATOR_BROWSER_FALLBACK=1` (default on) |
| 4 | Chromium through a proxy | proxies + browser |
| 5 | Hosted scrape API (Firecrawl / ScrapingBee) | `WEBOPERATOR_SCRAPE_API` set |

With nothing configured it behaves like the plain HTTP path plus an automatic browser
fallback for JavaScript-rendered pages.

| Env var | Default | Purpose |
|---------|---------|---------|
| `WEBOPERATOR_BROWSER_FALLBACK` | `1` | Escalate blocked/JS pages to headless Chromium. |
| `WEBOPERATOR_PROXIES` | _(none)_ | Comma-separated proxy URLs; sticky per domain with health tracking. |
| `WEBOPERATOR_SCRAPE_API` | _(none)_ | `firecrawl` or `scrapingbee` (needs `FIRECRAWL_API_KEY` / `SCRAPINGBEE_API_KEY`). |
| `WEBOPERATOR_DOMAIN_RPS` / `_BURST` | `3` / `5` | Per-domain rate limit (token bucket). |
| `WEBOPERATOR_BREAKER_THRESHOLD` / `_COOLDOWN` | `5` / `120` | Per-domain circuit breaker. |
| `WEBOPERATOR_NEGCACHE_TTL` | `300` | Seconds to remember a blocked URL. |
| `WEBOPERATOR_THIN_CONTENT_CHARS` | `200` | Below this extracted length a script-heavy page counts as a JS shell. |

## Runtime Data

Persistent cache:

```text
~/.weboperator-mcp/source_cache/
```

Persistent research memory:

```text
~/.weboperator-mcp/research_memory.json
```

Override cache location:

```bash
WEBOPERATOR_SOURCE_CACHE=/path/to/cache python server.py
```

## Calendars

`check_date_completeness` supports:

- `calendar`
- `business_day`
- `crypto_24_7`
- `forex_weekday`
- `us_business_day`
- `ru_business_day`

Pass explicit `holidays` for source-specific calendars. `us_business_day` and `ru_business_day` use the optional `holidays` package when installed.

## Tests

Offline unit and smoke tests:

```bash
python -m pytest \
  tests/test_tools_data.py \
  tests/test_tools_extra.py \
  tests/test_tools_search.py \
  tests/test_semantic.py \
  tests/test_benchmark.py \
  tests/test_scraper.py \
  tests/test_tools_browser.py \
  tests/test_server_dispatch.py \
  tests/test_mcp_smoke.py \
  -q
```

Optional live search test:

```bash
RUN_LIVE_WEB_TESTS=1 python -m pytest -m live
```

CI is defined in `.github/workflows/tests.yml`.

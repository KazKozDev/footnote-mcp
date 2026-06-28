# WebOperator MCP Server

MCP server and LangGraph research agent for source-grounded web research.

## Tool Surface

Discovery and reading:

| Tool | Description |
|------|-------------|
| `web_search` | Bing + DuckDuckGo search returning links and snippets. Snippets are discovery only. |
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

Recommended:

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

Interactive agent:

```bash
./agent.command
```

`agent.command` creates `.venv`, installs dependencies when `requirements.txt` changes, installs Playwright Chromium, and launches `agent/agent.py` from the virtual environment.

## MCP Client Config

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

Visible browser:

```json
{
  "mcpServers": {
    "weboperator": {
      "command": "python",
      "args": ["server.py", "--headed"],
      "cwd": "/path/to/weboperator-mcp"
    }
  }
}
```

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
  tests/test_agent_e2e.py \
  tests/test_tools_data.py \
  tests/test_tools_extra.py \
  tests/test_tools_search.py \
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

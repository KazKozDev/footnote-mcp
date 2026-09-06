# Tool reference

45 tools, all over stdio MCP. Counted from `list_tools()` in
[`src/footnote_mcp/server.py`](../src/footnote_mcp/server.py).

## Discovery and reading (12)

| Tool | Description |
|------|-------------|
| `web_search` | Configured providers plus zero-key Bing, DuckDuckGo, Brave, and Wiby; Marginalia is explicit-only. Snippets are discovery only. |
| `web_search_recent` | Search restricted to a recency window (day/week/month/year). |
| `web_deep_search` | Iteratively close evidence gaps across web/papers/encyclopedia/GitHub/archive sources; extracts tables/files, verifies individual facts, and returns an evidence ledger plus diagnostic funnel. |
| `web_read` | Fetch one URL, extract text, classify source quality, persist cache metadata. |
| `papers_search` | Search Crossref and arXiv through one normalized, zero-key paper contract. |
| `encyclopedia_search` | Search Wikipedia/Wikidata entities or run read-only Wikidata SPARQL. |
| `github_search` | Search public repositories, issues, code, or commits; authentication is optional. |
| `archive_search` | Find URL captures through Wayback Machine and Common Crawl, optionally extracting archived text. |
| `web_archive_fetch` | Find the closest Wayback Machine snapshot for a dead/changed URL. |
| `web_fetch_authenticated` | Fetch a page that needs cookies or custom headers. |
| `web_crawl` | Breadth-first crawl from a start URL, on-host by default. |
| `generate_search_queries` | Generate operator queries (`site:`, `filetype:csv`, API/data-table variants). |

## Structured data (9)

| Tool | Description |
|------|-------------|
| `web_extract_tables` | Parse HTML tables into `columns`/`rows` with source-URL provenance. |
| `web_detect_downloads` | Detect linked CSV/TSV/XLS/XLSX/PDF/JSON/XML files. |
| `web_parse_file` | Download and parse CSV/TSV/XLS/XLSX/PDF/JSON. |
| `web_fetch_json` | Fetch direct API/JSON endpoints into parsed JSON. |
| `check_date_completeness` | Validate required date coverage (day/week/month). |
| `resolve_units` | Detect currencies, currency pairs, measurement units. |
| `validate_unit_rows` | Reject rows with incompatible units or currency pairs. |
| `reconcile_time_series` | Align series on a key, compute deltas, flag missing keys/outliers. |
| `export_dataset` | Write consolidated rows to a `csv`/`xlsx`/`json` file. |

`check_date_completeness` supports the calendars `calendar`, `business_day`, `crypto_24_7`,
`forex_weekday`, `us_business_day`, and `ru_business_day` (pass explicit `holidays` for
source-specific ones; the `us_`/`ru_` variants use the optional `holidays` package).

## Source quality and verification (8)

| Tool | Description |
|------|-------------|
| `classify_source` | Classify official / aggregator / blog / forum / interactive / blocked / error. |
| `evidence_entailment` | Strict claim-vs-source checker: `heuristic`, `auto`, `ollama`, optional `local_nli`. |
| `corroborate_claim` | Triangulate a claim across excerpts (corroborated / conflicting / single_source / …). |
| `locate_claim_span` | Locate supporting sentence(s) with char offsets and a containment score. |
| `source_cache_get` / `source_cache_put` | Inspect and write persistent source-cache entries. |
| `build_research_debug_report` | Compact report of queries, URLs, source quality, verification gaps. |
| `startup_health_check` | Check parser, OCR, browser, and cache dependencies. |

## Controlled extraction recipes (6)

When generic parsers fail, synthesize a sandboxed parser:

| Tool | Description |
|------|-------------|
| `tool_spec_propose` | Propose a task-specific extraction recipe spec. |
| `tool_code_generate` | Generate a starter `extract(source_text, input_payload)` recipe. |
| `tool_code_validate` | Validate recipe code against a static safety allowlist. |
| `tool_code_run_sandboxed` | Run validated code in a limited subprocess (JSON output only). |
| `tool_promote` | Save a validated recipe as reusable memory (no server edit). |
| `recipe_registry` | Manage promoted recipes: `list` / `get` / `run` / `delete`. |

Recipe code may import only `csv`, `datetime`, `html`, `json`, `math`, `re`, and
`statistics`; `__import__`, `eval`, `exec`, `compile`, `open`, `getattr`, `setattr`,
`globals`, `locals`, `vars`, `input`, and `breakpoint` are rejected by the validator
([`tools_data/sandbox.py`](../src/footnote_mcp/tools_data/sandbox.py)).

## Browser fallback (10)

A controlled Chromium session for JS-heavy or interactive pages:

| Tool | Description |
|------|-------------|
| `web_navigate` · `web_snapshot` · `web_click` · `web_type` · `web_extract` · `web_scroll` | Drive a page via stable element refs. |
| `browser_set_date_range` · `browser_extract_tables` · `browser_extract_tables_for_date_range` | Set a date range, submit, extract visible tables. |
| `web_screenshot` | Save a PNG and optionally OCR text locked inside the image. |

Pass `--headed` to `footnote-mcp` to watch the browser tier work.

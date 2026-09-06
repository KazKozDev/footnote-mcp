# footnote-mcp — web research MCP server with claim verification

Search, extract data, and verify every claim against the source page.

Unlike search APIs (Tavily, Exa) or scrapers (Firecrawl) that return raw text or markdown, `footnote-mcp` is built for verification: every claim is checked against the raw source text with sentence-level citations and character offsets, with zero required API keys.

[Add to Cursor](https://cursor.com/install-mcp?name=footnote&config=eyJjb21tYW5kIjoiZm9vdG5vdGUtbWNwIn0%3D) · [Claude Desktop setup](#quick-start)

<video src="https://github.com/user-attachments/assets/47f66267-0210-47a7-8c21-12a889aeebb0" controls muted playsinline width="820">
  <img src="https://raw.githubusercontent.com/KazKozDev/footnote-mcp/main/assets/demo.gif" alt="footnote-mcp searching the web, extracting data, and verifying each claim against its source" width="820">
</video>

Runs without API keys · 45 tools · MIT licensed



<!-- mcp-name: io.github.KazKozDev/footnote-mcp -->

## Quick start

```bash
pip install footnote-mcp
python -m playwright install chromium   # browser tier for JS-heavy and blocked pages
```

`footnote-mcp` speaks MCP over stdio. Point a client at it — Claude Desktop's
`claude_desktop_config.json`, or Cursor's `~/.cursor/mcp.json`:

```json
{"mcpServers": {"footnote": {"command": "footnote-mcp"}}}
```

No API keys are needed to start. Restart the client and ask it to run `startup_health_check`:
it reports, one line each, whether the extractors, the PDF and spreadsheet parsers, the
browser, OCR and the cache directory are usable on this machine — so a missing optional piece
shows up now rather than halfway through a research task.

## Search the web without configuring an API key

Ask the assistant to look something up and `web_search` queries the zero-key providers — Bing,
DuckDuckGo, Brave, Wiby — alongside any keyed provider you configured, then deduplicates and
merges them into one ranking.

Every result carries the engines that returned it, so a page two independent indexes agree on
is distinguishable from one only a single engine found, and a relevance score. Snippets count
as discovery only: nothing at this stage is evidence yet, which is what the next two sections
are for.

Provider routing, keys, and semantic reranking: [docs/search-backends.md](docs/search-backends.md).

## Extract tables and files from a page into structured rows

`web_extract_tables` turns a page's HTML tables into named columns and rows, each set tagged
with the URL it came from and with the true total, so a truncated answer says so instead of
looking complete. Point it at a Wikipedia revenue table and you get `Rank`, `Name`,
`Industry`, `Revenue`, `Employees` as fields you can sort and compare, not a wall of text to
re-read.

Data a page links rather than renders is covered too: `web_detect_downloads` finds the
CSV/TSV/XLS/XLSX/PDF/JSON attached to it, `web_parse_file` parses them, and `web_fetch_json`
takes an API endpoint directly.

## Verify that a source actually supports a claim

The part a plain search tool does not do. `evidence_entailment` compares a claim to the source
text and returns a verdict; `corroborate_claim` triangulates across excerpts, and
`locate_claim_span` returns the supporting sentence with character offsets.

Given the claim *"Norway's battery-electric share of new passenger cars was 82.4% in 2023"*
and a source reading *"In 2023, battery-electric vehicles accounted for 82.4% of all new
passenger cars registered in Norway"*, the verdict comes back as a value your agent can act
on rather than a paragraph it has to interpret:

```json
{"status": "supported", "score": 0.778, "reason": "token overlap heuristic", "backend": "heuristic"}
```

The default backend is deterministic and offline, and it never quietly hands the decision to
another model: where it is not confident it returns `needs_review` with the spans it matched,
for you — or the assistant that called it, which already has both texts — to read. A local
LLM judge (`backend="ollama"`) and a local NLI model (`backend="local_nli"`) are there when you
want them. Measured accuracy of the deterministic path, which is the default:
[benchmarks/REPORT.md](benchmarks/REPORT.md).

## How it works

Search snippets are discovery, never evidence. Discovery merges several independent indexes;
fetching escalates through a ladder — plain HTTP, optional proxy, headless Chromium, optional
hosted scrape API — and stops at the cheapest tier returning real content. Extraction pulls
text, tables, and linked files, each cached with its source URL. Only then does verification
run, checking the claim against the fetched text before it counts. `web_deep_search` wraps the
whole loop: it decomposes requirements, re-searches unresolved gaps, and returns an evidence
ledger with a funnel showing where candidates were lost.

```
query → merged discovery → fetch ladder → extract (text · tables · files) → verify vs source → evidence
```

<details>
<summary>All 45 tools by category</summary>

### Search (8)
- `web_search` — Multi-engine search across zero-key providers (Bing, DuckDuckGo, Brave, Wiby) and optional metered APIs
- `web_search_recent` — Search restricted to a recency window (day, week, month, year)
- `web_deep_search` — Iterative multi-step research loop with evidence ledger and funnel diagnostics
- `papers_search` — Academic paper search via Crossref and arXiv
- `encyclopedia_search` — Wikipedia and Wikidata entity lookup plus read-only SPARQL
- `github_search` — Search public repositories, code, issues, and commits
- `archive_search` — Historical snapshots via Wayback Machine and Common Crawl
- `generate_search_queries` — Generate targeted operator queries (`site:`, `filetype:csv`)

### Fetch (4)
- `web_read` — Fetch URL, extract text, evaluate source quality, and cache snapshot
- `web_fetch_authenticated` — Fetch pages requiring custom cookies or session headers
- `web_archive_fetch` — Retrieve the nearest Wayback Machine snapshot for dead or changed URLs
- `web_crawl` — Breadth-first crawl following links from a starting URL

### Extract (15)
- `web_extract_tables` — Parse HTML tables into typed columns and rows with source URLs
- `web_detect_downloads` — Discover linked data files (CSV, TSV, XLS, XLSX, PDF, JSON, XML)
- `web_parse_file` — Download and parse tabular data and PDF documents
- `web_fetch_json` — Fetch direct REST API endpoints into parsed JSON
- `check_date_completeness` — Validate time series date continuity across calendar and market schedules
- `resolve_units` — Normalize currencies, units, and currency pairs
- `validate_unit_rows` — Detect and reject rows with conflicting units or currencies
- `reconcile_time_series` — Align series by key, compute deltas, and flag missing entries or outliers
- `export_dataset` — Save consolidated rows to CSV, XLSX, or JSON files
- `tool_spec_propose` — Propose task-specific extraction recipe specifications
- `tool_code_generate` — Generate starter Python extraction recipes
- `tool_code_validate` — Validate recipe code against an AST safety allowlist
- `tool_code_run_sandboxed` — Execute extraction code inside a restricted subprocess
- `tool_promote` — Persist validated recipes to local memory
- `recipe_registry` — List, inspect, run, and delete registered extraction recipes

### Verify (8)
- `evidence_entailment` — Evaluate whether source text entails a claim (heuristic, Ollama, or NLI)
- `corroborate_claim` — Triangulate claim consensus or conflict across multiple excerpts
- `locate_claim_span` — Locate supporting sentences with character offsets and containment scores
- `classify_source` — Classify domains (official, aggregator, blog, forum, interactive, blocked)
- `source_cache_get` — Retrieve cached page snapshots and provenance metadata
- `source_cache_put` — Store page contents and metadata into persistent cache
- `build_research_debug_report` — Compact diagnostic report of queries, sources, and verification gaps
- `startup_health_check` — Inspect availability of parsers, browser runtime, OCR, and cache directories

### Browser (10)
- `web_navigate` — Open URL in Chromium session (headless or `--headed`)
- `web_snapshot` — Inspect interactive DOM accessibility tree with stable element references
- `web_click` — Click interactive page elements by reference ID or CSS selector
- `web_type` — Enter text into form fields and input elements
- `web_scroll` — Scroll viewports or containers to reveal dynamic content
- `web_extract` — Extract targeted HTML elements or attributes
- `web_screenshot` — Capture page screenshots with optional Tesseract OCR
- `browser_set_date_range` — Manipulate dynamic web date-picker controls
- `browser_extract_tables` — Extract client-side rendered tables after DOM hydration
- `browser_extract_tables_for_date_range` — Automate date selection and table extraction cycles

Full parameters and schemas: [docs/tools.md](docs/tools.md).

</details>

## Requirements

- Python 3.10 or newer, on macOS, Linux or Windows — CI runs all six combinations
- Chromium via `python -m playwright install chromium`, for the browser tier and browser tools
- Any MCP client speaking stdio; config is documented for Claude Desktop and Cursor
- Optional, for `semantic: true`: an Ollama daemon, or `requirements-embed.txt` to run the same bge-m3 weights in-process — the only option where no daemon exists, such as Docker
- Optional: the system `tesseract` binary for OCR in `web_screenshot` and scanned PDFs
- No API keys, no account, no hosted service

## Limitations

- The offline entailment heuristic scores 100% on numeric and factual claims but 83% overall on the labelled set; purely semantic negation and paraphrase need `backend="ollama"`.
- The benchmark runner has no hard per-task timeout on Windows: it is built on `signal.setitimer`, which Windows lacks, so a hung task there runs unguarded. The server itself is unaffected.
- Zero-key providers are scraped, so results vary by IP. A refused search retries through a proxy and, for Bing and Brave, headless Chromium; DuckDuckGo answers Chromium with an error stub, so there it is cooldown or nothing.
- Semantic reranking is best-effort: with no embedding runtime reachable, the original ranking is returned unchanged.
- Generated recipes run in a subprocess that may import only `csv`, `datetime`, `html`, `json`, `math`, `re` and `statistics`, with `eval`, `exec`, `open` and `__import__` rejected — a validator, not a hardened sandbox.
- The hosted HTTP server holds per-user rate limits in memory; they reset on restart.

## Configuration

The server takes one flag, `--headed`, which shows the Chromium window instead of running it
invisibly — useful for watching the browser tier work on a page that keeps failing:

```json
{"mcpServers": {"footnote": {"command": "footnote-mcp", "args": ["--headed"]}}}
```

### Environment variables

Every variable is optional. The free tier (Bing, DuckDuckGo, Brave, Wiby) answers first; metered providers are called only when free results fall below `FOOTNOTE_MIN_FREE_RESULTS`. Paid search is the fallback, not the default: without keys, most queries are unaffected, and only narrow or obscure searches stay thin rather than being topped up.

| Variable | Effect when set | Effect when unset |
|---|---|---|
| `FOOTNOTE_SEARXNG_URL` | Self-hosted SearXNG joins the free tier, unmetered | Free tier is Bing, DuckDuckGo, Brave, Wiby |
| `TAVILY_API_KEY` | Tavily joins the metered rotation | Skipped; never called |
| `BRAVE_API_KEY` | Brave Search API joins the rotation, alongside scraped Brave | Only the scraped, keyless Brave is used |
| `GOOGLE_API_KEY` **and** `GOOGLE_CSE_ID` | Google Programmable Search joins the rotation | Skipped; never called |
| `FOOTNOTE_MIN_FREE_RESULTS` | Free results threshold to trigger metered fallback | `3` |
| `FOOTNOTE_PROVIDER_STRATEGY` | `merge` calls all providers; `cost_aware` calls metered only if needed | `cost_aware` |
| `GITHUB_TOKEN` | `github_search` runs at authenticated rate limits | Unauthenticated rate limits |
| `FOOTNOTE_RESEARCH_MODEL` | Ollama model for query planning and fact extraction in `web_deep_search` | Runs without planner |
| `FOOTNOTE_EMBED_MODEL` | Embedding model for `semantic: true` reranking | `bge-m3` |
| `FOOTNOTE_EMBED_BACKEND` | `ollama` needs the daemon; `local` loads the same weights in-process (`requirements-embed.txt`) | `auto`: daemon if running, else in-process, else ranking is unchanged |
| `FOOTNOTE_BROWSER_FALLBACK` | `0` disables Chromium browser fallback | Enabled |
| `FOOTNOTE_SEARCH_CACHE_TTL` | Search cache TTL in seconds (`0` disables) | `86400` |
| `FOOTNOTE_PROXIES` | Comma-separated proxy URLs for requests | Direct connections |
| `FOOTNOTE_SCRAPE_API` | Hosted scraper fallback (`firecrawl` or `scrapingbee`) with key | Browser tier is last fallback |
| `FOOTNOTE_SOURCE_CACHE` | Directory for raw cached pages | `~/.footnote-mcp/source_cache/` |

Full list and defaults: [.env.example](.env.example), [docs/fetching.md](docs/fetching.md), [docs/search-backends.md](docs/search-backends.md).

<details>
<summary>Docker, uvx, from source, OCR, tests</summary>

### Docker

```bash
docker run -i --rm ghcr.io/kazkozdev/footnote-mcp:latest   # bundles Chromium and Tesseract
```

```json
{"mcpServers": {"footnote": {"command": "docker",
  "args": ["run", "-i", "--rm", "ghcr.io/kazkozdev/footnote-mcp:latest"]}}}
```

### uvx / pipx / from source

```bash
uvx footnote-mcp
pipx install footnote-mcp
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

None of these fetch the browser. Run `python -m playwright install chromium` once as well, or
the browser tier and the browser tools are unavailable.

### OCR and local NLI

`pytesseract` needs the system binary (`brew install tesseract`). `evidence_entailment` with
`backend="local_nli"` needs `pip install -r requirements-nli.txt` and `FOOTNOTE_NLI_MODEL`.

### Tests

```bash
python -m pytest -q                              # offline; no network or keys
RUN_LIVE_WEB_TESTS=1 python -m pytest -m live    # opt-in live search
```

</details>

<br><br>

<div align="center">

![Claude Desktop](https://img.shields.io/badge/Claude_Desktop-333?style=flat-square&logo=anthropic&logoColor=fff) ![Cursor](https://img.shields.io/badge/Cursor-333?style=flat-square&logo=cursor&logoColor=fff)

[![tests](https://img.shields.io/github/actions/workflow/status/KazKozDev/footnote-mcp/tests.yml?style=flat-square&label=tests)](https://github.com/KazKozDev/footnote-mcp/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/footnote-mcp?style=flat-square)](https://pypi.org/project/footnote-mcp/) [![Python](https://img.shields.io/badge/python-3.10%2B-333?style=flat-square)](pyproject.toml) [![License](https://img.shields.io/badge/license-MIT-333?style=flat-square)](LICENSE)

[Issues](https://github.com/KazKozDev/footnote-mcp/issues) · [LICENSE](LICENSE) · [Tools](docs/tools.md) · [Hosting](docs/hosting.md) · [Benchmarks](benchmarks/REPORT.md) · [LinkedIn](https://www.linkedin.com/in/kazkozdev)

</div>


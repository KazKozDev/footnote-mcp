# footnote-mcp — web research MCP server with claim verification

Search, extract data, and verify every claim against the source page.

[Add to Cursor](cursor://anysphere.cursor-deeplink/mcp/install?name=footnote&config=eyJjb21tYW5kIjoiZm9vdG5vdGUtbWNwIn0%3D) · [Claude Desktop setup](#quick-start)

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

No API keys are needed to start. Ask the client to run `startup_health_check` to see what
this install can actually do (output trimmed):

```json
{"ok": true, "checks": {
  "trafilatura": {"ok": true}, "playwright": {"ok": true}, "pdfplumber": {"ok": true},
  "tesseract_binary": {"ok": true, "path": "/opt/homebrew/bin/tesseract"},
  "cache_dir": {"ok": true, "path": "/Users/you/.footnote-mcp/source_cache"}}}
```

## Search the web without configuring an API key

`web_search` queries the zero-key providers — Bing, DuckDuckGo, Brave, Wiby — alongside any
keyed provider you configured, then deduplicates and merges them into one ranking. Each result
records which engines returned it, so agreement across independent indexes is visible rather
than assumed. Snippets count as discovery only; nothing here is evidence yet.

```json
{
  "provider": "auto", "count": 2,
  "results": [{
    "title": "Almost every new car sold in Norway is electric - Our World in Data",
    "url": "https://ourworldindata.org/data-insights/almost-every-new-car-sold-in-norway-is-electric",
    "score": 1.444, "engines": ["brave", "ddg"]
  }]
}
```

Provider routing, keys, and semantic reranking: [docs/search-backends.md](docs/search-backends.md).

## Extract tables and files from a page into structured rows

`web_extract_tables` parses HTML tables into `columns`/`rows` carrying the source URL.
`web_detect_downloads`, `web_parse_file`, and `web_fetch_json` cover the CSV/XLS/XLSX/PDF/JSON
a page links instead of rendering. From a Wikipedia revenue table, `max_rows: 3`:

```json
{
  "columns": ["Rank", "Name", "Industry", "Revenue (USD millions)", "Employees"],
  "rows": [{"Rank": "1", "Name": "Walmart", "Industry": "Retail",
            "Revenue (USD millions)": "680,985", "Employees": "2,100,000"}],
  "row_count": 3, "total_row_count": 100, "truncated": true
}
```

## Verify that a source actually supports a claim

The part a plain search tool does not do. `evidence_entailment` compares a claim to the source
text and returns a verdict; `corroborate_claim` triangulates across excerpts, and
`locate_claim_span` returns the supporting sentence with character offsets.

Claim: *"Norway's battery-electric share of new passenger cars was 82.4% in 2023."*
Source excerpt: *"In 2023, battery-electric vehicles accounted for 82.4% of all new passenger
cars registered in Norway, up from 79.3% in 2022."*

```json
{"status": "supported", "score": 0.778, "reason": "token overlap heuristic", "backend": "heuristic"}
```

`heuristic` is offline and deterministic; `backend="ollama"` adds a local LLM judge and `auto`
escalates to it. Measured accuracy: [benchmarks/REPORT.md](benchmarks/REPORT.md).

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

## Configuration

| Option | Default | What it does |
|---|---|---|
| `--headed` | off | Show the Chromium window instead of running headless |

### Environment variables

All optional — the server starts and searches with none of them set.

| Variable | Required | What it does |
|---|---|---|
| `FOOTNOTE_SEARXNG_URL` | no | Zero-key SearXNG instance, tried first by `auto` |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` | no | Keyed search providers, merged with the zero-key ones |
| `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | no | Google Programmable Search provider |
| `GITHUB_TOKEN` | no | Raises the rate limit for `github_search` |
| `FOOTNOTE_RESEARCH_MODEL` | no | Ollama model for `web_deep_search` planning and extraction |
| `FOOTNOTE_EMBED_MODEL` | no | Embedding model for `semantic: true` (default `bge-m3`) |
| `FOOTNOTE_BROWSER_FALLBACK` | no | `0` disables the Chromium tier (default `1`) |
| `FOOTNOTE_SEARCH_CACHE_TTL` | no | Seconds a scraped search result is reused (default `86400`, `0` disables) |
| `FOOTNOTE_PROXIES` | no | Comma-separated proxy URLs, for the fetch ladder and refused searches |
| `FOOTNOTE_SCRAPE_API` | no | `firecrawl` or `scrapingbee`, with its matching key |
| `FOOTNOTE_SOURCE_CACHE` | no | Cache location (default `~/.footnote-mcp/source_cache/`) |

Every variable with its default: [.env.example](.env.example), [docs/fetching.md](docs/fetching.md).

## Requirements

- Python 3.10 or newer, on macOS, Linux or Windows — CI runs all six combinations
- Chromium via `python -m playwright install chromium`, for the browser tier and browser tools
- Any MCP client speaking stdio; config is documented for Claude Desktop and Cursor
- Optional: a local Ollama for `semantic: true` and the `ollama` entailment backend
- Optional: the system `tesseract` binary for OCR in `web_screenshot` and scanned PDFs
- No API keys, no account, no hosted service

## Limitations

- The offline entailment heuristic scores 100% on numeric and factual claims but 83% overall on the labelled set; purely semantic negation and paraphrase need `backend="ollama"`.
- The benchmark runner has no hard per-task timeout on Windows: it is built on `signal.setitimer`, which Windows lacks, so a hung task there runs unguarded. The server itself is unaffected.
- Zero-key providers are scraped, so results vary by IP. A refused search retries through a proxy and, for Bing and Brave, headless Chromium; DuckDuckGo answers Chromium with an error stub, so there it is cooldown or nothing.
- Semantic reranking is best-effort: with no Ollama reachable, the original ranking is returned unchanged.
- Generated recipes run in a subprocess limited to six stdlib imports, with `eval`, `exec`, `open`, and `__import__` rejected — a validator, not a hardened sandbox.
- The hosted HTTP server holds per-user rate limits in memory; they reset on restart.

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

![macOS](https://img.shields.io/badge/macOS-333?style=flat-square&logo=apple&logoColor=fff) ![Linux](https://img.shields.io/badge/Linux-333?style=flat-square&logo=linux&logoColor=fff) ![Windows](https://img.shields.io/badge/Windows-333?style=flat-square&logo=windows&logoColor=fff)

[![tests](https://img.shields.io/github/actions/workflow/status/KazKozDev/footnote-mcp/tests.yml?style=flat-square&label=tests)](https://github.com/KazKozDev/footnote-mcp/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/footnote-mcp?style=flat-square)](https://pypi.org/project/footnote-mcp/) [![Python](https://img.shields.io/badge/python-3.10%2B-333?style=flat-square)](pyproject.toml) [![License](https://img.shields.io/badge/license-MIT-333?style=flat-square)](LICENSE)

[Issues](https://github.com/KazKozDev/footnote-mcp/issues) · [Tools](docs/tools.md) · [Search backends](docs/search-backends.md) · [Fetching](docs/fetching.md) · [Hosting](docs/hosting.md) · [Benchmarks](benchmarks/REPORT.md) · [License](LICENSE) · [LinkedIn](https://www.linkedin.com/in/kazkozdev)

</div>

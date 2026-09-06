# footnote-mcp — web research MCP server with claim verification

Search, extract data, and verify every claim against the source page.

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

Every one is optional, but that is not the same as unused. The free tier — Bing, DuckDuckGo,
Brave and Wiby scraped without a key — answers first, on every query. A metered provider is
called only when the free tier comes back with fewer than `FOOTNOTE_MIN_FREE_RESULTS` strong
matches, and then just one of them, rotating between whichever are configured so a single
quota does not drain first. Paid search here is the fallback, not the default.

So keys are worth having but nothing breaks without them. With none set there is simply no
tier to fall back on: most queries are unaffected, and the narrow or obscure ones — where the
free engines return two weak hits instead of ten — stay thin rather than being topped up.
That is the whole difference a key buys.

| Variable | Effect when set | Effect when unset |
|---|---|---|
| `FOOTNOTE_SEARXNG_URL` | Self-hosted SearXNG joins the free tier, unmetered | Free tier is Bing, DuckDuckGo, Brave, Wiby |
| `TAVILY_API_KEY` | Tavily joins the metered rotation | Skipped; never called |
| `BRAVE_API_KEY` | Brave Search API joins the rotation, alongside scraped Brave | Only the scraped, keyless Brave is used |
| `GOOGLE_API_KEY` **and** `GOOGLE_CSE_ID` | Google Programmable Search joins the rotation — both are required, either alone does nothing | Skipped; never called |
| `FOOTNOTE_MIN_FREE_RESULTS` | Free results below which a metered call is worth spending | `3` |
| `FOOTNOTE_PROVIDER_STRATEGY` | `merge` calls every configured provider on every query | `cost_aware`: free first, one metered call only if needed |
| `GITHUB_TOKEN` | `github_search` runs at your account's rate limit | Works at GitHub's lower per-IP limit |
| `FOOTNOTE_RESEARCH_MODEL` | `web_deep_search` plans requirements and extracts facts with this Ollama model | Runs without a planner |
| `FOOTNOTE_EMBED_MODEL` | Model for `semantic: true` reranking | `bge-m3`; without Ollama, ranking is unchanged |
| `FOOTNOTE_BROWSER_FALLBACK` | `0` disables the Chromium tier | Enabled |
| `FOOTNOTE_SEARCH_CACHE_TTL` | Seconds a scraped search result is reused; `0` disables | `86400` |
| `FOOTNOTE_PROXIES` | Comma-separated proxies for the fetch ladder and refused searches | Direct connections only |
| `FOOTNOTE_SCRAPE_API` | `firecrawl` or `scrapingbee` as the last fetch tier, with its key | Ladder stops at the browser tier |
| `FOOTNOTE_SOURCE_CACHE` | Cache location | `~/.footnote-mcp/source_cache/` |

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

[![tests](https://img.shields.io/github/actions/workflow/status/KazKozDev/footnote-mcp/tests.yml?style=flat-square&label=tests)](https://github.com/KazKozDev/footnote-mcp/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/footnote-mcp?style=flat-square)](https://pypi.org/project/footnote-mcp/) [![Python](https://img.shields.io/badge/python-3.10%2B-333?style=flat-square)](pyproject.toml) [![License](https://img.shields.io/badge/license-MIT-333?style=flat-square)](LICENSE)

[Issues](https://github.com/KazKozDev/footnote-mcp/issues) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [LICENSE](LICENSE) · [Tools](docs/tools.md) · [Hosting](docs/hosting.md) · [Benchmarks](benchmarks/REPORT.md) · [LinkedIn](https://www.linkedin.com/in/kazkozdev)

</div>

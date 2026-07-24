from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"


async def _with_session(callback):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(SRC_PATH), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "footnote_mcp"], env=env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await callback(session)


async def _call_tool(session: ClientSession, name: str, args: dict | None = None) -> dict:
    result = await session.call_tool(name, args or {})
    text = "".join(getattr(content, "text", "") for content in result.content)
    return json.loads(text)


class _SmokeHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib callback name
        if self.path == "/data.csv":
            body = b"date,rate\n2026-05-01,90.1\n2026-05-02,91.2\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/data.json":
            body = b'{"rows":[{"date":"2026-05-01","rate":90.1},{"date":"2026-05-02","rate":91.2}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = b"""<!doctype html>
<html>
  <head>
    <title>Smoke MCP page</title>
    <meta property="article:published_time" content="2026-06-26T09:00:00">
  </head>
  <body>
    <main>
      <h1>Smoke MCP page</h1>
      <article>
        <p>This local smoke page verifies that web_read can fetch and extract article content.</p>
        <p>The content is deliberately plain so the extractor has a stable target in tests.</p>
      </article>
      <table>
        <caption>Daily rates</caption>
        <tr><th>date</th><th>rate</th></tr>
        <tr><td>2026-05-01</td><td>90.1</td></tr>
        <tr><td>2026-05-02</td><td>91.2</td></tr>
      </table>
      <a href="/data.csv">Download CSV</a>
    </main>
  </body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - stdlib callback signature
        return


@contextmanager
def _local_http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _skip_if_playwright_browser_missing():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright package is not installed")

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        if "playwright install" in str(exc).lower() or "executable doesn't exist" in str(exc).lower():
            pytest.skip("Playwright browser is not installed for this Python environment")
        raise


def test_mcp_lists_expected_tools():
    async def run(session: ClientSession):
        result = await session.list_tools()
        names = {tool.name for tool in result.tools}
        assert {
            "web_search",
            "web_deep_search",
            "papers_search",
            "encyclopedia_search",
            "github_search",
            "archive_search",
            "web_read",
            "web_navigate",
            "web_snapshot",
            "web_click",
            "web_type",
            "web_extract",
            "web_scroll",
            "web_extract_tables",
            "web_detect_downloads",
            "web_parse_file",
            "web_fetch_json",
            "check_date_completeness",
            "classify_source",
            "generate_search_queries",
            "resolve_units",
            "validate_unit_rows",
            "evidence_entailment",
            "tool_spec_propose",
            "tool_code_generate",
            "tool_code_validate",
            "tool_code_run_sandboxed",
            "tool_promote",
            "source_cache_get",
            "source_cache_put",
            "build_research_debug_report",
            "startup_health_check",
            "browser_extract_tables",
            "browser_set_date_range",
            "browser_extract_tables_for_date_range",
        } <= names

    asyncio.run(_with_session(run))


def test_web_read_extracts_local_page():
    async def run(session: ClientSession):
        with _local_http_server() as url:
            result = await _call_tool(session, "web_read", {"url": url, "use_cache": False})
            cached = await _call_tool(session, "web_read", {"url": url})
        assert result["url"].startswith("http://127.0.0.1:")
        assert result["title"] == "Smoke MCP page"
        assert "web_read can fetch and extract article content" in result["text"]
        assert result["text_length"] > 50
        assert result["cached"] is False
        assert cached["cached"] is True

    asyncio.run(_with_session(run))


def test_structured_data_tools_on_local_page():
    async def run(session: ClientSession):
        with _local_http_server() as url:
            tables = await _call_tool(session, "web_extract_tables", {"url": url, "use_cache": False})
            downloads = await _call_tool(session, "web_detect_downloads", {"url": url})
            csv_url = downloads["downloads"][0]["url"]
            parsed = await _call_tool(session, "web_parse_file", {"url": csv_url, "use_cache": False})
            fetched_json = await _call_tool(session, "web_fetch_json", {"url": url + "data.json", "use_cache": False})

        assert tables["table_count"] == 1
        assert tables["tables"][0]["columns"] == ["date", "rate"]
        assert tables["tables"][0]["rows"][0]["date"] == "2026-05-01"
        assert downloads["count"] == 1
        assert csv_url.endswith("/data.csv")
        assert parsed["file_type"] == "csv"
        assert parsed["tables"][0]["rows"][1]["rate"] == "91.2"
        assert fetched_json["json"]["rows"][1]["rate"] == 91.2

    asyncio.run(_with_session(run))


def test_validator_and_classifier_tools():
    async def run(session: ClientSession):
        completeness = await _call_tool(
            session,
            "check_date_completeness",
            {
                "start_date": "2026-05-01",
                "end_date": "2026-05-03",
                "actual_items": ["2026-05-01", "2026-05-03"],
            },
        )
        business_days = await _call_tool(
            session,
            "check_date_completeness",
            {
                "start_date": "2026-05-01",
                "end_date": "2026-05-04",
                "actual_items": ["2026-05-01", "2026-05-04"],
                "calendar": "business_day",
            },
        )
        source_type = await _call_tool(session, "classify_source", {"url": "https://data.gov/example"})
        queries = await _call_tool(session, "generate_search_queries", {"task": "EUR RUB exchange rate May 2026"})
        units = await _call_tool(session, "resolve_units", {"text": "euro to ruble daily rate in RUB"})
        row_validation = await _call_tool(
            session,
            "validate_unit_rows",
            {
                "expected_unit_or_pair": "EUR/RUB",
                "rows": [
                    {"pair": "EUR/RUB", "rate": "90.1"},
                    {"pair": "EUR/USD", "rate": "1.08"},
                ],
                "text_fields": ["pair"],
            },
        )
        entailment = await _call_tool(
            session,
            "evidence_entailment",
            {
                "claim": "The rate was 90.1 on 2026-05-01",
                "source_excerpt": "2026-05-01 rate 90.1",
                "backend": "heuristic",
            },
        )
        contradiction = await _call_tool(
            session,
            "evidence_entailment",
            {
                "claim": "The rate was 90.1 on 2026-05-01",
                "source_excerpt": "2026-05-01 rate 91.2",
                "backend": "heuristic",
            },
        )
        debug_report = await _call_tool(
            session,
            "build_research_debug_report",
            {
                "task": "EUR RUB exchange rate",
                "search_memory": {"attempted_queries": ["q"]},
                "sources": [{"url": "https://data.gov/example", "title": "Source", "content": "abc"}],
                "verification": {"task_complete": False, "gaps": ["missing"]},
            },
        )
        health = await _call_tool(session, "startup_health_check")

        assert completeness["complete"] is False
        assert completeness["missing_items"] == ["2026-05-02"]
        assert business_days["complete"] is True
        assert source_type["source_type"] == "official"
        assert any("filetype:csv" in query for query in queries["queries"])
        assert "EUR/RUB" in units["currency_pairs"]
        assert row_validation["accepted_count"] == 1
        assert row_validation["rejected_count"] == 1
        assert entailment["backend"] == "heuristic"
        assert entailment["status"] in {"supported", "partially_supported"}
        assert contradiction["status"] == "contradicted"
        assert debug_report["attempted_queries"] == ["q"]
        assert "checks" in health

    asyncio.run(_with_session(run))


def test_browser_tools_preserve_refs_in_one_session():
    _skip_if_playwright_browser_missing()

    html = """<!doctype html>
<html>
  <head><title>Browser Smoke</title></head>
  <body>
    <label for="q">Search</label>
    <input id="q" name="search" type="search" />
    <button type="button">Submit</button>
  </body>
</html>"""
    url = "data:text/html;charset=utf-8," + quote(html)

    async def run(session: ClientSession):
        nav = await _call_tool(session, "web_navigate", {"url": url})
        assert nav["title"] == "Browser Smoke"

        snapshot = await _call_tool(session, "web_snapshot")
        searchboxes = [el for el in snapshot["elements"] if el["role"] == "searchbox"]
        assert searchboxes

        typed = await _call_tool(
            session,
            "web_type",
            {"ref": searchboxes[0]["ref"], "text": "openai", "submit": False},
        )
        assert typed["ok"] is True

    asyncio.run(_with_session(run))


def test_browser_date_range_extracts_visible_table():
    _skip_if_playwright_browser_missing()

    html = """<!doctype html>
<html>
  <head><title>Date Range Smoke</title></head>
  <body>
    <label for="from">From date</label>
    <input id="from" name="fromDate" type="date" />
    <label for="to">To date</label>
    <input id="to" name="toDate" type="date" />
    <button type="button" onclick="document.querySelector('#status').textContent = document.querySelector('#from').value + '|' + document.querySelector('#to').value">Apply</button>
    <div id="status"></div>
    <table>
      <tr><th>date</th><th>rate</th></tr>
      <tr><td>2026-05-01</td><td>90.1</td></tr>
    </table>
  </body>
</html>"""
    url = "data:text/html;charset=utf-8," + quote(html)

    async def run(session: ClientSession):
        nav = await _call_tool(session, "web_navigate", {"url": url})
        assert nav["title"] == "Date Range Smoke"

        result = await _call_tool(
            session,
            "browser_extract_tables_for_date_range",
            {"start_date": "2026-05-01", "end_date": "2026-05-31"},
        )
        assert result["date_range"]["ok"] is True
        assert result["table_count"] == 1
        assert result["tables"][0]["rows"][0]["date"] == "2026-05-01"

    asyncio.run(_with_session(run))


@pytest.mark.live
def test_web_search_live_smoke():
    if os.environ.get("RUN_LIVE_WEB_TESTS") != "1":
        pytest.skip("set RUN_LIVE_WEB_TESTS=1 to hit external search engines")

    async def run(session: ClientSession):
        result = await _call_tool(session, "web_search", {"query": "OpenAI", "num": 2})
        assert result["count"] > 0
        assert any(item["url"].startswith("http") for item in result["results"])

    asyncio.run(_with_session(run))

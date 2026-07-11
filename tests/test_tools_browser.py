from __future__ import annotations

import asyncio
from urllib.parse import quote

import pytest

from footnote_mcp.tools_browser import WebBrowser


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


def _data_url(html: str) -> str:
    return "data:text/html;charset=utf-8," + quote(html)


def test_browser_extract_tables_ignores_hidden_tables_and_limits_rows():
    _skip_if_playwright_browser_missing()
    html = """<!doctype html>
<html>
  <head><title>Tables</title></head>
  <body>
    <table style="display:none"><tr><th>hidden</th></tr><tr><td>x</td></tr></table>
    <table>
      <caption>Visible</caption>
      <tr><th>date</th><th>rate</th></tr>
      <tr><td>2026-05-01</td><td>90.1</td></tr>
      <tr><td>2026-05-02</td><td>91.2</td></tr>
    </table>
  </body>
</html>"""

    async def run():
        browser = WebBrowser(headed=False)
        try:
            await browser.navigate(_data_url(html))
            result = await browser.extract_tables(max_tables=3, max_rows=1)
        finally:
            await browser.close()
        assert result["title"] == "Tables"
        assert result["table_count"] == 1
        assert result["tables"][0]["caption"] == "Visible"
        assert result["tables"][0]["row_count"] == 1
        assert result["tables"][0]["rows"][0]["rate"] == "90.1"

    asyncio.run(run())


def test_browser_set_date_range_fills_inputs_and_clicks_apply():
    _skip_if_playwright_browser_missing()
    html = """<!doctype html>
<html>
  <head><title>Date Inputs</title></head>
  <body>
    <label for="from">From date</label>
    <input id="from" name="fromDate" type="date" />
    <label for="to">To date</label>
    <input id="to" name="toDate" type="date" />
    <button type="button" onclick="document.body.setAttribute('data-applied', document.querySelector('#from').value + '|' + document.querySelector('#to').value)">Apply</button>
  </body>
</html>"""

    async def run():
        browser = WebBrowser(headed=False)
        try:
            await browser.navigate(_data_url(html))
            result = await browser.set_date_range("2026-05-01", "2026-05-31", submit=True)
            applied = await browser._page.evaluate("() => document.body.getAttribute('data-applied')")
        finally:
            await browser.close()
        assert result["ok"] is True
        assert result["clicked"] == "apply"
        assert applied == "2026-05-01|2026-05-31"

    asyncio.run(run())


def test_browser_set_date_range_reports_no_date_fields():
    _skip_if_playwright_browser_missing()
    html = """<!doctype html>
<html>
  <head><title>No Dates</title></head>
  <body><input name="q" type="text" /></body>
</html>"""

    async def run():
        browser = WebBrowser(headed=False)
        try:
            await browser.navigate(_data_url(html))
            result = await browser.set_date_range("2026-05-01", "2026-05-31", submit=True)
        finally:
            await browser.close()
        assert result["ok"] is False
        assert result["date_like_count"] == 0

    asyncio.run(run())


def test_browser_set_date_range_clicks_calendar_cells_without_inputs():
    _skip_if_playwright_browser_missing()
    html = """<!doctype html>
<html>
  <head><title>Calendar Cells</title></head>
  <body>
    <button data-date="2026-05-01" onclick="document.body.setAttribute('data-start', '2026-05-01')">1</button>
    <button data-date="2026-05-31" onclick="document.body.setAttribute('data-end', '2026-05-31')">31</button>
    <button onclick="document.body.setAttribute('data-applied', 'yes')">Apply</button>
  </body>
</html>"""

    async def run():
        browser = WebBrowser(headed=False)
        try:
            await browser.navigate(_data_url(html))
            result = await browser.set_date_range("2026-05-01", "2026-05-31", submit=True)
            start = await browser._page.evaluate("() => document.body.getAttribute('data-start')")
            end = await browser._page.evaluate("() => document.body.getAttribute('data-end')")
            applied = await browser._page.evaluate("() => document.body.getAttribute('data-applied')")
        finally:
            await browser.close()
        assert result["ok"] is True
        assert [item["field"] for item in result["clicked_dates"]] == ["start", "end"]
        assert start == "2026-05-01"
        assert end == "2026-05-31"
        assert applied == "yes"

    asyncio.run(run())


def test_browser_extract_tables_for_date_range_combines_steps():
    _skip_if_playwright_browser_missing()
    html = """<!doctype html>
<html>
  <head><title>Date Table</title></head>
  <body>
    <label for="from">Start</label>
    <input id="from" name="start" type="date" />
    <label for="to">End</label>
    <input id="to" name="end" type="date" />
    <button>Search</button>
    <table>
      <tr><th>date</th><th>rate</th></tr>
      <tr><td>2026-05-01</td><td>90.1</td></tr>
    </table>
  </body>
</html>"""

    async def run():
        browser = WebBrowser(headed=False)
        try:
            await browser.navigate(_data_url(html))
            result = await browser.extract_tables_for_date_range("2026-05-01", "2026-05-31")
        finally:
            await browser.close()
        assert result["date_range"]["ok"] is True
        assert result["table_count"] == 1
        assert result["tables"][0]["rows"][0]["date"] == "2026-05-01"

    asyncio.run(run())

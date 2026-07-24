from __future__ import annotations

import asyncio
import json

from footnote_mcp import server as server_module


def _decode(contents) -> dict:
    text = "".join(getattr(item, "text", "") for item in contents)
    return json.loads(text)


def test_list_tools_exposes_unique_names():
    async def run():
        tools = await server_module.list_tools()
        names = [tool.name for tool in tools]
        assert len(names) == len(set(names))
        assert {
            "web_search",
            "papers_search",
            "encyclopedia_search",
            "github_search",
            "archive_search",
            "web_read",
            "web_extract_tables",
            "web_parse_file",
            "web_fetch_json",
            "check_date_completeness",
            "validate_unit_rows",
            "evidence_entailment",
            "tool_code_run_sandboxed",
            "tool_promote",
            "build_research_debug_report",
            "startup_health_check",
            "browser_extract_tables_for_date_range",
        } <= set(names)

    asyncio.run(run())


def test_call_tool_dispatches_non_browser_tools(monkeypatch):
    calls = []

    def stub(name):
        def inner(**kwargs):
            calls.append((name, kwargs))
            return {"tool": name, "kwargs": kwargs}

        return inner

    for name in [
        "web_search",
        "web_deep_search",
        "papers_search",
        "encyclopedia_search",
        "github_search",
        "archive_search",
        "web_read",
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
    ]:
        monkeypatch.setattr(server_module, name, stub(name))

    cases = [
        ("web_search", {"query": "q", "lang": "en", "num": 3}, "web_search"),
        ("web_deep_search", {"query": "q", "lang": "en", "sources": ["github"]}, "web_deep_search"),
        ("papers_search", {"query": "q", "source": "crossref"}, "papers_search"),
        ("encyclopedia_search", {"query": "q", "source": "wikidata"}, "encyclopedia_search"),
        ("github_search", {"query": "q", "kind": "repositories"}, "github_search"),
        ("archive_search", {"url": "https://example.com", "source": "common_crawl"}, "archive_search"),
        ("web_read", {"url": "https://example.com", "use_cache": False}, "web_read"),
        ("web_extract_tables", {"url": "https://example.com", "max_tables": 2}, "web_extract_tables"),
        ("web_detect_downloads", {"url": "https://example.com", "max_links": 2}, "web_detect_downloads"),
        ("web_parse_file", {"url": "https://example.com/data.csv", "max_rows": 5}, "web_parse_file"),
        ("web_fetch_json", {"url": "https://example.com/data.json", "timeout": 5}, "web_fetch_json"),
        (
            "check_date_completeness",
            {"start_date": "2026-05-01", "end_date": "2026-05-02", "actual_items": [], "calendar": "business_day"},
            "check_date_completeness",
        ),
        ("classify_source", {"url": "https://example.com", "status_code": 403}, "classify_source"),
        ("generate_search_queries", {"task": "rates", "max_queries": 2}, "generate_search_queries"),
        ("resolve_units", {"text": "EUR/RUB"}, "resolve_units"),
        ("validate_unit_rows", {"rows": [{"pair": "EUR/RUB"}], "expected_unit_or_pair": "EUR/RUB"}, "validate_unit_rows"),
        (
            "evidence_entailment",
            {"claim": "a", "source_excerpt": "a", "backend": "heuristic", "model": "m"},
            "evidence_entailment",
        ),
        ("tool_spec_propose", {"task": "extract rows", "source_url": "https://example.com"}, "tool_spec_propose"),
        ("tool_code_generate", {"spec": {"name": "extract"}}, "tool_code_generate"),
        ("tool_code_validate", {"code": "def extract(source_text, input_payload):\n    return {}", "max_chars": 1000}, "tool_code_validate"),
        (
            "tool_code_run_sandboxed",
            {"code": "def extract(source_text, input_payload):\n    return {}", "source_text": "x"},
            "tool_code_run_sandboxed",
        ),
        (
            "tool_promote",
            {"name": "extract", "code": "def extract(source_text, input_payload):\n    return {}"},
            "tool_promote",
        ),
        ("source_cache_get", {"url": "https://example.com"}, "source_cache_get"),
        ("source_cache_put", {"url": "https://example.com", "payload": {"ok": True}}, "source_cache_put"),
        ("build_research_debug_report", {"task": "task"}, "build_research_debug_report"),
        ("startup_health_check", {}, "startup_health_check"),
    ]

    async def run():
        for tool_name, args, expected in cases:
            result = _decode(await server_module.call_tool(tool_name, args))
            assert result["tool"] == expected
            assert calls[-1][0] == expected

    asyncio.run(run())


def test_call_tool_dispatches_browser_tools(monkeypatch):
    class FakeBrowser:
        async def navigate(self, url):
            return {"method": "navigate", "url": url}

        async def snapshot(self):
            return {"method": "snapshot"}

        async def click(self, ref):
            return {"method": "click", "ref": ref}

        async def type(self, ref, text, submit=False):
            return {"method": "type", "ref": ref, "text": text, "submit": submit}

        async def extract(self, refs):
            return {"method": "extract", "refs": refs}

        async def scroll(self, direction):
            return {"method": "scroll", "direction": direction}

        async def extract_tables(self, max_tables=8, max_rows=100):
            return {"method": "extract_tables", "max_tables": max_tables, "max_rows": max_rows}

        async def set_date_range(self, start_date, end_date, submit=True):
            return {"method": "set_date_range", "start_date": start_date, "end_date": end_date, "submit": submit}

        async def extract_tables_for_date_range(self, start_date, end_date, max_tables=8, max_rows=100):
            return {
                "method": "extract_tables_for_date_range",
                "start_date": start_date,
                "end_date": end_date,
                "max_tables": max_tables,
                "max_rows": max_rows,
            }

    monkeypatch.setattr(server_module, "get_browser", lambda: FakeBrowser())

    cases = [
        ("web_navigate", {"url": "https://example.com"}, "navigate"),
        ("web_snapshot", {}, "snapshot"),
        ("web_click", {"ref": "@e1"}, "click"),
        ("web_type", {"ref": "@e1", "text": "hello", "submit": True}, "type"),
        ("web_extract", {"refs": "visible"}, "extract"),
        ("web_scroll", {"direction": "down"}, "scroll"),
        ("browser_extract_tables", {"max_tables": 2, "max_rows": 3}, "extract_tables"),
        ("browser_set_date_range", {"start_date": "2026-05-01", "end_date": "2026-05-31"}, "set_date_range"),
        (
            "browser_extract_tables_for_date_range",
            {"start_date": "2026-05-01", "end_date": "2026-05-31", "max_tables": 2, "max_rows": 3},
            "extract_tables_for_date_range",
        ),
    ]

    async def run():
        for tool_name, args, method in cases:
            result = _decode(await server_module.call_tool(tool_name, args))
            assert result["method"] == method

    asyncio.run(run())


def test_call_tool_handles_unknown_and_exceptions(monkeypatch):
    async def run_unknown():
        result = _decode(await server_module.call_tool("missing_tool", {}))
        assert result == {"error": "Unknown tool: missing_tool"}

    asyncio.run(run_unknown())

    def boom(**kwargs):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(server_module, "web_search", boom)

    async def run_error():
        result = _decode(await server_module.call_tool("web_search", {"query": "q"}))
        assert result == {"error": "network exploded"}

    asyncio.run(run_error())

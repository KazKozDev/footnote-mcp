#!/usr/bin/env python3
"""MCP server for web search + browser automation. ponytail: one file entry point.

Usage:
    python server.py              # headless (default)
    python server.py --headed     # visible browser window
"""

import argparse
import asyncio
import json
import sys
from copy import deepcopy

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, ServerCapabilities

from .tools_search import web_search, web_deep_search, web_read
from .tools_browser import WebBrowser
from .tools_extra import (
    corroborate_claim,
    export_dataset,
    locate_claim_span,
    recipe_registry,
    reconcile_time_series,
    scholarly_search,
    web_archive_fetch,
    web_crawl,
    web_fetch_authenticated,
    web_search_recent,
)
from .tools_data.cache import source_cache_get, source_cache_put
from .tools_data.classify import classify_source, generate_search_queries, resolve_units, validate_unit_rows
from .tools_data.dates import check_date_completeness
from .tools_data.entailment import evidence_entailment
from .tools_data.files import web_detect_downloads, web_extract_tables, web_fetch_json, web_parse_file
from .tools_data.health import build_research_debug_report, startup_health_check
from .tools_data.sandbox import (
    tool_code_generate,
    tool_code_run_sandboxed,
    tool_code_validate,
    tool_promote,
    tool_spec_propose,
)

server = Server("footnote")

_browser: WebBrowser | None = None
_headed: bool = False


def get_browser() -> WebBrowser:
    global _browser
    if _browser is None:
        _browser = WebBrowser(headed=_headed)
    return _browser


ToolSpec = tuple[str, tuple[str, ...], dict[str, object]]


def _tool_kwargs(arguments: dict, required: tuple[str, ...], defaults: dict[str, object]) -> dict:
    kwargs = {key: arguments[key] for key in required}
    for key, default in defaults.items():
        kwargs[key] = arguments[key] if key in arguments else deepcopy(default)
    return kwargs


SYNC_TOOLS: dict[str, ToolSpec] = {
    "web_search": ("web_search", ("query",), {"lang": "en", "num": 10, "provider": "auto", "semantic": False}),
    "web_deep_search": ("web_deep_search", ("query",), {"lang": "en"}),
    "web_read": ("web_read", ("url",), {"lang": "en", "use_cache": True}),
    "web_extract_tables": (
        "web_extract_tables",
        ("url",),
        {"lang": "en", "max_tables": 8, "max_rows": 80, "use_cache": True},
    ),
    "web_detect_downloads": ("web_detect_downloads", ("url",), {"lang": "en", "max_links": 50}),
    "web_parse_file": ("web_parse_file", ("url",), {"lang": "en", "max_rows": 200, "use_cache": True}),
    "web_fetch_json": ("web_fetch_json", ("url",), {"lang": "en", "use_cache": True, "timeout": 20}),
    "check_date_completeness": (
        "check_date_completeness",
        ("start_date", "end_date"),
        {"actual_items": [], "granularity": "day", "calendar": "calendar", "holidays": []},
    ),
    "classify_source": (
        "classify_source",
        ("url",),
        {"status_code": None, "content_type": "", "text_sample": ""},
    ),
    "generate_search_queries": ("generate_search_queries", ("task",), {"requirements": {}, "max_queries": 8}),
    "resolve_units": ("resolve_units", ("text",), {}),
    "validate_unit_rows": ("validate_unit_rows", ("expected_unit_or_pair",), {"rows": [], "text_fields": None}),
    "evidence_entailment": (
        "evidence_entailment",
        ("claim", "source_excerpt"),
        {"backend": "auto", "model": None},
    ),
    "tool_spec_propose": (
        "tool_spec_propose",
        ("task",),
        {"source_url": "", "observed_failure": "", "desired_output": "rows with date, value, unit, and source_url"},
    ),
    "tool_code_generate": ("tool_code_generate", (), {"spec": {}}),
    "tool_code_validate": ("tool_code_validate", ("code",), {"max_chars": 12000}),
    "tool_code_run_sandboxed": (
        "tool_code_run_sandboxed",
        ("code",),
        {"source_text": "", "input_payload": {}, "timeout": 5, "max_output_chars": 20000},
    ),
    "tool_promote": (
        "tool_promote",
        ("name", "code"),
        {"spec": {}, "sample_source_text": "", "input_payload": {}, "expected_min_rows": 0},
    ),
    "source_cache_get": ("source_cache_get", ("url",), {}),
    "source_cache_put": ("source_cache_put", ("url", "payload"), {}),
    "build_research_debug_report": (
        "build_research_debug_report",
        ("task",),
        {"requirements": {}, "search_memory": {}, "sources": [], "verification": {}},
    ),
    "startup_health_check": ("startup_health_check", (), {}),
    "web_archive_fetch": (
        "web_archive_fetch",
        ("url",),
        {"timestamp": "", "lang": "en", "fetch_text": True},
    ),
    "scholarly_search": ("scholarly_search", ("query",), {"source": "arxiv", "num": 10, "lang": "en"}),
    "web_search_recent": ("web_search_recent", ("query",), {"freshness": "month", "lang": "en", "num": 10}),
    "corroborate_claim": ("corroborate_claim", ("claim",), {"excerpts": [], "backend": "heuristic"}),
    "locate_claim_span": ("locate_claim_span", ("claim", "source_text"), {"max_spans": 3}),
    "recipe_registry": (
        "recipe_registry",
        ("action",),
        {"recipe_id": "", "source_text": "", "input_payload": {}},
    ),
    "web_fetch_authenticated": (
        "web_fetch_authenticated",
        ("url",),
        {"cookies": {}, "headers": {}, "lang": "en", "timeout": 20},
    ),
    "web_crawl": ("web_crawl", ("start_url",), {"max_pages": 10, "same_domain": True, "lang": "en"}),
    "export_dataset": ("export_dataset", ("rows",), {"format": "csv", "path": "", "columns": None}),
    "reconcile_time_series": ("reconcile_time_series", ("series",), {"on": "date", "value_field": "value"}),
}


BROWSER_TOOLS: dict[str, ToolSpec] = {
    "web_navigate": ("navigate", ("url",), {}),
    "web_snapshot": ("snapshot", (), {}),
    "web_click": ("click", ("ref",), {}),
    "web_type": ("type", ("ref", "text"), {"submit": False}),
    "web_extract": ("extract", ("refs",), {}),
    "web_scroll": ("scroll", ("direction",), {}),
    "browser_extract_tables": ("extract_tables", (), {"max_tables": 8, "max_rows": 100}),
    "browser_set_date_range": (
        "set_date_range",
        ("start_date", "end_date"),
        {"submit": True},
    ),
    "browser_extract_tables_for_date_range": (
        "extract_tables_for_date_range",
        ("start_date", "end_date"),
        {"max_tables": 8, "max_rows": 100},
    ),
    "web_screenshot": ("screenshot", (), {"full_page": False, "ocr": False}),
}


# ── Tool definitions ──

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the web. Uses a keyed provider (Tavily/Brave/Google) when an API key is set, otherwise falls back to scraping Bing + DuckDuckGo. Returns titles, URLs, snippets and scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "lang": {"type": "string", "description": "Language code: en, ru, etc.", "default": "en"},
                    "num": {"type": "integer", "description": "Max results to return", "default": 10},
                    "provider": {"type": "string", "description": "auto | tavily | brave | google | scrape", "default": "auto"},
                    "semantic": {"type": "boolean", "description": "Rerank results by meaning using local bge-m3 embeddings", "default": False},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="web_deep_search",
            description="Deep search: search Bing+DDG → fetch top pages → extract text → rerank chunks → return LLM-ready context. Slower but thorough.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "lang": {"type": "string", "description": "Language code", "default": "en"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="web_read",
            description="Fetch and extract text from a single URL. Returns extracted content (markdown).",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch and extract"},
                    "lang": {"type": "string", "description": "Language code", "default": "en"},
                    "use_cache": {"type": "boolean", "default": True},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_extract_tables",
            description="Fetch a page, parse HTML tables, and return structured columns/rows with source URL provenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "lang": {"type": "string", "default": "en"},
                    "max_tables": {"type": "integer", "default": 8},
                    "max_rows": {"type": "integer", "default": 80},
                    "use_cache": {"type": "boolean", "default": True},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_detect_downloads",
            description="Detect downloadable CSV/XLS/XLSX/PDF/JSON/XML links on a page.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "lang": {"type": "string", "default": "en"},
                    "max_links": {"type": "integer", "default": 50},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_parse_file",
            description="Download and parse CSV/TSV/XLSX/PDF/JSON files, returning structured rows or extracted text with provenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "lang": {"type": "string", "default": "en"},
                    "max_rows": {"type": "integer", "default": 200},
                    "use_cache": {"type": "boolean", "default": True},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_fetch_json",
            description="Fetch an API/JSON URL directly and return parsed JSON with source URL provenance and persistent cache.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "lang": {"type": "string", "default": "en"},
                    "use_cache": {"type": "boolean", "default": True},
                    "timeout": {"type": "integer", "default": 20},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="check_date_completeness",
            description="Validate date-range completeness for structured results. Supports day, week, and month granularity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "actual_items": {"type": "array", "items": {"type": "string"}},
                    "granularity": {"type": "string", "description": "day | week | month", "default": "day"},
                    "calendar": {"type": "string", "description": "calendar | business_day | crypto_24_7 | forex_weekday | us_business_day | ru_business_day", "default": "calendar"},
                    "holidays": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["start_date", "end_date", "actual_items"],
            },
        ),
        Tool(
            name="classify_source",
            description="Classify a source as official, aggregator, blog, forum, interactive, blocked, or error.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "status_code": {"type": "integer"},
                    "content_type": {"type": "string", "default": ""},
                    "text_sample": {"type": "string", "default": ""},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="generate_search_queries",
            description="Generate specialized search queries using operators like site:, filetype:, API, CSV, and data-table variants.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "requirements": {"type": "object", "default": {}},
                    "max_queries": {"type": "integer", "default": 8},
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="resolve_units",
            description="Resolve units, currencies, and currency pairs from text so incompatible rows can be rejected.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        Tool(
            name="validate_unit_rows",
            description="Reject structured rows that are incompatible with the expected unit, currency, or currency pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}},
                    "expected_unit_or_pair": {"type": "string"},
                    "text_fields": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rows", "expected_unit_or_pair"],
            },
        ),
        Tool(
            name="evidence_entailment",
            description="Strict entailment check for claim vs source excerpt. Supports heuristic, Ollama judge, or auto fallback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                    "backend": {"type": "string", "description": "auto | heuristic | ollama | local_nli", "default": "auto"},
                    "model": {"type": "string", "description": "Optional Ollama model name"},
                },
                "required": ["claim", "source_excerpt"],
            },
        ),
        Tool(
            name="tool_spec_propose",
            description="Propose a controlled task-specific extraction recipe spec when generic tools cannot extract structured rows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "source_url": {"type": "string", "default": ""},
                    "observed_failure": {"type": "string", "default": ""},
                    "desired_output": {"type": "string", "default": "rows with date, value, unit, and source_url"},
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="tool_code_generate",
            description="Generate a safe starter extraction recipe function for the proposed spec. The result must still be validated before running.",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {"type": "object", "default": {}},
                },
            },
        ),
        Tool(
            name="tool_code_validate",
            description="Statically validate task-specific extraction code against the safe recipe contract and allowlist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 12000},
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="tool_code_run_sandboxed",
            description="Run validated extraction recipe code in a separate limited subprocess. Code must define extract(source_text, input_payload).",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "source_text": {"type": "string", "default": ""},
                    "input_payload": {"type": "object", "default": {}},
                    "timeout": {"type": "integer", "default": 5},
                    "max_output_chars": {"type": "integer", "default": 20000},
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="tool_promote",
            description="Save a validated extraction recipe as reusable memory after an optional smoke test. Does not edit the MCP server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "spec": {"type": "object", "default": {}},
                    "code": {"type": "string"},
                    "sample_source_text": {"type": "string", "default": ""},
                    "input_payload": {"type": "object", "default": {}},
                    "expected_min_rows": {"type": "integer", "default": 0},
                },
                "required": ["name", "code"],
            },
        ),
        Tool(
            name="source_cache_get",
            description="Read persistent source cache entry for a URL.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
        Tool(
            name="source_cache_put",
            description="Write arbitrary parsed source payload into persistent source cache.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["url", "payload"],
            },
        ),
        Tool(
            name="build_research_debug_report",
            description="Build a compact diagnostic report for a research run: queries, URLs, source quality, and verification gaps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "requirements": {"type": "object", "default": {}},
                    "search_memory": {"type": "object", "default": {}},
                    "sources": {"type": "array", "items": {"type": "object"}, "default": []},
                    "verification": {"type": "object", "default": {}},
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="startup_health_check",
            description="Check optional parser, OCR, browser, and cache dependencies for this MCP server.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="web_navigate",
            description="Navigate the browser to a URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to navigate to"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_snapshot",
            description="Capture the current page state: URL, title, accessibility tree with stable refs (@e1, @e2...), and visible text. Use this before clicking/typing.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="web_click",
            description="Click an interactive element by its ref (e.g. @e3 from web_snapshot).",
            inputSchema={
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Element ref from web_snapshot, like @e3"},
                },
                "required": ["ref"],
            },
        ),
        Tool(
            name="web_type",
            description="Type text into an input field by its ref.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Input field ref from web_snapshot"},
                    "text": {"type": "string", "description": "Text to type"},
                    "submit": {"type": "boolean", "description": "Press Enter after typing", "default": False},
                },
                "required": ["ref", "text"],
            },
        ),
        Tool(
            name="web_extract",
            description="Extract text from the page. refs: comma-separated @eN, or 'visible' for all visible text, or 'all' for full HTML.",
            inputSchema={
                "type": "object",
                "properties": {
                    "refs": {"type": "string", "description": "Comma-separated refs, or 'visible', or 'all'"},
                },
                "required": ["refs"],
            },
        ),
        Tool(
            name="web_scroll",
            description="Scroll the page: up, down, top, bottom.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "description": "up | down | top | bottom"},
                },
                "required": ["direction"],
            },
        ),
        Tool(
            name="browser_extract_tables",
            description="Extract visible tables from the current browser page after navigation/interactions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_tables": {"type": "integer", "default": 8},
                    "max_rows": {"type": "integer", "default": 100},
                },
            },
        ),
        Tool(
            name="browser_set_date_range",
            description="Best-effort browser date-range setter for interactive pages with date inputs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "submit": {"type": "boolean", "default": True},
                },
                "required": ["start_date", "end_date"],
            },
        ),
        Tool(
            name="browser_extract_tables_for_date_range",
            description="Set date range in the current browser page, submit it, then extract visible tables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "max_tables": {"type": "integer", "default": 8},
                    "max_rows": {"type": "integer", "default": 100},
                },
                "required": ["start_date", "end_date"],
            },
        ),
        Tool(
            name="web_archive_fetch",
            description="Find the closest Wayback Machine snapshot of a URL (for dead or changed sources) and optionally read its text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timestamp": {"type": "string", "description": "Optional target YYYYMMDD or YYYYMMDDhhmmss", "default": ""},
                    "lang": {"type": "string", "default": "en"},
                    "fetch_text": {"type": "boolean", "default": True},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="scholarly_search",
            description="Search specialized corpora missing from general web search: arXiv (scientific papers) or Wikipedia (encyclopedic).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string", "description": "arxiv | wikipedia", "default": "arxiv"},
                    "num": {"type": "integer", "default": 10},
                    "lang": {"type": "string", "default": "en"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="web_search_recent",
            description="Web search restricted to a recency window via DuckDuckGo's date filter (day/week/month/year).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "freshness": {"type": "string", "description": "day | week | month | year", "default": "month"},
                    "lang": {"type": "string", "default": "en"},
                    "num": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="corroborate_claim",
            description="Triangulate a claim across multiple source excerpts; returns a corroboration verdict (corroborated/conflicting/single_source/...).",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "excerpts": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"source_url": {"type": "string"}, "text": {"type": "string"}}},
                    },
                    "backend": {"type": "string", "description": "heuristic | auto | ollama | local_nli", "default": "heuristic"},
                },
                "required": ["claim", "excerpts"],
            },
        ),
        Tool(
            name="locate_claim_span",
            description="Locate the sentence(s) in a source that best support a claim, with character offsets and a containment score (span-level provenance).",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source_text": {"type": "string"},
                    "max_spans": {"type": "integer", "default": 3},
                },
                "required": ["claim", "source_text"],
            },
        ),
        Tool(
            name="recipe_registry",
            description="Manage promoted extraction recipes: list, get, run, or delete saved recipes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "list | get | run | delete", "default": "list"},
                    "recipe_id": {"type": "string", "default": ""},
                    "source_text": {"type": "string", "default": ""},
                    "input_payload": {"type": "object", "default": {}},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="web_fetch_authenticated",
            description="Fetch a page that needs cookies or custom headers (logged-in or gated pages).",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "cookies": {"type": "object", "description": "name→value cookie map", "default": {}},
                    "headers": {"type": "object", "description": "extra request headers", "default": {}},
                    "lang": {"type": "string", "default": "en"},
                    "timeout": {"type": "integer", "default": 20},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_crawl",
            description="Breadth-first crawl from a start URL, fetching and extracting each page. Stays on the start host by default. Capped at 50 pages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_url": {"type": "string"},
                    "max_pages": {"type": "integer", "default": 10},
                    "same_domain": {"type": "boolean", "default": True},
                    "lang": {"type": "string", "default": "en"},
                },
                "required": ["start_url"],
            },
        ),
        Tool(
            name="export_dataset",
            description="Write extracted rows to a consolidated file (csv | xlsx | json) and return the path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}},
                    "format": {"type": "string", "description": "csv | xlsx | json", "default": "csv"},
                    "path": {"type": "string", "description": "Optional output path; defaults to cache exports dir", "default": ""},
                    "columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rows"],
            },
        ),
        Tool(
            name="reconcile_time_series",
            description="Align several time series on a common key, compute deltas vs the first series, and flag missing keys and outliers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "series": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"name": {"type": "string"}, "rows": {"type": "array", "items": {"type": "object"}}}},
                    },
                    "on": {"type": "string", "description": "Key field to align on", "default": "date"},
                    "value_field": {"type": "string", "default": "value"},
                },
                "required": ["series"],
            },
        ),
        Tool(
            name="web_screenshot",
            description="Capture a PNG screenshot of the current browser page, save it to disk, and optionally OCR text locked inside the image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "full_page": {"type": "boolean", "default": False},
                    "ocr": {"type": "boolean", "default": False},
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name in SYNC_TOOLS:
            func_name, required, defaults = SYNC_TOOLS[name]
            func = globals()[func_name]
            result = func(**_tool_kwargs(arguments, required, defaults))
        elif name in BROWSER_TOOLS:
            method_name, required, defaults = BROWSER_TOOLS[name]
            method = getattr(get_browser(), method_name)
            result = await method(**_tool_kwargs(arguments, required, defaults))
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except Exception as e:
        print(f"[footnote] tool '{name}' failed: {e}", file=sys.stderr)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def main():
    parser = argparse.ArgumentParser(description="footnote MCP server")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    args = parser.parse_args()

    global _headed
    _headed = args.headed

    init_opts = InitializationOptions(
        server_name="footnote",
        server_version="0.1.1",
        capabilities=ServerCapabilities(tools={}),
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, init_opts)


def cli():
    """Console-script entry point (sync wrapper around the async server)."""
    asyncio.run(main())

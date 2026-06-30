from __future__ import annotations

from datetime import date

from footnote_mcp import tools_search


def test_web_search_formats_results(monkeypatch):
    monkeypatch.setattr(
        tools_search,
        "search",
        lambda query, num=10, lang="en", provider="auto": [
            {"title": "A", "url": "https://example.com/a", "snippet": "sa", "score": 1.0, "engines": ["x"]},
            {"title": "B", "url": "https://example.com/b", "snippet": "sb", "score": 0.5, "engines": ["y"]},
        ],
    )

    result = tools_search.web_search("query", lang="en", num=2)

    assert result["query"] == "query"
    assert result["count"] == 2
    assert result["results"][0] == {
        "title": "A",
        "url": "https://example.com/a",
        "snippet": "sa",
        "score": 1.0,
        "engines": ["x"],
    }


def test_web_deep_search_formats_sources(monkeypatch):
    monkeypatch.setattr(
        tools_search,
        "search_extract_rerank",
        lambda query, lang="en": (["chunk"], [{"url": "https://example.com"}], {"https://example.com"}),
    )
    monkeypatch.setattr(
        tools_search,
        "build_llm_context",
        lambda ranked, results, fetched_urls=None: (
            "context",
            {0: 7},
            {0: {"title": "Title", "url": "https://example.com", "chunks": ["a", "b"]}},
        ),
    )

    result = tools_search.web_deep_search("query")

    assert result["context"] == "context"
    assert result["source_count"] == 1
    assert result["sources"] == [{"num": 7, "title": "Title", "url": "https://example.com", "chunks": 2}]


def test_web_read_fetches_extracts_classifies_and_caches(monkeypatch):
    cache = {}
    html = "<html><head><title>Page title</title></head><body><article>Hello source text.</article></body></html>"

    monkeypatch.setattr(tools_search, "_read_cache", lambda url: cache.get(url))

    def fake_write_cache(url, payload):
        existing = cache.get(url, {})
        existing.update(payload)
        cache[url] = existing

    monkeypatch.setattr(tools_search, "_write_cache", fake_write_cache)
    monkeypatch.setattr(tools_search, "fetch_page", lambda url, lang="en": (url, html, date(2026, 5, 1), None))
    monkeypatch.setattr(tools_search, "extract_content", lambda html, url=None: "Hello source text.")

    first = tools_search.web_read("https://data.gov/page", use_cache=True)
    second = tools_search.web_read("https://data.gov/page", use_cache=True)

    assert first["cached"] is False
    assert first["title"] == "Page title"
    assert first["text"] == "Hello source text."
    assert first["source_type"]["source_type"] == "official"
    assert second["cached"] is True
    assert second["text"] == "Hello source text."
    assert cache["https://data.gov/page"]["web_read"]["title"] == "Page title"


def test_web_read_bypasses_cache_when_requested(monkeypatch):
    calls = {"count": 0}
    cache = {"https://example.com": {"web_read": {"url": "https://example.com", "text": "cached"}}}

    monkeypatch.setattr(tools_search, "_read_cache", lambda url: cache.get(url))
    monkeypatch.setattr(tools_search, "_write_cache", lambda url, payload: None)

    def fake_fetch_page(url, lang="en"):
        calls["count"] += 1
        return url, "<html><title>Fresh</title></html>", None, None

    monkeypatch.setattr(tools_search, "fetch_page", fake_fetch_page)
    monkeypatch.setattr(tools_search, "extract_content", lambda html, url=None: "fresh")

    result = tools_search.web_read("https://example.com", use_cache=False)

    assert result["cached"] is False
    assert result["text"] == "fresh"
    assert calls["count"] == 1


def test_web_read_caches_error_and_empty(monkeypatch):
    writes = []
    monkeypatch.setattr(tools_search, "_read_cache", lambda url: None)
    monkeypatch.setattr(tools_search, "_write_cache", lambda url, payload: writes.append((url, payload)))

    monkeypatch.setattr(tools_search, "fetch_page", lambda url, lang="en": (url, None, None, "HTTP 500"))
    error = tools_search.web_read("https://example.com/error")
    assert error["error"] == "HTTP 500"
    assert writes[-1][1]["fetch_error"] == "HTTP 500"

    monkeypatch.setattr(tools_search, "fetch_page", lambda url, lang="en": (url, "", None, None))
    empty = tools_search.web_read("https://example.com/empty")
    assert empty["error"] == "Empty response body"
    assert writes[-1][1]["fetch_error"] == "Empty response body"

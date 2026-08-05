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
        "ollama_json_call",
        lambda model: "model-json",
    )
    monkeypatch.setattr(
        tools_search,
        "run_deep_research",
        lambda query, **kwargs: {
            "context": "context",
            "sources": [{"title": "Title", "url": "https://example.com"}],
            "context_length": 7,
            "source_count": 1,
            "answer_ready": True,
            "state": {"coverage": 1.0},
        },
    )

    result = tools_search.web_deep_search("query", model="m", max_iterations=3, max_fetch=12)

    assert result["context"] == "context"
    assert result["source_count"] == 1
    assert result["sources"] == [{"title": "Title", "url": "https://example.com"}]
    assert result["answer_ready"] is True
    assert result["model"] == "m"


def test_select_discovery_sources_routes_by_intent():
    assert tools_search.select_discovery_sources("Who is the creator of this GitHub repository?") == [
        "web",
        "encyclopedia",
        "github",
    ]
    assert tools_search.select_discovery_sources("Find DOI papers about retrieval") == ["web", "papers"]
    assert tools_search.select_discovery_sources("archive https://example.com old version") == ["web", "archive"]
    assert tools_search.select_discovery_sources("anything", ["github", "papers"]) == ["github", "papers"]


def test_discover_sources_keeps_direct_url_without_search_result(monkeypatch):
    monkeypatch.setattr(
        tools_search,
        "web_search",
        lambda *args, **kwargs: {"results": [], "count": 0},
    )

    results, routed, errors = tools_search.discover_sources(
        "Read https://example.gov/report and identify the value",
        num=5,
    )

    assert routed == ["web"]
    assert errors == {}
    assert results == [{
        "title": "https://example.gov/report",
        "url": "https://example.gov/report",
        "snippet": "Direct source supplied in the research query.",
        "score": 1.0,
        "engines": ["direct"],
    }]


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

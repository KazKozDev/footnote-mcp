from __future__ import annotations

import pytest

from footnote_mcp import search


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    for var in (
        "FOOTNOTE_SEARXNG_URL",
        "SEARXNG_URL",
        "TAVILY_API_KEY",
        "BRAVE_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
    ):
        monkeypatch.delenv(var, raising=False)


# ── provider order ──

def test_provider_order_auto_uses_only_keyed(monkeypatch):
    assert search._provider_order("auto") == []
    monkeypatch.setenv("FOOTNOTE_SEARXNG_URL", "http://localhost:8080")
    assert search._provider_order("auto") == ["searxng"]
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    assert search._provider_order("auto") == ["searxng", "brave"]
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert search._provider_order("auto") == ["searxng", "tavily", "brave"]


def test_provider_order_explicit_and_scrape():
    assert search._provider_order("google") == ["google"]
    assert search._provider_order("scrape") == []


def test_provider_order_google_needs_both_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert search._provider_order("auto") == []  # CSE id missing
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    assert search._provider_order("auto") == ["google"]


# ── provider parsing → merged shape ──

def test_search_tavily_parses(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    payload = {"results": [{"title": "A", "url": "https://a.com", "content": "snip a"},
                           {"title": "B", "url": "https://b.com", "content": "snip b"}]}
    monkeypatch.setattr(search.http, "post", lambda *a, **k: FakeResp(200, payload))
    out = search.search_tavily("q", num=5)
    assert [r["url"] for r in out] == ["https://a.com", "https://b.com"]
    assert out[0]["engines"] == ["tavily"]
    assert out[0]["score"] > out[1]["score"]


def test_search_brave_parses(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    payload = {"web": {"results": [{"title": "A", "url": "https://a.com", "description": "d"}]}}
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(200, payload))
    out = search.search_brave("q")
    assert out[0]["url"] == "https://a.com"
    assert out[0]["snippet"] == "d"
    assert out[0]["engines"] == ["brave"]


def test_search_google_parses(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    payload = {"items": [{"title": "A", "link": "https://a.com", "snippet": "s"}]}
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(200, payload))
    out = search.search_google("q")
    assert out[0]["url"] == "https://a.com"
    assert out[0]["engines"] == ["google"]


def test_search_searxng_parses_zero_key_json(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searx.test/")
    payload = {"results": [{"title": "A", "url": "https://a.com", "content": "result text"}]}
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(200, payload))

    out = search.search_searxng("q")

    assert out[0]["url"] == "https://a.com"
    assert out[0]["snippet"] == "result text"
    assert out[0]["engines"] == ["searxng"]


def test_provider_without_key_returns_empty():
    assert search.search_searxng("q") == []
    assert search.search_tavily("q") == []
    assert search.search_brave("q") == []
    assert search.search_google("q") == []


def test_provider_http_error_raises(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(429, {}))
    with pytest.raises(RuntimeError):
        search.search_brave("q")


# ── search() routing + fallback ──

def test_search_uses_api_provider_when_keyed(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(search, "search_tavily",
                        lambda q, num=10, lang="en": [{"title": "T", "url": "https://t.com", "snippet": "s", "score": 1.0, "engines": ["tavily"]}])
    # scraping must NOT be called
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: pytest.fail("bing should not run"))
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: pytest.fail("ddg should not run"))
    out = search.search("q", num=5)
    assert out[0]["url"] == "https://t.com"


def test_search_falls_back_to_scrape_when_provider_fails(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setattr(search, "search_brave", lambda q, num=10, lang="en": (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: [{"title": "Bg", "url": "https://bing.com/x", "snippet": ""}])
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [])
    out = search.search("q", num=5)
    assert any("bing.com" in r["url"] for r in out)


def test_search_scrape_when_no_keys(monkeypatch):
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: [{"title": "Bg", "url": "https://bing.com/x", "snippet": ""}])
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [{"title": "Dg", "url": "https://ddg.com/y", "snippet": ""}])
    out = search.search("q", num=5)
    urls = {r["url"] for r in out}
    assert "https://bing.com/x" in urls and "https://ddg.com/y" in urls

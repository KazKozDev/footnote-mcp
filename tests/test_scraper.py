from __future__ import annotations

import time

import pytest

from footnote_mcp import scraper


# ── block detector ──

def test_detect_block_http_status():
    assert scraper.detect_block(403, "<html>x</html>")[0] is True
    assert scraper.detect_block(429, "<html>x</html>")[0] is True


def test_detect_block_error_without_html():
    blocked, reason = scraper.detect_block(0, None, "HTTP 500")
    assert blocked is True and reason == "HTTP 500"


def test_detect_block_challenge_and_js_markers():
    assert scraper.detect_block(200, "<html>Just a moment... cf-chl</html>")[1] == "challenge_or_captcha"
    assert scraper.detect_block(200, "<html>Please enable JavaScript to run this app</html>")[1] == "js_required"


def test_detect_block_thin_js_shell(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_THIN_CONTENT_CHARS", "200")
    shell = "<html><head>" + ("<meta x='1'>" * 200) + "</head><body><div id=root></div><script src=a.js></script></body></html>"
    blocked, reason = scraper.detect_block(200, shell)
    assert blocked is True and reason == "thin_js_shell"


def test_detect_block_good_page():
    html = "<html><body><article>" + ("Real readable content. " * 50) + "</article></body></html>"
    assert scraper.detect_block(200, html)[0] is False


# ── rate limiter ──

def test_rate_limiter_paces_after_burst():
    rl = scraper.DomainRateLimiter(rps=10, burst=2)
    assert rl.acquire("d") == 0.0
    assert rl.acquire("d") == 0.0
    waited = rl.acquire("d")  # burst exhausted → must wait ~1/rps
    assert waited > 0


def test_rate_limiter_disabled_when_rps_zero():
    rl = scraper.DomainRateLimiter(rps=0, burst=0)
    assert rl.acquire("d") == 0.0


# ── circuit breaker ──

def test_circuit_breaker_opens_after_threshold():
    cb = scraper.CircuitBreaker(threshold=3, cooldown=60)
    assert cb.is_open("d") is False
    for _ in range(3):
        cb.record_failure("d")
    assert cb.is_open("d") is True
    cb.record_success("d")
    assert cb.is_open("d") is False


# ── negative cache ──

def test_negative_cache_roundtrip(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_NEGCACHE_TTL", "60")
    nc = scraper.NegativeCache()
    nc.put("https://x.com", "http_403")
    assert nc.get("https://x.com") == "http_403"


def test_negative_cache_disabled(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_NEGCACHE_TTL", "0")
    nc = scraper.NegativeCache()
    nc.put("https://x.com", "http_403")
    assert nc.get("https://x.com") is None


# ── proxy pool ──

def test_proxy_pool_sticky_and_health(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_PROXIES", "http://p1:8000,http://p2:8000")
    pool = scraper.ProxyPool()
    assert pool.available() is True
    first = pool.get("a.com")
    assert pool.get("a.com") == first  # sticky per domain
    pool.report(first, ok=False)       # mark dead
    monkeypatch.setenv("FOOTNOTE_PROXY_COOLDOWN", "300")
    second = pool.get("a.com", rotate=True)
    assert second != first             # avoids dead proxy


def test_proxy_pool_empty_when_unset(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_PROXIES", "")
    pool = scraper.ProxyPool()
    assert pool.available() is False
    assert pool.get("a.com") is None


# ── orchestrator: tier-1 success ──

def test_fetch_tier1_http_success():
    def http_fn(url, lang="en"):
        return url, "<html><body><article>" + ("Good content. " * 50) + "</article></body></html>", None, None

    res = scraper.fetch("https://x.com", http_fn=http_fn)
    assert res["tier"] == "http"
    assert res["blocked"] is False
    assert res["html"]


# ── orchestrator: escalate to browser on block ──

def test_fetch_escalates_to_browser(monkeypatch):
    def http_fn(url, lang="en"):
        return url, None, None, "HTTP 403"

    good_html = "<html><body><article>" + ("Rendered content. " * 50) + "</article></body></html>"
    monkeypatch.setattr(scraper._RENDERER, "render", lambda url, lang="en", proxy=None, timeout=None: (good_html, 200, None))

    res = scraper.fetch("https://x.com", http_fn=http_fn, allow_browser=True)
    assert res["tier"] == "browser"
    assert res["blocked"] is False
    assert "Rendered content" in res["html"]
    tiers = [t["tier"] for t in res["tiers_tried"]]
    assert tiers == ["http", "browser"]


# ── orchestrator: external tier ──

def test_fetch_escalates_to_external(monkeypatch):
    def http_fn(url, lang="en"):
        return url, None, None, "HTTP 403"

    good_html = "<html><body><article>" + ("External content. " * 50) + "</article></body></html>"
    monkeypatch.setattr(scraper, "_scrape_external", lambda url, lang="en": (good_html, 200, None))

    res = scraper.fetch("https://x.com", http_fn=http_fn, allow_browser=False, allow_external=True)
    assert res["tier"] == "external"
    assert "External content" in res["html"]


# ── orchestrator: graceful degrade returns best partial html ──

def test_fetch_degrades_to_partial_html_when_all_fail():
    # thin JS shell → flagged blocked, but no other tier available → return it anyway
    shell = "<html><head>" + ("<meta x='1'>" * 300) + "</head><body><div id=root></div><script src=a></script></body></html>"

    def http_fn(url, lang="en"):
        return url, shell, None, None

    res = scraper.fetch("https://x.com", http_fn=http_fn, allow_browser=False)
    assert res["blocked"] is True
    assert res["html"] == shell  # graceful: still hand back what we got


# ── orchestrator: hard failure (no html anywhere) ──

def test_fetch_hard_failure_returns_error():
    def http_fn(url, lang="en"):
        return url, None, None, "HTTP 500"

    res = scraper.fetch("https://x.com", http_fn=http_fn, allow_browser=False)
    assert res["blocked"] is True
    assert res["html"] is None
    assert res["error"] == "HTTP 500"


# ── orchestrator: negative cache short-circuits ──

def test_fetch_negative_cache_short_circuits(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_NEGCACHE_TTL", "60")
    scraper._NEG_CACHE.put("https://x.com", "http_403")
    calls = {"n": 0}

    def http_fn(url, lang="en"):
        calls["n"] += 1
        return url, "<html>x</html>", None, None

    res = scraper.fetch("https://x.com", http_fn=http_fn)
    assert res["tier"] == "negative_cache"
    assert calls["n"] == 0  # never hit the network

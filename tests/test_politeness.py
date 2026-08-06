"""Politeness applied at the request funnel: pacing, rate-limit backoff,
bounded crawling, per-host parallelism, and conditional requests."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from footnote_mcp import fetch, http_cache, politeness, scraper, tools_extra


class FakeResp:
    def __init__(self, status_code=200, text="<html><body>ok</body></html>", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


@pytest.fixture(autouse=True)
def _clean_politeness():
    politeness.reset_state()
    yield
    politeness.reset_state()


# ── Retry-After parsing ──

def test_retry_after_reads_delta_seconds_and_respects_default():
    resp = FakeResp(429, "", {"Retry-After": "7"})
    assert politeness.retry_after_seconds(resp, default=2.0) == 7.0
    # The caller's own backoff wins when the server asks for less.
    assert politeness.retry_after_seconds(FakeResp(429), default=2.0) == 2.0


def test_retry_after_reads_http_date_form():
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    resp = FakeResp(503, "", {"Retry-After": format_datetime(when)})

    seconds = politeness.retry_after_seconds(resp)

    assert 20 <= seconds <= 35


def test_retry_after_is_capped():
    resp = FakeResp(429, "", {"Retry-After": "99999"})
    assert politeness.retry_after_seconds(resp, cap=60.0) == 60.0


def test_is_block_error_matches_refusal_statuses_only():
    assert politeness.is_block_error("HTTP 429") is True
    assert politeness.is_block_error("HTTP 403") is True
    assert politeness.is_block_error("HTTP 404") is False
    assert politeness.is_block_error(None) is False


# ── item 1: the limiter sits in _get, so nothing can route around it ──

def test_every_get_takes_a_token_for_its_domain(monkeypatch):
    taken = []
    monkeypatch.setattr(fetch.RATE_LIMITER, "acquire", lambda domain: taken.append(domain) or 0.0)
    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp())

    fetch._get("https://data.gov.example/tables/1")
    fetch._get("https://other.example/x")

    assert taken == ["data.gov.example", "other.example"]


def test_scraper_and_fetch_share_one_limiter():
    # Two buckets would mean neither enforces the configured rate.
    assert scraper._RATE_LIMITER is fetch.RATE_LIMITER
    assert scraper._BREAKER is fetch.BREAKER
    assert scraper._NEG_CACHE is politeness.NEG_CACHE


# ── item 2: status-aware retry ──

def test_get_waits_out_a_429_then_retries(monkeypatch):
    slept = []
    responses = [FakeResp(429, "", {"Retry-After": "2"}), FakeResp(200, "<html>ok</html>")]
    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: slept.append(s))

    resp = fetch._get("https://slow.example/a", max_retries=2)

    assert resp.status_code == 200
    assert slept == [2.0]


def test_get_hands_back_the_refusal_when_the_wait_exceeds_the_cap(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_RETRY_AFTER_MAX_SECONDS", "5")
    slept = []
    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp(429, "", {"Retry-After": "600"}))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: slept.append(s))

    resp = fetch._get("https://slow.example/a", max_retries=2)

    # Blocking a tool call for ten minutes helps nobody; the breaker keeps the
    # next callers away instead.
    assert resp.status_code == 429
    assert slept == []


def test_repeated_refusals_open_the_breaker_for_that_domain(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_BREAKER_THRESHOLD", "2")
    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp(429, ""))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)

    for _ in range(2):
        fetch._get("https://blocked.example/a", max_retries=0)

    assert politeness.BREAKER.is_open("blocked.example") is True
    assert politeness.BREAKER.is_open("fine.example") is False


def test_a_successful_response_clears_earlier_failures(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_BREAKER_THRESHOLD", "3")
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp(503, ""))
    fetch._get("https://flaky.example/a", max_retries=0)
    fetch._get("https://flaky.example/a", max_retries=0)

    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp(200, "<html>ok</html>"))
    fetch._get("https://flaky.example/a", max_retries=0)

    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp(503, ""))
    fetch._get("https://flaky.example/a", max_retries=0)

    assert politeness.BREAKER.is_open("flaky.example") is False


def test_a_404_is_returned_untouched(monkeypatch):
    slept = []
    monkeypatch.setattr(fetch.http, "get", lambda *a, **k: FakeResp(404, ""))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: slept.append(s))

    assert fetch._get("https://x.example/missing", max_retries=2).status_code == 404
    assert slept == []  # a missing page is not a rate limit


# ── item 3: the crawl stops at the first refusal ──

def test_web_crawl_stops_on_the_first_refusal(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_NEGCACHE_TTL", "60")
    calls = []
    page = '<html><body><a href="/a">A</a><a href="/b">B</a><a href="/c">C</a>text</body></html>'

    def fake_fetch_page(url, lang="en"):
        calls.append(url)
        if len(calls) == 1:
            return url, page, None, None
        return url, None, None, "HTTP 429"

    monkeypatch.setattr(tools_extra, "fetch_page", fake_fetch_page)

    result = tools_extra.web_crawl("https://host.example/", max_pages=10)

    assert len(calls) == 2  # start page, one refusal, then stop
    assert "429" in result["stopped_reason"]
    assert politeness.NEG_CACHE.get(calls[1]) is not None


def test_web_crawl_keeps_going_past_an_ordinary_error(monkeypatch):
    page = '<html><body><a href="/a">A</a><a href="/b">B</a>text</body></html>'

    def fake_fetch_page(url, lang="en"):
        if url.endswith("/a"):
            return url, None, None, "HTTP 404"
        return url, page, None, None

    monkeypatch.setattr(tools_extra, "fetch_page", fake_fetch_page)

    result = tools_extra.web_crawl("https://host.example/", max_pages=4)

    assert "stopped_reason" not in result
    assert result["pages_crawled"] >= 3


# ── item 4: parallelism across hosts, never within one ──

def test_parallel_fetch_never_hits_one_host_concurrently(monkeypatch):
    lock = threading.Lock()
    inflight: dict[str, int] = {}
    peak = {"per_host": 0, "hosts": 0}

    def fake_fetch(url, query, lang):
        host = politeness.domain_of(url)
        with lock:
            inflight[host] = inflight.get(host, 0) + 1
            peak["per_host"] = max(peak["per_host"], inflight[host])
            peak["hosts"] = max(peak["hosts"], sum(1 for n in inflight.values() if n))
        time.sleep(0.02)  # long enough for a same-host overlap to be observable
        with lock:
            inflight[host] -= 1
        return url, {"text": "x", "pub_date": None}

    monkeypatch.setattr(fetch, "_fetch_and_extract", fake_fetch)

    urls = [f"https://a.example/{i}" for i in range(5)] + [f"https://b.example/{i}" for i in range(5)]
    results = fetch.fetch_pages_parallel(urls)

    assert len(results) == 10
    assert peak["per_host"] == 1  # one host is never fetched by two workers at once
    assert peak["hosts"] == 2     # but the two hosts do run in parallel


def test_parallel_fetch_skips_a_host_whose_circuit_is_open(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_BREAKER_THRESHOLD", "1")
    politeness.BREAKER.record_failure("blocked.example")
    seen = []

    monkeypatch.setattr(
        fetch, "_fetch_and_extract",
        lambda url, q, lang: (seen.append(url) or (url, {"text": "x", "pub_date": None})),
    )

    fetch.fetch_pages_parallel(["https://blocked.example/1", "https://ok.example/1"])

    assert seen == ["https://ok.example/1"]


# ── item 5: conditional requests ──

def test_stored_validators_are_sent_and_a_304_is_served_from_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTNOTE_HTTP_CACHE", "1")
    monkeypatch.setenv("FOOTNOTE_SOURCE_CACHE", str(tmp_path))
    url = "https://gov.example/report"
    body = "<html><body>Annual report 2026</body></html>"

    assert http_cache.store(url, FakeResp(200, body, {"ETag": '"v1"'})) is True

    sent = {}

    def fake_get(u, lang="en", cookies=None, max_retries=2, timeout=15, extra_headers=None, proxies=None):
        sent.update(extra_headers or {})
        return FakeResp(304, "", {})

    monkeypatch.setattr(fetch, "_get", fake_get)

    fetched_url, html, _pub, err = fetch.fetch_page(url)

    assert sent == {"If-None-Match": '"v1"'}
    assert err is None
    assert html == body  # served locally; no body crossed the wire


def test_a_response_without_a_validator_is_not_stored(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTNOTE_HTTP_CACHE", "1")
    monkeypatch.setenv("FOOTNOTE_SOURCE_CACHE", str(tmp_path))
    url = "https://nocache.example/page"

    assert http_cache.store(url, FakeResp(200, "<html>x</html>", {})) is False
    assert http_cache.conditional_headers(url) == {}
    assert http_cache.load(url) is None


def test_a_304_without_a_local_copy_falls_back_to_a_full_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTNOTE_HTTP_CACHE", "1")
    monkeypatch.setenv("FOOTNOTE_SOURCE_CACHE", str(tmp_path))
    responses = [FakeResp(304, "", {}), FakeResp(200, "<html>fresh</html>", {})]
    monkeypatch.setattr(fetch, "_get", lambda *a, **k: responses.pop(0))

    _url, html, _pub, err = fetch.fetch_page("https://gov.example/gone-from-cache")

    assert err is None
    assert html == "<html>fresh</html>"


def test_http_cache_is_off_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTNOTE_HTTP_CACHE", "0")
    monkeypatch.setenv("FOOTNOTE_SOURCE_CACHE", str(tmp_path))
    url = "https://gov.example/report"

    assert http_cache.store(url, FakeResp(200, "<html>x</html>", {"ETag": '"v1"'})) is False
    assert list(tmp_path.glob("**/*.json")) == []

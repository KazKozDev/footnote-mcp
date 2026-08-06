"""Shared test fixtures.

The scraper escalation ladder is configurable via env. By default in tests we keep
it offline and deterministic: browser fallback off, no proxies, no external API,
no rate-limit pacing, no negative-cache carryover, and no on-disk HTTP validator
cache. Tests that exercise the ladder opt in explicitly (via env or by passing
allow_* flags).
"""

import pytest

from footnote_mcp import scraper


@pytest.fixture(autouse=True)
def _offline_scraper_defaults(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_BROWSER_FALLBACK", "0")
    monkeypatch.setenv("FOOTNOTE_PROXIES", "")
    monkeypatch.delenv("FOOTNOTE_SCRAPE_API", raising=False)
    monkeypatch.setenv("FOOTNOTE_DOMAIN_RPS", "0")        # disable pacing
    monkeypatch.setenv("FOOTNOTE_NEGCACHE_TTL", "0")      # disable negative cache
    monkeypatch.setenv("FOOTNOTE_HTTP_CACHE", "0")        # never touch the real cache dir
    scraper.reset_state()
    yield
    scraper.reset_state()

"""Shared test fixtures.

The scraper escalation ladder is configurable via env. By default in tests we keep
it offline and deterministic: browser fallback off, no proxies, no external API,
no rate-limit pacing, and no negative-cache carryover. Tests that exercise the
ladder opt in explicitly (via env or by passing allow_* flags).
"""

import pytest

import scraper


@pytest.fixture(autouse=True)
def _offline_scraper_defaults(monkeypatch):
    monkeypatch.setenv("WEBOPERATOR_BROWSER_FALLBACK", "0")
    monkeypatch.setenv("WEBOPERATOR_PROXIES", "")
    monkeypatch.delenv("WEBOPERATOR_SCRAPE_API", raising=False)
    monkeypatch.setenv("WEBOPERATOR_DOMAIN_RPS", "0")        # disable pacing
    monkeypatch.setenv("WEBOPERATOR_NEGCACHE_TTL", "0")      # disable negative cache
    scraper.reset_state()
    yield
    scraper.reset_state()

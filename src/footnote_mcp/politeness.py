"""Per-domain politeness shared by every outbound request.

The rate limiter, circuit breaker and negative cache live here rather than inside
``scraper`` so that ``fetch._get`` — the single funnel every tool's HTTP request
passes through — can apply them. Pacing enforced at the funnel cannot be skipped
by calling a lower-level helper: previously only ``web_read`` and deep research
went through the escalation ladder, while table extraction, file parsing, JSON
endpoints, crawling and archive lookups reached the network unpaced.
"""

from __future__ import annotations

import os
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

# Statuses that mean "the other side is refusing us", not "this page is missing".
BLOCK_STATUSES = (401, 403, 407, 429, 503)
# Of those, the ones worth waiting out and retrying rather than giving up on.
RETRY_STATUSES = (429, 503)


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def retry_after_seconds(response=None, default: float = 0.0, cap: float = 3600.0) -> float:
    """Seconds to wait before retrying, honoring ``Retry-After`` when present.

    Accepts both forms the RFC allows: delta-seconds and an HTTP date. The result
    is never below ``default`` (the caller's own backoff) nor above ``cap``.
    """
    try:
        seconds = max(0.0, float(default))
    except (TypeError, ValueError):
        seconds = 0.0

    headers = getattr(response, "headers", None) or {}
    raw = ""
    try:
        raw = str(headers.get("Retry-After") or headers.get("retry-after") or "").strip()
    except Exception:
        raw = ""

    if raw:
        try:
            seconds = max(seconds, float(raw))
        except ValueError:
            try:
                when = parsedate_to_datetime(raw)
                if when is not None:
                    import datetime as _dt

                    now = _dt.datetime.now(when.tzinfo) if when.tzinfo else _dt.datetime.now()
                    seconds = max(seconds, (when - now).total_seconds())
            except Exception:
                pass

    return max(0.0, min(seconds, cap))


def is_block_error(error) -> bool:
    """True when a ``fetch_page``-style error string reports a refusal status."""
    text = str(error or "").lower()
    if not text:
        return False
    return any(f"http {status}" in text for status in BLOCK_STATUSES)


class DomainRateLimiter:
    """Token-bucket limiter per domain. rps<=0 disables pacing."""

    def __init__(self, rps=None, burst=None):
        self._rps = rps
        self._burst = burst
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}

    def _params(self):
        rps = self._rps if self._rps is not None else float(os.getenv("FOOTNOTE_DOMAIN_RPS", "3"))
        burst = self._burst if self._burst is not None else float(os.getenv("FOOTNOTE_DOMAIN_BURST", "5"))
        return rps, burst

    def acquire(self, domain: str) -> float:
        rps, burst = self._params()
        if rps <= 0:
            return 0.0
        with self._lock:
            tokens, last = self._state.get(domain, (burst, time.monotonic()))
            now = time.monotonic()
            tokens = min(burst, tokens + (now - last) * rps)
            if tokens >= 1:
                self._state[domain] = (tokens - 1, now)
                wait = 0.0
            else:
                wait = (1 - tokens) / rps
                self._state[domain] = (0.0, now + wait)
        if wait > 0:
            time.sleep(wait)
        return wait

    def reset(self):
        with self._lock:
            self._state.clear()


class CircuitBreaker:
    """Open per-domain after N consecutive failures; skip expensive tiers while open."""

    def __init__(self, threshold=None, cooldown=None):
        self._threshold = threshold
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._fail: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def _params(self):
        threshold = self._threshold if self._threshold is not None else int(os.getenv("FOOTNOTE_BREAKER_THRESHOLD", "5"))
        cooldown = self._cooldown if self._cooldown is not None else float(os.getenv("FOOTNOTE_BREAKER_COOLDOWN", "120"))
        return threshold, cooldown

    def is_open(self, domain: str) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until.get(domain, 0.0)

    def record_failure(self, domain: str):
        threshold, cooldown = self._params()
        with self._lock:
            n = self._fail.get(domain, 0) + 1
            self._fail[domain] = n
            if n >= threshold:
                self._open_until[domain] = time.monotonic() + cooldown

    def record_success(self, domain: str):
        with self._lock:
            self._fail.pop(domain, None)
            self._open_until.pop(domain, None)

    def reset(self):
        with self._lock:
            self._fail.clear()
            self._open_until.clear()


class NegativeCache:
    """Remember recently-blocked URLs so we don't immediately retry them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, float]] = {}

    def get(self, url: str):
        with self._lock:
            entry = self._entries.get(url)
            if entry and time.monotonic() < entry[1]:
                return entry[0]
            if entry:
                self._entries.pop(url, None)
            return None

    def put(self, url: str, reason: str):
        ttl = float(os.getenv("FOOTNOTE_NEGCACHE_TTL", "300"))
        if ttl <= 0:
            return
        with self._lock:
            self._entries[url] = (reason, time.monotonic() + ttl)

    def reset(self):
        with self._lock:
            self._entries.clear()


# ── Process-wide shared state ──────────────────────────────────────────────
# One set of buckets for the whole process: the limiter is only meaningful when
# every caller shares it.

RATE_LIMITER = DomainRateLimiter()
BREAKER = CircuitBreaker()
NEG_CACHE = NegativeCache()


def reset_state():
    """Reset limiter/breaker/negative-cache state (used by tests)."""
    RATE_LIMITER.reset()
    BREAKER.reset()
    NEG_CACHE.reset()

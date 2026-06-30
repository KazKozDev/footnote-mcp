"""Escalation-ladder page fetcher (anti-bot / scale).

Cheapest method first; escalate only when the result looks blocked or empty:

  tier 1  http            curl_cffi with browser TLS impersonation (fast path)
  tier 2  http_proxy      retry through a rotating proxy
  tier 3  browser         headless Chromium (executes JavaScript)
  tier 4  browser_proxy   Chromium through a proxy
  tier 5  external         hosted scrape API (Firecrawl / ScrapingBee)

Cross-cutting: a block/quality detector decides when to escalate; a per-domain
rate limiter and circuit breaker keep us polite; a negative cache avoids retrying
known-dead URLs. Everything is optional and degrades to the plain http path when
nothing is configured (zero-config == previous behaviour).
"""

from __future__ import annotations

import os
import queue
import random
import re
import threading
import time
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as http

from . import core
from .fetch import _get, fetch_page

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


# ── Block / quality detection ──────────────────────────────────────────────

_CHALLENGE_MARKERS = (
    "just a moment", "cf-chl", "cf-browser-verification", "challenge-platform",
    "/cdn-cgi/challenge", "g-recaptcha", "h-captcha", "hcaptcha", "recaptcha",
    "are you a robot", "verify you are human", "px-captcha", "captcha-delivery",
)
_JS_REQUIRED_MARKERS = (
    "enable javascript", "please enable js", "javascript is required",
    "you need to enable javascript", "javascript to run this app",
)


def _strip_tags(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())
    except Exception:
        return re.sub(r"<[^>]+>", " ", html or "")


def _thin_threshold() -> int:
    return int(os.getenv("FOOTNOTE_THIN_CONTENT_CHARS", "200"))


def detect_block(status, html, error=None):
    """Return (blocked: bool, reason: str) for a fetch result."""
    if error and not html:
        return True, str(error)
    if status in (401, 403, 407, 429, 503):
        return True, f"http_{status}"
    html_l = (html or "").lower()
    if not html_l.strip():
        return True, "empty"
    for marker in _CHALLENGE_MARKERS:
        if marker in html_l:
            return True, "challenge_or_captcha"
    for marker in _JS_REQUIRED_MARKERS:
        if marker in html_l:
            return True, "js_required"
    # A JS-rendered shell: big HTML with scripts but almost no readable text.
    if len(html_l) > 2000 and "<script" in html_l and len(_strip_tags(html)) < _thin_threshold():
        return True, "thin_js_shell"
    return False, ""


# ── Per-domain politeness: rate limiter + circuit breaker + negative cache ──

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


# ── Proxy pool: sticky-per-domain + health ─────────────────────────────────

class ProxyPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._dead: dict[str, float] = {}
        self._sticky: dict[str, str] = {}

    def _list(self):
        return [p.strip() for p in os.getenv("FOOTNOTE_PROXIES", "").split(",") if p.strip()]

    def available(self) -> bool:
        return bool(self._list())

    def _is_dead(self, proxy: str) -> bool:
        return time.monotonic() < self._dead.get(proxy, 0.0)

    def get(self, domain: str, rotate: bool = False):
        proxies = self._list()
        if not proxies:
            return None
        with self._lock:
            if rotate:
                self._sticky.pop(domain, None)
            current = self._sticky.get(domain)
            if current and current in proxies and not self._is_dead(current):
                return current
            alive = [p for p in proxies if not self._is_dead(p)]
            choice = random.choice(alive or proxies)
            self._sticky[domain] = choice
            return choice

    def report(self, proxy, ok: bool):
        if not proxy:
            return
        with self._lock:
            if ok:
                self._dead.pop(proxy, None)
            else:
                self._dead[proxy] = time.monotonic() + float(os.getenv("FOOTNOTE_PROXY_COOLDOWN", "300"))

    def reset(self):
        with self._lock:
            self._dead.clear()
            self._sticky.clear()


# ── Headless browser renderer: one reused Chromium in a dedicated thread ────

class BrowserRenderer:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._error: str | None = None

    def _ensure(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # playwright not installed
            self._error = f"playwright unavailable: {exc}"
            self._serve_errors()
            return
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    url, lang, proxy, timeout, reply = item
                    try:
                        ctx_args = {"locale": lang or "en", "user_agent": _UA}
                        if proxy:
                            ctx_args["proxy"] = {"server": proxy}
                        context = browser.new_context(**ctx_args)
                        context.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
                        page = context.new_page()
                        page.goto(url, wait_until="domcontentloaded", timeout=int(timeout * 1000))
                        try:
                            page.wait_for_load_state("networkidle", timeout=4000)
                        except Exception:
                            pass
                        html = page.content()
                        context.close()
                        reply.put((html, 200, None))
                    except Exception as exc:
                        reply.put((None, None, str(exc)))
        except Exception as exc:
            self._error = f"browser loop failed: {exc}"
            self._serve_errors()

    def _serve_errors(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            item[-1].put((None, None, self._error))

    def render(self, url, lang="en", proxy=None, timeout=None):
        timeout = timeout or float(os.getenv("FOOTNOTE_BROWSER_TIMEOUT", "25"))
        self._ensure()
        if self._error:
            return None, None, self._error
        reply: queue.Queue = queue.Queue()
        self._queue.put((url, lang, proxy, timeout, reply))
        try:
            return reply.get(timeout=timeout + 15)
        except Exception as exc:
            return None, None, f"render timeout: {exc}"


# ── External hosted scrape APIs (tier 5) ────────────────────────────────────

def _scrape_external(url, lang="en"):
    provider = os.getenv("FOOTNOTE_SCRAPE_API", "").lower()
    if provider == "firecrawl":
        key = os.getenv("FIRECRAWL_API_KEY")
        if not key:
            return None, None, "firecrawl key missing"
        resp = http.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["html"]},
            timeout=60,
        )
        if resp.status_code != 200:
            return None, resp.status_code, f"firecrawl HTTP {resp.status_code}"
        data = resp.json()
        html = (data.get("data") or {}).get("html") or data.get("html")
        return html, 200, (None if html else "firecrawl empty")
    if provider == "scrapingbee":
        key = os.getenv("SCRAPINGBEE_API_KEY")
        if not key:
            return None, None, "scrapingbee key missing"
        resp = http.get(
            "https://app.scrapingbee.com/api/v1/",
            params={"api_key": key, "url": url, "render_js": "true"},
            timeout=60,
        )
        if resp.status_code != 200:
            return None, resp.status_code, f"scrapingbee HTTP {resp.status_code}"
        return resp.text, 200, None
    return None, None, "no external scrape api configured"


# ── Module-level shared state ──────────────────────────────────────────────

_RATE_LIMITER = DomainRateLimiter()
_BREAKER = CircuitBreaker()
_NEG_CACHE = NegativeCache()
_PROXIES = ProxyPool()
_RENDERER = BrowserRenderer()


def reset_state():
    """Reset all in-memory limiter/breaker/cache/proxy state (used by tests)."""
    _RATE_LIMITER.reset()
    _BREAKER.reset()
    _NEG_CACHE.reset()
    _PROXIES.reset()


def _browser_enabled(override):
    if override is not None:
        return override
    return _env_bool("FOOTNOTE_BROWSER_FALLBACK", True)


def _proxy_enabled(override):
    if override is not None:
        return override
    return _PROXIES.available()


def _external_enabled(override):
    if override is not None:
        return override
    return bool(os.getenv("FOOTNOTE_SCRAPE_API", "").strip())


def _pub_date(html):
    try:
        from .extract import _extract_publish_date

        return _extract_publish_date(html)
    except Exception:
        return None


def _http_fetch_proxy(url, lang, proxy):
    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        resp = _get(url, lang=lang, timeout=core.FETCH_TIMEOUT, max_retries=1, proxies=proxies)
        if resp.status_code == 200:
            return url, resp.text, _pub_date(resp.text), None
        return url, None, None, f"HTTP {resp.status_code}"
    except Exception as exc:
        return url, None, None, str(exc)


def _result(url, final_url, html, pub_date, tier, tiers, blocked, error=None):
    return {
        "url": url,
        "final_url": final_url or url,
        "html": html,
        "pub_date": pub_date,
        "tier": tier,
        "tiers_tried": [{"tier": t, "status": s, "reason": r} for t, s, r in tiers],
        "blocked": blocked,
        "error": error,
    }


def fetch(url, lang="en", http_fn=None, allow_browser=None, allow_proxy=None, allow_external=None):
    """Fetch a page through the escalation ladder. Returns a result dict.

    ``http_fn`` is the tier-1 HTTP fetcher (defaults to ``fetch.fetch_page``);
    injectable so callers/tests can supply their own.
    """
    http_fn = http_fn or fetch_page
    domain = _domain(url)
    tiers: list[tuple] = []

    reason = _NEG_CACHE.get(url)
    if reason:
        return _result(url, url, None, None, "negative_cache", [("negative_cache", None, reason)], blocked=True,
                       error=f"recently blocked: {reason}")

    breaker_open = _BREAKER.is_open(domain)
    _RATE_LIMITER.acquire(domain)

    best_final = best_html = best_pub = None  # best partial result seen, for graceful degrade

    # tier 1: plain HTTP via injected fetcher
    final_url, html, pub_date, error = http_fn(url, lang=lang)
    status = 200 if (html and not error) else 0
    blocked, why = detect_block(status, html, error)
    tiers.append(("http", status or None, why or "ok"))
    if html and not blocked:
        _BREAKER.record_success(domain)
        return _result(url, final_url, html, pub_date, "http", tiers, blocked=False)
    if html:
        best_final, best_html, best_pub = final_url, html, pub_date
    # Carry the real fetch error (HTTP 500, exception) for the user-facing field;
    # 'why' (e.g. "empty") stays in tiers_tried for observability only.
    first_error = error

    # tier 2: HTTP retry through a proxy
    if _proxy_enabled(allow_proxy) and _PROXIES.available() and not breaker_open:
        proxy = _PROXIES.get(domain)
        f2, h2, p2, e2 = _http_fetch_proxy(url, lang, proxy)
        s2 = 200 if (h2 and not e2) else 0
        b2, why2 = detect_block(s2, h2, e2)
        tiers.append(("http_proxy", s2 or None, why2 or "ok"))
        if h2 and not b2:
            _PROXIES.report(proxy, ok=True)
            _BREAKER.record_success(domain)
            return _result(url, f2, h2, p2, "http_proxy", tiers, blocked=False)
        _PROXIES.report(proxy, ok=False)
        if h2 and not best_html:
            best_final, best_html, best_pub = f2, h2, p2

    # tier 3: headless browser
    if _browser_enabled(allow_browser) and not breaker_open:
        h3, s3, e3 = _RENDERER.render(url, lang=lang, proxy=None)
        b3, why3 = detect_block(s3 or 0, h3, e3)
        tiers.append(("browser", s3, why3 or "ok"))
        if h3 and not b3:
            _BREAKER.record_success(domain)
            return _result(url, url, h3, _pub_date(h3), "browser", tiers, blocked=False)
        if h3 and not best_html:
            best_final, best_html, best_pub = url, h3, _pub_date(h3)

        # tier 4: browser through a proxy
        if _proxy_enabled(allow_proxy) and _PROXIES.available():
            proxy = _PROXIES.get(domain, rotate=True)
            h4, s4, e4 = _RENDERER.render(url, lang=lang, proxy=proxy)
            b4, why4 = detect_block(s4 or 0, h4, e4)
            tiers.append(("browser_proxy", s4, why4 or "ok"))
            if h4 and not b4:
                _PROXIES.report(proxy, ok=True)
                _BREAKER.record_success(domain)
                return _result(url, url, h4, _pub_date(h4), "browser_proxy", tiers, blocked=False)
            _PROXIES.report(proxy, ok=False)
            if h4 and not best_html:
                best_final, best_html, best_pub = url, h4, _pub_date(h4)

    # tier 5: external hosted scrape API
    if _external_enabled(allow_external) and not breaker_open:
        h5, s5, e5 = _scrape_external(url, lang)
        b5, why5 = detect_block(s5 or 0, h5, e5)
        tiers.append(("external", s5, why5 or "ok"))
        if h5 and not b5:
            _BREAKER.record_success(domain)
            return _result(url, url, h5, _pub_date(h5), "external", tiers, blocked=False)
        if h5 and not best_html:
            best_final, best_html, best_pub = url, h5, _pub_date(h5)

    # all tiers exhausted: degrade gracefully to the best partial html we got
    _BREAKER.record_failure(domain)
    _NEG_CACHE.put(url, first_error or (tiers[-1][2] if tiers else "blocked"))
    if best_html:
        return _result(url, best_final, best_html, best_pub, tiers[0][0], tiers, blocked=True,
                       error=None)
    return _result(url, final_url, None, None, tiers[-1][0] if tiers else None, tiers, blocked=True,
                   error=first_error)

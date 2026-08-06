from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests as http

from . import http_cache
from .diagnostics import log
from .politeness import BREAKER, RATE_LIMITER, RETRY_STATUSES, domain_of, retry_after_seconds


def _imp():
    from . import core

    return random.choice(core.IMPERSONATE)


def _headers(lang="en"):
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": f"{lang},en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _max_retry_sleep() -> float:
    """Longest we are willing to block inside one request waiting out a limit."""
    try:
        return max(0.0, float(os.getenv("FOOTNOTE_RETRY_AFTER_MAX_SECONDS", "30")))
    except ValueError:
        return 30.0


def _backoff(attempt: int) -> float:
    return min(2.0 * (2**attempt), 8.0) + random.uniform(0.0, 0.5)


def _get(url, lang="en", cookies=None, max_retries=2, timeout=15, extra_headers=None, proxies=None):
    """Single HTTP GET with TLS impersonation, per-domain pacing and retries.

    Every outbound request in the server funnels through here, so this is where
    politeness is enforced: a token bucket paces requests per domain, and a 429 or
    503 is waited out (honoring ``Retry-After``) instead of being hammered. The
    circuit breaker is *recorded* here but not enforced — a single tool call the
    user explicitly asked for still goes out; loops (crawl, parallel fetch) consult
    the breaker themselves and stop.

    ``extra_headers`` (optional) are merged on top of the default browser headers,
    which lets callers attach auth tokens or custom headers for gated pages.
    ``proxies`` (optional) routes the request through a proxy, e.g.
    ``{"http": "http://host:port", "https": "http://host:port"}``.
    """
    headers = _headers(lang)
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    domain = domain_of(url)
    last_err = None
    for attempt in range(max_retries + 1):
        RATE_LIMITER.acquire(domain)
        try:
            resp = http.get(
                url,
                headers=headers,
                cookies=cookies,
                impersonate=_imp(),
                timeout=timeout,
                allow_redirects=True,
                proxies=proxies,
            )
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(random.uniform(0.5, 1.5))
                continue
            BREAKER.record_failure(domain)
            raise last_err

        if resp.status_code not in RETRY_STATUSES:
            if resp.status_code < 400:
                BREAKER.record_success(domain)
            return resp

        # The server is refusing, not failing. Retrying immediately is what turns a
        # soft rate limit into a ban, so wait for as long as it asked — unless that
        # is longer than a tool call may reasonably block, in which case hand the
        # refusal back and let the breaker keep later calls away.
        server_wait = retry_after_seconds(resp)
        wait = server_wait if server_wait > 0 else _backoff(attempt)
        if attempt >= max_retries or wait > _max_retry_sleep():
            BREAKER.record_failure(domain)
            log.warning("[FETCH] %s refused with HTTP %s", domain, resp.status_code)
            return resp
        log.info("[FETCH] %s returned HTTP %s; waiting %.1fs", domain, resp.status_code, wait)
        time.sleep(wait)

    # Unreachable: every path above either returns or raises.
    raise last_err or RuntimeError(f"no response for {url}")


def fetch_page(url, lang="en"):
    """Fetch a single page → return (url, raw_html, publish_date, error).

    Sends the stored validators for this URL, so an unchanged page comes back as a
    bodiless 304 and is served from the local copy.
    """
    from . import core
    from .extract import _extract_publish_date

    try:
        validators = http_cache.conditional_headers(url)
        resp = _get(url, lang, timeout=core.FETCH_TIMEOUT, max_retries=1, extra_headers=validators or None)
        if resp.status_code == 304:
            cached = http_cache.load(url)
            if cached:
                return (url, cached, _extract_publish_date(cached), None)
            # Unchanged against a copy we no longer hold: ask again in full.
            resp = _get(url, lang, timeout=core.FETCH_TIMEOUT, max_retries=1)
        if resp.status_code == 200:
            http_cache.store(url, resp)
            pub_date = _extract_publish_date(resp.text)
            return (url, resp.text, pub_date, None)
        return (url, None, None, f"HTTP {resp.status_code}")
    except Exception as exc:
        return (url, None, None, str(exc))


def _fetch_and_extract(url, query, lang):
    """Fetch one URL and extract usable text. Returns (final_url, payload) or None."""
    from . import core
    from .extract import extract_content, is_content_page

    try:
        fetched_url, html, pub_date, err = fetch_page(url, lang=lang)
        if err:
            log.info("[FETCH] failed %s... - %s", url[:60], err)
            return None

        text = extract_content(html, url=fetched_url)
        if not text or len(text) <= 50:
            log.info("[FETCH] rejected %s... - empty after extraction", url[:60])
            return None
        if not is_content_page(text, query=query, lang=lang):
            log.info("[FETCH] rejected %s... - low-quality content", url[:60])
            return None

        text = text[:core.MAX_CONTENT_CHARS]
        date_str = pub_date.strftime("%Y-%m-%d") if pub_date else "unknown"
        log.info("[FETCH] ok %s... - %s chars, date=%s", url[:60], len(text), date_str)
        return fetched_url, {"text": text, "pub_date": pub_date}
    except Exception as exc:
        log.warning("[FETCH] failed %s... - %s", url[:60], exc)
        return None


def _fetch_host_serial(host, host_urls, query, lang):
    """Fetch one host's URLs one at a time, stopping if the host starts refusing."""
    fetched = {}
    for index, url in enumerate(host_urls):
        if BREAKER.is_open(host):
            log.warning("[FETCH] %s circuit open; skipping %s remaining URL(s)", host, len(host_urls) - index)
            break
        result = _fetch_and_extract(url, query, lang)
        if result:
            fetched[result[0]] = result[1]
    return fetched


def fetch_pages_parallel(urls, query=None, lang="en"):
    """Fetch multiple pages in parallel and return extracted text by URL.

    Parallelism is across hosts, never within one: ten workers all landing on the
    same domain is the shape of a request burst that gets an IP blocked. URLs from
    one host are fetched sequentially (and paced by the limiter in ``_get``).
    """
    from . import core

    by_host: dict[str, list] = {}
    for url in urls:
        by_host.setdefault(domain_of(url), []).append(url)
    if not by_host:
        return {}

    results = {}
    workers = max(1, min(core.FETCH_WORKERS, len(by_host)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_host_serial, host, host_urls, query, lang): host
            for host, host_urls in by_host.items()
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                results.update(future.result())
            except Exception as exc:
                log.warning("[FETCH] host %s failed - %s", host, exc)

    return results

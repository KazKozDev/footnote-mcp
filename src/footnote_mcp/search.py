from __future__ import annotations

import base64
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlencode, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as http

from .diagnostics import log
from .fetch import _get
from .politeness import retry_after_seconds


_SEARCH_STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "what", "where", "when",
    "как", "для", "или", "что", "это", "где", "когда", "при", "про",
}
_GENERIC_SEARCH_TERMS = {
    "context", "documentation", "docs", "guide", "github", "language", "model", "official",
    "programming", "protocol", "search", "today", "tutorial", "weather",
    "документация", "официальный", "официальная", "поиск", "погода", "сегодня",
}

_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}
_PROVIDER_COOLDOWN_LOCK = threading.Lock()
_PROVIDER_COOLDOWN_DEFAULTS = {"brave": 300.0, "ddg": 120.0}
_PROVIDER_FALLBACK_COOLDOWN = 180.0
# Engines that can rest. Used to tell "this one is resting" from "there is
# nothing left to ask", which must never happen.
_BACKOFF_ENGINES = ("bing", "ddg", "brave", "marginalia", "wiby")


def _provider_cooldown_seconds(engine, response=None):
    """Return a bounded rate-limit cooldown, honoring Retry-After when present."""
    env_name = f"FOOTNOTE_{engine.upper()}_COOLDOWN_SECONDS"
    default = _PROVIDER_COOLDOWN_DEFAULTS.get(engine, _PROVIDER_FALLBACK_COOLDOWN)
    try:
        seconds = max(0.0, float(os.getenv(env_name, default)))
    except (TypeError, ValueError):
        seconds = default
    return retry_after_seconds(response, default=seconds, cap=3600.0)


def _start_provider_cooldown(engine, response=None):
    seconds = _provider_cooldown_seconds(engine, response)
    if seconds <= 0:
        return
    until = time.monotonic() + seconds
    with _PROVIDER_COOLDOWN_LOCK:
        _PROVIDER_COOLDOWN_UNTIL[engine] = max(_PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0), until)
    log.warning("[%s] Rate limited; cooling down for %.0fs", engine.upper(), seconds)


def _provider_on_cooldown(engine):
    now = time.monotonic()
    with _PROVIDER_COOLDOWN_LOCK:
        until = _PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0)
        remaining = until - now
        if remaining <= 0:
            _PROVIDER_COOLDOWN_UNTIL.pop(engine, None)
            return False
        everything_resting = all(
            _PROVIDER_COOLDOWN_UNTIL.get(name, 0.0) > now for name in _BACKOFF_ENGINES
        )
    if everything_resting:
        # Backoff exists to stop paying for a broken endpoint, not to switch
        # search off. With nothing else left to ask, a blocked attempt still
        # beats returning no candidates at all.
        log.warning("[%s] Every provider is resting; querying anyway", engine.upper())
        return False
    log.debug("[%s] Cooldown active; skipping request for %.0fs", engine.upper(), remaining)
    return True


def _search_terms(text):
    return {
        token
        for token in re.findall(r"[\w]+", unquote(text).lower(), flags=re.UNICODE)
        if len(token) > 2 and token not in _SEARCH_STOPWORDS
    }


def _validate_result_relevance(query, results, engine):
    """Reject result pages with no meaningful lexical connection to the query.

    Some zero-key sources return HTTP 200 with results unrelated to the query.
    Require either a distinctive query term or broad coverage across generic
    terms, then discard individual rows with no lexical connection to the query.
    """
    query_terms = _search_terms(query)
    if not query_terms or not results:
        return results

    matched_terms = set()
    overlaps = []
    for result in results:
        result_terms = _search_terms(
            f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}"
        )
        overlap = query_terms & result_terms
        overlaps.append(overlap)
        matched_terms.update(overlap)

    distinctive_terms = query_terms - _GENERIC_SEARCH_TERMS
    broad_coverage = 1 if len(query_terms) == 1 else max(2, (len(query_terms) * 3 + 4) // 5)
    has_distinctive_match = bool(matched_terms & distinctive_terms)
    has_broad_coverage = len(matched_terms) >= broad_coverage
    if not has_distinctive_match and not has_broad_coverage:
        log.warning(
            "[%s] Rejected unrelated result page for query %r (matched %s/%s query terms)",
            engine.upper(),
            query,
            len(matched_terms),
            len(query_terms),
        )
        return []

    return [
        result
        for result, overlap in zip(results, overlaps)
        if (overlap & distinctive_terms) or len(overlap) >= broad_coverage
    ]


def _dedupe_source_results(results):
    """Deduplicate one provider without turning repeated rows into rank votes."""
    deduped = {}
    order = []
    for result in results:
        url = str(result.get("url") or "")
        if not url:
            continue
        norm = _normalize_url(url)
        if norm not in deduped:
            deduped[norm] = dict(result)
            order.append(norm)
            continue
        current = deduped[norm]
        if len(str(result.get("title") or "")) > len(str(current.get("title") or "")):
            current["title"] = result["title"]
        if len(str(result.get("snippet") or "")) > len(str(current.get("snippet") or "")):
            current["snippet"] = result["snippet"]
        for key in ("attribution", "license"):
            if result.get(key) and not current.get(key):
                current[key] = result[key]
    return [deduped[norm] for norm in order]


def _prepare_source_results(query, results, engine):
    """Apply the mandatory per-source relevance and deduplication contract."""
    relevant = _validate_result_relevance(query, results, engine)
    return _dedupe_source_results(relevant)


def _bing_unwrap_url(href):
    if "bing.com/ck/a" not in href:
        return href
    try:
        parsed = parse_qs(urlparse(href).query)
        if "u" in parsed:
            raw = parsed["u"][0]
            if raw.startswith("a1"):
                raw = raw[2:]
            decoded = base64.urlsafe_b64decode(raw).decode("utf-8", errors="ignore")
            if decoded.startswith("http"):
                return decoded
        if "r" in parsed and parsed["r"][0].startswith("http"):
            return parsed["r"][0]
    except Exception:
        pass
    return href


def search_bing(query, num=None, lang="en", debug=False):
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    params = {"q": query, "count": min(num + 5, 30), "setlang": lang}
    if lang == "en":
        params["cc"] = "US"
        params["setmkt"] = "en-US"

    url = f"https://www.bing.com/search?{urlencode(params)}"
    if debug:
        log.debug("[BING] %s", url)

    try:
        resp = _get(url, lang)
    except Exception as exc:
        log.warning("[BING] Request failed: %s", exc)
        return []

    if debug:
        with open("debug_bing.html", "w", encoding="utf-8") as handle:
            handle.write(resp.text)
        log.debug("[BING] Status %s, %s bytes -> debug_bing.html", resp.status_code, len(resp.text))

    if resp.status_code != 200:
        log.warning("[BING] HTTP %s", resp.status_code)
        return []

    response_text = resp.text.lower()
    if "one last step" in response_text and ("captcha" in response_text or "challenge" in response_text):
        log.warning("[BING] Blocked by anti-bot challenge for query %r", query)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    def _add(title, href, snippet=""):
        if not title or not href:
            return
        real_url = _bing_unwrap_url(href)
        if not real_url.startswith("http"):
            return
        host = urlparse(real_url).hostname or ""
        if host.endswith("bing.com") or host.endswith("microsoft.com"):
            return
        norm = real_url.split("?")[0].split("#")[0].rstrip("/").lower()
        if norm in seen:
            return
        seen.add(norm)
        results.append({"title": title.strip(), "url": real_url, "snippet": (snippet or "").strip()})

    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a") or li.select_one("a[href]")
        if not a or not a.get("href", ""):
            continue
        title = a.get_text(strip=True)
        link = a["href"]
        snippet = ""
        for sel in ["div.b_caption p", "p.b_lineclamp2", "p.b_lineclamp3", "p.b_lineclamp4", "div.b_caption .b_snippet"]:
            el = li.select_one(sel)
            if el:
                snippet = el.get_text(" ", strip=True)
                break
        if not snippet:
            cap = li.select_one("div.b_caption")
            if cap:
                snippet = cap.get_text(" ", strip=True)[:300]
        _add(title, link, snippet)

    if not results:
        for h2 in soup.select("h2"):
            a = h2.select_one("a[href]")
            if a and a.get("href", ""):
                _add(a.get_text(strip=True), a["href"])

    had_parsed_results = bool(results)
    results = _prepare_source_results(query, results, "bing")

    if not results and not had_parsed_results:
        log.warning("[BING] Parser returned 0 results for query %r; search markup may have changed", query)

    if debug:
        log.debug("[BING] %s results", len(results))
    return results[:num]


def _ddg_extract_real_url(href):
    if "duckduckgo.com" in href and "uddg=" in href:
        parsed = parse_qs(urlparse(href).query)
        if "uddg" in parsed:
            return parsed["uddg"][0]
    return href


def search_ddg(query, num=None, lang="en", debug=False, df=""):
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    if _provider_on_cooldown("ddg"):
        return []

    # df = DuckDuckGo freshness filter: d (day), w (week), m (month), y (year).
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    if df in ("d", "w", "m", "y"):
        url += f"&df={df}"

    if debug:
        log.debug("[DDG] %s", url)

    try:
        resp = _get(url, lang)
    except Exception as exc:
        log.warning("[DDG] Request failed: %s", exc)
        return []

    if debug:
        with open("debug_ddg.html", "w", encoding="utf-8") as handle:
            handle.write(resp.text)
        log.debug("[DDG] Status %s, %s bytes -> debug_ddg.html", resp.status_code, len(resp.text))

    if resp.status_code in (202, 429):
        _start_provider_cooldown("ddg", resp)
        return []
    if resp.status_code != 200:
        log.warning("[DDG] HTTP %s", resp.status_code)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    def _add(title, link, snippet=""):
        link = _ddg_extract_real_url(link)
        if not title or not link.startswith("http"):
            return
        host = urlparse(link).hostname or ""
        if host.endswith("duckduckgo.com"):
            return
        norm = _normalize_url(link)
        if norm in seen:
            return
        seen.add(norm)
        results.append({"title": title, "url": link, "snippet": snippet})

    for div in soup.select("div.result, div.web-result"):
        a = div.select_one("a.result__a")
        if not a:
            continue
        title = a.get_text(strip=True)
        link = a.get("href", "")
        sn = div.select_one("a.result__snippet, div.result__snippet")
        snippet = sn.get_text(strip=True) if sn else ""
        _add(title, link, snippet)
        if len(results) >= num:
            break

    if not results:
        for a in soup.select("a.result__a[href], h2 a[href], a[href]"):
            _add(a.get_text(" ", strip=True), a.get("href", ""))
            if len(results) >= num:
                break

    had_parsed_results = bool(results)
    results = _prepare_source_results(query, results, "ddg")

    if not results and not had_parsed_results:
        log.warning("[DDG] Parser returned 0 results for query %r; search markup may have changed", query)

    if debug:
        log.debug("[DDG] %s results", len(results))
    return results[:num]


def search_brave_scrape(query, num=None, lang="en", debug=False):
    """Scrape search.brave.com HTML (no API key needed)."""
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    if _provider_on_cooldown("brave"):
        return []

    url = f"https://search.brave.com/search?{urlencode({'q': query, 'source': 'web', 'spellcheck': '0'})}"
    if debug:
        log.debug("[BRAVE] %s", url)

    try:
        resp = _get(url, lang, extra_headers={"Referer": "https://search.brave.com/"})
    except Exception as exc:
        log.warning("[BRAVE] Request failed: %s", exc)
        return []

    if debug:
        with open("debug_brave.html", "w", encoding="utf-8") as handle:
            handle.write(resp.text)
        log.debug("[BRAVE] Status %s, %s bytes -> debug_brave.html", resp.status_code, len(resp.text))

    if resp.status_code == 429:
        _start_provider_cooldown("brave", resp)
        return []
    if resp.status_code != 200:
        log.warning("[BRAVE] HTTP %s", resp.status_code)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    blocks = (soup.select("div.snippet[data-type='web']")
              or soup.select("div.snippet")
              or soup.select("#results .snippet"))

    results = []
    seen = set()

    for block in blocks:
        a = block.select_one("a[href^='http']")
        if not a:
            continue
        link = str(a.get("href", ""))
        host = urlparse(link).hostname or ""
        if not link.startswith("http") or host.endswith("brave.com"):
            continue
        norm = _normalize_url(link)
        if norm in seen:
            continue
        seen.add(norm)

        title_el = (block.select_one(".title")
                    or block.select_one("div[class*='title']")
                    or a)
        title = title_el.get_text(" ", strip=True)
        if not title:
            continue

        desc_el = (block.select_one(".generic-snippet .content")
                   or block.select_one(".snippet-description")
                   or block.select_one("div[class*='description']")
                   or block.select_one("div[class*='snippet-content']")
                   or block.select_one("p"))
        snippet = desc_el.get_text(" ", strip=True) if desc_el else ""

        results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= num:
            break

    had_parsed_results = bool(results)
    results = _prepare_source_results(query, results, "brave")

    if not results and not had_parsed_results:
        low = resp.text.lower()
        if any(word in low for word in ("captcha", "unusual traffic", "challenge")):
            log.warning("[BRAVE] Blocked by anti-bot challenge for query %r", query)
        else:
            log.warning("[BRAVE] Parser returned 0 results for query %r; search markup may have changed", query)

    if debug:
        log.debug("[BRAVE] %s results", len(results))
    return results[:num]


def search_wiby(query, num=None, lang="en", debug=False):
    """Search Wiby's public zero-key JSON endpoint."""
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    url = f"https://wiby.me/json/?{urlencode({'q': query})}"
    if debug:
        log.debug("[WIBY] %s", url)

    try:
        resp = _get(
            url,
            lang,
            max_retries=0,
            timeout=10,
            extra_headers={"Accept": "application/json"},
        )
    except Exception as exc:
        log.warning("[WIBY] Request failed: %s", exc)
        return []

    if resp.status_code != 200:
        log.warning("[WIBY] HTTP %s", resp.status_code)
        return []

    try:
        payload = resp.json()
    except Exception as exc:
        log.warning("[WIBY] Invalid JSON response: %s", exc)
        return []
    if not isinstance(payload, list):
        log.warning("[WIBY] Unexpected JSON payload")
        return []

    results = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        link = str(item.get("URL") or "").strip()
        title = str(item.get("Title") or "").strip()
        if not link.startswith("http") or not title:
            continue
        snippet = str(item.get("Snippet") or item.get("Description") or "").strip()
        results.append({
            "title": title,
            "url": link,
            "snippet": snippet,
            "attribution": "https://wiby.me/",
        })
        if len(results) >= num:
            break

    results = _prepare_source_results(query, results, "wiby")
    if debug:
        log.debug("[WIBY] %s results", len(results))
    return results


def search_marginalia(query, num=None, lang="en", debug=False):
    """Search Marginalia's shared public zero-registration API."""
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    count = max(1, min(num, 20))
    url = f"https://api.marginalia.nu/public/search/{quote(query, safe='')}?{urlencode({'count': count})}"
    if debug:
        log.debug("[MARGINALIA] %s", url)

    try:
        resp = _get(
            url,
            lang,
            max_retries=0,
            timeout=10,
            extra_headers={"Accept": "application/json"},
        )
    except Exception as exc:
        log.warning("[MARGINALIA] Request failed: %s", exc)
        return []

    if resp.status_code != 200:
        log.warning("[MARGINALIA] HTTP %s", resp.status_code)
        return []

    try:
        payload = resp.json()
    except Exception as exc:
        log.warning("[MARGINALIA] Invalid JSON response: %s", exc)
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        log.warning("[MARGINALIA] Unexpected JSON payload")
        return []

    license_name = str(payload.get("license") or "CC-BY-NC-SA 4.0").strip()
    results = []
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        link = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not link.startswith("http") or not title:
            continue
        results.append({
            "title": title,
            "url": link,
            "snippet": str(item.get("description") or "").strip(),
            "license": license_name,
            "attribution": "https://search.marginalia.nu/",
        })
        if len(results) >= count:
            break

    results = _prepare_source_results(query, results, "marginalia")
    if debug:
        log.debug("[MARGINALIA] %s results", len(results))
    return results


def _normalize_url(url):
    url = url.split("?")[0].split("#")[0]
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("www.", "")
    return url.rstrip("/").lower()



def merge_results(
    bing_results,
    ddg_results,
    brave_results=None,
    wiby_results=None,
    marginalia_results=None,
    num=20,
):
    return _merge_engine_results(
        {
            "bing": bing_results,
            "ddg": ddg_results,
            "brave": brave_results or [],
            "wiby": wiby_results or [],
            "marginalia": marginalia_results or [],
        },
        num=num,
    )


def _merge_engine_results(engine_results, num=20):
    """Merge any set of prepared providers into one deduplicated ranking."""
    merged = {}
    engine_weights = {"wiby": 0.65, "marginalia": 0.75}

    for engine_name, results in engine_results.items():
        engine_weight = engine_weights.get(engine_name, 1.0)
        for rank, result in enumerate(results, 1):
            norm = _normalize_url(result["url"])
            position_score = engine_weight / rank
            if norm in merged:
                if engine_name not in merged[norm]["engines"]:
                    merged[norm]["score"] += position_score
                    merged[norm]["engines"].add(engine_name)
                if len(result["snippet"]) > len(merged[norm]["snippet"]):
                    merged[norm]["snippet"] = result["snippet"]
                if len(result["title"]) > len(merged[norm]["title"]):
                    merged[norm]["title"] = result["title"]
                if result.get("attribution"):
                    merged[norm]["attributions"].add(result["attribution"])
                if result.get("license"):
                    merged[norm]["licenses"].add(result["license"])
            else:
                merged[norm] = {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "score": position_score,
                    "engines": {engine_name},
                    "attributions": {result["attribution"]} if result.get("attribution") else set(),
                    "licenses": {result["license"]} if result.get("license") else set(),
                }

    # Agreement bonus: +30% per extra engine that also returned the URL.
    for entry in merged.values():
        overlap = len(entry["engines"]) - 1
        if overlap > 0:
            entry["score"] *= 1.0 + 0.3 * overlap

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    output = []
    for entry in ranked[:num]:
        item = {
            "title": entry["title"],
            "url": entry["url"],
            "snippet": entry["snippet"],
            "score": round(entry["score"], 3),
            "engines": sorted(entry["engines"]),
        }
        if entry["attributions"]:
            item["attributions"] = sorted(entry["attributions"])
        if entry["licenses"]:
            item["licenses"] = sorted(entry["licenses"])
        output.append(item)
    return output


# ── Keyed API search providers (reliable; used before scraping when a key is set) ──

def _normalize_api(items, engine):
    """Map provider results into the merged search shape, scored by rank."""
    out = []
    for rank, item in enumerate(items, 1):
        url = item.get("url", "")
        if not url:
            continue
        out.append({
            "title": (item.get("title") or "").strip(),
            "url": url,
            "snippet": (item.get("snippet") or "").strip(),
            "score": round(1.0 / rank, 3),
            "engines": [engine],
        })
    return out


_FRESHNESS_WINDOWS = {
    "day": "day", "d": "day",
    "week": "week", "w": "week",
    "month": "month", "m": "month",
    "year": "year", "y": "year",
}


def _freshness_window(freshness):
    """Normalize a recency window, or "" when the query is not time-boxed.

    One vocabulary for every provider: each maps it to its own parameter name, so a
    recency-filtered search is no longer stuck on the single engine that happened to
    implement it.
    """
    return _FRESHNESS_WINDOWS.get(str(freshness or "").strip().lower(), "")


def search_tavily(query, num=10, lang="en", freshness=""):
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    payload = {"query": query, "max_results": min(num, 20), "search_depth": "basic"}
    window = _freshness_window(freshness)
    if window:
        payload["time_range"] = window  # day | week | month | year
    resp = http.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"tavily HTTP {resp.status_code}")
    results = resp.json().get("results", []) or []
    items = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in results]
    return _prepare_source_results(query, _normalize_api(items, "tavily"), "tavily")


def search_brave(query, num=10, lang="en", freshness=""):
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        return []
    params = {"q": query, "count": min(num, 20)}
    window = _freshness_window(freshness)
    if window:
        params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[window]
    resp = http.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params=params,
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"brave HTTP {resp.status_code}")
    results = (resp.json().get("web", {}) or {}).get("results", []) or []
    items = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")} for r in results]
    return _prepare_source_results(query, _normalize_api(items, "brave"), "brave")


def search_google(query, num=10, lang="en", freshness=""):
    key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not (key and cx):
        return []
    params = {"key": key, "cx": cx, "q": query, "num": min(num, 10), "hl": lang}
    window = _freshness_window(freshness)
    if window:
        params["dateRestrict"] = {"day": "d1", "week": "w1", "month": "m1", "year": "y1"}[window]
    resp = http.get(
        "https://www.googleapis.com/customsearch/v1",
        params=params,
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"google HTTP {resp.status_code}")
    results = resp.json().get("items", []) or []
    items = [{"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")} for r in results]
    return _prepare_source_results(query, _normalize_api(items, "google"), "google")


def _searxng_url() -> str:
    return (os.getenv("FOOTNOTE_SEARXNG_URL") or os.getenv("SEARXNG_URL") or "").strip().rstrip("/")


def search_searxng(query, num=10, lang="en"):
    """Search a configured SearXNG instance through its zero-key JSON API."""
    base_url = _searxng_url()
    if not base_url:
        return []
    resp = http.get(
        f"{base_url}/search",
        params={
            "q": query,
            "format": "json",
            "language": lang,
            "categories": "general",
            "safesearch": 0,
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"searxng HTTP {resp.status_code}")
    try:
        results = resp.json().get("results", []) or []
    except Exception as exc:
        raise RuntimeError(f"searxng invalid JSON response: {exc}") from exc
    items = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("content", ""),
        }
        for result in results[: max(1, min(num, 50))]
    ]
    return _prepare_source_results(query, _normalize_api(items, "searxng"), "searxng")


# Map provider name → function name; resolved via globals() at call time so the
# function is looked up live (testable, and survives reassignment).
_DIRECT_PROVIDERS = {
    "searxng": "search_searxng",
    "tavily": "search_tavily",
    "brave": "search_brave",
    "google": "search_google",
    "wiby": "search_wiby",
    "marginalia": "search_marginalia",
}


def _provider_order(provider):
    """Decide which keyed providers to try, in priority order.

    auto / auto+marginalia: configured SearXNG, then every provider that has its key set.
    A specific name forces just that provider. 'scrape' skips configured providers.
    Zero-key Bing/DDG/Brave/Wiby discovery is always the final fallback.
    Marginalia remains available only as an explicit provider because its shared
    public endpoint can be too slow for the default latency-sensitive path.
    """
    provider = (provider or "auto").lower()
    keyed = {
        "searxng": bool(_searxng_url()),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "brave": bool(os.getenv("BRAVE_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
    }
    if provider in _DIRECT_PROVIDERS:
        return [provider]
    if provider == "scrape":
        return []
    return [name for name in ("searxng", "tavily", "brave", "google") if keyed[name]]


# Providers that bill per call. A metered quota is a finite resource shared by every
# question, so calls rotate through them instead of draining the first one listed.
_METERED_PROVIDERS = ("tavily", "brave", "google")
_metered_cursor = 0
_metered_cursor_lock = threading.Lock()


def _metered_pool():
    """Configured metered providers that are not currently resting."""
    keyed = {
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "brave": bool(os.getenv("BRAVE_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
    }
    return [name for name in _METERED_PROVIDERS if keyed[name] and not _provider_on_cooldown(name)]


def _next_metered_provider(skip=()):
    """Next metered provider in round-robin order, or None when none is available."""
    global _metered_cursor
    pool = [name for name in _metered_pool() if name not in skip]
    if not pool:
        return None
    with _metered_cursor_lock:
        name = pool[_metered_cursor % len(pool)]
        _metered_cursor += 1
    return name


def _min_free_results():
    """Strong free results below which spending a metered call is worth it."""
    try:
        return max(0, int(os.getenv("FOOTNOTE_MIN_FREE_RESULTS", "3")))
    except ValueError:
        return 3


def _strong_match_count(query, engine_results):
    """Free results that actually look like answers, not merely on-topic pages.

    The per-provider relevance filter is deliberately permissive — one distinctive
    term is enough to survive it. Counting by that bar makes a page like
    "Geography of Spain" look like a hit for "Spain events August 2026" and keeps a
    metered provider that would have answered properly from ever being asked. A
    strong match covers at least half of the distinctive query terms.
    """
    query_terms = _search_terms(query)
    distinctive = query_terms - _GENERIC_SEARCH_TERMS or query_terms
    if not distinctive:
        return sum(len(results) for results in engine_results.values())

    needed = 1 if len(distinctive) == 1 else max(2, (len(distinctive) + 1) // 2)
    seen = set()
    strong = 0
    for results in engine_results.values():
        for result in results:
            url = _normalize_url(str(result.get("url") or ""))
            if not url or url in seen:
                continue
            terms = _search_terms(
                f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}"
            )
            if len(distinctive & terms) >= needed:
                seen.add(url)
                strong += 1
    return strong


def _provider_strategy():
    return (os.getenv("FOOTNOTE_PROVIDER_STRATEGY", "cost_aware") or "cost_aware").strip().lower()


def _run_free_providers(query, num, lang, debug, with_marginalia=False, freshness=""):
    """Query everything that costs nothing: self-hosted SearXNG and the scrapers."""
    from . import core

    window = _freshness_window(freshness)
    free: dict[str, list] = {"ddg": []}
    jobs: dict[str, tuple] = {
        # Only DuckDuckGo's HTML endpoint takes a date filter, so a time-boxed query
        # asks it alone rather than diluting the merge with undated engines.
        "ddg": (search_ddg, (query, core.NUM_PER_ENGINE, lang, debug, window[:1] if window else "")),
    }
    if not window:
        free.update({"bing": [], "brave": [], "wiby": []})
        jobs.update({
            "bing": (search_bing, (query, core.NUM_PER_ENGINE, lang, debug)),
            "brave": (search_brave_scrape, (query, core.NUM_PER_ENGINE, lang, debug)),
            "wiby": (search_wiby, (query, core.NUM_PER_ENGINE, lang, debug)),
        })
        if with_marginalia:
            free["marginalia"] = []
            jobs["marginalia"] = (search_marginalia, (query, core.NUM_PER_ENGINE, lang, debug))
    if _searxng_url():
        # Self-hosted: keyed, but unmetered, so it belongs with the free tier.
        free["searxng"] = []
        jobs["searxng"] = (search_searxng, (query, num, lang))

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(fn, *args): name for name, (fn, args) in jobs.items()}
        for future in as_completed(futures):
            engine = futures[future]
            try:
                free[engine] = _prepare_source_results(query, future.result(), engine)
            except Exception as exc:
                log.warning("[%s] Error: %s", engine.upper(), exc)
    return free


def _call_metered(query, num, lang, name, freshness=""):
    """One metered provider call. Returns prepared results (possibly empty)."""
    try:
        fn = globals()[_DIRECT_PROVIDERS[name]]
        window = _freshness_window(freshness)
        results = fn(query, num=num, lang=lang, freshness=window) if window else fn(query, num=num, lang=lang)
        prepared = _prepare_source_results(query, results, name)
        log.info("[SEARCH] metered provider=%s -> %s results", name, len(prepared))
        return prepared
    except Exception as exc:
        log.warning("[SEARCH] metered provider %s failed: %s", name, exc)
        return []


def search(query, num=20, lang="en", debug=False, provider="auto", freshness=""):
    """Merge providers into one ranking. ``freshness`` time-boxes every provider."""
    requested_provider = (provider or "auto").lower()

    # An explicitly selected provider remains isolated by definition.
    if requested_provider in _DIRECT_PROVIDERS:
        name = requested_provider
        try:
            results = _call_metered(query, num, lang, name, freshness) if _freshness_window(freshness) \
                else _prepare_source_results(
                    query, globals()[_DIRECT_PROVIDERS[name]](query, num=num, lang=lang), name
                )
            if results:
                log.info("[SEARCH] provider=%s -> %s results", name, len(results))
                return _merge_engine_results({name: results}, num=num)
            log.info("[SEARCH] provider=%s returned 0 results, trying next", name)
        except Exception as exc:
            log.warning("[SEARCH] provider %s failed: %s", name, exc)
        return []

    with_marginalia = requested_provider == "auto+marginalia"
    combined = _run_free_providers(query, num, lang, debug, with_marginalia, freshness)
    free_count = _strong_match_count(query, combined)

    if requested_provider != "scrape" and _provider_strategy() != "merge":
        # Cost-aware: the free tier answers most queries outright. A metered credit is
        # spent only when free results are too thin to be worth ranking, and the
        # provider that gets charged rotates so one quota does not empty first.
        threshold = _min_free_results()
        if free_count >= threshold:
            log.info("[SEARCH] free tier returned %s strong results (>= %s); no metered call",
                     free_count, threshold)
        else:
            tried = []
            while True:
                name = _next_metered_provider(skip=tried)
                if not name:
                    if not tried:
                        log.info("[SEARCH] free tier thin (%s results) and no metered provider available",
                                 free_count)
                    break
                tried.append(name)
                prepared = _call_metered(query, num, lang, name, freshness)
                if prepared:
                    combined[name] = _prepare_source_results(
                        query, combined.get(name, []) + prepared, name
                    )
                    break
                # An empty answer is not a usable one; try one alternate, then stop.
                if len(tried) >= 2:
                    break
    elif requested_provider != "scrape":
        # Legacy merge strategy: every configured provider contributes to every query.
        for name in _provider_order("auto"):
            if name == "searxng" and "searxng" in combined:
                continue
            prepared = _call_metered(query, num, lang, name, freshness)
            if prepared:
                combined[name] = _prepare_source_results(
                    query, combined.get(name, []) + prepared, name
                )

    counts = ", ".join(f"{name.title()}: {len(results)}" for name, results in combined.items())
    log.info("[MERGE] %s; merging", counts)
    return _merge_engine_results(combined, num=num)

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


def _provider_cooldown_seconds(engine, response=None):
    """Return a bounded rate-limit cooldown, honoring Retry-After when present."""
    env_name = f"FOOTNOTE_{engine.upper()}_COOLDOWN_SECONDS"
    try:
        seconds = max(0.0, float(os.getenv(env_name, _PROVIDER_COOLDOWN_DEFAULTS[engine])))
    except (TypeError, ValueError):
        seconds = _PROVIDER_COOLDOWN_DEFAULTS[engine]
    headers = getattr(response, "headers", {}) or {}
    try:
        seconds = max(seconds, float(headers.get("Retry-After", 0)))
    except (TypeError, ValueError):
        pass
    return min(seconds, 3600.0)


def _start_provider_cooldown(engine, response=None):
    seconds = _provider_cooldown_seconds(engine, response)
    if seconds <= 0:
        return
    until = time.monotonic() + seconds
    with _PROVIDER_COOLDOWN_LOCK:
        _PROVIDER_COOLDOWN_UNTIL[engine] = max(_PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0), until)
    log.warning("[%s] Rate limited; cooling down for %.0fs", engine.upper(), seconds)


def _provider_on_cooldown(engine):
    with _PROVIDER_COOLDOWN_LOCK:
        until = _PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0)
        remaining = until - time.monotonic()
        if remaining <= 0:
            _PROVIDER_COOLDOWN_UNTIL.pop(engine, None)
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


def search_tavily(query, num=10, lang="en"):
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    resp = http.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "max_results": min(num, 20), "search_depth": "basic"},
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"tavily HTTP {resp.status_code}")
    results = resp.json().get("results", []) or []
    items = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in results]
    return _prepare_source_results(query, _normalize_api(items, "tavily"), "tavily")


def search_brave(query, num=10, lang="en"):
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        return []
    resp = http.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params={"q": query, "count": min(num, 20)},
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"brave HTTP {resp.status_code}")
    results = (resp.json().get("web", {}) or {}).get("results", []) or []
    items = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")} for r in results]
    return _prepare_source_results(query, _normalize_api(items, "brave"), "brave")


def search_google(query, num=10, lang="en"):
    key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not (key and cx):
        return []
    resp = http.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": key, "cx": cx, "q": query, "num": min(num, 10), "hl": lang},
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


def search(query, num=20, lang="en", debug=False, provider="auto"):
    from . import core

    requested_provider = (provider or "auto").lower()
    direct_results = {}

    # An explicitly selected provider remains isolated by definition.
    if requested_provider in _DIRECT_PROVIDERS:
        name = requested_provider
        try:
            results = globals()[_DIRECT_PROVIDERS[name]](query, num=num, lang=lang)
            results = _prepare_source_results(query, results, name)
            if results:
                log.info("[SEARCH] provider=%s -> %s results", name, len(results))
                return _merge_engine_results({name: results}, num=num)
            log.info("[SEARCH] provider=%s returned 0 results, trying next", name)
        except Exception as exc:
            log.warning("[SEARCH] provider %s failed: %s", name, exc)
        return []

    # In auto mode, every configured provider contributes to the final merge.
    configured_order = [] if requested_provider == "scrape" else _provider_order("auto")
    for name in configured_order:
        try:
            results = globals()[_DIRECT_PROVIDERS[name]](query, num=num, lang=lang)
            prepared = _prepare_source_results(query, results, name)
            if prepared:
                direct_results[name] = prepared
                log.info("[SEARCH] provider=%s -> %s results", name, len(prepared))
            else:
                log.info("[SEARCH] provider=%s returned 0 results", name)
        except Exception as exc:
            log.warning("[SEARCH] provider %s failed: %s", name, exc)

    # 2. Fallback: query the latency-bounded zero-key sources in parallel and merge.
    # The opt-in auto+marginalia mode adds the slower shared Marginalia endpoint
    # without changing the latency contract of the default auto mode.
    scraped = {"bing": [], "ddg": [], "brave": [], "wiby": []}
    if requested_provider == "auto+marginalia":
        scraped["marginalia"] = []

    with ThreadPoolExecutor(max_workers=len(scraped)) as pool:
        futures = {
            pool.submit(search_bing, query, core.NUM_PER_ENGINE, lang, debug): "bing",
            pool.submit(search_ddg, query, core.NUM_PER_ENGINE, lang, debug): "ddg",
            pool.submit(search_brave_scrape, query, core.NUM_PER_ENGINE, lang, debug): "brave",
            pool.submit(search_wiby, query, core.NUM_PER_ENGINE, lang, debug): "wiby",
        }
        if "marginalia" in scraped:
            futures[pool.submit(search_marginalia, query, core.NUM_PER_ENGINE, lang, debug)] = "marginalia"
        for future in as_completed(futures):
            engine = futures[future]
            try:
                scraped[engine] = _prepare_source_results(query, future.result(), engine)
            except Exception as exc:
                log.warning("[%s] Error: %s", engine.upper(), exc)

    counts = ", ".join(f"{name.title()}: {len(results)}" for name, results in scraped.items())
    log.info("[MERGE] %s; merging", counts)
    combined = dict(direct_results)
    for name, results in scraped.items():
        combined[name] = _prepare_source_results(query, combined.get(name, []) + results, name)
    return _merge_engine_results(combined, num=num)

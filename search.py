from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, quote_plus, urlencode, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as http

from diagnostics import log
from fetch import _get


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
    import core

    if num is None:
        num = core.NUM_PER_ENGINE

    params = {"q": query, "count": min(num + 5, 30), "setlang": lang, "cc": lang}
    if lang == "en":
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

    if not results:
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
    import core

    if num is None:
        num = core.NUM_PER_ENGINE

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

    if not results:
        log.warning("[DDG] Parser returned 0 results for query %r; search markup may have changed", query)

    if debug:
        log.debug("[DDG] %s results", len(results))
    return results[:num]


def _normalize_url(url):
    url = url.split("?")[0].split("#")[0]
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("www.", "")
    return url.rstrip("/").lower()



def merge_results(bing_results, ddg_results, num=20):
    merged = {}

    for engine_name, results in [("bing", bing_results), ("ddg", ddg_results)]:
        for rank, result in enumerate(results, 1):
            norm = _normalize_url(result["url"])
            position_score = 1.0 / rank
            if norm in merged:
                merged[norm]["score"] += position_score
                merged[norm]["engines"].add(engine_name)
                if len(result["snippet"]) > len(merged[norm]["snippet"]):
                    merged[norm]["snippet"] = result["snippet"]
                if len(result["title"]) > len(merged[norm]["title"]):
                    merged[norm]["title"] = result["title"]
            else:
                merged[norm] = {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result["snippet"],
                    "score": position_score,
                    "engines": {engine_name},
                }

    for entry in merged.values():
        if len(entry["engines"]) >= 2:
            entry["score"] *= 1.3

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return [
        {
            "title": entry["title"],
            "url": entry["url"],
            "snippet": entry["snippet"],
            "score": round(entry["score"], 3),
            "engines": sorted(entry["engines"]),
        }
        for entry in ranked[:num]
    ]


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
    return _normalize_api(items, "tavily")


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
    return _normalize_api(items, "brave")


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
    return _normalize_api(items, "google")


# Map provider name → function name; resolved via globals() at call time so the
# function is looked up live (testable, and survives reassignment).
_API_PROVIDERS = {"tavily": "search_tavily", "brave": "search_brave", "google": "search_google"}


def _provider_order(provider):
    """Decide which keyed providers to try, in priority order.

    auto: every provider that has its key set (tavily → brave → google).
    A specific name forces just that provider. 'scrape' skips APIs entirely.
    Scraping Bing+DDG is always the final fallback regardless.
    """
    provider = (provider or "auto").lower()
    keyed = {
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "brave": bool(os.getenv("BRAVE_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
    }
    if provider in _API_PROVIDERS:
        return [provider]
    if provider == "scrape":
        return []
    return [name for name in ("tavily", "brave", "google") if keyed[name]]


def search(query, num=20, lang="en", debug=False, provider="auto"):
    import core

    # 1. Keyed API providers first — reliable, no scraping. First non-empty wins.
    for name in _provider_order(provider):
        try:
            results = globals()[_API_PROVIDERS[name]](query, num=num, lang=lang)
            if results:
                log.info("[SEARCH] provider=%s -> %s results", name, len(results))
                return results[:num]
            log.info("[SEARCH] provider=%s returned 0 results, trying next", name)
        except Exception as exc:
            log.warning("[SEARCH] provider %s failed: %s", name, exc)

    # 2. Fallback: scrape Bing + DDG in parallel and merge (original behaviour).
    bing_r = []
    ddg_r = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(search_bing, query, core.NUM_PER_ENGINE, lang, debug): "bing",
            pool.submit(search_ddg, query, core.NUM_PER_ENGINE, lang, debug): "ddg",
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                results = future.result()
                if engine == "bing":
                    bing_r = results
                else:
                    ddg_r = results
            except Exception as exc:
                log.warning("[%s] Error: %s", engine.upper(), exc)

    log.info("[MERGE] Bing: %s, DDG: %s; merging", len(bing_r), len(ddg_r))
    return merge_results(bing_r, ddg_r, num=num)

from __future__ import annotations

import base64
import functools
import hashlib
import inspect
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlencode, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as http

from .diagnostics import log
from .fetch import _get
from .politeness import retry_after_seconds


# Words that carry no topic. A result matching only one of these has shown no
# connection to the query, and treating one as evidence is how a natural-language
# question let dictionary pages through: "how many countries use the euro" made
# "many" a distinctive term, so Bing's definition-of-MANY rows matched it and
# were certified relevant. Tokens of two characters or fewer are dropped by
# _search_terms already, so only longer function words need listing here.
_SEARCH_STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "what", "where", "when",
    # interrogatives and the auxiliaries that frame a question
    "how", "why", "who", "whom", "whose", "which",
    "does", "did", "are", "was", "were", "been", "being",
    "will", "would", "can", "could", "should", "must", "have", "has", "had",
    # bare quantifiers and pronouns
    "many", "much", "more", "most", "some", "any", "all",
    "you", "your", "its", "they", "them", "their", "there", "here", "not", "but", "than", "then",
    "как", "для", "или", "что", "это", "где", "когда", "при", "про",
    "почему", "зачем", "кто", "кого", "кому", "чей",
    "какой", "какая", "какое", "какие", "сколько", "чего", "чему", "чем",
    "был", "была", "было", "были", "будет", "есть", "может", "можно", "нужно",
    "если", "чтобы", "также", "тоже", "они", "них", "его", "она", "оно",
}
_GENERIC_SEARCH_TERMS = {
    "context", "documentation", "docs", "guide", "github", "language", "model", "official",
    "programming", "protocol", "search", "today", "tutorial", "weather",
    "документация", "официальный", "официальная", "поиск", "погода", "сегодня",
}

_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}
_PROVIDER_COOLDOWN_LOCK = threading.Lock()
# Cooldowns also survive a restart: the ban lives at the provider, keyed to our
# IP, so a fresh process must not walk straight back into it. Deadlines are
# persisted as wall-clock epochs; the in-memory copy stays monotonic.
_PERSISTED_COOLDOWNS_LOADED = False
_PERSISTED_COOLDOWN_UNTIL: dict[str, float] = {}


def _cooldown_state_path() -> Path:
    root = os.getenv("FOOTNOTE_SOURCE_CACHE", "").strip() or "~/.footnote-mcp/source_cache"
    return Path(root).expanduser() / "provider_cooldowns.json"
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


def _load_persisted_cooldowns():
    """Read cooldown deadlines left behind by an earlier process."""
    global _PERSISTED_COOLDOWNS_LOADED
    if _PERSISTED_COOLDOWNS_LOADED:
        return
    _PERSISTED_COOLDOWNS_LOADED = True
    try:
        raw = json.loads(_cooldown_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    now = time.time()
    for engine, until in raw.items():
        try:
            until = float(until)
        except (TypeError, ValueError):
            continue
        if until > now:
            _PERSISTED_COOLDOWN_UNTIL[str(engine)] = until


def _save_persisted_cooldowns():
    """Best-effort: a lost cooldown file costs politeness, never correctness."""
    now = time.time()
    live = {name: until for name, until in _PERSISTED_COOLDOWN_UNTIL.items() if until > now}
    try:
        path = _cooldown_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(live), encoding="utf-8")
    except OSError as exc:
        log.debug("[cooldown] could not persist state: %s", exc)


def _start_provider_cooldown(engine, response=None):
    seconds = _provider_cooldown_seconds(engine, response)
    if seconds <= 0:
        return
    until = time.monotonic() + seconds
    with _PROVIDER_COOLDOWN_LOCK:
        _load_persisted_cooldowns()
        _PROVIDER_COOLDOWN_UNTIL[engine] = max(_PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0), until)
        _PERSISTED_COOLDOWN_UNTIL[engine] = max(
            _PERSISTED_COOLDOWN_UNTIL.get(engine, 0.0), time.time() + seconds
        )
        _save_persisted_cooldowns()
    log.warning("[%s] Rate limited; cooling down for %.0fs", engine.upper(), seconds)


def _provider_on_cooldown(engine):
    now = time.monotonic()
    with _PROVIDER_COOLDOWN_LOCK:
        _load_persisted_cooldowns()
        carried = _PERSISTED_COOLDOWN_UNTIL.get(engine, 0.0) - time.time()
        if carried > 0:
            # Deadline set before this process started; fold it into the
            # monotonic clock so the rest of the logic stays unchanged.
            _PROVIDER_COOLDOWN_UNTIL[engine] = max(
                _PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0), now + carried
            )
        until = _PROVIDER_COOLDOWN_UNTIL.get(engine, 0.0)
        remaining = until - now
        if remaining <= 0:
            _PROVIDER_COOLDOWN_UNTIL.pop(engine, None)
            _PERSISTED_COOLDOWN_UNTIL.pop(engine, None)
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


# ── Search-result cache ────────────────────────────────────────────────────
#
# A repeated query used to spend a fresh request against the provider's rate
# limit for an answer we already had. Results are keyed by everything that can
# change them and kept on disk, so the budget survives a restart too. Only
# non-empty result sets are stored: a block must never be cached as "no hits".

_SEARCH_CACHE_DEFAULT_TTL = 86400.0


def _search_cache_dir() -> Path:
    root = os.getenv("FOOTNOTE_SEARCH_CACHE", "").strip() or "~/.footnote-mcp/search_cache"
    return Path(root).expanduser()


def _search_cache_ttl() -> float:
    try:
        return max(0.0, float(os.getenv("FOOTNOTE_SEARCH_CACHE_TTL", _SEARCH_CACHE_DEFAULT_TTL)))
    except (TypeError, ValueError):
        return _SEARCH_CACHE_DEFAULT_TTL


def _search_cache_key(engine, query, lang, num, extra="") -> str:
    raw = "|".join(str(part) for part in (engine, query, lang, num, extra))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _search_cache_get(key):
    ttl = _search_cache_ttl()
    if ttl <= 0:
        return None
    path = _search_cache_dir() / f"{key}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if time.time() - float(payload.get("stored_at", 0.0)) > ttl:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    results = payload.get("results")
    return results if isinstance(results, list) and results else None


def _search_cache_put(key, results):
    if not results or _search_cache_ttl() <= 0:
        return
    try:
        directory = _search_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.json").write_text(
            json.dumps({"stored_at": time.time(), "results": results}), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        log.debug("[cache] could not store %s results: %s", key[:8], exc)


def _cached_search(engine):
    """Serve a scraped provider's results from disk when they are still fresh.

    Applied to the scraped engines only: the keyed APIs are paid for and their
    callers expect live answers, while these are the ones whose budget is a
    rate limit we keep running into.
    """
    def wrap(fn):
        # Bind against the wrapped function's own signature rather than
        # restating it here: a provider that grows a parameter (ddg's df) used
        # to reach a wrapper that took four positionals and raise a TypeError
        # naming the provider, which reads as a broken provider rather than a
        # stale decorator.
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            from . import core

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            call = dict(bound.arguments)
            query = call["query"]
            lang = call.get("lang", "en")
            num = call.get("num")

            resolved = core.NUM_PER_ENGINE if num is None else num
            extra = {k: v for k, v in call.items() if k not in ("query", "num", "lang", "debug")}
            key = _search_cache_key(
                engine, query, lang, resolved,
                "&".join(f"{k}={v}" for k, v in sorted(extra.items())),
            )
            hit = _search_cache_get(key)
            if hit is not None:
                log.debug("[%s] cache hit for %r", engine.upper(), query)
                return hit[:resolved]
            results = fn(**call)
            _search_cache_put(key, results)
            return results
        return inner
    return wrap


# ── Escalation for a refused search request ────────────────────────────────
#
# The scraped engines used to give up on the first 202/429 and cool the
# provider down for minutes. Two tiers already exist in scraper.py for pages;
# a refused search page is the same problem, so they are reused here:
# a proxy (a different exit address) and headless Chromium (a real browser
# fingerprint executing the page's JavaScript).

# Rendering a search page in Chromium only helps where the engine serves a real
# page to a real browser. Measured against each endpoint:
#   search.brave.com  -> 20 result snippets, parses
#   www.bing.com      -> 10 li.b_algo, parses
#   html/lite.duckduckgo.com -> a 273-byte "please email us" stub
#   duckduckgo.com    -> 56 KB shell with no result nodes
# So DuckDuckGo opts out: sending it through the browser would trade a clean
# refusal for an empty page that parses to nothing and never trips the cooldown.
_BROWSER_TIER_ENGINES = ("bing", "brave")

# A rendered search page that is this small is an error stub, not results.
_MIN_RENDERED_SEARCH_BYTES = 2000


def _search_fetch(engine, url, lang="en", headers=None, debug=False):
    """Fetch a search-results page. Returns (html, refusal).

    `refusal` is the 202/429 response only when every tier was refused; the
    caller cools the provider down on that. A non-refusal failure returns
    (None, None) so the caller can try another endpoint.
    """
    from . import scraper

    try:
        resp = _get(url, lang, extra_headers=headers)
    except Exception as exc:
        log.warning("[%s] request failed: %s", engine.upper(), exc)
        return None, None

    if resp.status_code == 200:
        return resp.text, None
    if resp.status_code not in (202, 429):
        log.warning("[%s] HTTP %s", engine.upper(), resp.status_code)
        return None, None

    refusal = resp
    domain = urlparse(url).hostname or engine

    # tier 2: a different exit address.
    if scraper._proxy_enabled(None) and scraper._PROXIES.available():
        proxy = scraper._PROXIES.get(domain, rotate=True)
        if proxy:
            try:
                retry = _get(
                    url, lang, extra_headers=headers, max_retries=1,
                    proxies={"http": proxy, "https": proxy},
                )
            except Exception as exc:
                scraper._PROXIES.report(proxy, ok=False)
                log.debug("[%s] proxy attempt failed: %s", engine.upper(), exc)
            else:
                ok = retry.status_code == 200
                scraper._PROXIES.report(proxy, ok=ok)
                if ok:
                    log.debug("[%s] answered through a proxy after HTTP %s", engine.upper(), refusal.status_code)
                    return retry.text, None
                refusal = retry if retry.status_code in (202, 429) else refusal

    # tier 3: a real browser, for the engines that serve one a real page.
    if engine.split("/")[0] in _BROWSER_TIER_ENGINES and scraper._browser_enabled(None):
        html, _status, error = scraper._RENDERER.render(url, lang=lang)
        if error:
            log.debug("[%s] browser tier failed: %s", engine.upper(), error)
        elif not html or len(html) < _MIN_RENDERED_SEARCH_BYTES:
            log.debug("[%s] browser tier returned %s bytes; treating as refused",
                      engine.upper(), len(html or ""))
        elif scraper.detect_block(200, html)[0]:
            log.debug("[%s] browser tier hit a block page", engine.upper())
        else:
            log.debug("[%s] answered through the browser tier after HTTP %s",
                      engine.upper(), refusal.status_code)
            return html, None

    return None, refusal


# Bing answers a natural-language question by matching a word in it rather than
# the topic: "how many countries use the euro" comes back as dictionary entries
# for MANY. Stripping the interrogative frame — and only the leading frame, so
# word order and every content word survive — asks the same question as
# keywords. Applied to Bing alone: it is the engine measured to need it.
_QUESTION_LEAD_WORDS = {
    "how", "what", "which", "who", "whom", "whose", "why", "where", "when",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "has", "have", "had", "many", "much",
    "как", "какой", "какая", "какое", "какие", "сколько", "почему", "зачем",
    "кто", "что", "где", "когда", "чему", "чего",
}


def _strip_question_frame(query):
    """Drop the leading question words from a query, keeping the rest verbatim."""
    # An operator query is a precise instrument; never rewrite one.
    if any(marker in query for marker in ('"', "site:", "filetype:", "inurl:", "intitle:", " OR ")):
        return query
    # A leading "-" is the exclusion operator; a hyphen inside a word is not.
    if any(word.startswith("-") for word in query.split()):
        return query

    stripped = query.strip().rstrip("?").strip()
    words = stripped.split()
    index = 0
    while index < len(words) and words[index].lower() in _QUESTION_LEAD_WORDS:
        index += 1

    # Refuse to rewrite when the frame is the whole query, or when too little is
    # left to search for: a two-word remainder is a weaker query, not a better one.
    remainder = words[index:]
    if index == 0 or len(remainder) < 2:
        return query
    return " ".join(remainder)


@_cached_search("bing")
def search_bing(query, num=None, lang="en", debug=False):
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    sent = _strip_question_frame(query)
    if sent != query:
        log.debug("[BING] question reframed: %r -> %r", query, sent)
    params = {"q": sent, "count": min(num + 5, 30), "setlang": lang}
    if lang == "en":
        params["cc"] = "US"
        params["setmkt"] = "en-US"

    url = f"https://www.bing.com/search?{urlencode(params)}"
    if debug:
        log.debug("[BING] %s", url)

    html, refusal = _search_fetch("bing", url, lang, debug=debug)
    if refusal is not None:
        _start_provider_cooldown("bing", refusal)
        return []
    if not html:
        return []

    if debug:
        with open("debug_bing.html", "w", encoding="utf-8") as handle:
            handle.write(html)
        log.debug("[BING] %s bytes -> debug_bing.html", len(html))

    response_text = html.lower()
    if "one last step" in response_text and ("captcha" in response_text or "challenge" in response_text):
        log.warning("[BING] Blocked by anti-bot challenge for query %r", query)
        return []

    soup = BeautifulSoup(html, "html.parser")
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


# Requests to the search endpoints are made to look like a submission from the
# DuckDuckGo home page rather than a URL typed into the address bar, which is
# what a bare Sec-Fetch-Site: none says.
_DDG_FORM_HEADERS = {
    "Referer": "https://duckduckgo.com/",
    "Origin": "https://duckduckgo.com",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "navigate",
}

# Tried in order. Measured: the two hosts share one rate-limit budget — a lite
# request sent immediately after html refuses comes back 202 as well — so lite
# is NOT a way around a block, and a refusal must not be retried against it.
# It is kept only as a second shot when html fails some other way: a 5xx, a
# transport error, or markup that stops parsing.
_DDG_ENDPOINTS = (
    ("html", "https://html.duckduckgo.com/html/?q={q}"),
    ("lite", "https://lite.duckduckgo.com/lite/?q={q}"),
)


def _parse_ddg_lite(soup, add, num, results):
    """The lite endpoint is a bare table: one <tr> per link, snippet in a
    following cell that holds no link of its own."""
    for row in soup.select("tr"):
        a = row.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        snippet_cells = [td.get_text(strip=True) for td in row.find_all("td") if not td.find("a")]
        add(title, a["href"], " ".join(part for part in snippet_cells if part).strip())
        if len(results) >= num:
            break


@_cached_search("ddg")
def search_ddg(query, num=None, lang="en", debug=False, df=""):
    from . import core

    if num is None:
        num = core.NUM_PER_ENGINE

    if _provider_on_cooldown("ddg"):
        return []

    # df = DuckDuckGo freshness filter: d (day), w (week), m (month), y (year).
    suffix = f"&df={df}" if df in ("d", "w", "m", "y") else ""

    results = []
    seen = set()
    rate_limited = None

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

    for tag, template in _DDG_ENDPOINTS:
        if results:
            break
        url = template.format(q=quote_plus(query)) + suffix
        if debug:
            log.debug("[DDG/%s] %s", tag, url)

        html, refusal = _search_fetch(f"ddg/{tag}", url, lang, headers=_DDG_FORM_HEADERS, debug=debug)

        if refusal is not None:
            # Shared budget across the two hosts, and every tier has already
            # been tried. Poking the other endpoint would only collect a second
            # refusal from a provider that has said no. Stop here.
            rate_limited = refusal
            break
        if not html:
            continue

        if debug:
            with open(f"debug_ddg_{tag}.html", "w", encoding="utf-8") as handle:
                handle.write(html)
            log.debug("[DDG/%s] %s bytes", tag, len(html))

        soup = BeautifulSoup(html, "html.parser")
        if tag == "lite":
            _parse_ddg_lite(soup, _add, num, results)
        else:
            for div in soup.select("div.result, div.web-result"):
                a = div.select_one("a.result__a")
                if not a:
                    continue
                sn = div.select_one("a.result__snippet, div.result__snippet")
                _add(a.get_text(strip=True), a.get("href", ""), sn.get_text(strip=True) if sn else "")
                if len(results) >= num:
                    break

            if not results:
                for a in soup.select("a.result__a[href], h2 a[href], a[href]"):
                    _add(a.get_text(" ", strip=True), a.get("href", ""))
                    if len(results) >= num:
                        break

    if not results and rate_limited is not None:
        _start_provider_cooldown("ddg", rate_limited)
        return []

    had_parsed_results = bool(results)
    results = _prepare_source_results(query, results, "ddg")

    if not results and not had_parsed_results:
        log.warning("[DDG] Parser returned 0 results for query %r; search markup may have changed", query)

    if debug:
        log.debug("[DDG] %s results", len(results))
    return results[:num]


@_cached_search("brave")
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

    html, refusal = _search_fetch(
        "brave", url, lang, headers={"Referer": "https://search.brave.com/"}, debug=debug
    )
    if refusal is not None:
        _start_provider_cooldown("brave", refusal)
        return []
    if not html:
        return []

    if debug:
        with open("debug_brave.html", "w", encoding="utf-8") as handle:
            handle.write(html)
        log.debug("[BRAVE] %s bytes -> debug_brave.html", len(html))

    soup = BeautifulSoup(html, "html.parser")
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
        low = html.lower()
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


def reset_state():
    """Drop provider backoff, on disk as well as in memory. Used by tests."""
    global _PERSISTED_COOLDOWNS_LOADED
    with _PROVIDER_COOLDOWN_LOCK:
        _PROVIDER_COOLDOWN_UNTIL.clear()
        _PERSISTED_COOLDOWN_UNTIL.clear()
        _PERSISTED_COOLDOWNS_LOADED = False
        try:
            _cooldown_state_path().unlink()
        except OSError:
            pass

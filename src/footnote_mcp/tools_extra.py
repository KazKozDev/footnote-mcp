"""Extended research tools: archive/recent search, corroboration,
span provenance, recipe registry, authenticated fetch, crawl, dataset export,
and time-series reconciliation.

These build on the same conventions as tools_data/tools_search: plain functions
returning JSON-serializable dicts, source-URL provenance, and reuse of the shared
fetch/extract/cache helpers.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup

from . import core
from .extract import extract_content
from .fetch import _get, fetch_page
from .search import search_ddg
from .tools_data.cache import CACHE_DIR
from .tools_data.classify import classify_source
from .tools_data.entailment import evidence_entailment
from .tools_data.files import _fetch_bytes
from .tools_data.sandbox import _load_recipe_store, _recipe_rows, _save_recipe_store, tool_code_run_sandboxed


def _page_title(html: str) -> str:
    try:
        tag = BeautifulSoup(html, "html.parser").find("title")
        return tag.get_text(strip=True) if tag else ""
    except Exception:
        return ""


# ── 1. Web archive (Wayback Machine) ──

def web_archive_fetch(url: str, timestamp: str = "", lang: str = "en", fetch_text: bool = True) -> dict:
    """Find the closest Wayback Machine snapshot for a URL and optionally read it.

    Useful when a live source is dead (404) or has changed since it was cited.
    ``timestamp`` is an optional YYYYMMDD or YYYYMMDDhhmmss target.
    """
    api = "http://archive.org/wayback/available?url=" + quote(url, safe="")
    if timestamp:
        api += "&timestamp=" + quote(str(timestamp), safe="")
    data, _ct, err = _fetch_bytes(api, lang=lang, timeout=20)
    if err or not data:
        return {"url": url, "archived": False, "error": err or "no response from wayback"}
    try:
        info = json.loads(data.decode("utf-8", errors="replace"))
    except Exception as exc:
        return {"url": url, "archived": False, "error": f"bad wayback response: {exc}"}

    closest = (info.get("archived_snapshots") or {}).get("closest") or {}
    if not closest.get("available") or not closest.get("url"):
        return {"url": url, "archived": False, "error": "no snapshot found"}

    snapshot_url = closest["url"]
    result = {
        "url": url,
        "archived": True,
        "snapshot_url": snapshot_url,
        "snapshot_timestamp": closest.get("timestamp"),
        "status": closest.get("status"),
    }
    if fetch_text:
        fetched_url, html, pub_date, ferr = fetch_page(snapshot_url, lang=lang)
        if ferr or not html:
            result["fetch_error"] = ferr or "empty snapshot body"
        else:
            text = extract_content(html, url=fetched_url) or ""
            result["title"] = _page_title(html)
            result["text"] = text[: core.MAX_CONTENT_CHARS]
            result["text_length"] = len(text)
            result["published"] = str(pub_date) if pub_date else None
    return result


# ── 2. Freshness-filtered search ──

def web_search_recent(query: str, freshness: str = "month", lang: str = "en", num: int = 10) -> dict:
    """Search with a recency window via DuckDuckGo's date filter.

    ``freshness`` is day | week | month | year (or d | w | m | y).
    """
    fmap = {"day": "d", "week": "w", "month": "m", "year": "y", "d": "d", "w": "w", "m": "m", "y": "y"}
    df = fmap.get((freshness or "month").lower(), "m")
    results = search_ddg(query, num=num, lang=lang, df=df)
    return {
        "query": query,
        "freshness": freshness,
        "df": df,
        "count": len(results),
        "results": [{"title": r["title"], "url": r["url"], "snippet": r.get("snippet", "")} for r in results],
    }


# ── 3. Cross-source corroboration ──

def corroborate_claim(claim: str, excerpts: list[dict], backend: str = "heuristic") -> dict:
    """Triangulate a claim across multiple source excerpts.

    ``excerpts`` is a list of ``{"source_url": str, "text": str}``. Each is checked
    with evidence_entailment; results are aggregated into a corroboration verdict.
    """
    excerpts = excerpts or []
    per_source = []
    counts: dict[str, int] = {"supported": 0, "contradicted": 0, "neutral": 0, "unsupported": 0}
    supporting_domains: set[str] = set()

    for ex in excerpts:
        if not isinstance(ex, dict):
            continue
        text = ex.get("text", "") or ""
        src = ex.get("source_url", "") or ""
        judged = evidence_entailment(claim, text, backend=backend)
        status = judged.get("status", "unsupported")
        counts[status] = counts.get(status, 0) + 1
        host = (urlparse(src).hostname or "").lower() if src else ""
        if status == "supported" and host:
            supporting_domains.add(host)
        per_source.append({
            "source_url": src,
            "status": status,
            "score": judged.get("score"),
            "reason": judged.get("reason", ""),
        })

    total = len(per_source)
    supporting = counts.get("supported", 0)
    contradicting = counts.get("contradicted", 0)
    independent_domains = len(supporting_domains)

    if total == 0:
        verdict = "no_evidence"
    elif supporting and contradicting:
        verdict = "conflicting"
    elif supporting >= 2 and independent_domains >= 2:
        verdict = "corroborated"
    elif supporting >= 1:
        verdict = "single_source"
    elif contradicting:
        verdict = "contradicted"
    else:
        verdict = "unverified"

    return {
        "claim": claim,
        "verdict": verdict,
        "agreement": round(supporting / total, 3) if total else 0.0,
        "supporting": supporting,
        "contradicting": contradicting,
        "independent_supporting_domains": independent_domains,
        "counts": counts,
        "per_source": per_source,
    }


# ── 5. Span-level provenance ──

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "that", "this", "with", "from",
    "have", "has", "had", "not", "but", "its", "into", "than", "then", "they",
    "their", "which", "what", "when", "where", "who", "will", "would", "can",
    "could", "been", "being", "also", "such", "per", "via", "about",
}


def _tokenize(text: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and w not in _STOPWORDS}


def locate_claim_span(claim: str, source_text: str, max_spans: int = 3) -> dict:
    """Locate the sentence(s) in a source that best support a claim.

    Returns spans with character offsets and a containment score, giving
    span-level provenance instead of a whole-document citation.
    """
    import re
    text = source_text or ""
    claim_tokens = _tokenize(claim)
    if not claim_tokens or not text.strip():
        return {"claim": claim, "spans": [], "best_score": 0.0}

    spans = []
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        sentence = m.group(0).strip()
        if not sentence:
            continue
        s_tokens = _tokenize(sentence)
        overlap = claim_tokens & s_tokens
        if not overlap:
            continue
        spans.append({
            "text": sentence,
            "start": m.start(),
            "end": m.end(),
            "score": round(len(overlap) / len(claim_tokens), 3),
            "matched_terms": sorted(overlap),
        })

    spans.sort(key=lambda s: s["score"], reverse=True)
    top = spans[: max(1, max_spans)]
    return {"claim": claim, "spans": top, "best_score": top[0]["score"] if top else 0.0}


# ── 6. Recipe registry (list / get / run / delete) ──

def recipe_registry(action: str, recipe_id: str = "", source_text: str = "", input_payload: dict | None = None) -> dict:
    """Manage promoted extraction recipes saved by tool_promote.

    ``action``: list | get | run | delete.
    """
    action = (action or "list").lower()
    store = _load_recipe_store()
    recipes = store.get("recipes", {})

    if action == "list":
        return {
            "action": "list",
            "count": len(recipes),
            "recipes": [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "created_at": r.get("created_at"),
                    "plays": r.get("plays", 0),
                    "wins": r.get("wins", 0),
                }
                for r in recipes.values()
            ],
        }
    if action == "get":
        recipe = recipes.get(recipe_id)
        return {"action": "get", "found": recipe is not None, "recipe": recipe}
    if action == "delete":
        existed = recipe_id in recipes
        if existed:
            del recipes[recipe_id]
            store["recipes"] = recipes
            _save_recipe_store(store)
        return {"action": "delete", "deleted": existed, "recipe_id": recipe_id}
    if action == "run":
        recipe = recipes.get(recipe_id)
        if not recipe:
            return {"action": "run", "found": False, "error": f"recipe not found: {recipe_id}"}
        run = tool_code_run_sandboxed(recipe.get("code", ""), source_text or "", input_payload or {})
        recipe["plays"] = recipe.get("plays", 0) + 1
        if run.get("ok"):
            recipe["wins"] = recipe.get("wins", 0) + 1
        store["recipes"][recipe_id] = recipe
        _save_recipe_store(store)
        return {
            "action": "run",
            "found": True,
            "recipe_id": recipe_id,
            "ok": run.get("ok"),
            "result": run.get("result"),
            "rows": _recipe_rows(run.get("result")) if run.get("ok") else [],
        }
    return {"action": action, "error": f"unsupported action: {action}"}


# ── 7. Authenticated / cookie fetch ──

def web_fetch_authenticated(url: str, cookies: dict | None = None, headers: dict | None = None,
                            lang: str = "en", timeout: int = 20) -> dict:
    """Fetch a page that needs cookies or custom headers (logged-in / gated pages).

    ``cookies`` is a name→value map; ``headers`` adds/overrides request headers.
    """
    try:
        resp = _get(url, lang=lang, cookies=cookies or None, timeout=timeout, max_retries=1,
                    extra_headers=headers or None)
    except Exception as exc:
        return {"url": url, "authenticated": True, "error": str(exc)}

    content_type = resp.headers.get("content-type", "")
    if resp.status_code != 200:
        return {
            "url": url,
            "status_code": resp.status_code,
            "error": f"HTTP {resp.status_code}",
            "content_type": content_type,
            "source_type": classify_source(url, status_code=resp.status_code),
        }

    html = resp.text
    text = extract_content(html, url=url) or ""
    return {
        "url": url,
        "status_code": 200,
        "title": _page_title(html),
        "text": text[: core.MAX_CONTENT_CHARS],
        "text_length": len(text),
        "content_type": content_type,
        "authenticated": True,
    }


# ── 8. Controlled same-domain crawl ──

def web_crawl(start_url: str, max_pages: int = 10, same_domain: bool = True, lang: str = "en") -> dict:
    """Breadth-first crawl from a start URL, fetching and extracting each page.

    Stays within the start host by default. Capped at 50 pages for safety.
    """
    max_pages = max(1, min(int(max_pages or 10), 50))
    start_host = (urlparse(start_url).hostname or "").lower()
    seen: set[str] = set()
    queue: deque[str] = deque([start_url])
    pages = []

    while queue and len(pages) < max_pages:
        url = queue.popleft().split("#")[0]
        if url in seen:
            continue
        seen.add(url)

        fetched_url, html, pub_date, err = fetch_page(url, lang=lang)
        if err or not html:
            pages.append({"url": url, "error": err or "empty body", "text_length": 0})
            continue

        text = extract_content(html, url=fetched_url) or ""
        pages.append({
            "url": fetched_url,
            "title": _page_title(html),
            "text": text[:2000],
            "text_length": len(text),
            "published": str(pub_date) if pub_date else None,
        })

        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(fetched_url, a["href"]).split("#")[0]
                if not link.startswith("http"):
                    continue
                host = (urlparse(link).hostname or "").lower()
                if same_domain and host != start_host:
                    continue
                if link not in seen and (len(seen) + len(queue)) < max_pages * 5:
                    queue.append(link)
        except Exception:
            pass

    return {
        "start_url": start_url,
        "same_domain": same_domain,
        "pages_crawled": len(pages),
        "pages": pages,
    }


# ── 9. Consolidated dataset export ──

def export_dataset(rows: list[dict], format: str = "csv", path: str = "", columns: list[str] | None = None) -> dict:
    """Write extracted rows to a consolidated file (csv | xlsx | json).

    Without ``path``, writes a timestamped file under the cache ``exports`` dir.
    """
    fmt = (format or "csv").lower()
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return {"exported": False, "error": "rows must be a list of objects"}

    cols = list(columns) if columns else []
    if not cols:
        for r in rows:
            for key in r.keys():
                if key not in cols:
                    cols.append(key)

    if path:
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = CACHE_DIR / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "json" if fmt == "json" else fmt
        out_path = out_dir / f"dataset-{datetime.now().strftime('%Y%m%d-%H%M%S')}.{ext}"

    try:
        if fmt == "csv":
            with out_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow({c: r.get(c, "") for c in cols})
        elif fmt == "xlsx":
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.append(cols)
            for r in rows:
                ws.append([r.get(c, "") for c in cols])
            wb.save(out_path)
        elif fmt == "json":
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        else:
            return {"exported": False, "error": f"unsupported format: {fmt}"}
    except Exception as exc:
        return {"exported": False, "error": str(exc)}

    return {"exported": True, "format": fmt, "path": str(out_path), "row_count": len(rows), "columns": cols}


# ── 10. Time-series reconciliation ──

def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _detect_outliers(value_map: dict) -> list[str]:
    values = [v for v in value_map.values() if v is not None]
    if len(values) < 4:
        return []
    median = statistics.median(values)
    deviations = [abs(v - median) for v in values]
    mad = statistics.median(deviations)
    # When most values are identical the MAD collapses to 0; fall back to the mean
    # absolute deviation so a lone spike is still caught.
    scale = mad if mad > 0 else (sum(deviations) / len(deviations))
    if scale == 0:
        return []
    return sorted(k for k, v in value_map.items() if v is not None and abs(v - median) > 3 * scale)


def reconcile_time_series(series: list[dict], on: str = "date", value_field: str = "value") -> dict:
    """Align several series on a common key, compute deltas, and flag gaps/outliers.

    ``series`` is a list of ``{"name": str, "rows": [{on: ..., value_field: ...}]}``.
    Deltas are computed for each series against the first one.
    """
    series = series or []
    names: list[str] = []
    maps: dict[str, dict] = {}
    all_keys: set[str] = set()

    for i, s in enumerate(series):
        if not isinstance(s, dict):
            continue
        name = s.get("name") or f"series_{i + 1}"
        value_map: dict[str, float | None] = {}
        for r in (s.get("rows") or []):
            if not isinstance(r, dict):
                continue
            key = str(r.get(on, "")).strip()
            if not key:
                continue
            value_map[key] = _to_float(r.get(value_field))
            all_keys.add(key)
        names.append(name)
        maps[name] = value_map

    keys = sorted(all_keys)
    base = names[0] if names else None
    gaps: dict[str, list[str]] = {name: [] for name in names}
    aligned = []

    for key in keys:
        row: dict = {on: key}
        for name in names:
            value = maps[name].get(key)
            row[name] = value
            if value is None:
                gaps[name].append(key)
        if base is not None and row.get(base) is not None:
            for name in names[1:]:
                if row.get(name) is not None:
                    row[f"delta_{name}_vs_{base}"] = round(row[name] - row[base], 6)
        aligned.append(row)

    return {
        "on": on,
        "value_field": value_field,
        "series_names": names,
        "key_count": len(keys),
        "aligned": aligned,
        "missing_keys": gaps,
        "outliers": {name: _detect_outliers(maps[name]) for name in names},
    }

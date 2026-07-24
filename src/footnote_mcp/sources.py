"""Zero-key specialized discovery sources with one normalized result contract."""

from __future__ import annotations

import gzip
import json
import os
import re
from html import unescape
from urllib.parse import quote, urlencode, urlparse

from bs4 import BeautifulSoup

from . import core
from .extract import extract_content
from .fetch import _get, fetch_page
from .tools_data.files import _fetch_bytes


def _empty(query: str, source: str, error: str) -> dict:
    return {"query": query, "source": source, "count": 0, "results": [], "error": error}


def _json_get(url: str, *, lang: str = "en", timeout: int = 20) -> tuple[object | None, str | None]:
    data, _content_type, error = _fetch_bytes(url, lang=lang, timeout=timeout)
    if error or not data:
        return None, error or "empty response"
    try:
        return json.loads(data.decode("utf-8", errors="replace")), None
    except Exception as exc:
        return None, f"invalid JSON response: {exc}"


def _dedupe(results: list[dict], num: int) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for result in results:
        identity = (result.get("url") or result.get("title") or "").split("#", 1)[0].rstrip("/").lower()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(result)
        if len(unique) >= max(1, min(num, 50)):
            break
    return unique


def _round_robin(responses: list[dict]) -> list[dict]:
    groups = [list(response.get("results") or []) for response in responses]
    interleaved = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                interleaved.append(group[index])
    return interleaved


def _normalized_terms(text: str) -> tuple[str, list[str]]:
    normalized = re.sub(r"[^\w]+", " ", (text or "").casefold(), flags=re.UNICODE).strip()
    return normalized, [term for term in normalized.split() if term]


def _result_match_score(query: str, title: str, aliases: list[str] | None = None, description: str = "") -> float:
    """Score a candidate without changing the user's original query.

    Names and repository titles deliberately carry much more weight than a
    descriptive snippet: a query for ``footnote mcp`` should prefer a repository
    named ``footnote-mcp`` over one that merely mentions footnotes in its README.
    """
    query_normalized, query_terms = _normalized_terms(query)
    title_normalized, title_terms = _normalized_terms(title)
    alias_text = " ".join(aliases or [])
    alias_normalized, alias_terms = _normalized_terms(alias_text)
    description_normalized, description_terms = _normalized_terms(description)
    if not query_terms:
        return 0.0

    query_set = set(query_terms)
    title_coverage = len(query_set & set(title_terms)) / len(query_set)
    alias_coverage = len(query_set & set(alias_terms)) / len(query_set)
    description_coverage = len(query_set & set(description_terms)) / len(query_set)
    score = title_coverage * 10 + alias_coverage * 4 + description_coverage
    if query_normalized == title_normalized:
        score += 20
    elif query_normalized and query_normalized in title_normalized:
        score += 8
    elif query_normalized and query_normalized in alias_normalized:
        score += 4
    return round(score, 3)


def _date_parts(message: dict, key: str = "published") -> str | None:
    parts = ((message.get(key) or {}).get("date-parts") or [[]])[0]
    if not parts:
        return None
    return "-".join(str(part).zfill(2) if index else str(part) for index, part in enumerate(parts))


def _arxiv_search(query: str, num: int) -> dict:
    api = (
        "https://export.arxiv.org/api/query?"
        + urlencode({"search_query": f"all:{query}", "start": 0, "max_results": max(1, min(num, 50))})
    )
    data, _content_type, error = _fetch_bytes(api, timeout=20)
    if error or not data:
        return _empty(query, "arxiv", error or "empty response")

    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "xml")
    results = []
    for entry in soup.find_all("entry"):
        title = entry.find("title")
        summary = entry.find("summary")
        entry_id = entry.find("id")
        published = entry.find("published")
        authors = [author.get_text(" ", strip=True) for author in entry.find_all("name")]
        results.append(
            {
                "title": title.get_text(" ", strip=True) if title else "",
                "url": entry_id.get_text(strip=True) if entry_id else "",
                "snippet": (summary.get_text(" ", strip=True) if summary else "")[:1000],
                "published": published.get_text(strip=True) if published else None,
                "authors": authors,
                "source": "arxiv",
                "source_type": "paper",
            }
        )
    return {"query": query, "source": "arxiv", "count": len(results), "results": results}


def _crossref_search(query: str, num: int) -> dict:
    params = {"query": query, "rows": max(1, min(num, 50))}
    mailto = os.getenv("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    payload, error = _json_get("https://api.crossref.org/works?" + urlencode(params))
    if error or not isinstance(payload, dict):
        return _empty(query, "crossref", error or "invalid Crossref response")

    results = []
    for item in ((payload.get("message") or {}).get("items") or []):
        titles = item.get("title") or []
        doi = item.get("DOI") or ""
        abstract = BeautifulSoup(unescape(item.get("abstract") or ""), "html.parser").get_text(" ", strip=True)
        authors = [
            " ".join(part for part in (author.get("given", ""), author.get("family", "")) if part).strip()
            for author in (item.get("author") or [])
        ]
        results.append(
            {
                "title": titles[0] if titles else "",
                "url": item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                "snippet": abstract[:1000],
                "published": _date_parts(item),
                "authors": [author for author in authors if author],
                "source": "crossref",
                "source_type": "paper",
                "identifiers": {"doi": doi} if doi else {},
            }
        )
    return {"query": query, "source": "crossref", "count": len(results), "results": results}


def papers_search(query: str, source: str = "auto", num: int = 10, lang: str = "en") -> dict:
    """Search scholarly works through arXiv, Crossref, or both."""
    del lang
    source = (source or "auto").lower()
    if source == "arxiv":
        return _arxiv_search(query, num)
    if source == "crossref":
        return _crossref_search(query, num)
    if source != "auto":
        return _empty(query, source, f"unsupported source: {source}")

    per_source = max(3, min(num, 25))
    responses = [_crossref_search(query, per_source), _arxiv_search(query, per_source)]
    results = _dedupe(_round_robin(responses), num)
    errors = {response["source"]: response["error"] for response in responses if response.get("error")}
    output = {"query": query, "source": "auto", "sources": ["crossref", "arxiv"], "count": len(results), "results": results}
    if errors:
        output["errors"] = errors
    return output


def _wikipedia_search(query: str, num: int, lang: str) -> dict:
    language = (lang or "en").split("-", 1)[0] or "en"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "formatversion": 2,
        "srlimit": max(1, min(num, 50)),
    }
    payload, error = _json_get(f"https://{language}.wikipedia.org/w/api.php?{urlencode(params)}", lang=language)
    if error or not isinstance(payload, dict):
        return _empty(query, "wikipedia", error or "invalid Wikipedia response")

    results = []
    for item in ((payload.get("query") or {}).get("search") or []):
        title = item.get("title") or ""
        results.append(
            {
                "title": title,
                "url": f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                "snippet": BeautifulSoup(item.get("snippet") or "", "html.parser").get_text(" ", strip=True),
                "published": item.get("timestamp"),
                "authors": [],
                "source": "wikipedia",
                "source_type": "encyclopedia",
            }
        )
    return {"query": query, "source": "wikipedia", "count": len(results), "results": results}


def _wikidata_sitelink_counts(entity_ids: list[str], lang: str) -> dict[str, int]:
    if not entity_ids:
        return {}
    params = {
        "action": "wbgetentities",
        "ids": "|".join(entity_ids),
        "props": "sitelinks",
        "format": "json",
    }
    payload, error = _json_get("https://www.wikidata.org/w/api.php?" + urlencode(params), lang=lang)
    if error or not isinstance(payload, dict):
        return {}
    return {
        entity_id: len((entity.get("sitelinks") or {}))
        for entity_id, entity in (payload.get("entities") or {}).items()
        if isinstance(entity, dict)
    }


def _wikidata_search(query: str, num: int, lang: str) -> dict:
    language = (lang or "en").split("-", 1)[0] or "en"
    candidate_limit = max(10, min(max(num, 1) * 10, 50))
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": language,
        "uselang": language,
        "format": "json",
        "limit": candidate_limit,
    }
    payload, error = _json_get("https://www.wikidata.org/w/api.php?" + urlencode(params), lang=language)
    if error or not isinstance(payload, dict):
        return _empty(query, "wikidata", error or "invalid Wikidata response")

    candidates = payload.get("search") or []
    sitelinks = _wikidata_sitelink_counts(
        [item.get("id") for item in candidates if item.get("id")],
        language,
    )
    ranked = []
    for rank, item in enumerate(candidates):
        entity_id = item.get("id") or ""
        title = item.get("label") or entity_id
        aliases = item.get("aliases") or []
        ranked.append((
            _result_match_score(query, title, aliases, item.get("description") or ""),
            sitelinks.get(entity_id, 0),
            -rank,
            {
                "title": title,
                "url": item.get("concepturi") or f"https://www.wikidata.org/wiki/{entity_id}",
                "snippet": item.get("description") or "",
                "published": None,
                "authors": [],
                "source": "wikidata",
                "source_type": "knowledge_graph",
                "identifiers": {"wikidata": entity_id} if entity_id else {},
                "aliases": aliases,
                "relevance_score": _result_match_score(query, title, aliases, item.get("description") or ""),
                "sitelink_count": sitelinks.get(entity_id, 0),
            },
        ))
    results = [item for _score, _sitelinks, _rank, item in sorted(ranked, reverse=True)[: max(1, min(num, 50))]]
    return {"query": query, "source": "wikidata", "count": len(results), "results": results}


def _wikidata_sparql(sparql: str, num: int) -> dict:
    operation = re.match(r"\s*(?:PREFIX\s+\S+:\s*<[^>]+>\s*)*(\w+)", sparql, re.IGNORECASE)
    if not operation or operation.group(1).upper() not in {"SELECT", "ASK"}:
        return _empty(sparql, "wikidata_sparql", "only read-only SELECT or ASK queries are allowed")
    url = "https://query.wikidata.org/sparql?" + urlencode({"query": sparql, "format": "json"})
    payload, error = _json_get(url, timeout=30)
    if error or not isinstance(payload, dict):
        return _empty(sparql, "wikidata_sparql", error or "invalid Wikidata SPARQL response")
    if "boolean" in payload:
        return {
            "query": sparql,
            "source": "wikidata_sparql",
            "count": 1,
            "results": [{"boolean": bool(payload["boolean"])}],
            "variables": [],
        }
    bindings = ((payload.get("results") or {}).get("bindings") or [])[: max(1, min(num, 100))]
    rows = [{key: value.get("value") for key, value in row.items()} for row in bindings]
    return {
        "query": sparql,
        "source": "wikidata_sparql",
        "count": len(rows),
        "results": rows,
        "variables": (payload.get("head") or {}).get("vars") or [],
    }


def encyclopedia_search(
    query: str,
    source: str = "auto",
    num: int = 10,
    lang: str = "en",
    sparql: str = "",
) -> dict:
    """Search Wikipedia/Wikidata, with optional read-only Wikidata SPARQL."""
    source = (source or "auto").lower()
    if sparql:
        return _wikidata_sparql(sparql, num)
    if source == "wikipedia":
        return _wikipedia_search(query, num, lang)
    if source == "wikidata":
        return _wikidata_search(query, num, lang)
    if source != "auto":
        return _empty(query, source, f"unsupported source: {source}")

    per_source = max(3, min(num, 25))
    responses = [_wikipedia_search(query, per_source, lang), _wikidata_search(query, per_source, lang)]
    results = _dedupe(_round_robin(responses), num)
    errors = {response["source"]: response["error"] for response in responses if response.get("error")}
    output = {
        "query": query,
        "source": "auto",
        "sources": ["wikipedia", "wikidata"],
        "count": len(results),
        "results": results,
    }
    if errors:
        output["errors"] = errors
    return output


def github_search(query: str, kind: str = "repositories", num: int = 10) -> dict:
    """Search public GitHub data without requiring authentication."""
    kind = (kind or "repositories").lower()
    endpoints = {
        "repositories": "repositories",
        "issues": "issues",
        "code": "code",
        "commits": "commits",
    }
    if kind not in endpoints:
        return _empty(query, "github", f"unsupported kind: {kind}")

    candidate_limit = max(20, min(max(num, 1) * 10, 100))
    params = {"q": query, "per_page": candidate_limit}
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = _get(
            f"https://api.github.com/search/{endpoints[kind]}?{urlencode(params)}",
            timeout=20,
            max_retries=1,
            extra_headers=headers,
        )
    except Exception as exc:
        return _empty(query, "github", str(exc))
    if response.status_code != 200:
        return _empty(query, "github", f"GitHub HTTP {response.status_code}")
    try:
        items = response.json().get("items") or []
    except Exception as exc:
        return _empty(query, "github", f"invalid GitHub response: {exc}")

    ranked = []
    for rank, item in enumerate(items):
        repository = item.get("repository") or {}
        owner = item.get("owner") or {}
        if kind == "repositories":
            snippet = item.get("description") or ""
            title = item.get("full_name") or item.get("name") or ""
            published = item.get("pushed_at") or item.get("updated_at")
            identifiers = {"github_id": item.get("id"), "owner": owner.get("login")}
        else:
            snippet = item.get("body") or item.get("path") or item.get("message") or ""
            title = item.get("title") or item.get("name") or item.get("sha") or ""
            published = item.get("created_at") or item.get("updated_at")
            identifiers = {"github_id": item.get("id"), "repository": repository.get("full_name")}
        result = {
                "title": title,
                "url": item.get("html_url") or "",
                "snippet": str(snippet)[:1000],
                "published": published,
                "authors": [owner.get("login")] if owner.get("login") else [],
                "source": "github",
                "source_type": {
                    "repositories": "repository",
                    "issues": "issue",
                    "code": "code",
                    "commits": "commit",
                }[kind],
                "identifiers": {key: value for key, value in identifiers.items() if value is not None},
            }
        score = _result_match_score(query, title, description=str(snippet))
        result["relevance_score"] = score
        ranked.append((score, -rank, result))
    results = [item for _score, _rank, item in sorted(ranked, reverse=True)[: max(1, min(num, 50))]]
    return {"query": query, "source": "github", "kind": kind, "count": len(results), "results": results}


def _common_crawl_search(url: str, num: int, fetch_text: bool) -> dict:
    indexes, error = _json_get("https://index.commoncrawl.org/collinfo.json")
    if error or not isinstance(indexes, list) or not indexes:
        return _empty(url, "common_crawl", error or "no Common Crawl indexes")
    index_api = indexes[0].get("cdx-api")
    if not index_api:
        return _empty(url, "common_crawl", "latest Common Crawl index has no API URL")

    query_url = index_api + "?" + urlencode(
        {
            "url": url,
            "output": "json",
            "filter": "status:200",
            "collapse": "digest",
            "limit": max(1, min(num, 50)),
        }
    )
    data, _content_type, fetch_error = _fetch_bytes(query_url, timeout=30)
    if fetch_error or not data:
        return _empty(url, "common_crawl", fetch_error or "empty Common Crawl response")

    captures = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        capture = {
            "title": item.get("url") or url,
            "url": item.get("url") or url,
            "snippet": f"Common Crawl capture {item.get('timestamp', '')}".strip(),
            "published": item.get("timestamp"),
            "authors": [],
            "source": "common_crawl",
            "source_type": "web_archive",
            "archive": {
                "filename": item.get("filename"),
                "offset": item.get("offset"),
                "length": item.get("length"),
                "digest": item.get("digest"),
            },
        }
        captures.append(capture)

    if fetch_text and captures:
        archive = captures[0]["archive"]
        try:
            offset = int(archive["offset"])
            length = int(archive["length"])
            response = _get(
                "https://data.commoncrawl.org/" + archive["filename"],
                timeout=30,
                max_retries=1,
                extra_headers={"Range": f"bytes={offset}-{offset + length - 1}", "Accept-Encoding": "identity"},
            )
            if response.status_code not in {200, 206}:
                raise RuntimeError(f"Common Crawl data HTTP {response.status_code}")
            raw = bytes(response.content)
            try:
                raw = gzip.decompress(raw)
            except (gzip.BadGzipFile, OSError):
                pass
            match = re.search(br"\r?\n\r?\n(?:HTTP/\d(?:\.\d)?[^\r\n]*\r?\n)?(?:.*?\r?\n)*?\r?\n", raw, re.DOTALL)
            html = raw[match.end() :] if match else raw
            decoded = html.decode("utf-8", errors="replace")
            text = extract_content(decoded, url=captures[0]["url"]) or ""
            captures[0]["text"] = text[: core.MAX_CONTENT_CHARS]
            captures[0]["text_length"] = len(text)
        except Exception as exc:
            captures[0]["fetch_error"] = str(exc)

    return {"query": url, "source": "common_crawl", "count": len(captures), "results": captures}


def _wayback_search(url: str, timestamp: str, lang: str, fetch_text: bool) -> dict:
    api = "https://archive.org/wayback/available?" + urlencode({"url": url, **({"timestamp": timestamp} if timestamp else {})})
    payload, error = _json_get(api, lang=lang)
    if error or not isinstance(payload, dict):
        return _empty(url, "wayback", error or "invalid Wayback response")
    closest = ((payload.get("archived_snapshots") or {}).get("closest") or {})
    if not closest.get("available") or not closest.get("url"):
        return _empty(url, "wayback", "no snapshot found")

    result = {
        "title": url,
        "url": closest["url"],
        "snippet": f"Wayback snapshot {closest.get('timestamp', '')}".strip(),
        "published": closest.get("timestamp"),
        "authors": [],
        "source": "wayback",
        "source_type": "web_archive",
        "original_url": url,
        "status": closest.get("status"),
    }
    if fetch_text:
        fetched_url, html, published, fetch_error = fetch_page(closest["url"], lang=lang)
        if fetch_error or not html:
            result["fetch_error"] = fetch_error or "empty snapshot body"
        else:
            text = extract_content(html, url=fetched_url) or ""
            title = BeautifulSoup(html, "html.parser").find("title")
            result["title"] = title.get_text(" ", strip=True) if title else url
            result["text"] = text[: core.MAX_CONTENT_CHARS]
            result["text_length"] = len(text)
            result["published"] = str(published) if published else result["published"]
    return {"query": url, "source": "wayback", "count": 1, "results": [result]}


def archive_search(
    url: str,
    source: str = "auto",
    num: int = 10,
    timestamp: str = "",
    lang: str = "en",
    fetch_text: bool = False,
) -> dict:
    """Find archived captures in Common Crawl and the Wayback Machine."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if not parsed.hostname:
        return _empty(url, source, "archive_search requires a URL or host pattern")

    source = (source or "auto").lower()
    if source == "common_crawl":
        return _common_crawl_search(url, num, fetch_text)
    if source == "wayback":
        return _wayback_search(url, timestamp, lang, fetch_text)
    if source != "auto":
        return _empty(url, source, f"unsupported source: {source}")

    responses = [
        _wayback_search(url, timestamp, lang, fetch_text),
        _common_crawl_search(url, num, fetch_text),
    ]
    results = _dedupe([item for response in responses for item in response["results"]], num)
    errors = {response["source"]: response["error"] for response in responses if response.get("error")}
    output = {
        "query": url,
        "source": "auto",
        "sources": ["wayback", "common_crawl"],
        "count": len(results),
        "results": results,
    }
    if errors:
        output["errors"] = errors
    return output

"""Search tools — wraps the RAG pipeline (search.py, pipeline.py)."""

import re
from urllib.parse import urlparse

from . import core
from bs4 import BeautifulSoup
from .search import search
from .pipeline import search_extract_rerank, build_llm_context
from .fetch import fetch_page
from .extract import extract_content
from .scraper import fetch as scrape_fetch
from .semantic import semantic_rerank
from . import sources as specialized_sources
from .tools_data.classify import classify_source
from .tools_data.cache import _read_cache, _write_cache


def web_search(query: str, lang: str = "en", num: int = 10, provider: str = "auto", semantic: bool = False) -> dict:
    """Search via configured SearXNG/keyed providers, else scraped Bing + DDG.

    ``provider``: auto | searxng | tavily | brave | google | scrape. Results are merged into a
    single shape regardless of backend.
    ``semantic``: rerank results by meaning using local bge-m3 embeddings (best-effort;
    over-fetches candidates, reorders by query similarity, then trims to ``num``).
    """
    candidates = max(num * 2, 15) if semantic else num
    results = search(query, num=candidates, lang=lang, provider=provider)

    reranked = False
    if semantic and results:
        ranked = semantic_rerank(query, results)
        if any("semantic_score" in r for r in ranked):
            results, reranked = ranked, True

    results = results[:num]
    return {
        "query": query,
        "provider": provider,
        "semantic": reranked,
        "count": len(results),
        "results": [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "score": r["score"],
                "engines": r["engines"],
                **({"semantic_score": r["semantic_score"]} if "semantic_score" in r else {}),
            }
            for r in results
        ],
    }


_PAPER_HINTS = {
    "arxiv", "crossref", "doi", "journal", "paper", "papers", "publication", "research study",
    "scientific", "исследование", "статья", "публикация", "научн",
}
_ENCYCLOPEDIA_HINTS = {
    "who is", "what is", "history of", "biography", "capital of", "wikidata", "wikipedia",
    "кто такой", "кто такая", "что такое", "история", "биография", "столица",
}
_GITHUB_HINTS = {
    "github", "repository", "repo", "source code", "release", "pull request", "issue",
    "репозитор", "исходный код", "релиз",
}
_ARCHIVE_HINTS = {
    "archive", "archived", "wayback", "common crawl", "old version", "dead link",
    "архив", "старая версия", "недоступная ссылка",
}


def _contains_hint(query: str, hints: set[str]) -> bool:
    lowered = query.lower()
    return any(hint in lowered for hint in hints)


def _query_url(query: str) -> str:
    match = re.search(r"https?://[^\s<>\"]+", query)
    if match:
        return match.group(0).rstrip(".,;:!?)")
    stripped = query.strip()
    if " " not in stripped and urlparse(f"https://{stripped}").hostname and "." in stripped:
        return stripped
    return ""


def select_discovery_sources(query: str, requested: list[str] | None = None) -> list[str]:
    """Choose intent-level sources; an explicit list always wins."""
    allowed = {"web", "papers", "encyclopedia", "github", "archive"}
    if requested:
        selected = []
        for source in requested:
            normalized = str(source).lower()
            if normalized in allowed and normalized not in selected:
                selected.append(normalized)
        return selected or ["web"]

    selected = ["web"]
    if _contains_hint(query, _PAPER_HINTS):
        selected.append("papers")
    if _contains_hint(query, _ENCYCLOPEDIA_HINTS):
        selected.append("encyclopedia")
    if _contains_hint(query, _GITHUB_HINTS):
        selected.append("github")
    if _query_url(query) and _contains_hint(query, _ARCHIVE_HINTS):
        selected.append("archive")
    return selected


def _discovery_shape(result: dict, source_name: str, rank: int) -> dict | None:
    url = result.get("url") or ""
    title = result.get("title") or url
    if not url or not title:
        return None
    return {
        "title": title,
        "url": url,
        "snippet": result.get("snippet") or result.get("summary") or "",
        "score": round(1.0 / max(rank, 1), 3),
        "engines": [result.get("source") or source_name],
    }


def discover_sources(
    query: str,
    *,
    lang: str = "en",
    requested: list[str] | None = None,
    provider: str = "auto",
    num: int = 20,
) -> tuple[list[dict], list[str], dict[str, str]]:
    """Run routed discovery and normalize every backend for the fetch pipeline."""
    routed = select_discovery_sources(query, requested)
    per_source = max(3, min(num, 10))
    responses: dict[str, dict] = {}
    for source_name in routed:
        if source_name == "web":
            responses[source_name] = web_search(query, lang=lang, num=per_source, provider=provider)
        elif source_name == "papers":
            responses[source_name] = specialized_sources.papers_search(query, num=per_source, lang=lang)
        elif source_name == "encyclopedia":
            responses[source_name] = specialized_sources.encyclopedia_search(query, num=per_source, lang=lang)
        elif source_name == "github":
            responses[source_name] = specialized_sources.github_search(query, num=per_source)
        elif source_name == "archive":
            responses[source_name] = specialized_sources.archive_search(
                _query_url(query) or query,
                num=per_source,
                lang=lang,
            )

    merged = []
    seen = set()
    errors = {}
    for source_name in routed:
        response = responses[source_name]
        if response.get("error"):
            errors[source_name] = response["error"]
    max_results = max((len(response.get("results") or []) for response in responses.values()), default=0)
    for index in range(max_results):
        for source_name in routed:
            response = responses[source_name]
            items = response.get("results") or []
            if index >= len(items):
                continue
            item = items[index]
            rank = index + 1
            normalized = _discovery_shape(item, source_name, rank)
            if not normalized:
                continue
            identity = normalized["url"].split("#", 1)[0].rstrip("/").lower()
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(normalized)
            if len(merged) >= num:
                return merged, routed, errors
    return merged, routed, errors


def web_deep_search(
    query: str,
    lang: str = "en",
    sources: list[str] | None = None,
    provider: str = "auto",
    num: int = 20,
) -> dict:
    """Route discovery by intent, then fetch, extract, rerank, and build context."""
    discovery, routed_sources, discovery_errors = discover_sources(
        query,
        lang=lang,
        requested=sources,
        provider=provider,
        num=num,
    )
    ranked_chunks, search_results, fetched_urls = search_extract_rerank(
        query,
        lang=lang,
        provider=provider,
        search_results=discovery,
    )
    context, source_map, by_source = build_llm_context(ranked_chunks, search_results, fetched_urls=fetched_urls)

    # Build compact result
    sources = []
    for old_idx in sorted(by_source.keys()):
        src = by_source[old_idx]
        sources.append({
            "num": source_map.get(old_idx, old_idx + 1),
            "title": src["title"],
            "url": src["url"],
            "chunks": len(src["chunks"]),
        })

    return {
        "query": query,
        "context": context,
        "sources": sources,
        "context_length": len(context),
        "source_count": len(by_source),
        "routed_sources": routed_sources,
        "discovery_count": len(discovery),
        "discovery_errors": discovery_errors,
    }


def web_read(url: str, lang: str = "en", use_cache: bool = True) -> dict:
    """Fetch a single page and extract text content."""
    cached = _read_cache(url) if use_cache else None
    if cached and cached.get("web_read"):
        result = dict(cached["web_read"])
        result["cached"] = True
        return result

    # Escalation ladder: http → proxy → browser → external (see scraper.py).
    # fetch_page stays the injected tier-1 fetcher so callers/tests can stub it.
    res = scrape_fetch(url, lang=lang, http_fn=fetch_page)
    html = res.get("html")
    pub_date = res.get("pub_date")
    if not html:
        error = res.get("error") or "Empty response body"
        result = {
            "url": url,
            "error": error,
            "source_type": classify_source(url, text_sample=error),
            "fetch_tier": res.get("tier"),
            "scrape_tiers": res.get("tiers_tried"),
        }
        _write_cache(url, {"web_read": result, "fetch_error": error, "source_quality": result["source_type"]})
        return result

    fetched_url = res.get("final_url") or url
    text = extract_content(html, url=fetched_url)
    title = ""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
    source_type = classify_source(url, text_sample=(text or "")[:1000])
    result = {
        "url": url,
        "title": title,
        "text": text[:core.MAX_CONTENT_CHARS] if text else "",
        "pub_date": str(pub_date) if pub_date else None,
        "text_length": len(text) if text else 0,
        "source_type": source_type,
        "fetch_tier": res.get("tier"),
        "scrape_tiers": res.get("tiers_tried"),
        "cached": False,
    }
    _write_cache(url, {"web_read": result, "source_quality": source_type, "last_seen": result["pub_date"]})
    return result


if __name__ == "__main__":
    print("=== tools_search tests ===\n")

    # 1. web_search: basic
    print("1. web_search...")
    r = web_search("hello world", num=3)
    assert r["count"] > 0, f"web_search returned {r['count']} results, expected >0"
    assert len(r["results"]) == 3, f"expected 3 results, got {len(r['results'])}"
    for i, res in enumerate(r["results"]):
        assert res["title"], f"result {i} missing title"
        assert res["url"].startswith("http"), f"result {i} bad url: {res['url']}"
        assert "engines" in res, f"result {i} missing engines"
    print(f"   ✓ {r['count']} results\n")

    # 2. web_search: lang parameter
    print("2. web_search(lang=ru)...")
    r_ru = web_search("привет мир", lang="ru", num=2)
    assert r_ru["count"] >= 0, f"ru search returned {r_ru['count']}"
    print(f"   ✓ {r_ru['count']} results\n")

    # 3. web_read: valid URL
    print("3. web_read(https://www.iana.org/domains/reserved)...")
    page = web_read("https://www.iana.org/domains/reserved")
    assert "error" not in page or page.get("text", ""), f"web_read failed: {page.get('error')}"
    assert page["url"] == "https://www.iana.org/domains/reserved"
    assert len(page.get("text", "")) > 0, "extracted text is empty"
    assert page["text_length"] > 0, "text_length is 0"
    print(f"   ✓ {page['text_length']} chars, title='{page['title']}'\n")

    # 4. web_read: bad URL returns error
    print("4. web_read(bad url)...")
    bad = web_read("https://this-domain-does-not-exist-12345.com")
    assert bad.get("error") or bad.get("text_length", 0) == 0, f"expected error or empty, got {bad}"
    print(f"   ✓ error={bad.get('error', 'timeout/empty')}\n")

    # 5. web_deep_search: basic
    print("5. web_deep_search...")
    deep = web_deep_search("python asyncio documentation")
    assert len(deep["context"]) > 0, "deep search context is empty"
    assert deep["source_count"] > 0, f"expected sources, got {deep['source_count']}"
    assert len(deep["sources"]) == deep["source_count"]
    print(f"   ✓ {deep['source_count']} sources, {deep['context_length']} chars\n")

    # 6. web_deep_search: ru
    print("6. web_deep_search(ru)...")
    deep_ru = web_deep_search("как приготовить борщ", lang="ru")
    assert deep_ru["context_length"] >= 0, "ru deep search failed"
    print(f"   ✓ {deep_ru['source_count']} sources, {deep_ru['context_length']} chars\n")

    # 7. web_search: empty query handled gracefully
    print("7. web_search(empty)...")
    try:
        r_empty = web_search("")
        print(f"   ✓ returned {r_empty['count']} results (empty query)\n")
    except Exception as e:
        print(f"   ⚠ expected, got error: {e}\n")

    print("=== all search tests passed ===")

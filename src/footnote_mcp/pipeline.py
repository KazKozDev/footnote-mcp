from __future__ import annotations

from .diagnostics import log


def search_extract_rerank(query, num_fetch=None, lang="en", debug=False):
    """
    Full pipeline:
      1. Search Bing+DDG → merge
      2. Take top N results
      3. Parallel fetch + extraction
      4. Chunk extracted text
      5. Rerank chunks vs query
      6. Return ranked context chunks with metadata
    """
    from . import core
    from .extract import chunk_text, filter_low_quality_chunks
    from .fetch import fetch_pages_parallel
    from .rerank import filter_results_by_relevance, rerank_chunks
    from .search import search

    if num_fetch is None:
        num_fetch = core.TOP_N_FETCH

    log.info("QUERY: %s", query)

    all_results = search(query, num=20, lang=lang, debug=debug)
    if not all_results:
        return [], [], set()

    threshold = 0.25 if lang == "ru" else 0.30
    log.info("[PIPELINE] Pre-filtering %s results by semantic relevance (threshold=%s)", len(all_results), threshold)
    all_results = filter_results_by_relevance(query, all_results, threshold=threshold, lang=lang)

    if not all_results:
        log.info("[PIPELINE] No relevant results after pre-filtering")
        return [], [], set()

    top_results = all_results[:num_fetch]
    log.info("[PIPELINE] Top %s results to fetch:", len(top_results))
    for i, result in enumerate(top_results, 1):
        engines = ", ".join(result["engines"])
        log.info("  %s. [%s] (score=%s) %s", i, engines, result["score"], result["title"][:60])
        log.info("     %s", result["url"][:80])

    log.info("[PIPELINE] Fetching %s pages in parallel", len(top_results))
    urls = [result["url"] for result in top_results]
    fetched = fetch_pages_parallel(urls, query=query, lang=lang)

    if not fetched:
        log.info("[PIPELINE] No pages fetched successfully; using snippets only")
        chunks_with_meta = []
        snippet_urls = set()
        for i, result in enumerate(top_results):
            if result["snippet"]:
                chunks_with_meta.append(
                    {
                        "text": result["snippet"],
                        "source_idx": i,
                        "source_url": result["url"],
                        "source_title": result["title"],
                        "chunk_idx": 0,
                    }
                )
                snippet_urls.add(result["url"])
        ranked = rerank_chunks(query, chunks_with_meta, top_k=12, lang=lang)
        return ranked, top_results, snippet_urls

    log.info("[PIPELINE] Chunking %s pages", len(fetched))
    all_chunks = []

    for i, result in enumerate(top_results):
        page_data = fetched.get(result["url"])
        if not page_data:
            if result["snippet"]:
                all_chunks.append(
                    {
                        "text": result["snippet"],
                        "source_idx": i,
                        "source_url": result["url"],
                        "source_title": result["title"],
                        "chunk_idx": 0,
                        "pub_date": None,
                    }
                )
            continue

        text = page_data["text"]
        pub_date = page_data.get("pub_date")
        chunks = filter_low_quality_chunks(chunk_text(text, lang=lang))
        log.info("  Source %s: %s chunks from %s", i + 1, len(chunks), result["title"][:50])

        for ci, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "source_idx": i,
                    "source_url": result["url"],
                    "source_title": result["title"],
                    "chunk_idx": ci,
                    "pub_date": pub_date,
                }
            )

    log.info("  Total chunks: %s", len(all_chunks))

    ranked = rerank_chunks(query, all_chunks, top_k=12, lang=lang)
    log.info("  Selected top %s chunks:", len(ranked))
    for chunk in ranked:
        if core.HAS_EMBEDDINGS:
            log.info(
                "    src=%s chunk=%s rel=%.3f (bm25=%.3f sem=%.3f) - %s...",
                chunk["source_idx"] + 1,
                chunk["chunk_idx"],
                chunk["relevance"],
                chunk["bm25"],
                chunk["semantic"],
                chunk["text"][:50],
            )
        else:
            log.info(
                "    src=%s chunk=%s rel=%.4f - %s...",
                chunk["source_idx"] + 1,
                chunk["chunk_idx"],
                chunk["relevance"],
                chunk["text"][:60],
            )

    fetched_urls = set(fetched.keys()) if fetched else set()
    return ranked, top_results, fetched_urls


def build_llm_context(ranked_chunks, search_results, fetched_urls=None, renumber_sources=True):
    if not ranked_chunks:
        return "No relevant content found.", {}, {}

    by_source = {}
    skipped_ghost_sources = set()
    for chunk in ranked_chunks:
        idx = chunk["source_idx"]
        url = chunk["source_url"]
        if fetched_urls is not None and url not in fetched_urls:
            skipped_ghost_sources.add((idx, chunk["source_title"], url))
            continue
        if idx not in by_source:
            by_source[idx] = {"title": chunk["source_title"], "url": url, "chunks": []}
        by_source[idx]["chunks"].append(chunk)

    if skipped_ghost_sources:
        log.info("[CONTEXT] Filtered out %s sources with failed fetch:", len(skipped_ghost_sources))
        for idx, title, url in sorted(skipped_ghost_sources):
            log.info("  [%s] %s... - %s", idx + 1, title[:60], url[:60])

    filtered_sources = {}
    for idx, source_data in by_source.items():
        if source_data["chunks"]:
            filtered_sources[idx] = source_data

    by_source = filtered_sources
    if not by_source:
        return "No relevant sources found.", {}, {}

    if renumber_sources:
        source_mapping = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(sorted(by_source.keys()))}
    else:
        source_mapping = {idx: idx + 1 for idx in by_source.keys()}

    parts = []
    for old_idx in sorted(by_source.keys()):
        src = by_source[old_idx]
        src_num = source_mapping[old_idx]
        parts.append(f"[{src_num}] {src['title']}")
        for chunk in src["chunks"]:
            parts.append(chunk["text"].replace("**", ""))
        parts.append("")

    return "\n".join(parts), source_mapping, by_source

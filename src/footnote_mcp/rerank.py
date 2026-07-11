from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .diagnostics import log

_EMBEDDING_MODEL = None
_CROSS_ENCODER_MODEL = None
_FACTUAL_DATA_PATTERNS = (
    re.compile(r"\d+[.,]\d+"),
    re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d+\s*%"),
    re.compile(r"[\$€₽£¥]\s*\d+|руб|usd|eur|btc", re.IGNORECASE),
    re.compile(r"\d{4,}"),
    re.compile(r"\d{1,2}:\d{2}"),
)


def _tokenize(text):
    # ponytail: hand-rolled stopwords (35 ru + 35 en). ceiling: ~100 words. upgrade: nltk.corpus.stopwords when adding 3rd+ language.
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    _stop = {
        "в",
        "и",
        "на",
        "с",
        "по",
        "для",
        "не",
        "что",
        "это",
        "как",
        "из",
        "за",
        "к",
        "до",
        "от",
        "при",
        "или",
        "но",
        "а",
        "то",
        "все",
        "так",
        "может",
        "быть",
        "год",
        "года",
        "уже",
        "более",
    }
    _stop_en = {
        "the",
        "is",
        "at",
        "which",
        "on",
        "a",
        "an",
        "as",
        "are",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
    }
    return [t for t in tokens if len(t) > 2 and t not in (_stop | _stop_en)]


def _get_embedding_model(lang="en"):
    from . import core

    global _EMBEDDING_MODEL
    if not core.HAS_EMBEDDINGS:
        return None
    if _EMBEDDING_MODEL is None:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if lang == "ru":
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            log.info("[EMBEDDINGS] Loading %s (multilingual for Russian)", model_name)
        else:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            log.info("[EMBEDDINGS] Loading %s (English)", model_name)
        _EMBEDDING_MODEL = core.SentenceTransformer(model_name)
        log.info("[EMBEDDINGS] Model loaded")
    return _EMBEDDING_MODEL


def _get_cross_encoder_model(lang="en"):
    from . import core

    global _CROSS_ENCODER_MODEL
    if not core.HAS_EMBEDDINGS:
        return None
    if _CROSS_ENCODER_MODEL is None:
        if lang == "ru":
            model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            log.info("[CROSS-ENCODER] Loading %s (multilingual for Russian)", model_name)
        else:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            log.info("[CROSS-ENCODER] Loading %s (English)", model_name)
        _CROSS_ENCODER_MODEL = core.CrossEncoder(model_name)
        log.info("[CROSS-ENCODER] Model loaded")
    return _CROSS_ENCODER_MODEL


def _semantic_similarity(query, texts, lang="en"):
    model = _get_embedding_model(lang)
    if model is None:
        return [0.0] * len(texts)
    try:
        query_embedding = model.encode(query, convert_to_tensor=False)
        text_embeddings = model.encode(texts, convert_to_tensor=False)
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(text_embeddings))[0].tolist()
    except Exception as exc:
        log.warning("[EMBEDDINGS] Error: %s", exc)
        return [0.0] * len(texts)


def filter_results_by_relevance(query, results, threshold=0.25, lang="en"):
    from . import core

    if not results or not core.HAS_EMBEDDINGS:
        return results
    try:
        texts = [r["title"] + " " + r.get("snippet", "") for r in results]
        sims = _semantic_similarity(query, texts, lang=lang)
        filtered = []
        dropped_count = 0
        for result, sim in zip(results, sims):
            if sim >= threshold:
                filtered.append(result)
            else:
                dropped_count += 1
                log.info("  [PRE-FILTER] Dropped (sim=%.3f): %s...", sim, result["title"][:60])
        if dropped_count > 0:
            log.info("  [PRE-FILTER] Kept %s/%s results (threshold=%s)", len(filtered), len(results), threshold)
        return filtered
    except Exception as exc:
        log.warning("  [PRE-FILTER] Error: %s, returning all results", exc)
        return results


def _has_factual_data(text):
    return sum(bool(pattern.search(text)) for pattern in _FACTUAL_DATA_PATTERNS) >= 2


def _cross_encoder_rerank(query, chunks, top_k, lang="en"):
    model = _get_cross_encoder_model(lang)
    if not model or len(chunks) <= top_k:
        return chunks
    try:
        scores = model.predict([[query, chunk["text"]] for chunk in chunks])
        for i, chunk in enumerate(chunks):
            chunk["cross_encoder_score"] = float(scores[i])
            chunk["relevance"] = chunk["relevance"] * 0.6 + scores[i] * 0.4
        chunks.sort(key=lambda x: -x["relevance"])
        log.info("  [CROSS-ENCODER] Reranked %s chunks", len(chunks))
        return chunks[:top_k]
    except Exception as exc:
        log.warning("  [WARN] Cross-encoder reranking failed: %s", exc)
        return chunks[:top_k]




def rerank_chunks(query, chunks_with_meta, top_k=None, lang="en"):
    from . import core

    if top_k is None:
        top_k = core.TOTAL_CONTEXT_CHUNKS
    if not chunks_with_meta:
        return []

    query_tokens = _tokenize(query)
    # ponytail: static 0.7/0.3 BM25/semantic ratio. ceiling: no per-lang tuning. upgrade: per-language split when non-English results get <70% user-rated relevance.
    if lang == "ru":
        bm25_ratio = 0.5
        semantic_ratio = 0.5
    else:
        bm25_ratio = 0.7
        semantic_ratio = 0.3

    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks_with_meta])
    bm25_scores = list(bm25.get_scores(query_tokens))
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    if max_bm25 > 0:
        bm25_scores = [score / max_bm25 for score in bm25_scores]

    semantic_scores = [0.0] * len(chunks_with_meta)
    if core.HAS_EMBEDDINGS:
        semantic_scores = _semantic_similarity(query, [c["text"] for c in chunks_with_meta], lang=lang)

    scored = []
    for i, chunk in enumerate(chunks_with_meta):
        hybrid_score = bm25_ratio * bm25_scores[i] + semantic_ratio * semantic_scores[i]
        if chunk["chunk_idx"] < 3:
            hybrid_score *= 1.1
        if any(token in chunk.get("source_title", "").lower() for token in query_tokens):
            hybrid_score *= 1.2
        if _has_factual_data(chunk["text"]):
            hybrid_score *= 1.15
        scored.append(
            {
                **chunk,
                "relevance": round(hybrid_score, 4),
                "bm25": round(bm25_scores[i], 4),
                "semantic": round(semantic_scores[i], 4),
            }
        )

    scored.sort(key=lambda x: (-x["relevance"], x["source_idx"], x["chunk_idx"]))

    deduplicated = []
    for chunk in scored:
        chunk_tokens = _tokenize(chunk["text"])
        is_duplicate = False
        for existing in deduplicated:
            existing_tokens = _tokenize(existing["text"])
            if chunk_tokens and existing_tokens:
                common = set(chunk_tokens) & set(existing_tokens)
                similarity = len(common) / max(len(chunk_tokens), len(existing_tokens))
                if similarity > 0.85:
                    is_duplicate = True
                    break
        if not is_duplicate:
            deduplicated.append(chunk)
    if len(scored) - len(deduplicated) > 0:
        log.info("  [DEDUP] Removed %s duplicate chunks", len(scored) - len(deduplicated))

    from urllib.parse import urlparse

    chunks_per_page = 5 if top_k >= 25 else 4 if top_k >= 15 else core.TOP_CHUNKS_PER_PAGE
    per_source = {}
    per_domain = {}
    selected = []

    for chunk in deduplicated:
        src = chunk["source_idx"]
        if per_source.get(src, 0) >= chunks_per_page:
            continue
        try:
            domain = urlparse(chunk.get("source_url", "")).netloc.replace("www.", "")
            if per_domain.get(domain, 0) >= 3 and selected:
                avg_rel = sum(c["relevance"] for c in selected) / len(selected)
                if chunk["relevance"] < avg_rel * 0.9:
                    continue
        except Exception:
            domain = "unknown"
        per_source[src] = per_source.get(src, 0) + 1
        per_domain[domain] = per_domain.get(domain, 0) + 1
        selected.append(chunk)
        if len(selected) >= top_k * 2:
            break

    selected = _cross_encoder_rerank(query, selected, top_k=top_k, lang=lang) if core.HAS_EMBEDDINGS else selected[:top_k]

    if selected:
        unique_domains = len(set(per_domain.keys()))
        log.info("  [DIVERSITY] %s chunks from %s sources, %s domains", len(selected), len(per_source), unique_domains)

    return selected

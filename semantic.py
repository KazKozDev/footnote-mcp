"""Semantic (embedding) reranking via a local ollama embedding model (bge-m3).

Keyword search engines rank by lexical overlap; this reorders their results by
*meaning* — cosine similarity between the query and each result in bge-m3 space.
Best-effort: if ollama or the model is unavailable, callers get the original order.
"""

from __future__ import annotations

import math
import os

from curl_cffi import requests as http

from diagnostics import log


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def embed_model() -> str:
    return os.getenv("WEBOPERATOR_EMBED_MODEL", "bge-m3")


def embed_texts(texts, model=None, timeout=30):
    """Embed a list of texts with the ollama embedding model.

    Tries the batch ``/api/embed`` endpoint first, falling back to the singular
    ``/api/embeddings`` (one call per text) for older ollama builds.
    """
    texts = list(texts)
    if not texts:
        return []
    model = model or embed_model()
    host = _ollama_host()

    try:
        resp = http.post(f"{host}/api/embed", json={"model": model, "input": texts}, timeout=timeout)
        if resp.status_code == 200:
            embeddings = resp.json().get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
    except Exception as exc:
        log.warning("[SEMANTIC] /api/embed failed (%s); falling back to /api/embeddings", exc)

    out = []
    for text in texts:
        resp = http.post(f"{host}/api/embeddings", json={"model": model, "prompt": text}, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"embeddings HTTP {resp.status_code}")
        embedding = resp.json().get("embedding")
        if not embedding:
            raise RuntimeError("no embedding returned")
        out.append(embedding)
    return out


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_rerank(query, results, text_fields=("title", "snippet"), model=None, timeout=30):
    """Reorder search results by semantic similarity to the query.

    Each returned result gains a ``semantic_score``. On any embedding failure the
    original results are returned unchanged (best-effort enhancement).
    """
    if not results:
        return results
    try:
        texts = [" ".join(str(r.get(f, "")) for f in text_fields).strip() for r in results]
        vectors = embed_texts([query] + texts, model=model, timeout=timeout)
        query_vec, doc_vecs = vectors[0], vectors[1:]
        scored = []
        for result, vec in zip(results, doc_vecs):
            item = dict(result)
            item["semantic_score"] = round(_cosine(query_vec, vec), 4)
            scored.append(item)
        scored.sort(key=lambda x: x["semantic_score"], reverse=True)
        return scored
    except Exception as exc:
        log.warning("[SEMANTIC] rerank unavailable: %s", exc)
        return results

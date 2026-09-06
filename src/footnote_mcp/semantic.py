"""Semantic (embedding) reranking with bge-m3.

Keyword search engines rank by lexical overlap; this reorders their results by
*meaning* — cosine similarity between the query and each result in bge-m3 space.

Two runtimes serve the same model. ``ollama`` (the default) talks to a running
daemon, which costs nothing extra where one is already installed. ``local`` loads
``BAAI/bge-m3`` through transformers in this process, which needs no daemon at
all — the only option inside a container, where the default silently cannot work.
``FOOTNOTE_EMBED_BACKEND`` chooses; ``auto`` tries the daemon and falls back.

Best-effort throughout: if no runtime is available, callers get the original order.
"""

from __future__ import annotations

import math
import os

from curl_cffi import requests as http

from .diagnostics import log


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def embed_model() -> str:
    return os.getenv("FOOTNOTE_EMBED_MODEL", "bge-m3")


def embed_backend() -> str:
    return (os.getenv("FOOTNOTE_EMBED_BACKEND", "auto") or "auto").strip().lower()


def _local_model_id(model: str | None) -> str:
    """Ollama's short tag and the Hub repo id name the same weights."""
    name = model or embed_model()
    return "BAAI/bge-m3" if name in ("bge-m3", "bge-m3:latest") else name


_LOCAL_CACHE: dict = {}


def _embed_texts_local(texts, model=None):
    """Embed in-process with transformers. CLS pooling, L2-normalised, as bge-m3 expects."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "local embedding backend needs transformers and torch: "
            "pip install -r requirements-embed.txt"
        ) from exc

    model_id = _local_model_id(model)
    if model_id not in _LOCAL_CACHE:
        log.info("[SEMANTIC] loading %s in-process (first call downloads the weights)", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        net = AutoModel.from_pretrained(model_id)
        net.eval()
        _LOCAL_CACHE[model_id] = (tokenizer, net)
    tokenizer, net = _LOCAL_CACHE[model_id]

    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        hidden = net(**batch).last_hidden_state[:, 0]          # CLS token
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
    return hidden.tolist()


def _embed_texts_ollama(texts, model=None, timeout=30):
    """Embed through the ollama daemon.

    Tries the batch ``/api/embed`` endpoint first, falling back to the singular
    ``/api/embeddings`` (one call per text) for older ollama builds.
    """
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


def embed_texts(texts, model=None, timeout=30):
    """Embed a list of texts with whichever runtime is configured and available."""
    texts = list(texts)
    if not texts:
        return []
    backend = embed_backend()
    if backend == "local":
        return _embed_texts_local(texts, model=model)
    if backend == "ollama":
        return _embed_texts_ollama(texts, model=model, timeout=timeout)
    try:
        return _embed_texts_ollama(texts, model=model, timeout=timeout)
    except Exception as exc:
        log.info("[SEMANTIC] ollama unavailable (%s); trying the in-process backend", exc)
        return _embed_texts_local(texts, model=model)


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

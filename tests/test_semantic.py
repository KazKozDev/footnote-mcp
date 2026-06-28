from __future__ import annotations

import pytest

import semantic


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


# ── embed_texts ──

def test_embed_texts_uses_batch_endpoint(monkeypatch):
    calls = {"batch": 0}

    def fake_post(url, json=None, timeout=30):
        if url.endswith("/api/embed"):
            calls["batch"] += 1
            return FakeResp(200, {"embeddings": [[1.0, 0.0]] * len(json["input"])})
        pytest.fail("should not hit singular endpoint")

    monkeypatch.setattr(semantic.http, "post", fake_post)
    out = semantic.embed_texts(["a", "b", "c"])
    assert len(out) == 3
    assert calls["batch"] == 1


def test_embed_texts_falls_back_to_singular(monkeypatch):
    def fake_post(url, json=None, timeout=30):
        if url.endswith("/api/embed"):
            return FakeResp(500, {})  # batch unsupported
        return FakeResp(200, {"embedding": [0.5, 0.5]})

    monkeypatch.setattr(semantic.http, "post", fake_post)
    out = semantic.embed_texts(["a", "b"])
    assert out == [[0.5, 0.5], [0.5, 0.5]]


def test_embed_texts_empty():
    assert semantic.embed_texts([]) == []


# ── cosine ──

def test_cosine_basic():
    assert semantic._cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert semantic._cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert semantic._cosine([1, 0], [0, 0]) == 0.0


# ── semantic_rerank ──

def test_semantic_rerank_orders_by_similarity(monkeypatch):
    results = [
        {"title": "banana bread", "snippet": "dessert"},
        {"title": "capital of France", "snippet": "Paris"},
    ]
    # query vec [1,0]; doc0 orthogonal, doc1 aligned → doc1 should win
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    monkeypatch.setattr(semantic, "embed_texts", lambda texts, model=None, timeout=30: vectors)

    ranked = semantic.semantic_rerank("capital of France", results)
    assert ranked[0]["title"] == "capital of France"
    assert ranked[0]["semantic_score"] > ranked[1]["semantic_score"]


def test_semantic_rerank_graceful_on_failure(monkeypatch):
    results = [{"title": "a", "snippet": ""}, {"title": "b", "snippet": ""}]

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(semantic, "embed_texts", boom)
    out = semantic.semantic_rerank("q", results)
    assert out == results  # unchanged order, no crash


def test_semantic_rerank_empty():
    assert semantic.semantic_rerank("q", []) == []

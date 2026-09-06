from __future__ import annotations

import importlib.util
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "run_search_benchmarks", ROOT / "benchmarks" / "run_search_benchmarks.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_decrypt_string_round_trip():
    plaintext = "private benchmark text"
    password = "canary"
    digest = __import__("hashlib").sha256(password.encode()).digest()
    raw = plaintext.encode()
    key = digest * (len(raw) // len(digest)) + digest[: len(raw) % len(digest)]
    encrypted = bytes(a ^ b for a, b in zip(raw, key))
    encoded = __import__("base64").b64encode(encrypted).decode()
    assert runner.decrypt_string(encoded, password) == plaintext


def test_source_merge_deduplicates_across_queries():
    groups = [
        [{"title": "A", "url": "https://example.com/a?x=1", "snippet": "s", "score": 1.0, "engines": ["ddg"]}],
        [{"title": "A2", "url": "https://example.com/a#top", "snippet": "long snippet", "score": 0.5, "engines": ["brave"]}],
    ]
    out = runner.merge_query_results(groups)
    assert len(out) == 1
    assert out[0]["query_hits"] == 2
    assert out[0]["engines"] == ["brave", "ddg"]
    assert out[0]["snippet"] == "long snippet"


def test_retrieval_metrics():
    ranked = ["a", "x", "b"]
    relevant = {"a", "b"}
    assert runner._recall(ranked, relevant, 1) == 0.5
    assert runner._recall(ranked, relevant, 3) == 1.0
    assert 0 < runner._ndcg(ranked, relevant, 3) < 1


def test_query_focused_document_can_select_late_evidence():
    text = "irrelevant filler " * 200 + "\n\nThe target zephyr protocol was created in 2026."
    focused = runner.query_focused_document("zephyr protocol creation", text, max_chunks=1)
    assert "zephyr protocol" in focused


def test_deepsearch_sampling_keeps_answer_mix():
    rows = ([{"answer_type": "Set Answer", "id": str(i)} for i in range(70)]
            + [{"answer_type": "Single Answer", "id": str(i + 70)} for i in range(30)])
    out = runner.stratified_deepsearch_sample(rows, 20, seed=42)
    assert len(out) == 20
    assert sum(row["answer_type"] == "Set Answer" for row in out) == 13


def test_ollama_json_retries_invalid_model_output(monkeypatch):
    responses = iter(["not json", '{"queries": ["recovered"]}'])
    calls = []

    class FakeClient:
        def __init__(self, timeout):
            assert 0 < timeout <= 60

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=next(responses)))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    result = runner._ollama_json("model", [{"role": "user", "content": "return json"}], retries=2)

    assert result == {"queries": ["recovered"]}
    assert len(calls) == 2
    assert all(call["think"] is False for call in calls)
    assert calls[1]["messages"][-1]["content"].startswith("The previous response was invalid")


def test_ollama_json_recovers_fenced_object_without_retry(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, timeout):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content='```json\n{"items": []}\n```'))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    assert runner._ollama_json("model", [{"role": "user", "content": "json"}]) == {"items": []}
    assert len(calls) == 1


def test_judge_rejects_insufficient_evidence_without_model_call(monkeypatch):
    monkeypatch.setattr(runner, "_ollama_json", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model must not run")))
    grade = runner.judge_answer(
        "Question", "Reference answer", "INSUFFICIENT_EVIDENCE", "Set Answer", "model"
    )
    assert grade == {
        "correct": False,
        "precision": 0.0,
        "recall": 0.0,
        "method": "deterministic_insufficient",
    }


@pytest.mark.skipif(
    not hasattr(signal, "setitimer"),
    reason="_run_with_hard_timeout documents itself as Unix-only: without "
           "signal.setitimer it runs the task unguarded, which is what Windows gets",
)
def test_hard_task_timeout_interrupts_wall_clock_work():
    started = time.monotonic()
    try:
        runner._run_with_hard_timeout(0.05, time.sleep, 1.0)
    except runner.HardTaskTimeout:
        pass
    else:
        raise AssertionError("hard timeout did not interrupt task")
    assert time.monotonic() - started < 0.3

"""Claim-vs-source entailment (heuristic, local NLI, ollama, OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import re
from urllib.request import Request, urlopen



def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : idx + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
    return {}


def _fold_plural(token: str) -> str:
    """Count a bare plural as its singular.

    A table names its unit in the column header — "Revenue (USD millions)" —
    while the claim about one row says "million". That is the same word, and
    scoring it as a miss cost a correct numeric claim its verdict. Deliberately
    only a trailing "s": this is a matcher, not a stemmer, and folding harder
    would start merging words that differ.
    """
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token[-2].isdigit() else token


def _tokens(text: str) -> set:
    return {_fold_plural(token.lower()) for token in re.findall(r"[A-Za-z0-9]{3,}", text)}


def _heuristic_entailment(claim: str, source_excerpt: str) -> dict:
    claim_tokens = _tokens(claim)
    source_tokens = _tokens(source_excerpt)
    if not claim_tokens:
        return {"status": "unsupported", "score": 0.0, "reason": "empty claim", "backend": "heuristic"}
    overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", claim))
    source_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", source_excerpt))
    missing_numbers = [number for number in numbers if number not in source_numbers]
    claim_dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", claim))
    source_dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", source_excerpt))
    if claim_dates and claim_dates & source_dates and missing_numbers and source_numbers:
        return {
            "status": "contradicted",
            "score": round(overlap, 3),
            "reason": f"same dated evidence contains different numbers: {missing_numbers}",
            "backend": "heuristic",
        }
    if missing_numbers:
        return {
            "status": "unsupported",
            "score": round(overlap, 3),
            "reason": f"numbers missing from source: {missing_numbers}",
            "backend": "heuristic",
        }
    if overlap >= 0.75:
        status = "supported"
    elif overlap >= 0.45:
        status = "partially_supported"
    else:
        status = "unsupported"
    return {"status": status, "score": round(overlap, 3), "reason": "token overlap heuristic", "backend": "heuristic"}


def _local_nli_entailment(claim: str, source_excerpt: str, model: str | None = None) -> dict:
    model = model or os.getenv("FOOTNOTE_NLI_MODEL") or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    try:
        from transformers import pipeline
    except ImportError:
        return {"error": "transformers is not installed; install requirements-nli.txt", "backend": "local_nli", "model": model}

    try:
        classifier = pipeline("text-classification", model=model, top_k=None)
        outputs = classifier({"text": source_excerpt[:4000], "text_pair": claim[:1000]})
    except Exception as exc:
        return {"error": f"local NLI failed: {exc}", "backend": "local_nli", "model": model}

    rows = outputs[0] if outputs and isinstance(outputs[0], list) else outputs
    scores = {}
    for row in rows or []:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        if "entail" in label:
            scores["entailment"] = max(scores.get("entailment", 0.0), score)
        elif "contrad" in label:
            scores["contradiction"] = max(scores.get("contradiction", 0.0), score)
        elif "neutral" in label:
            scores["neutral"] = max(scores.get("neutral", 0.0), score)
    entailment = scores.get("entailment", 0.0)
    contradiction = scores.get("contradiction", 0.0)
    neutral = scores.get("neutral", 0.0)
    if contradiction >= 0.6 and contradiction > entailment:
        status = "contradicted"
        score = contradiction
    elif entailment >= 0.7:
        status = "supported"
        score = entailment
    elif entailment >= 0.35 and entailment >= neutral:
        status = "partially_supported"
        score = entailment
    else:
        status = "unsupported"
        score = max(neutral, 1.0 - entailment)
    return {
        "status": status,
        "score": round(score, 3),
        "reason": f"local NLI scores: {scores}",
        "backend": "local_nli",
        "model": model,
    }


_JUDGE_SYSTEM_PROMPT = """You are a strict evidence entailment judge.
Use only the source excerpt.
Return JSON only with:
{"status":"supported|partially_supported|unsupported|contradicted","score":0.0-1.0,"reason":"short reason"}
Definitions:
- supported: the source directly entails the whole claim.
- partially_supported: the source supports part of the claim but leaves a material part unstated.
- unsupported: the source does not provide enough evidence for the claim.
- contradicted: the source states facts that conflict with the claim.
Do not use outside knowledge."""

_VALID_STATUSES = {"supported", "partially_supported", "unsupported", "contradicted"}


def _judge_messages(claim: str, source_excerpt: str) -> list:
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"CLAIM:\n{claim[:2000]}\n\nSOURCE_EXCERPT:\n{source_excerpt[:6000]}\n\nJudge entailment."},
    ]


def _parse_judge_reply(content: str, backend: str, model: str) -> dict:
    parsed = _extract_json_object(content)
    status = str(parsed.get("status", "")).lower()
    if status not in _VALID_STATUSES:
        return {"error": f"{backend} judge returned invalid status", "raw": content[:1000],
                "backend": backend, "model": model}
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "status": status,
        "score": max(0.0, min(1.0, score)),
        "reason": str(parsed.get("reason", ""))[:500],
        "backend": backend,
        "model": model,
    }


def _ollama_entailment(claim: str, source_excerpt: str, model: str | None = None, timeout: int = 25) -> dict:
    model = model or os.getenv("FOOTNOTE_ENTAILMENT_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"
    endpoint = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": _judge_messages(claim, source_excerpt),
        "stream": False,
        "options": {"temperature": 0},
    }
    req = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = (payload.get("message") or {}).get("content", "")
    return _parse_judge_reply(content, "ollama", model)


def _openai_entailment(claim: str, source_excerpt: str, model: str | None = None, timeout: int = 120) -> dict:
    """Judge through any OpenAI-compatible /v1/chat/completions server.

    Covers the servers that speak that dialect and not Ollama's /api/chat —
    llama.cpp, vLLM, LM Studio, a local router, or a hosted provider. The base
    URL points at the /v1 root; the API key is optional, since a local server
    usually wants none.
    """
    model = model or os.getenv("FOOTNOTE_ENTAILMENT_MODEL") or os.getenv("OPENAI_MODEL") or "auto"
    base = (os.getenv("FOOTNOTE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            or "http://127.0.0.1:8080/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    key = os.getenv("FOOTNOTE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": model,
        "messages": _judge_messages(claim, source_excerpt),
        "temperature": 0,
        "stream": False,
    }
    req = Request(f"{base}/chat/completions", data=json.dumps(body).encode("utf-8"),
                  headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return _parse_judge_reply(content, "openai", model)


def _supporting_spans(claim: str, source_excerpt: str) -> list:
    """Best-effort sentence spans for a caller that has to judge for itself."""
    try:
        from ..tools_extra import locate_claim_span
    except Exception:  # pragma: no cover - import guard only
        return []
    try:
        return locate_claim_span(claim, source_excerpt, max_spans=3).get("spans", [])
    except Exception:
        return []


# A judge that could not be reached has said nothing about the claim. Reporting
# that as "unsupported" — which the error paths used to do, with a score of 0.0 —
# turns a network blip into a verdict against the evidence, in the one tool whose
# whole job is telling "the source does not support this" apart from "we could
# not check". The deterministic result stands in, flagged for review, and the
# transport failure is named.
def _unreachable_judge(heuristic: dict, backend: str, claim: str, source_excerpt: str, error: str) -> dict:
    return {
        **heuristic,
        "needs_review": True,
        "judge_error": error,
        "review_reason": (
            f"the {backend} judge could not be reached, so this is the deterministic "
            "result only; read the quoted spans or retry"
        ),
        "requested_backend": backend,
        "spans": _supporting_spans(claim, source_excerpt),
    }


_LLM_BACKENDS = {"ollama", "local_nli", "openai"}


def evidence_entailment(claim: str, source_excerpt: str, backend: str = "auto", model: str | None = None) -> dict:
    backend = (backend or "auto").lower()
    heuristic = _heuristic_entailment(claim, source_excerpt)
    if backend == "heuristic":
        return heuristic
    if backend not in _LLM_BACKENDS | {"auto"}:
        return {"status": "unsupported", "score": 0.0, "reason": f"unknown backend: {backend}", "backend": backend}
    if backend == "auto":
        if heuristic["status"] in {"supported", "contradicted"} and heuristic["score"] >= 0.75:
            return heuristic
        # Uncertain. The old behaviour escalated to a local 7B judge here, which
        # handed the hardest calls to the weakest participant: an MCP client
        # already holds both the claim and the excerpt, and is better placed to
        # read them than qwen2.5 is. So say plainly that this one needs a human
        # or the caller's own judgement, and hand over the spans to read.
        return {
            **heuristic,
            "needs_review": True,
            "review_reason": (
                "the deterministic check is not confident; the caller should read the quoted "
                "spans and decide, or re-run with an explicit backend"
            ),
            "spans": _supporting_spans(claim, source_excerpt),
            "explicit_backends": sorted(_LLM_BACKENDS),
        }

    judges = {
        "local_nli": _local_nli_entailment,
        "ollama": _ollama_entailment,
        "openai": _openai_entailment,
    }
    try:
        judged = judges[backend](claim=claim, source_excerpt=source_excerpt, model=model)
    except Exception as exc:
        return _unreachable_judge(heuristic, backend, claim, source_excerpt, f"{type(exc).__name__}: {exc}")
    if judged.get("error"):
        return _unreachable_judge(heuristic, backend, claim, source_excerpt, judged["error"])
    judged["heuristic_precheck"] = heuristic
    return judged

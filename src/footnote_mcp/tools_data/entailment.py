"""Claim-vs-source entailment (heuristic, local NLI, ollama)."""

from __future__ import annotations

import csv
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import ollama


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


def _heuristic_entailment(claim: str, source_excerpt: str) -> dict:
    claim_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", claim)}
    source_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", source_excerpt)}
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


def _ollama_entailment(claim: str, source_excerpt: str, model: str | None = None, timeout: int = 25) -> dict:
    model = model or os.getenv("FOOTNOTE_ENTAILMENT_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"
    endpoint = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    system = """You are a strict evidence entailment judge.
Use only the source excerpt.
Return JSON only with:
{"status":"supported|partially_supported|unsupported|contradicted","score":0.0-1.0,"reason":"short reason"}
Definitions:
- supported: the source directly entails the whole claim.
- partially_supported: the source supports part of the claim but leaves a material part unstated.
- unsupported: the source does not provide enough evidence for the claim.
- contradicted: the source states facts that conflict with the claim.
Do not use outside knowledge."""
    user = f"CLAIM:\n{claim[:2000]}\n\nSOURCE_EXCERPT:\n{source_excerpt[:6000]}\n\nJudge entailment."
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    req = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = (payload.get("message") or {}).get("content", "")
    parsed = _extract_json_object(content)
    status = str(parsed.get("status", "")).lower()
    if status not in {"supported", "partially_supported", "unsupported", "contradicted"}:
        return {"error": "Ollama judge returned invalid status", "raw": content[:1000], "backend": "ollama", "model": model}
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "status": status,
        "score": max(0.0, min(1.0, score)),
        "reason": str(parsed.get("reason", ""))[:500],
        "backend": "ollama",
        "model": model,
    }


def evidence_entailment(claim: str, source_excerpt: str, backend: str = "auto", model: str | None = None) -> dict:
    backend = (backend or "auto").lower()
    heuristic = _heuristic_entailment(claim, source_excerpt)
    if backend == "heuristic":
        return heuristic
    if backend not in {"auto", "ollama", "local_nli"}:
        return {"status": "unsupported", "score": 0.0, "reason": f"unknown backend: {backend}", "backend": backend}
    if backend == "auto" and heuristic["status"] in {"supported", "contradicted"} and heuristic["score"] >= 0.75:
        return heuristic
    if backend == "local_nli":
        judged = _local_nli_entailment(claim=claim, source_excerpt=source_excerpt, model=model)
        if judged.get("error"):
            return {"status": "unsupported", "score": 0.0, "reason": judged["error"], "backend": "local_nli", "fallback": heuristic}
        judged["heuristic_precheck"] = heuristic
        return judged
    try:
        judged = _ollama_entailment(claim=claim, source_excerpt=source_excerpt, model=model)
        if judged.get("error"):
            if backend == "ollama":
                return {"status": "unsupported", "score": 0.0, "reason": judged["error"], "backend": "ollama", "fallback": heuristic}
            return {**heuristic, "fallback_reason": judged["error"]}
        judged["heuristic_precheck"] = heuristic
        return judged
    except Exception as exc:
        if backend == "ollama":
            return {"status": "unsupported", "score": 0.0, "reason": f"Ollama entailment failed: {exc}", "backend": "ollama", "fallback": heuristic}
        return {**heuristic, "fallback_reason": f"Ollama entailment unavailable: {exc}"}

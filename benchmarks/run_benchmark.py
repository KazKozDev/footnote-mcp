#!/usr/bin/env python3
"""Trustworthiness benchmark + demo for the footnote verification stack.

Measures how well the server distinguishes claims that are *supported* by their
source from claims that are not (unsupported / contradicted). Also demos the
corroboration and span-provenance tools end to end.

Usage:
    python benchmarks/run_benchmark.py                 # heuristic (offline, deterministic)
    python benchmarks/run_benchmark.py --backend ollama  # LLM judge (needs ollama)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools_data import evidence_entailment  # noqa: E402
from tools_extra import corroborate_claim, locate_claim_span  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "entailment_cases.json"


def load_cases(path=CASES_PATH):
    return json.loads(Path(path).read_text())["cases"]


def _binary(status: str) -> str:
    # Trust decision: only an explicit "supported" verdict clears a claim.
    return "supported" if status == "supported" else "not_supported"


def evaluate(cases, backend="heuristic"):
    per_case = []
    for case in cases:
        verdict = evidence_entailment(case["claim"], case["source"], backend=backend)
        predicted = _binary(verdict.get("status", "unsupported"))
        per_case.append({
            "id": case["id"],
            "category": case["category"],
            "label": case["label"],
            "predicted": predicted,
            "status": verdict.get("status"),
            "correct": predicted == case["label"],
        })

    def metrics(rows):
        if not rows:
            return {"n": 0, "accuracy": 0.0, "catch_rate": 0.0, "precision_supported": 0.0}
        n = len(rows)
        correct = sum(r["correct"] for r in rows)
        not_sup = [r for r in rows if r["label"] == "not_supported"]
        caught = sum(1 for r in not_sup if r["predicted"] == "not_supported")
        pred_sup = [r for r in rows if r["predicted"] == "supported"]
        true_sup = sum(1 for r in pred_sup if r["label"] == "supported")
        return {
            "n": n,
            "accuracy": round(correct / n, 3),
            "catch_rate": round(caught / len(not_sup), 3) if not_sup else None,
            "precision_supported": round(true_sup / len(pred_sup), 3) if pred_sup else None,
        }

    categories = sorted({c["category"] for c in cases})
    data_domain = [r for r in per_case if r["category"] in ("numeric", "factual")]
    return {
        "backend": backend,
        "overall": metrics(per_case),
        "data_domain": metrics(data_domain),
        "by_category": {cat: metrics([r for r in per_case if r["category"] == cat]) for cat in categories},
        "per_case": per_case,
    }


def _demo():
    print("\n── Demo: corroborate_claim (triangulation across sources) ──")
    result = corroborate_claim(
        "The capital of France is Paris.",
        [
            {"source_url": "https://a.gov/p", "text": "Paris is the capital of France."},
            {"source_url": "https://b.org/p", "text": "France's capital city is Paris."},
        ],
        backend="heuristic",
    )
    print(f"  verdict={result['verdict']}  supporting={result['supporting']}  "
          f"independent_domains={result['independent_supporting_domains']}")

    print("\n── Demo: locate_claim_span (span-level provenance) ──")
    span = locate_claim_span(
        "capital of France is Paris",
        "The sky is blue. The capital of France is Paris. Water boils at 100 C.",
    )
    top = span["spans"][0]
    print(f"  best_score={span['best_score']}  offsets=({top['start']},{top['end']})  text={top['text']!r}")


def _print_report(report):
    b = report["backend"]
    dd = report["data_domain"]
    ov = report["overall"]
    print(f"\n=== Trustworthiness benchmark (backend={b}) ===")
    print(f"Data-domain claims (numeric+factual): n={dd['n']}  accuracy={dd['accuracy']:.0%}  "
          f"catch_rate={_pct(dd['catch_rate'])}  precision_supported={_pct(dd['precision_supported'])}")
    print(f"Overall (incl. semantic):             n={ov['n']}  accuracy={ov['accuracy']:.0%}  "
          f"catch_rate={_pct(ov['catch_rate'])}  precision_supported={_pct(ov['precision_supported'])}")
    print("\nBy category:")
    for cat, m in report["by_category"].items():
        print(f"  {cat:9} n={m['n']}  accuracy={m['accuracy']:.0%}  catch_rate={_pct(m['catch_rate'])}")
    misses = [r for r in report["per_case"] if not r["correct"]]
    if misses:
        print("\nMisses:")
        for r in misses:
            print(f"  [{r['category']}] {r['id']}: predicted {r['predicted']} (status={r['status']}), expected {r['label']}")


def _pct(value):
    return "n/a" if value is None else f"{value:.0%}"


def main():
    parser = argparse.ArgumentParser(description="footnote trustworthiness benchmark")
    parser.add_argument("--backend", default="heuristic", help="heuristic | ollama | local_nli | auto")
    parser.add_argument("--write", default="", help="Optional path to write a markdown report")
    args = parser.parse_args()

    cases = load_cases()
    report = evaluate(cases, backend=args.backend)
    _print_report(report)
    _demo()

    if args.write:
        Path(args.write).write_text(_markdown(report))
        print(f"\nWrote {args.write}")


def _markdown(report):
    dd, ov = report["data_domain"], report["overall"]
    lines = [
        f"# Trustworthiness benchmark (backend={report['backend']})",
        "",
        "| Set | n | Accuracy | Unsupported-claim catch rate | Precision on 'supported' |",
        "|-----|---|----------|------------------------------|--------------------------|",
        f"| Data domain (numeric+factual) | {dd['n']} | {dd['accuracy']:.0%} | {_pct(dd['catch_rate'])} | {_pct(dd['precision_supported'])} |",
        f"| Overall (incl. semantic) | {ov['n']} | {ov['accuracy']:.0%} | {_pct(ov['catch_rate'])} | {_pct(ov['precision_supported'])} |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

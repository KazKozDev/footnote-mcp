from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_benchmark", _ROOT / "benchmarks" / "run_benchmark.py")
run_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_benchmark)


def test_dataset_is_balanced_and_well_formed():
    cases = run_benchmark.load_cases()
    assert len(cases) >= 15
    labels = {c["label"] for c in cases}
    assert labels == {"supported", "not_supported"}
    for c in cases:
        assert c["claim"] and c["source"] and c["category"]


def test_heuristic_is_perfect_on_data_domain():
    # The server's purpose is source-grounded data claims (numeric/factual). The
    # offline heuristic verifier must be flawless there: never bless an unsupported
    # claim, always catch one. This guards the headline trust number.
    report = run_benchmark.evaluate(run_benchmark.load_cases(), backend="heuristic")
    dd = report["data_domain"]
    assert dd["accuracy"] == 1.0
    assert dd["catch_rate"] == 1.0
    assert dd["precision_supported"] == 1.0


def test_heuristic_overall_is_strong():
    report = run_benchmark.evaluate(run_benchmark.load_cases(), backend="heuristic")
    assert report["overall"]["accuracy"] >= 0.8

#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

OUTPUT_DIR="benchmarks/results/pilot-provenance-v3-task704-post-unit-2026-08-05"
mkdir -p "$OUTPUT_DIR"
.venv/bin/python benchmarks/run_search_benchmarks.py \
  --suites deepsearchqa --deepsearchqa-ids 704 \
  --model qwen3.5:cloud --judge-model qwen3.5:cloud \
  --provider auto+marginalia --json-retries 2 --model-timeout 60 \
  --research-timeout 300 --task-timeout 360 \
  --max-iterations 4 --initial-fetch 8 --fetch-growth 4 --max-fetch 28 \
  --initial-chunks 8 --chunk-growth 4 --max-chunks 24 \
  --max-structured-files 6 --max-structured-rows 120 \
  --max-extraction-context-chars 16000 --stall-limit 2 \
  --entailment-backend heuristic --output-dir "$OUTPUT_DIR" \
  2>&1 | tee -a "$OUTPUT_DIR/run.log"

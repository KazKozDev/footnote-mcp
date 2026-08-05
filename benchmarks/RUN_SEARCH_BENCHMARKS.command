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

mkdir -p benchmarks/results/pilot-2026-08-05
export FOOTNOTE_BRAVE_COOLDOWN_SECONDS=0
export FOOTNOTE_DDG_COOLDOWN_SECONDS=0
.venv/bin/python benchmarks/run_search_benchmarks.py \
  --browsecomp-limit 100 \
  --deepsearchqa-limit 150 \
  --livebrowsecomp-limit 50 \
  --model qwen3.5:cloud \
  --judge-model qwen3.5:cloud \
  --provider auto \
  --json-retries 0 \
  --output-dir benchmarks/results/pilot-2026-08-05 \
  2>&1 | tee -a benchmarks/results/pilot-2026-08-05/run.log

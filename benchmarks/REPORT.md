# Trustworthiness benchmark

Does a claim actually follow from its source excerpt? 18 cases, binary decision:
only an explicit `supported` verdict clears a claim. Categories: `numeric` and
`factual` are the server's data domain; `semantic` are meaning-only cases —
negation and paraphrase — that token overlap cannot reach by construction.

## By backend

| Backend | Set | n | Accuracy | Unsupported-claim catch rate | Precision on 'supported' |
|---------|-----|---|----------|------------------------------|--------------------------|
| heuristic | Data domain (numeric+factual) | 15 | 100% | 100% | 100% |
| heuristic | Overall (incl. semantic) | 18 | 83% | 78% | 80% |
| ollama, qwen3.5:9b | Data domain (numeric+factual) | 15 | 73% | 100% | 100% |
| ollama, qwen3.5:9b | Overall (incl. semantic) | 18 | 78% | 100% | 100% |
| openai-compatible router | Data domain (numeric+factual) | 15 | 93% | 100% | 100% |
| openai-compatible router | Overall (incl. semantic) | 18 | 94% | 100% | 100% |

By category:

| Backend | numeric (10) | factual (5) | semantic (3) |
|---------|--------------|-------------|--------------|
| heuristic | 100% | 100% | 0% |
| ollama, qwen3.5:9b | 70% | 80% | 100% |
| openai-compatible router | 100% | 80% | 100% |

## Reading it

The deterministic backend is exact on the data domain and blind to meaning: it
misses all three semantic cases, which is what `needs_review` exists for. A
local 9B judge inverts that trade — it solves every semantic case but marks
four true data-domain claims unsupported, scoring below the heuristic on the
domain this server is built for. A stronger judge keeps both halves.

Every backend reaches 100% catch rate except the heuristic, and no backend
scores below 100% precision on `supported` except the heuristic. Nothing here
lets a false claim through; the differences are all in false negatives.

The remaining miss for the stronger judges is `fact-everest`, where the source
says "tallest measured above sea level" and the claim drops the qualifier. The
judge answers `partially_supported` with that reason. The label is arguably the
thing that is wrong.

## Reproducing

```bash
python benchmarks/run_benchmark.py                      # heuristic, offline, deterministic
FOOTNOTE_ENTAILMENT_MODEL=qwen3.5:9b-mlx \
  python benchmarks/run_benchmark.py --backend ollama   # local judge via Ollama
FOOTNOTE_OPENAI_BASE_URL=http://127.0.0.1:8080/v1 FOOTNOTE_ENTAILMENT_MODEL=auto \
  python benchmarks/run_benchmark.py --backend openai   # any /v1/chat/completions server
```

`--write` regenerates a single-backend version of this file; the table above was
assembled from the three runs named here. Judge rows depend on the model behind
the endpoint and are not deterministic the way the heuristic row is.

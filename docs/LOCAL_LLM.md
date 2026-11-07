# Running CQA on an open-weight model (vLLM / Ollama / any OpenAI-compatible server)

CQA already speaks the OpenAI chat-completions protocol (`cqa/llm.py`, `CQA_PROVIDER=openai`): native
tool calling first, JSON-mode fallback on HTTP 400. Nothing else changes — same checklists, prompts,
evidence validation and scoring as the cloud path.

## 1. Deploy on RunPod (vLLM)

```bash
# needs RUNPOD_API_KEY in env or ~/.hermes/.env; generates CQA_LOCAL_LLM_TOKEN there on first run
python scripts/runpod_vllm.py up        # Qwen/Qwen3-32B-AWQ, A6000 → A40 → L40S, community then secure
python scripts/runpod_vllm.py wait      # polls https://<podId>-8000.proxy.runpod.net/v1/models (~3 min)
python scripts/runpod_vllm.py status
python scripts/runpod_vllm.py down      # TERMINATES the pod (no storage left behind)
```

`up` starts `vllm/vllm-openai:latest` with
`--model Qwen/Qwen3-32B-AWQ --served-model-name qwen3-32b-awq --max-model-len 32768 --gpu-memory-utilization 0.92
--api-key $CQA_LOCAL_LLM_TOKEN --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching
--default-chat-template-kwargs '{"enable_thinking": false}'` and spawns a detached watchdog that
terminates the pod after `--max-hours` (default 1.5) no matter what. Options: `--model`, `--served-name`,
`--gpu "NVIDIA A40"`, `--max-len`, `--no-no-think` (keep Qwen3 thinking), `--extra ...`.

Point CQA at it (single model, no jury):

```bash
export CQA_PROVIDER=openai CQA_ENSEMBLE=0 CQA_MODEL=qwen3-32b-awq \
       CQA_OPENAI_BASE_URL=https://<podId>-8000.proxy.runpod.net/v1 CQA_OPENAI_API_KEY=$CQA_LOCAL_LLM_TOKEN
python -m cqa evaluate --input data/DEV.json --concurrency 16
```

Any other OpenAI-compatible server works the same way, e.g. **Ollama** (default base URL already
`http://127.0.0.1:11434/v1`): `ollama pull qwen3:32b && CQA_PROVIDER=openai CQA_ENSEMBLE=0 CQA_MODEL=qwen3:32b python -m cqa gold`.
If the server lacks tool calling, set `CQA_OPENAI_TOOLS=0` (JSON mode). A local jury is possible with
`CQA_OPENAI_JURY=1 CQA_JURORS=... CQA_JUDGE=...`, not benchmarked here.

## 2. Benchmark (2026-11-07)

Setup: `Qwen/Qwen3-32B-AWQ` (4-bit AWQ, thinking disabled), vLLM latest, 1× **A40 48 GB, RunPod secure
cloud, $0.59/h** (A6000 and community A40 were unavailable), max-model-len 32k, CQA concurrency 16,
single model. Reproduce: `python scripts/bench_models.py run --base-url ... --model qwen3-32b-awq --tag X --price-hr 0.59`.

### Gold sets (vs hand labels)

| set | model | overall agr. | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | total MAE | gross P/R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gold (16) | Qwen3-32B-AWQ | **0.883** | 1.00 | 0.56 | 0.94 | 0.75 | 1.00 | 0.88 | 0.94 | 1.00 | 6.86 | 1.0 / 1.0 (3 TP) |
| gold (16) | cloud jury (haiku+sonnet, opus judge)* | 0.992 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 1.00 | 0.43 | 1.0 / 1.0 |
| holdout (10) | Qwen3-32B-AWQ | **0.887** | 1.00 | 0.70 | 0.90 | 0.70 | 1.00 | 0.80 | 1.00 | 1.00 | 6.38 | 1.0 / 1.0 (1 TP) |

\* from `results/gold_report.json` (earlier run; the gold set was used for tuning, so it is optimistic for the cloud).

### DEV, 65 real dialogs — agreement with the cloud jury (`results/dev_jury_v5.json`)

| metric | value |
|---|---|
| per-criterion exact match | 01 0.97 · 02 0.71 · 03 0.82 · 04 0.66 · 05 0.83 · 06 0.74 · 07 0.77 · 08 0.98 |
| overall exact match | **0.81** |
| total-score MAE vs cloud | **9.67** points |
| gross-error flag agreement | 0.985 (local 4 flags vs cloud 5) |
| incomplete / failed | 0 / 0 |
| wall time | 1043 s → **16.0 s/dialog** at concurrency 16 |
| GPU cost | **$0.0026/dialog** (vs cloud jury ≈ $0.16/dialog: $10.28 for 65, v7) |

Total RunPod spend for the whole experiment (deploy, ~3 min load, gold+holdout+DEV, terminate):
**$0.28** (balance $6.04 → $5.76).

## 3. Verdict & limitations

- Usable as a cheap/offline **first pass** (≈60× cheaper per dialog), not a replacement for the
  cloud jury: ~11 pp lower gold agreement, MAE ≈ 7–10 points; weakest on 02 (need discovery),
  04 (answer correctness) and 06 (tone/scale) — the nuanced criteria. Gross-error detection held up.
- Prompts are tuned for Claude; no per-model tweaking was done (fair comparison), so a tuned prompt or a
  local jury could close part of the gap.
- The checklists/prompts were being edited concurrently during the benchmark; the cloud DEV file (v5)
  and this run may differ slightly in prompt version — DEV numbers are agreement with the cloud, not accuracy.
- Throughput is bounded by one A40 and the RunPod proxy; 16 s/dialog is wall-clock with 16 parallel
  requests. Sampling: temperature 0 via the OpenAI path.
- Always `down` the pod: idle A40 secure costs $0.59/h. The watchdog is a safety net, not a scheduler.

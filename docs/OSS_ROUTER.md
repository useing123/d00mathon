# Open-weight first pass + cascade router (preset `cascade`)

Status: **partial** (code freeze). Measured 2026-11-07/08. Code: `cqa/router.py`, `scripts/bench_oss.py`,
`tests/test_router.py`; results in `results/oss/` and `results/bench_oss_qwen3-8b_*.json`.

## 1. Models checked (HF API, `https://huggingface.co/api/models/<repo>`)

| repo | exists | vLLM quant | access | run? |
|---|---|---|---|---|
| Qwen/Qwen3-8B-AWQ | yes | awq | open | **yes** (A40) |
| Qwen/Qwen3-14B-AWQ / -FP8 | yes | awq / fp8 | open | skipped (time) |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4, Qwen/Qwen3.5-27B-FP8 / -GPTQ-Int4 | yes | gptq / fp8 | open | skipped (time) |
| ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g (ungated mirror; google/gemma-3-* are gated) | yes | compressed-tensors | open | skipped |
| issai/LLama-3.1-KazLLM-1.0-8B, -70B-AWQ4 | yes | – / awq | **gated** (no HF token) | not runnable here |
| Qwen/Qwen3-32B-AWQ | yes | awq | open | earlier run, docs/LOCAL_LLM.md |

GPU: 1× A40 48 GB, RunPod **secure** $0.59/h (community A40 unavailable). vLLM flags (`scripts/runpod_vllm.py up`):
`vllm/vllm-openai:latest --model Qwen/Qwen3-8B-AWQ --served-model-name qwen3-8b-awq --max-model-len 32768
--gpu-memory-utilization 0.92 --enable-auto-tool-choice --tool-call-parser hermes --max-num-seqs 32
--enable-prefix-caching --default-chat-template-kwargs '{"enable_thinking": false}'`.
Next model on the same pod: `python scripts/bench_oss.py swap --model Qwen/Qwen3-14B-AWQ --served qwen3-14b-awq`
(podEditJob; keeps the GPU). `CQA_LOCAL_ADAPTER=<path|hf repo>` adds `--enable-lora --max-lora-rank 64
--lora-modules cqa-adapter=<adapter>` and the router then calls `cqa-adapter` in JSON-only mode (CQA_OPENAI_TOOLS=0).

## 2. Router design (3 tiers)

1. **Tier 1** scores all 8 criteria: an open-weight model on vLLM (`CQA_ROUTER_FIRST=local`) or Jev
   (`CQA_ROUTER_FIRST=jev`, default — see below). Local confidence is logprob-based: a second, tiny
   **decision call** (same prompt, JSON-schema enum with only `value`, `top_logprobs=20`) gives the option
   distribution; confidence = p(value of the full reasoned answer). Reading logprobs inside the full answer is
   useless (after the `reasoning` field p(value) ≈ 1.0 — measured). Alignment check: argmax option == value the
   decision call emitted (32/32 on a probe; a mismatch sets confidence 0 → escalate). The snap decision and the
   reasoned answer can legitimately differ (e.g. 02 reasoned NA, snap YES p=0.94) → `decision_disagree` → escalate.
   The local model also returns evidence quotes; failed validation → escalate.
2. **Tier 2 haiku** when: local error/invalid, evidence failed, p < T1, 04/05/06 with p < HARD_CONF, 08=YES,
   decision disagreement, or (speed mode) local endpoint down / slower than `CQA_ROUTER_LOCAL_TIMEOUT`.
3. **Tier 3 sonnet** decides when Haiku failed, Haiku confidence < HAIKU_LOW, or Haiku ≠ tier 1 and Haiku confidence < T2.

Every escalated verdict carries Haiku/Sonnet evidence. Results: per criterion `routing` {tier, reasons, values,
confidences, option probs, latency, models}; `meta.routing` {tier_share, reasons, cloud_usd_by_model}.
Settings: `CQA_ROUTER_T1/T2/HARD/HARD_CONF/HAIKU_LOW/JUDGE/MODE/LOCAL_TIMEOUT/LOCAL_MODEL`.
Thresholds tuned ONLY on the tuned gold (16) by offline replay of all-tier samples (`bench_oss.py tune`,
objective: max agreement, then min cost proxy tier2 + 15×tier3, 04/05/06 hard set fixed). Held-out run once per config.

## 3. Results

| config | gold (16, tuned) | held-out (10) | DEV agr. w/ jury (65) | $/dialog | s/dialog | tier1 / haiku / sonnet |
|---|---|---|---|---|---|---|
| Qwen3-8B-AWQ alone | 0.781 (MAE 14.7) | 0.775 (MAE 12.5, gross recall 0/1) | not finished¹ | $0.00085 GPU² | 5.2 | 100/0/0 |
| Qwen3-32B-AWQ alone (earlier run) | 0.883 | 0.887 | 0.81 (v5) | $0.0026 | 16 | 100/0/0 |
| cascade Qwen3-8B → haiku → sonnet (T1 .999, HARD_CONF 1.01, T2 .7) | 0.953 (replay) | **0.863** (MAE 13.9) | – | $0.0017 cloud + ~$0.0011 GPU | 7.0 | 34/65/1 % (held-out) |
| Jev alone (Jev agent, 3 runs) | 0.974 | 0.933 | – | $0.00053 | 0.4 | 100/0/0 |
| Jev → haiku (T1 .7, HARD_CONF .9) | 0.984 (replay) | 0.925 | – | $0.0011+$0.0005 | 0.9 | 82/18/0 % |
| **Jev → haiku → sonnet (T1 .7, HARD_CONF .9, T2 .7)** | 0.992 (replay) | **0.963** (MAE 2.46) | **0.948** (MAE 3.71, gross 5/5) | $0.0042 held-out, $0.0146 DEV | 1.3 / 2.8 | 82/15/2 % held-out; 73/19/7 % DEV |
| claude jury (bench_matrix) | 0.992 | 0.95 | reference | $0.0745 held-out, $0.158 DEV | 4.45 | – |
| haiku alone | 0.969 | 0.912 | 0.871 | $0.0044 | 3.1 | – |
| gpt-5.3-codex | 0.977 | 0.975 | – | $0.057 | 3.05 | – |
| gpt-6-luna | 0.984 | 0.944 | 0.885 | $0.0015 | 3.2 | – |

¹ the 8B DEV run hung on 2/520 criteria (degenerate generation → retries); `CQA_OPENAI_MAX_TOKENS` (default 2000)
was added to the OpenAI-compatible path; not re-run before the freeze. ² A40 $0.59/h × wall ÷ dialogs on gold+held-out.
DEV reference = `results/dev_jury_v7.json`.

**Cheapest config within run-to-run noise (±2/80 = ±0.025) of the claude jury (0.95 held-out):** Jev alone
(0.933 pooled, $0.00053/dialog, no evidence quotes). Cheapest *with evidence on every escalated criterion and above
the jury*: **Jev → haiku → sonnet** (0.963 held-out, 0.948 DEV agreement, ~$0.004–0.015/dialog, ≈18× cheaper than the jury on held-out
($0.0745), ≈11× on DEV ($0.158)) — the recommended `cascade` preset default. No local open-weight config tested reaches the band:
Qwen3-8B is too weak (02/04 especially) and its routing escalates 65% of criteria.

## 4. Caveats

- Gold sets are synthetic and labelled by us; held-out is 10 dialogs (one label = 1.25 pp). Single runs.
- DEV "agreement" = agreement with our own claude jury, not ground truth.
- Tuning on 16 dialogs overfits; several thresholds tie. Replay numbers on the tuned gold are optimistic.
- Jev tier-1 criteria that are not escalated have **no evidence quotes** (Jev returns probabilities only).
- Only one open-weight model was run in this sweep (time); 14B / Qwen3.5 / Gemma are scripted but unmeasured.
- bench_matrix.json was not modified (no new rows) to keep /models untouched.

## 5. Deploy

```bash
# Jev first pass (default): needs OPENROUTER key for cqa/jev.py + ANTHROPIC_API_KEY
CQA_PRESET=cascade python -m cqa evaluate --input data/DEV.json
# open-weight first pass
python scripts/runpod_vllm.py up --model Qwen/Qwen3-32B-AWQ --gpu "NVIDIA A40" && python scripts/runpod_vllm.py wait
CQA_PRESET=cascade CQA_ROUTER_FIRST=local CQA_ROUTER_LOCAL_MODEL=qwen3-32b-awq \
  CQA_OPENAI_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 CQA_OPENAI_API_KEY=$CQA_LOCAL_LLM_TOKEN \
  CQA_ROUTER_MODE=speed CQA_ROUTER_LOCAL_TIMEOUT=30 python -m cqa evaluate --input data/DEV.json
python scripts/runpod_vllm.py down   # always
```

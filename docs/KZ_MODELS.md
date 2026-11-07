# Kazakh-capable open models — held-out check (2026-11-08)

Protocol: `results/kz_models/run_local_only.py <served> <url> <tag>` = `scripts/bench_oss.py cascade --suite holdout
--first local --no-logprobs` with escalation disabled (tier 1 only, same prompts/scoring as Qwen3-8B 0.775).
Held-out = 10 dialogs (2 KZ, 5 RU, 3 EN by id). Subsets: `results/kz_models/subsets.py`. vLLM `latest`,
`--max-model-len 16384 --language-model-only`, thinking off. GPU: 1× A40 secure $0.59/h.
$/dialog = GPU $/h × s/dialog / 3600 (concurrency 8).

| model | held-out agr. | MAE | gross P/R | 01 02 03 04 05 06 07 08 | KZ (2) | RU (5) | EN (3) | s/dialog (A40) | $/dialog |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen/Qwen3.8-27B-FP8** | **0.963** | 4.3 | 1.0 / 1.0 | 1 .9 1 .8 1 1 1 1 | 1.000 | 0.925 | 1.000 | 26.6 | $0.0044 |
| issai/Qwen3.5-9B-Kazakh | 0.887 | 8.21 | 1.0 / 1.0 | 1 .7 .8 .7 1 .9 1 1 | 0.938 | 0.850 | 0.917 | 13.4 | $0.0022 |
| ref: Qwen3-8B-AWQ | 0.775 | 12.5 | –/0 | | | | | 5.2 | $0.00085 |
| ref: Jev alone | 0.934 | | | | | | | 0.4 | $0.00053 |
| ref: cascade Jev→haiku→sonnet | 0.963 | 2.46 | | | | | | 1.3 | $0.0042 |
| ref: Claude jury | 0.95 | | | | | | | 4.45 | $0.073 |

ISSAI 9B left 1 criterion unparsed (hold-07 crit 06 → incomplete). n=10, so ±0.025 is noise.

**Verdict — best local model: Qwen/Qwen3.8-27B-FP8** (0.963 held-out, = cascade, > Jev 0.934, gross 1/1), but 26.6 s/dialog
on an A40 (~65× slower than Jev). Usable as tier 1: `CQA_ROUTER_FIRST=local CQA_ROUTER_LOCAL_MODEL=qwen3.8-27b-fp8`
(served name) with `CQA_OPENAI_BASE_URL=<pod>/v1`. Single run, not yet confirmed on tuned gold/DEV.
ISSAI 9B-Kazakh: good on KZ but weaker on RU; candidate LoRA base, not a replacement.

## Not run / skipped (one line each)
- issai/Qwen3.5-4B-Kazakh: run invalid (my swap-wait checked only liveness, requests hit the old model) → no number; rerun needed.
- H100 runs (27B speed, ISSAI 35B-A3B fp8): H100 SXM secure pod `aizsygogxjecjt` ($3.99/h; PCIe unavailable) did not serve within ~11 min (still loading/unknown error, logs not checked) → no H100 numbers; pod STOPPED.
- ai-sage/GigaChat3.1-10B-A1.8B, issai/Qwen3.5-35B-A3B-Kazakh, WAL2-Ternary-27B (needs PrismML llama.cpp fork; card reports weak long reasoning/termination), LitJev, qwen-jev-like, openjev, vLLM guided_choice decision mode, `local-decision`/`cascade-local` presets: **not done** (out of time). No numbers exist for them.
- LumieraLM/Lumiera_v2: no config.json, no architecture, files named "model (1).safetensors", card claims impossible bases (gpt-5-omni, claude-5.0-opus) → untrustworthy, skipped.
- CohereLabs/aya-101: 13B mT5 encoder-decoder, poor vLLM support, not tuned for long structured JSON → skipped.
- MaziyarPanahi/Mistral-7B-Instruct-Aya-101-GGUF: low priority (7B Q4 unlikely > Qwen3-8B 0.775) → not run.
- alphaedge-ai/Qwen3-Embedding-kaz-32768: embedding model (used by another agent as signal) → skipped. salyamq/smq-tokenizer-kz: tokenizer only → skipped.
- Qwen/Qwen3.5-9B official: superseded by the ISSAI Kazakh 9B. AMAImedia NeoHorse1-Heretic: abliterated/uncensored, unacceptable for bank QA. nur-dev/farabi-0.6B-agent-rag and issai/foggen-gemma3-270m: too small for 8-criterion judging. AMAImedia/Ling-3.1-Flash GGUF: 572B MoE, beyond budget.
- ai-forever/SMITH-Exp: retrieval-tool-loop specialist, not a judge. ai-forever Zarya: text-diffusion custom arch, RU/EN only → skipped.

## Pods (STOPPED, not terminated)
- `aizsygogxjecjt` — H100 SXM, secure, $3.99/h running; image `vllm/vllm-openai:latest`,
  `--model Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b-fp8 --max-model-len 16384 --language-model-only` (+ runpod_vllm.py defaults).
  Status EXITED (verified via API). Note: with `volumeInGb 0` the weights are on the container disk, which RunPod may not keep across stop; restart may re-download. Stopped-pod storage bills ~$0.006/h for 60 GB container disk (estimate; check the dashboard).
  Restart: GraphQL `podResume(input:{podId:"aizsygogxjecjt", gpuCount:1})`, then `results/kz_models/waitmodel.sh qwen3.8-27b-fp8`.
  The H100 could not be brought up in time, so whether it actually serves is unverified.
- A40 `xtv6v09yyeyhds`: terminated (before the stop-not-terminate change).
- Balance: $5.16 before → $4.09 after (also includes other agents' pods, e.g. cqa-ft, during the window).

## issai/Qwen3.5-4B-Kazakh, ai-sage/GigaChat3.1-10B-A1.8B-bf16 (pod cqa-a1, A40 $0.59/h)
Same protocol (`run_local_only.py`, tier 1 only: 80/80 criteria at tier 1, no escalation). `/v1/models` checked to return the exact served id before running.

| model | held-out agr. | MAE | gross P/R | 01 02 03 04 05 06 07 08 | KZ (2) | RU (5) | EN (3) | s/dialog (A40) | $/dialog |
|---|---|---|---|---|---|---|---|---|---|
| issai/Qwen3.5-4B-Kazakh | 0.838 | 9.84 | 1.0 / 1.0 | 1 .8 .7 .7 1 1 .5 1 | 0.875 | 0.825 | 0.833 | 7.6 | $0.0012 |
| ai-sage/GigaChat3.1-10B-A1.8B-bf16 | — | | | | | | | | |

- 4B-Kazakh: 0 incomplete, 0 errors. Below the 9B (0.887) and 27B (0.963), above Qwen3-8B-AWQ (0.775); weak on 07 (0.5).
- GigaChat3.1-10B-A1.8B-bf16: **no number**. The pod was switched to it with `podEditJob` (vllm `latest`, `--max-model-len 16384`, then again with `--trust-remote-code`). The container never came up: uptime −1 and proxy 404 for about 20 min across both tries. I couldn't get the logs through the API, so the cause is unknown (possibly DeepseekV3/MLA on an A40 with vLLM `latest`). I stopped to stay within budget.
- Pod `ic49nwbfubw3ev` (cqa-a1): STOPPED, EXITED verified. About 40 min × $0.59/h ≈ $0.39.

## Decision models (Qwen3.8-27B-FP8 tier 1, held-out 10 dialogs, 1× A40 secure $0.59/h, 2026-11-08)

The same question builder (`cqa/jev.py`, endpoint via `CQA_JEV_BASE_URL` / `CQA_JEV_MODEL`) feeds every method.
Client: `scripts/bench_kz.py decision|tune|cascade`. Pod-side servers: `scripts/decision_server.py` (vllmchoice / qjl / openjev); LitJev runs natively
(`litjev --backend sglang`, `/v1/systemone`). Results use the shared schema in `results/kz_models/<method>__<model>.json` (per-dialog data
included; tuned-gold runs are `__gold.json`, raw dialogs are in `raw/`). $/dialog = GPU cost only, sequential (concurrency 1–4).

| method | model | agr. | MAE | gross P/R | KZ | RU | EN | s/dialog | $/dialog | escal. |
|---|---|---|---|---|---|---|---|---|---|---|
| cascade: vllm-choice → haiku (→ sonnet) | Qwen3.8-27B-FP8 | 0.950 | 4.42 | 1/1 | 1.0 | 0.90 | 1.0 | 7.65 | $0.0013 GPU + ~$0.00008 cloud* | 13.8 % |
| vLLM guided_choice decision mode (ours) | Qwen3.8-27B-FP8 | 0.938 | 4.58 | 1/1 | 0.938 | 0.90 | 1.0 | 7.3 | $0.0012 | – |
| qwen-jev-like (sting8k, MIT) | Qwen3.8-27B-FP8 | 0.912 | 9.41 | 1/1 | 1.0 | 0.85 | 0.958 | 2.7 | $0.00045 | – |
| LitJev (zhengxuyu, Apache-2.0; SGLang, FP8) | Qwen3.8-27B-FP8 | 0.863 | 7.16 | 1/1 | 0.938 | 0.825 | 0.875 | 13.5 | $0.0022 | – |
| LitJev | issai/Qwen3.5-9B-Kazakh | 0.812 | 10.34 | –/0 | 0.875 | 0.775 | 0.833 | 3.8 | $0.0006 | – |
| openjev 4B v5 (AlexWortega, MIT) | qwen3.5-4b-nli-v5 | 0.713 | 14.22 | –/0 | 0.812 | 0.55 | 0.917 | 6.4 | $0.0011 | – |
| ref: vLLM generate (full prompt) | Qwen3.8-27B-FP8 | 0.963 | 4.3 | 1/1 | 1.0 | 0.925 | 1.0 | 26.6 | $0.0044 | – |
| ref: Jev 1.14 / Jev cascade | – | 0.934 / 0.963 | | | | | | 0.4 / 1.3 | $0.0005 / $0.0042 | |

Tuned gold (16) agreement: vllm-choice 0.953, qwen-jev-like 0.875, LitJev-27B 0.859, LitJev-9B 0.719, openjev 0.727.
Cascade thresholds came from offline replay on tuned gold only (`bench_kz.py tune`; Haiku/Sonnet samples from `results/oss/jev_all_gold.json`):
T1 0.5, HARD_CONF 0.95 (04/05/06), T2 0 → replay agreement 1.0 on gold (overfit, so expect less). Held-out result: 0.950.
*Cloud $ is low because the Haiku calls hit the disk cache (`cloud_usd` counts only uncached calls). The real Haiku cost at 13.8 % escalation is roughly $0.0006/dialog (estimate).

**Verdicts.** Best decision method: our vLLM guided_choice (0.938, 3.6× faster than full generation, about 1 label lower). The cascade on top reaches
0.950 at 7.7 s/dialog and is the best self-hosted tier 1, but it does not beat Jev→haiku→sonnet (0.963, 1.3 s) or plain Qwen3.8 generation (0.963).
qwen-jev-like is the fastest (2.7 s) and is weak on RU. LitJev is below both (content readout, uncalibrated). openjev (EN NLI cross-encoder) fails on RU and is not usable.
With n=10, differences of ≤0.025 are noise. Caveats: concurrent agents edited `checklists/freedom_chat_v1.yaml` during this window, so later runs
(qwen-jev-like, openjev, LitJev-9B) may have seen slightly different rule text than vllm-choice/LitJev-27B. LitJev's SGLang server fails under
concurrent requests ("event loop is already running") and was run at concurrency 1. Each config was run once.

Presets: `local` (Qwen3.8-27B-FP8 on own vLLM, `CQA_OPENAI_BASE_URL`, `CQA_ROUTER_LOCAL_MODEL`) and `cascade-local` (tier 1 = `decision_server.py
vllmchoice` at `CQA_JEV_BASE_URL`, default `http://127.0.0.1:8101/v1/decisions`, thresholds above).

**Pod for tomorrow:** `hieibio5fqyeuj` (cqa-kz, A40 secure, runpod/pytorch image, SSH key `~/.ssh/cqa_runpod`, host alias `kz` in ~/.ssh/config, so the port changes on resume).
Status STOPPED (EXITED). Weights are on the container disk, which may be wiped on resume. Resume: GraphQL `podResume(input:{podId:"hieibio5fqyeuj", gpuCount:1})`, then
`ssh kz 'bash /root/serve.sh Qwen/Qwen3.8-27B-FP8 qwen3.8-27b-fp8 --language-model-only'` (≈6 min) +
`python decision_server.py vllmchoice --port 8101 --model qwen3.8-27b-fp8 --key $(cat /root/.tok)`, and tunnel `ssh -L 18000:…:8000 -L 18101:…:8101 kz`.
(Pod port 8001 is taken by RunPod nginx.) LitJev: `bash /root/lj.sh <model> <log>` (venv /root/lj-env, sglang 0.5.23).

## Agent a2 run (pod `cqa-a2`, A40 $0.59/h) — 2026-11-08

| model | held-out agr. | MAE | gross P/R | 01 02 03 04 05 06 07 08 | KZ (2) | RU (5) | EN (3) | s/dialog (A40) | $/dialog |
|---|---|---|---|---|---|---|---|---|---|
| issai/Qwen3.5-35B-A3B-Kazakh (vLLM `--quantization fp8`, len 8192) | 0.912 | 6.83 | 1.0 / 1.0 | 1 .7 .9 .8 1 1 .9 1 | 0.938 | 0.850 | 1.000 | 8.52 | $0.0014 |
| armanibadboy/WAL2-Ternary-27B (e022 PQ2_0, llama-server) | **no number** | | | | | | | | |

- ISSAI 35B-A3B: fits an A40 with fp8 weight-only at 8192 ctx, served in ~4 min; 3× faster than Qwen3.8-27B-FP8 but scores lower (0.912 vs 0.963). Same protocol (`run_local_only.py`, tier 1 only, thinking off).
- WAL2: PrismML fork @9a9394a built with CUDA (sm_86) in < 20 min; llama-server served `wal2-ternary-27b` (fingerprint b10709-9a9394a89), 16384 ctx/slot ×4, `--reasoning-budget 0`. Sanity check: ~41 tok/s decode, but answered "Astrakhan" for the capital of Kazakhstan. The held-out run did not finish in 25 min (calls generating up to 2000 tokens), so it was killed at the wrap-up order → no metrics.
- Pods `j1pqx9v9bfnnn2` (WAL2) and `d4acpnmevboih8` (ISSAI), both named cqa-a2: **STOPPED (EXITED, verified)**, not terminated.

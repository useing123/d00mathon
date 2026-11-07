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

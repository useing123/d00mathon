# Self-hosted vs frontier on the organisers' control set (FBroker, 10 dialogs)

Same checklist (`freedom_chat_v1`), same prompts, same scoring code; only the model changes.

| | Claude jury (Haiku + Sonnet, Opus judge) | Qwen3.8-27B-FP8, self-hosted (vLLM, 1x A40) |
|---|---|---|
| where the data goes | external API | our own GPU, nothing leaves the perimeter |
| mean total score | 71.71 | 71.96 |
| gross-error flags | 1 (p2-kz-003) | 1 (p2-kz-003) |
| GPU / API cost for 10 dialogs | $1.33 API | ~$0.05 of A40 time ($0.59/h) |

**Agreement between the two on the control set:** 75 of 80 criterion verdicts identical (0.938);
identical total score in 7 of 10 dialogs; same gross-error flag.

All 5 disagreements are on borderline criteria (07 closing x3, 06 style x1, 03 identification x1) in
3 dialogs (p2-kz-001, p2-kz-002, p2-en-001). For scale: the Claude jury itself flips 7.2 % of verdicts
between identical runs on DEV (docs/EVALUATION.md, run-to-run variance), so the self-hosted model differs
from the frontier jury by about as much as the jury differs from itself.

This is agreement, not accuracy: the organisers' ground truth for FBroker is not public. On our synthetic
held-out set (EXPERIMENTS.md) Qwen3.8-27B scores 0.963 vs 0.950 for the jury (difference not significant).

Submitted file: `results/prod/results.json` (Claude jury). Raw runs: `results/control/`.
Reproduce: `python scripts/runpod_vllm.py up --model Qwen/Qwen3.8-27B-FP8 --served-name qwen3.8-27b-fp8 --gpu "NVIDIA A40" --max-len 16384 --extra=--language-model-only`,
then `CQA_PROVIDER=openai CQA_ENSEMBLE=0 CQA_MODEL=qwen3.8-27b-fp8 CQA_OPENAI_BASE_URL=<pod>/v1 python -m cqa evaluate --group FBroker`.

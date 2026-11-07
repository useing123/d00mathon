#!/usr/bin/env bash
# QLoRA fine-tune on a RunPod GPU pod (executed once 2026-11-08, see docs/FINETUNE.md). See docs/FINETUNE.md.
#
# RunPod template (create in the console → Templates → New):
#   name            cqa-qlora
#   image           runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   GPU             1x RTX 4090 (24 GB, Qwen3-8B / Gemma-3-12B) or 1x A40 (48 GB, also Qwen3-14B)
#   container disk  40 GB; volume 60 GB mounted at /workspace (HF cache + adapters survive pod stop)
#   env             HF_TOKEN (gated models / push), HF_HOME=/workspace/hf, BASE=Qwen/Qwen3-8B
#   ports           none needed (pull the adapter with runpodctl / scp, or --push-to-hub)
# Upload: the repo + data/finetune/{train,val}.jsonl (git-ignored: build locally with scripts/finetune_dataset.py
# and copy it; DEV transcripts are the organizers' data — do not push the dataset to a public hub).
#
# Usage on the pod:  bash scripts/finetune_runpod.sh [extra finetune_lora.py args]
#        locally:    ESTIMATE_ONLY=1 bash scripts/finetune_runpod.sh   # token count + $ per epoch, no GPU
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-Qwen/Qwen3-8B}
OUT=${OUT:-/workspace/runs/cqa-$(basename "$BASE" | tr 'A-Z' 'a-z')-lora}
PY=${PY:-python}

if [[ "${ESTIMATE_ONLY:-0}" == "1" ]]; then
  exec $PY scripts/finetune_lora.py --estimate-only --base "$BASE" "$@"
fi

$PY -m pip install -q "transformers==5.20.2" "peft==0.21.2" "trl==1.17.1" "bitsandbytes==0.50.2" "accelerate==1.15.0" datasets pyyaml httpx  # pinned: measured run 2026-11-08 (torch 2.14)
[[ -f data/finetune/train.jsonl ]] || $PY scripts/finetune_dataset.py
$PY scripts/finetune_lora.py --estimate-only --base "$BASE"
# single GPU; for multi-GPU: accelerate launch --num_processes N (DDP; each rank loads its own 4-bit copy)
accelerate launch --num_processes 1 --mixed_precision bf16 scripts/finetune_lora.py \
  --base "$BASE" --out "$OUT" --select-on eval_agreement "$@"
echo "adapter -> $OUT   (serve: vllm serve $BASE --enable-lora --lora-modules cqa=$OUT --max-lora-rank 16)"

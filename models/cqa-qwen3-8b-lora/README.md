---
base_model: Qwen/Qwen3-8B
library_name: peft
license: apache-2.0
language: [ru, kk, en]
tags: [lora, qlora, call-center, quality-assurance, cqa]
---

# cqa-qwen3-8b-lora

A QLoRA adapter for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). It scores one call-center QA criterion per
request for RU/KZ/EN support dialogs and returns a strict JSON verdict. It is the local "tier 1" model of the CQA
pipeline: https://github.com/useing123/d00mathon

**Prototype. This adapter was trained on 0.32 of an epoch and does not beat the base model on agreement.** See the results below.

## Format

The prompt is exactly the one CQA sends in OpenAI JSON mode (`CQA_OPENAI_TOOLS=0`):

- `system`: the rubric instructions + `Верни ТОЛЬКО JSON-объект по схеме: <schema>`
- `user`: transcript block + criterion block

The answer is compact JSON:
`{"reasoning","value","justification","evidence","missing_where","na_reason","confidence"}`.
`value` is YES/NO/NA or a score, depending on the criterion. Serve the model with thinking disabled.

## Training data

- 798 train rows / 82 val rows (one row = one dialog × criterion). The split is by conversation, with no leakage. Language mix in train: RU 307 · KZ 307 · EN 184.
- Sources:
  - 128 synthetic gold examples written by the team;
  - 410 rows distilled from the project's cloud LLM jury (Claude Haiku/Sonnet + Opus judge) on the organizers' DEV dialogs. Only unanimous or ≥0.9-confidence verdicts were kept.
- There are 0 human-labelled rows. The held-out gold set was excluded from training.
- This card and the weights ship no raw transcripts. The dataset itself is not published.

## Training

QLoRA: nf4 4-bit with double quant, bf16. LoRA r=16, α=32, dropout 0.05 on q/k/v/o/gate/up/down. Loss is computed on the completion only. lr 1e-4 (cosine), effective batch 16, max_len 8192.
**16 optimizer steps = 256 rows ≈ 0.32 epoch.** Training was capped by the time budget.
Train loss went 1.07 → 0.67; val loss 0.622, val token accuracy 0.845.
Hardware: 1× RTX A5000 24 GB, 27 min. The whole pod session cost ≈ $0.22.
Stack: torch 2.14, transformers 5.20.2, peft 0.21.2, trl 1.17.1, bitsandbytes 0.50.2.

## Results (CQA `gold` eval, vLLM 0.33, JSON mode, greedy)

| set | model | agreement | total MAE | gross-flag P / R |
|---|---|---|---|---|
| held-out (10 dialogs) | Qwen3-8B base | 0.787 | 12.17 | 0.0 / 0.0 |
| held-out (10 dialogs) | **+ this adapter** | 0.787 | **10.43** | 0.0 / 0.0 |
| tuned gold (16, *in train*) | Qwen3-8B base | 0.789 | 14.77 | 0.67 / 0.67 |
| tuned gold (16, *in train*) | **+ this adapter** | 0.805 | **8.95** | 0.75 / 1.0 |
| held-out | reference: Jev 1.14 (project's rule/decision model) | **0.934** | **4.83** | 1.0 / 1.0 |

Per-criterion agreement on held-out, base → adapter:

| criterion | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
|---|---|---|---|---|---|---|---|---|
| base → adapter | 1.0 → 1.0 | 0.5 → 0.4 | 0.7 → 0.9 | 0.7 → 0.5 | 1.0 → 1.0 | 0.8 → 0.9 | 0.9 → 0.8 | 0.7 → 0.8 |

The tuned gold dialogs were part of the training data, so that row measures fit, not generalisation.

## Serve

```bash
vllm serve Qwen/Qwen3-8B --enable-lora --lora-modules cqa=./cqa-qwen3-8b-lora --max-lora-rank 16 \
  --max-model-len 16384 --default-chat-template-kwargs '{"enable_thinking": false}'
# in the CQA repo:
CQA_PROVIDER=openai CQA_ENSEMBLE=0 CQA_MODEL=cqa CQA_OPENAI_TOOLS=0 \
CQA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 CQA_OPENAI_API_KEY=x python -m cqa gold --dir examples/gold_holdout
# as tier 1 of the cascade router:
CQA_LOCAL_ADAPTER=cqa CQA_ROUTER_LOCAL_MODEL=cqa CQA_OPENAI_TOOLS=0 python -m cqa evaluate --preset cascade ...
```

You need a 24 GB GPU. bf16 base + LoRA fits on an RTX A5000/4090 at 16k context.

## Limitations

- Very small data, and most of it is distilled from our own LLM jury, so the adapter copies the jury's biases.
- Only 0.32 epoch was trained. Held-out agreement is unchanged (0.787). MAE improved, but per-criterion results are mixed: 02/04 got worse. With 10 held-out dialogs, one item is 0.1 on a criterion, so all deltas are within noise.
- The held-out set is synthetic and small. The adapter is well below the project's Jev 1.14 model.
- It is not a safety or compliance tool. Keep a human in the loop.

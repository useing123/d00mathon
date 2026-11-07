# Fine-tuning a local CQA decision model (QLoRA)

> **Status (2026-11-08): one real QLoRA run was executed and evaluated.** Qwen3-8B was trained for 16 steps (≈0.32 epoch, capped by the time
> budget) on 1× RTX A5000 for 27 min, and the pod session cost ≈ $0.22. On the held-out gold set the adapter **does not beat base on agreement**
> (0.787 vs 0.787). It lowers total MAE (12.17 → 10.43) and is far below Jev 1.14 (0.934 / MAE 4.83). Numbers:
> `results/finetune/summary.json`; details in [Measured run](#measured-run-2026-11-08).

## Run on your own GPU

Primary download: https://huggingface.co/useing/cqa-qwen3-8b-lora · mirror (GitHub Release):
`https://github.com/useing123/d00mathon/releases/download/finetune-v1/cqa-qwen3-8b-lora.tar.gz` (~80 MB).
You need one 24 GB GPU: bf16 Qwen3-8B + LoRA at 16k context fits on an RTX A5000/4090.

```bash
curl -LO https://github.com/useing123/d00mathon/releases/download/finetune-v1/cqa-qwen3-8b-lora.tar.gz && tar xzf cqa-qwen3-8b-lora.tar.gz
# or: huggingface-cli download useing/cqa-qwen3-8b-lora --local-dir cqa-qwen3-8b-lora
vllm serve Qwen/Qwen3-8B --enable-lora --lora-modules cqa=./cqa-qwen3-8b-lora --max-lora-rank 16 \
     --max-model-len 16384 --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000
export CQA_PROVIDER=openai CQA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 CQA_OPENAI_API_KEY=x CQA_OPENAI_TOOLS=0
# standalone single model:
CQA_ENSEMBLE=0 CQA_MODEL=cqa python -m cqa gold --dir examples/gold_holdout
# as tier 1 of the cascade:
CQA_LOCAL_ADAPTER=cqa CQA_ROUTER_LOCAL_MODEL=cqa python -m cqa evaluate --preset cascade ...
```

## Measured run (2026-11-08)

* Data: `make finetune-data` (prompt v10, rebuilt on the current rubric): 798 train / 82 val, 0 leakage, ~1.95 M train tokens
  (chars/3.5). Sources: gold 128 + distill 410 (dev_jury_v7), human 0. Same stats as above.
* Stack (pinned, works): torch 2.14.0, transformers 5.20.2, peft 0.21.2, trl 1.17.1, bitsandbytes 0.50.2, accelerate 1.15.0,
  vllm 0.33.0, all in one env on a `runpod/pytorch:2.8.0` image.
* Throughput on the A5000: **~78 s per optimizer step** (16 rows × ~2.4k tokens ≈ 500 tok/s, i.e. 3–4× slower than the planning
  estimate below). A full epoch would be ~50 steps ≈ 65 min, so the run was capped with `--max-steps 16`.
* Loss: 1.07 (step 5) → 0.70 (10) → 0.67 (15). Val loss 0.622, val token accuracy 0.845. The in-training generation check
  (8 val rows, 400 new tokens) gave agreement 0.5 with parse rate 0.75. Long answers were truncated, so this number is not meaningful.
* Eval: vLLM serving the base + `cqa` LoRA, `enable_thinking=false`, `CQA_OPENAI_TOOLS=0`, `CQA_ENSEMBLE=0`, `--fresh`.
  Base and adapter ran concurrently on the same GPU: wall time ≈ 17 s/dialog (8 criteria, ~16 parallel requests).

| set | model | agreement | total MAE | gross P / R | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out 10 | Qwen3-8B base | 0.787 | 12.17 | 0.0 / 0.0 (fp2 fn1) | 1.0 | 0.5 | 0.7 | 0.7 | 1.0 | 0.8 | 0.9 | 0.7 |
| held-out 10 | + adapter | 0.787 | 10.43 | 0.0 / 0.0 (fp1 fn1) | 1.0 | 0.4 | 0.9 | 0.5 | 1.0 | 0.9 | 0.8 | 0.8 |
| gold 16 (in train) | Qwen3-8B base | 0.789 | 14.77 | 0.67 / 0.67 | 1.0 | 0.5 | 0.81 | 0.81 | 0.88 | 0.75 | 0.69 | 0.88 |
| gold 16 (in train) | + adapter | 0.805 | 8.95 | 0.75 / 1.0 | 1.0 | 0.56 | 0.88 | 0.69 | 0.88 | 0.63 | 0.88 | 0.94 |
| held-out 10 | **Jev 1.14** (docs/JEV.md) | **0.934** | **4.83** | 1.0 / 1.0 | | | | | | | | |

**Verdict:** after 0.32 epoch the adapter **does not beat base** on held-out agreement (a tie). It lowers MAE by ~15%, moves 03/06/08
up and 02/04/07 down, and is **0.15 below Jev 1.14**. The goal "beat Jev" was **not met**. On the 16 gold dialogs, which are in
train, it gains +0.016 agreement and −5.8 MAE, which only shows fit. Caveats: tiny data, mostly distilled from our own jury
(it copies the jury's biases); held-out is 10 synthetic dialogs (one item = 0.1 per criterion, so every delta is within noise); 1/3 of an
epoch; the adapter was trained on the Qwen3 template with thinking enabled by default but served with `enable_thinking=false`, a small
prompt mismatch.

**Not done (deadline):** the full epoch; a KZ base. `issai/LLama-3.1-KazLLM-1.0-8B` is gated (auto) and CC-BY-NC, so it doesn't qualify.
`issai/Qwen3.5-9B-Kazakh` is ungated, Apache-2.0 and has a `chat_template.jinja`, but it is a multimodal
`Qwen3_5ForConditionalGeneration`, so the training script would need adapting. It is the next candidate. No external data was added.

## Why

* **Cheaper tier 1 in the cascade** (`cqa/router.py`): every criterion goes to a local model first, and Claude
  only sees escalations. A model that agrees with the reviewers more often (and is confident only when it's right)
  escalates less, so it costs less.
* **Better RU/KZ**: open 8–14B models are weakest on Kazakh and on the Russian-language rubric. A small LoRA trained on
  the exact prompt format is the cheapest way to fix the format and calibration problems the OSS benchmarks
  showed (`results/bench_oss_*`).

## Data sources (and how far to trust each one)

| source | where | weight | caveat |
|---|---|---|---|
| `human` | `examples/dev_labels.json` (`python -m cqa label`), `results/overrides/<group>.json`, resolved appeals `results/appeals/<group>.json` | 3 | the only source that can make the model *better than the teacher*. Currently **0 examples** (no labels/overrides/appeals yet). |
| `gold` | `examples/gold/*.json` (synthetic) | 2 | synthetic dialogs written by us; targets have the value plus a short rationale and **no evidence quotes**. `examples/gold_holdout` is refused by the builder because it is the eval holdout. |
| `distill` | `results/dev_jury_v*.json` (newest run with the current `PROMPT_VERSION`) | 1 | **Copies the teacher's biases.** The jury (Haiku + Sonnet, Opus judge) agrees with the organizers' labels on only part of DEV. Distillation at best reproduces the jury, cheaper. Filters: jurors unanimous **or** confidence ≥ 0.9; verdicts with an `evidence_warning` are dropped. `stale_rubric` counts verdicts whose `criterion_hash` differs from the current checklist, meaning the rubric has changed since the teacher ran. |

To get real gains you need human labels. The flywheel: run the cascade → reviewers appeal or override in the UI →
`cqa label` on disagreements → rebuild the dataset → retrain. Human rows win dedupe against gold and distill rows for the
same (dialog, criterion).

## Dataset builder (`cqa/finetune/dataset.py`)

* One example = one (dialog, criterion). Chat format: `system` / `user` / `assistant`.
* **Prompt is identical to inference**: it is built with `cqa.prompts.transcript_block` + `criterion_block` +
  `data.format_transcript/format_facts` (the same calls the evaluator makes), in the OpenAI-compatible **JSON mode**
  of `cqa.llm.LLM._openai` (`system + "Верни ТОЛЬКО JSON-объект по схеме: " + schema`;
  `user = transcript + "\n\n" + criterion`). `tests/test_finetune.py::test_prompt_identical_to_inference` captures
  the evaluator's real backend call and asserts byte equality. ⇒ Serve the adapter with `CQA_OPENAI_TOOLS=0`.
* Target = compact JSON in schema order (`reasoning, value, justification, evidence, missing_where, na_reason,
  confidence[, client_requests]`). Human/gold targets use confidence 0.9. Distill targets keep the teacher's confidence, so the router's
  confidence thresholds stay meaningful.
* Dedupe on (transcript hash, checklist, criterion); priority human > gold > distill.
* Split **by conversation** (conversations with identical transcripts are merged into one unit), stratified by
  (language, real/synthetic). Default is 15% val. Augmentation, weight repetition (weight → integer copies) and language
  balancing are applied to **train only, after the split**. Val never contains duplicates, and nothing leaks between splits
  (`leakage_conversations` in stats.json must be `[]`).
* Language balance: each language is upsampled to `target × count(largest language)`. Defaults: RU 1.0, KZ 1.0, EN 0.6.
* PII-light: no new PII is introduced. Augmentation uses fictional names and re-drawn digits. The output goes to
  `data/finetune/`, which is git-ignored because it contains DEV transcripts (organizers' data); don't push it to a public hub.

### Augmentation (`cqa/finetune/augment.py`, `--augment`)

Name swaps (after "меня зовут / менің атым / my name is") and multi-digit number swaps are applied consistently to the
transcript and the target quotes. An example is dropped if a quote stops being verbatim. Back-translation
(KZ/RU → EN → back) uses a pluggable translator. The default is a no-op; the LLM translator is used only with
`CQA_FT_TRANSLATOR=llm` (it costs API calls).

### Current dataset (built 2026-11-08 on the repo data)

`python scripts/finetune_dataset.py` → sources: human 0, gold 128 (16 dialogs × 8), distill 410 from
`dev_jury_v7.json` (skipped: 71 low agreement, 39 evidence warnings). After dedupe: 538 unique examples.

| split | rows | convs | by source | by language | ≈tokens (chars/3.5) |
|---|---|---|---|---|---|
| train | 798 | 69 | distill 515 · gold 283 | RU 307 · KZ 307 · EN 184 (238 upsampled) | 1.95 M |
| val | 82 | 12 | distill 58 · gold 24 | RU 40 · KZ 21 · EN 21 | 0.20 M |

The values are skewed: 01 = YES ×102 vs NO ×17, 06 = 10 ×84, and 08 = NO ×109 vs YES ×8 in train. Per-criterion/value counts are in
`data/finetune/stats.json`. This is a **small** dataset. Expect better format and calibration, not a large accuracy jump.

## Training (`scripts/finetune_lora.py`)

QLoRA: 4-bit nf4 with double quant and bf16 compute. LoRA r=16, α=32, dropout 0.05 on `q,k,v,o,gate,up,down_proj`.
Completion-only loss: the labels are -100 on every prompt token, and the completion is `assistant JSON + EOS` appended
to the exact inference prompt tokens. `max_len` 8192; longer examples are **dropped**, never truncated, in real runs.
Packing off, gradient checkpointing, cosine schedule with 3% warmup, lr 1e-4, effective batch 16, 2 epochs.
Eval every 25 steps on val: `eval_loss` plus a generation callback that greedily decodes 64 val rows, parses the JSON and logs
`eval_agreement` / `eval_json_parse_rate` / `eval_agree_<criterion>`. Early stopping uses patience 3, and the best checkpoint is
selected on `--select-on eval_loss|eval_agreement`. The adapter is saved to `--out`. `--merge` writes `<out>-merged` (bf16) and
`--push-to-hub <repo>` pushes privately. Base models: `Qwen/Qwen3-8B` (default), `Qwen/Qwen3-14B`, `google/gemma-3-12b-it`.
Gemma-3 needs a recent transformers and an HF token for the gated weights.

```bash
python scripts/finetune_dataset.py [--augment]                     # data/finetune/{train,val}.jsonl + stats.json
python scripts/finetune_lora.py --mock                             # no deps: mask + collate + 2 steps + metric
python scripts/finetune_lora.py --dry-run --max-len 1024           # tiny random Llama, CPU, 2 real steps (needs torch+transformers)
python scripts/finetune_lora.py --estimate-only [--base Qwen/Qwen3-14B]   # tokens + $ (tokenizer if available, else chars/3.5)
# on a GPU pod (RunPod template described at the top of the script):
bash scripts/finetune_runpod.sh                                    # BASE=Qwen/Qwen3-14B bash ... for 14B (A40)
```

`--dry-run` falls back to `--mock` when torch/transformers are missing. The tests always run the mock path. The
tiny-model test is skipped without transformers or when offline.

## Cost estimate (planning numbers; measured A5000 throughput was ~500 tok/s — planning numbers in `cqa/finetune/train_utils.py:GPUS`)

The estimate assumes these QLoRA throughputs: 8B: 4090 ≈ 1700 tok/s, A40 ≈ 1400 tok/s. 14B: A40 ≈ 850 tok/s. It adds 25% overhead
for loading, checkpoints and generation eval. RunPod list prices: A40 $0.40/h, 4090 $0.35/h (check the current prices). For the current
train set (~1.95 M tokens/epoch by chars/3.5):

| base | GPU | per epoch | 2 epochs |
|---|---|---|---|
| Qwen3-8B | RTX 4090 | ~0.45 h · **$0.16** | $0.32 |
| Qwen3-8B | A40 | ~0.55 h · **$0.22** | $0.44 |
| Qwen3-14B | A40 | ~0.9 h · **$0.36** | $0.73 |

Add roughly 10–20 minutes of pod time for setup, the model download and merging. Training cost is negligible. The real cost is human labelling.

## Evaluation and serving

```bash
# serve base + adapter (OpenAI-compatible); adapter name "cqa"
vllm serve Qwen/Qwen3-8B --enable-lora --lora-modules cqa=runs/cqa-qwen3-8b-lora --max-lora-rank 16 \
     --max-model-len 16384 --port 8000
# gold metrics through the adapter (JSON mode = the trained format)
CQA_PROVIDER=openai CQA_ENSEMBLE=0 CQA_MODEL=cqa CQA_OPENAI_TOOLS=0 \
CQA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 CQA_OPENAI_API_KEY=x \
  python -m cqa gold --dir examples/gold_holdout --out results/gold_holdout_lora.json
# compare against the un-tuned base: same command with CQA_MODEL=Qwen/Qwen3-8B
```

Cascade: `CQA_LOCAL_ADAPTER` names the LoRA module to use as tier 1. Pass it as the router's local model:
`export CQA_LOCAL_ADAPTER=cqa CQA_ROUTER_LOCAL_MODEL=$CQA_LOCAL_ADAPTER CQA_OPENAI_TOOLS=0` and use
`--preset cascade`. The router itself is owned by `cqa/router.py`. Wiring `CQA_LOCAL_ADAPTER` natively there is a
one-line follow-up.

Judge a trained adapter **only** on `examples/gold_holdout` plus human-labelled DEV dialogs that are not in train. Agreement
with the jury on DEV mostly measures how well the model imitates the teacher.

## What is mock vs real

| piece | status |
|---|---|
| dataset builder, stats, split, balance, dedupe | real, run on the repo data |
| augmentation swaps | real; back-translation hook real but off by default (no translator) |
| `--mock` trainer | toy unigram "model" + majority-value policy; proves masking/collation/metric only |
| `--dry-run` | real transformers/peft/trl code path on a tiny random model (CPU) — executed once in a scratch venv |
| QLoRA GPU training + vLLM eval | **executed once** (16 steps, see Measured run); merge/push not executed |
| throughput & $ numbers | estimates, not measurements |

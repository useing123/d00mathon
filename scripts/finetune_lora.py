#!/usr/bin/env python
"""QLoRA SFT for the CQA per-criterion decision model (transformers + peft + trl, imported lazily).

Real run (GPU, NOT executed in this repo yet — see docs/FINETUNE.md):
    python scripts/finetune_dataset.py                      # -> data/finetune/{train,val}.jsonl
    accelerate launch scripts/finetune_lora.py --base Qwen/Qwen3-8B --out runs/cqa-qwen3-8b-lora
CPU proofs:
    python scripts/finetune_lora.py --mock                  # no deps at all: masking, collator, 2 steps, metric
    python scripts/finetune_lora.py --dry-run               # tiny random HF model on CPU, 2 real optimizer steps
    python scripts/finetune_lora.py --estimate-only         # token count + RunPod cost per epoch
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import config  # noqa: E402
from cqa.finetune import train_utils as tu  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Qwen3 / Gemma-3 / Llama
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def load_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def has_ml_stack():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def make_agreement_callback(tokenizer, val_rows, n, max_new_tokens):
    """Generation-based eval: greedy-decode the JSON answer for n val rows, parse it, per-criterion exact agreement.
    Writes eval_agreement / eval_json_parse_rate into the metrics dict (so early stopping can use them)."""
    import torch
    from transformers import TrainerCallback

    rows = val_rows[:n]

    class AgreementCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
            if model is None or not rows:
                return
            was_training = model.training
            model.eval()
            preds = []
            with torch.no_grad():
                for r in rows:
                    ids = torch.tensor([tu.prompt_ids(tokenizer, r["messages"])], device=model.device)
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                    preds.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
            m = tu.agreement(preds, rows)
            if metrics is not None:
                metrics["eval_agreement"] = m["agreement"]
                metrics["eval_json_parse_rate"] = m["json_parse_rate"]
                for k, v in m["per_criterion"].items():
                    metrics[f"eval_agree_{k}"] = v
            print(f"[eval step {state.global_step}] agreement={m['agreement']} parse={m['json_parse_rate']} "
                  f"per_criterion={m['per_criterion']}", flush=True)
            if was_training:
                model.train()

    return AgreementCallback()


def compat_kwargs(cls, kw):
    """Map/trim TrainingArguments kwargs across transformers 4.x/5.x and trl versions."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(cls)}
    kw = dict(kw)
    if "warmup_ratio" in kw and "warmup_ratio" not in names:   # v5: warmup_steps accepts a float ratio
        kw["warmup_steps"] = kw.pop("warmup_ratio")
    dropped = sorted(k for k in kw if k not in names)
    if dropped:
        print(f"(ignoring unsupported training args for {cls.__name__}: {dropped})", flush=True)
    return {k: v for k, v in kw.items() if k in names}


def _dtype_kw(torch_dtype):
    import transformers
    return {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch_dtype}


def train(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

    dry = a.dry_run
    base = TINY if dry and a.base == DEFAULT_BASE else a.base
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=a.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_rows = load_rows(a.train, 8 if dry else None)
    val_rows = load_rows(a.val, 4 if dry else None)
    trunc = "middle" if dry else "drop"
    from datasets import Dataset

    def feats(rows):
        out, dropped = [], 0
        for r in rows:
            t = tu.tokenize_example(r, tok, a.max_len, trunc)
            if t is None:
                dropped += 1
            else:
                out.append(t)
        return out, dropped

    tr, dtr = feats(train_rows)
    va, dva = feats(val_rows)
    print(f"tokenized train={len(tr)} (dropped>{a.max_len}: {dtr}) val={len(va)} (dropped {dva})", flush=True)
    masked = sum(x == tu.IGNORE for t in tr for x in t["labels"]) / max(sum(len(t["labels"]) for t in tr), 1)
    print(f"prompt-masked label share: {masked:.3f}", flush=True)
    collator = tu.PadCollator(pad_id=tok.pad_token_id)

    if dry:   # CPU, fp32, no quantization; LoRA still applied so peft is exercised when present
        model = AutoModelForCausalLM.from_pretrained(base, **_dtype_kw(torch.float32))
    else:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, **_dtype_kw(torch.bfloat16),
                                                     attn_implementation=a.attn, trust_remote_code=a.trust_remote_code,
                                                     device_map={"": int(os.environ.get("LOCAL_RANK", 0))})
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if not dry:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                    gradient_checkpointing_kwargs={"use_reentrant": False})
        targets = [m for m in LORA_TARGETS if any(n.endswith(m) for n, _ in model.named_modules())] or "all-linear"
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                                 target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    except ImportError:
        if not dry:
            raise
        print("peft not installed — dry-run trains the tiny model's full weights", flush=True)

    common = dict(output_dir=a.out, per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
                  gradient_accumulation_steps=1 if dry else a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
                  warmup_ratio=0.03, num_train_epochs=a.epochs, max_steps=2 if dry else a.max_steps, logging_steps=1 if dry else 5,
                  eval_strategy="steps", eval_steps=1 if dry else a.eval_steps, save_strategy="steps",
                  save_steps=1 if dry else a.eval_steps, save_total_limit=2, load_best_model_at_end=True,
                  metric_for_best_model=a.select_on, greater_is_better=a.select_on != "eval_loss",
                  bf16=not dry, gradient_checkpointing=not dry, report_to=[] if dry else a.report_to,
                  remove_unused_columns=False, use_cpu=dry, seed=a.seed,
                  gradient_checkpointing_kwargs={"use_reentrant": False} if not dry else None)
    callbacks = [make_agreement_callback(tok, val_rows, 2 if dry else a.eval_generate, 64 if dry else a.max_new_tokens),
                 EarlyStoppingCallback(early_stopping_patience=a.patience)]
    train_ds, eval_ds = Dataset.from_list(tr), Dataset.from_list(va)
    try:   # trl path: pre-tokenized dataset + our completion-only labels; packing off
        from trl import SFTConfig, SFTTrainer
        cfg = SFTConfig(**compat_kwargs(SFTConfig, dict(common, packing=False, max_length=a.max_len,
                                                        dataset_kwargs={"skip_prepare_dataset": True})))
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator,
                             processing_class=tok, callbacks=callbacks)
    except ImportError:
        if not dry:
            raise
        from transformers import Trainer, TrainingArguments
        print("trl not installed — dry-run uses transformers.Trainer with the same collator", flush=True)
        targs = TrainingArguments(**compat_kwargs(TrainingArguments, common))
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                          data_collator=collator, callbacks=callbacks)
    res = trainer.train()
    print(json.dumps({"train": res.metrics}, default=str), flush=True)
    trainer.save_model(a.out)          # LoRA adapter (adapter_model.safetensors + adapter_config.json)
    tok.save_pretrained(a.out)
    if a.merge and not dry:
        merge(base, a.out, a.out.rstrip("/") + "-merged", a.trust_remote_code)
    if a.push_to_hub:
        trainer.model.push_to_hub(a.push_to_hub, private=True)
        tok.push_to_hub(a.push_to_hub, private=True)
    return res.metrics


def merge(base, adapter, out, trust_remote_code=False):
    """Merge the adapter into a bf16 copy of the base (needs ~2x model size of RAM/VRAM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(base, **_dtype_kw(torch.bfloat16), trust_remote_code=trust_remote_code)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    print(f"merged model -> {out}")


def estimate(a):
    rows, vrows = load_rows(a.train), load_rows(a.val)
    tok = None
    if not a.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.base)
        except Exception as e:   # noqa: BLE001 — offline / not installed → heuristic
            print(f"(tokenizer unavailable: {type(e).__name__}; using chars/3.5)", file=sys.stderr)
    n, how = tu.count_tokens(rows, tok)
    nv, _ = tu.count_tokens(vrows, tok)
    est = tu.estimate_cost(n, a.base, a.epochs, nv)
    est.update(token_count_method=how, train_rows=len(rows), val_rows=len(vrows), val_tokens=nv)
    print(json.dumps(est, indent=1))
    return est


DEFAULT_BASE = "Qwen/Qwen3-8B"


def parse_args(argv=None):
    d = config.DATA_DIR / "finetune"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help="Qwen/Qwen3-8B | Qwen/Qwen3-14B | google/gemma-3-12b-it")
    ap.add_argument("--train", default=str(d / "train.jsonl"))
    ap.add_argument("--val", default=str(d / "val.jsonl"))
    ap.add_argument("--out", default="runs/cqa-lora")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (time budget); -1 = full epochs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--eval-generate", type=int, default=64, help="val rows decoded per eval for the agreement metric")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--select-on", default="eval_loss", choices=["eval_loss", "eval_agreement"])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--report-to", nargs="*", default=[])
    ap.add_argument("--merge", action="store_true", help="also save a merged bf16 model (<out>-merged)")
    ap.add_argument("--push-to-hub", default="", help="repo id (private) to push the adapter to")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true", help=f"CPU: {TINY}, 2 steps (falls back to --mock without torch)")
    ap.add_argument("--mock", action="store_true", help="dependency-free pipeline check (tokenise/mask/collate/metric)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true", help="estimate with chars/3.5")
    return ap.parse_args(argv)


def ensure_dataset(a):
    if Path(a.train).exists() and Path(a.val).exists():
        return
    print("dataset missing — building it (python -m cqa.finetune.dataset)", flush=True)
    from cqa.finetune.dataset import build
    build(Path(a.train).parent)


def main(argv=None):
    a = parse_args(argv)
    ensure_dataset(a)
    if a.estimate_only:
        return estimate(a)
    if a.mock or (a.dry_run and not has_ml_stack()):
        if a.dry_run:
            print("torch/transformers not installed — running --mock instead of the tiny-model dry run", flush=True)
        res = tu.run_mock(load_rows(a.train, 16), load_rows(a.val, 32), steps=2)
        print(json.dumps(res, ensure_ascii=False))
        return res
    return train(a)


if __name__ == "__main__":
    main()

"""Pure-python training helpers shared by scripts/finetune_lora.py and the tests (no torch needed here):
completion-only tokenisation/masking, padding collator, JSON-answer agreement metric, token count + cost estimate,
and a dependency-free MOCK trainer that runs the whole pipeline on CPU in milliseconds.
"""
import json
import math
from collections import Counter, defaultdict

from .dataset import parse_answer

IGNORE = -100

# QLoRA (4-bit nf4, r=16, grad checkpointing, bf16, seq ~4-8k) training throughput, tokens/s — CONSERVATIVE planning
# numbers, not measured on this project. Prices: RunPod on-demand list (community cloud), USD/h; check before renting.
GPUS = {
    "A40": {"usd_h": 0.40, "vram_gb": 48, "tok_s": {"8b": 1400, "12b": 950, "14b": 850}},
    "RTX4090": {"usd_h": 0.35, "vram_gb": 24, "tok_s": {"8b": 1700, "12b": 1100, "14b": 950}},
    "A100-80G": {"usd_h": 1.20, "vram_gb": 80, "tok_s": {"8b": 3200, "12b": 2200, "14b": 1900}},
}


def size_class(base_model):
    m = base_model.lower()
    return "14b" if "14b" in m else "12b" if "12b" in m else "8b"


# ---------------------------------------------------------------- tokenisation with prompt masking

def _render_fallback(messages, add_generation_prompt):
    """ChatML-ish template for tokenizers without a chat template (tiny test models / mock)."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def prompt_ids(tokenizer, messages):
    """Token ids of system+user + generation prompt — what the inference server renders (default template kwargs)."""
    msgs = [m for m in messages if m["role"] != "assistant"]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):   # BatchEncoding / dict (newer transformers return_dict default)
            ids = ids["input_ids"]
        return [int(x) for x in ids]
    return list(tokenizer(_render_fallback(msgs, True), add_special_tokens=False)["input_ids"])


def tokenize_example(row, tokenizer, max_len=8192, truncate="drop"):
    """→ {"input_ids", "labels", "attention_mask"} with labels = IGNORE on every prompt token (completion-only loss).
    The completion is the assistant JSON + EOS (<|im_end|> for Qwen), appended to the exact inference prompt
    tokens, so the trained boundary is identical to serving. truncate: 'drop' (return None; used for real
    training — never train on a mutilated prompt) or 'middle' (cut the middle of the prompt; dry-run only)."""
    p = prompt_ids(tokenizer, row["messages"])
    ans = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    eos = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
    c = list(tokenizer(ans + eos, add_special_tokens=False)["input_ids"])
    if len(p) + len(c) > max_len:
        if truncate == "drop":
            return None
        keep = max(max_len - len(c), 8)
        head = keep // 2
        p = p[:head] + p[-(keep - head):]
        c = c[: max_len - len(p)]
    ids = p + c
    return {"input_ids": ids, "labels": [IGNORE] * len(p) + c, "attention_mask": [1] * len(ids)}


class PadCollator:
    """Right-pads input_ids (pad id), labels (IGNORE) and attention_mask (0). Returns torch tensors if torch is
    importable and as_tensors=True, else python lists (mock / tests)."""

    def __init__(self, pad_id=0, as_tensors=True, pad_to_multiple_of=8):
        self.pad_id, self.as_tensors, self.mult = pad_id, as_tensors, pad_to_multiple_of

    def __call__(self, features):
        n = max(len(f["input_ids"]) for f in features)
        if self.mult:
            n = int(math.ceil(n / self.mult) * self.mult)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            k = n - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_id] * k)
            batch["labels"].append(list(f["labels"]) + [IGNORE] * k)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * k)
        if self.as_tensors:
            try:
                import torch
            except ImportError:
                return batch
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch


# ---------------------------------------------------------------- metric

def agreement(pred_texts, rows):
    """Per-criterion exact agreement of the parsed JSON `value` against the row's target value."""
    per = defaultdict(lambda: [0, 0])
    parsed = 0
    for txt, row in zip(pred_texts, rows, strict=True):
        js = parse_answer(txt)
        gold = str(row["meta"]["value"]).upper()
        cid = row["meta"]["criterion_id"]
        ok = js is not None and str(js.get("value", "")).strip().upper() == gold
        parsed += js is not None
        per[cid][0] += ok
        per[cid][1] += 1
    tot = sum(v[1] for v in per.values())
    return {"agreement": round(sum(v[0] for v in per.values()) / tot, 4) if tot else 0.0,
            "json_parse_rate": round(parsed / tot, 4) if tot else 0.0,
            "per_criterion": {k: round(v[0] / v[1], 4) for k, v in sorted(per.items())}, "n": tot}


# ---------------------------------------------------------------- tokens + cost

def count_tokens(rows, tokenizer=None):
    """Exact with a tokenizer (prompt+completion), else chars/3.5 (Cyrillic-heavy text ≈ 3–4 chars/token on Qwen)."""
    if tokenizer is not None:
        n = 0
        for r in rows:
            t = tokenize_example(r, tokenizer, max_len=10 ** 9)
            n += len(t["input_ids"])
        return n, "tokenizer"
    return int(sum(len(m["content"]) for r in rows for m in r["messages"]) / 3.5), "chars/3.5"


def estimate_cost(train_tokens, base_model="Qwen/Qwen3-8B", epochs=1, eval_tokens=0, evals_per_epoch=4, overhead=1.25):
    """USD and hours per GPU. overhead covers model load, checkpointing, generation-based eval."""
    sc = size_class(base_model)
    out = {}
    for gpu, g in GPUS.items():
        if sc == "14b" and g["vram_gb"] < 24:
            continue
        tok = train_tokens * epochs + eval_tokens * evals_per_epoch * epochs / 3   # eval fwd ≈ 1/3 of train step
        h = tok / g["tok_s"][sc] / 3600 * overhead
        out[gpu] = {"hours": round(h, 2), "usd": round(h * g["usd_h"], 2), "usd_per_epoch": round(h * g["usd_h"] / epochs, 2),
                    "assumed_tok_s": g["tok_s"][sc], "usd_h": g["usd_h"]}
    return {"base_model": base_model, "size_class": sc, "train_tokens": train_tokens, "epochs": epochs, "gpus": out}


# ---------------------------------------------------------------- MOCK (no deps)

class CharTokenizer:
    """Deterministic byte-level tokenizer for mock mode (no chat template → ChatML fallback is exercised)."""
    eos_token = "<|im_end|>"
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [b + 1 for b in text.encode("utf-8")]}

    def decode(self, ids):
        return bytes(i - 1 for i in ids if i > 0).decode("utf-8", "replace")


class MockModel:
    """A unigram LM over completion tokens + a per-criterion majority-value 'policy' for generation.
    Learns only from label positions (proves the masking reaches the loss) — a sanity check, not a model."""

    def __init__(self, vocab=258):
        self.counts = [1.0] * vocab
        self.prior = defaultdict(Counter)

    def loss(self, batch):
        tot = sum(self.counts)
        n, nll = 0, 0.0
        for row in batch["labels"]:
            for t in row:
                if t != IGNORE:
                    nll -= math.log(self.counts[t] / tot)
                    n += 1
        return nll / max(n, 1), n

    def step(self, batch, rows, lr=1.0):
        for row in batch["labels"]:
            for t in row:
                if t != IGNORE:
                    self.counts[t] += lr
        for r in rows:
            self.prior[r["meta"]["criterion_id"]][r["meta"]["value"]] += 1

    def generate(self, row):
        c = self.prior.get(row["meta"]["criterion_id"])
        v = c.most_common(1)[0][0] if c else "YES"
        return json.dumps({"reasoning": "mock", "value": v, "justification": "mock", "evidence": [], "confidence": 0.5},
                          ensure_ascii=False)


def run_mock(train_rows, val_rows, steps=2, batch_size=2, max_len=4096, log=print):
    """End-to-end CPU pipeline: tokenise (masking) → collate → 'train' `steps` steps → generation eval + metric."""
    tok = CharTokenizer()
    feats = [(r, tokenize_example(r, tok, max_len=max_len, truncate="middle")) for r in train_rows]
    feats = [(r, f) for r, f in feats if f]
    coll = PadCollator(pad_id=tok.pad_token_id, as_tensors=False)
    model = MockModel()
    hist = []
    for s in range(steps):
        chunk = feats[s * batch_size:(s + 1) * batch_size] or feats[:batch_size]
        batch = coll([f for _, f in chunk])
        masked = sum(1 for row in batch["labels"] for t in row if t == IGNORE)
        before, n = model.loss(batch)
        model.step(batch, [r for r, _ in chunk])
        after, _ = model.loss(batch)
        hist.append({"step": s + 1, "loss_before": round(before, 4), "loss_after": round(after, 4),
                     "completion_tokens": n, "masked_tokens": masked, "seq_len": len(batch["input_ids"][0])})
        log(f"[mock] step {s + 1}/{steps} loss {before:.4f} -> {after:.4f}  completion_tokens={n} masked={masked}")
    preds = [model.generate(r) for r in val_rows]
    m = agreement(preds, val_rows)
    log(f"[mock] eval agreement={m['agreement']} parse={m['json_parse_rate']} per_criterion={m['per_criterion']}")
    return {"history": hist, "eval": m}

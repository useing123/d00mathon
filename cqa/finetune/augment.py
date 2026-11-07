"""Optional train-split augmentation for RU/KZ (and EN): name/number swaps and back-translation hooks.

Applied AFTER the conversation split, to train examples only, so augmented copies never leak into val.
The translator is pluggable: the default is a no-op (no back-translation at all); an LLM translator is used only
when explicitly enabled (CQA_FT_TRANSLATOR=llm, costs API calls). Swaps never introduce real PII: replacement
names come from a fixed fictional list, numbers are re-drawn digit by digit (same length/format).
Every substitution is applied consistently to the transcript AND the target (justification/evidence quotes), and
an example is dropped if an evidence quote is no longer a verbatim substring of its turn.
"""
import copy
import hashlib
import os
import random
import re

from .. import data

FAKE_NAMES = {"RU": ["Алия", "Дмитрий", "Ольга", "Тимур", "Марина", "Сергей"],
              "KZ": ["Айгерім", "Нұрлан", "Әсел", "Ерлан", "Дана", "Бауыржан"],
              "EN": ["Anna", "James", "Maria", "Daniel", "Sofia", "Alex"]}
NAME_RE = re.compile(r"(?:меня зовут|менің атым|атым|my name is|this is|i am|i'm)\s+([A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][a-zа-яёәғқңөұүһі]+)", re.I)
NUM_RE = re.compile(r"(?<![\d:])\d{2,}(?![\d:])")   # 2+ digit numbers, not timestamps like 12:30


class NoopTranslator:
    enabled = False

    def __call__(self, text, src, tgt):
        return text


class LLMTranslator:
    """Back-translation through the configured cqa LLM provider (explicit opt-in only)."""
    enabled = True

    def __init__(self, llm=None):
        from ..llm import LLM
        self.llm = llm or LLM()

    def __call__(self, text, src, tgt):
        tool = {"name": "translate", "description": "Return the translation.",
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
        out = self.llm.call(f"Translate from {src} to {tgt}. Keep names, numbers and tone. Translation only.", text, "", tool)
        return (out or {}).get("text") or text


def _rng(*parts):
    return random.Random(int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def _map_strings(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_map_strings(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: (_map_strings(v, fn) if k not in ("value", "turn", "confidence", "status", "essential") else v)
                for k, v in obj.items()}
    return obj


def evidence_ok(rec, target):
    by = {t.get("index"): t.get("text") or "" for t in rec.get("transcript") or []}
    return all(e.get("quote", "") in by.get(e.get("turn"), "") for e in target.get("evidence") or [])


class Augmenter:
    def __init__(self, translator=None, swap_names=True, swap_numbers=True, pivot="EN", seed=13):
        self.translator = translator or NoopTranslator()
        self.swap_names, self.swap_numbers, self.pivot, self.seed = swap_names, swap_numbers, pivot, seed

    @classmethod
    def from_env(cls):
        tr = LLMTranslator() if os.environ.get("CQA_FT_TRANSLATOR", "").lower() == "llm" else NoopTranslator()
        return cls(translator=tr)

    def substitution(self, ex):
        """Build a {old: new} map from the transcript (names after self-introductions, multi-digit numbers)."""
        rng = _rng(self.seed, ex["dialog_hash"], ex["criterion_id"])
        subs = {}
        texts = [t.get("text") or "" for t in ex["_rec"].get("transcript") or []]
        if self.swap_names:
            pool = FAKE_NAMES.get(ex["language"], FAKE_NAMES["RU"])
            for tx in texts:
                for m in NAME_RE.finditer(tx):
                    name = m.group(1)
                    if name not in subs:
                        cands = [n for n in pool if n != name]
                        subs[name] = rng.choice(cands)
        if self.swap_numbers:
            for tx in texts:
                for m in NUM_RE.finditer(tx):
                    s = m.group(0)
                    if s not in subs:
                        subs[s] = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in s[1:])
        return subs

    @staticmethod
    def apply(text, subs):
        if not subs or not text:
            return text
        pat = re.compile("|".join(sorted((re.escape(k) for k in subs), key=len, reverse=True)))
        return pat.sub(lambda m: subs[m.group(0)], text)

    def _rebuild(self, ex, fn, tag):
        new = dict(ex)
        rec = copy.deepcopy(ex["_rec"])
        for t in rec.get("transcript") or []:
            t["text"] = fn(t.get("text") or "")
        rec["hash"] = data.dialog_hash(rec)
        tgt = _map_strings(copy.deepcopy(ex["target"]), fn)
        if not evidence_ok(rec, tgt):
            return None
        new.update(_rec=rec, target=tgt, dialog_hash=rec["hash"], origin=ex["origin"] + f"+aug:{tag}", augmented=True,
                   weight=ex["weight"] * 0.5)
        return new

    def swap(self, ex):
        subs = self.substitution(ex)
        return self._rebuild(ex, lambda s: self.apply(s, subs), "swap") if subs else None

    def back_translate(self, ex):
        if not getattr(self.translator, "enabled", False) or ex["language"] == self.pivot:
            return None
        src = ex["language"]
        return self._rebuild(ex, lambda s: self.translator(self.translator(s, src, self.pivot), self.pivot, src) if s else s,
                             "bt")

    def augment_examples(self, examples, langs=("KZ", "RU")):
        """Returns NEW examples only (caller concatenates). One swap copy (+ one back-translation if enabled)
        per KZ/RU example."""
        out = []
        for ex in examples:
            if ex["language"] not in langs:
                continue
            for fn in (self.swap, self.back_translate):
                n = fn(ex)
                if n is not None:
                    out.append(n)
        return out

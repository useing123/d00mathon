"""SFT dataset builder for a per-criterion CQA decision model (chat format: system / user / assistant).

Sources (highest priority first; one example = one (dialog, criterion) pair):
  human    `cqa label` outputs (examples/dev_labels.json), human overrides (results/overrides/<group>.json) and
           resolved appeals (results/appeals/<group>.json)                         weight 3
  gold     synthetic gold sets (examples/gold; examples/gold_holdout is NEVER used — it is the eval holdout)  weight 2
  distill  cloud-jury verdicts on DEV (results/dev_jury_v*.json); only when the jurors agreed or the judge's
           confidence >= min_conf. The teacher's value + justification + evidence become the target.     weight 1

The prompt is built with exactly the same functions the evaluator uses (cqa.prompts + cqa.data), in the
OpenAI-compatible JSON mode of cqa.llm.LLM._openai (system + JSON-schema suffix; user = transcript block +
"\\n\\n" + criterion block), so train and inference match. Serve the adapter with CQA_OPENAI_TOOLS=0.

Split is by conversation (dialogs with identical transcripts are grouped), stratified by language and source;
language balancing (KZ upsampling) and weight-repetition are applied to the TRAIN split only, after the split,
so duplicates never leak into val. Output goes to data/finetune/ (git-ignored: it contains DEV transcripts).
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .. import checklist as ckl, config, data, prompts

SOURCES = ("human", "gold", "distill")
PRIORITY = {s: i for i, s in enumerate(SOURCES)}
DEFAULT_WEIGHTS = {"human": 3.0, "gold": 2.0, "distill": 1.0}
DEFAULT_LANG_TARGETS = {"RU": 1.0, "KZ": 1.0, "EN": 0.6}   # min share of the largest language's TRAIN count
LANGS = ("RU", "KZ", "EN")
JSON_MODE_SUFFIX = "\nВерни ТОЛЬКО JSON-объект по схеме: "
SCHEMA_KEYS = ("reasoning", "value", "justification", "evidence", "missing_where", "na_reason", "confidence",
               "client_requests")


def default_out_dir():
    return config.DATA_DIR / "finetune"


# ---------------------------------------------------------------- prompt (identical to inference)

def system_prompt():
    """System message exactly as cqa.llm.LLM._openai sends it in JSON mode."""
    return prompts.SYSTEM + JSON_MODE_SUFFIX + json.dumps(prompts.tool_schema()["input_schema"], ensure_ascii=False)


def transcript_block(rec, checklist):
    """Same call as Evaluator.evaluate_criterion."""
    a = checklist["assumptions"]
    return prompts.transcript_block(rec, data.format_transcript(rec, a.get("ignore_internal_hold_turns", True)),
                                    data.format_facts(data.compute_facts(rec, a.get("long_wait_seconds"))))


def build_prompt(rec, checklist, cr, tx_block=None):
    """(system, user) — user = cached_user + "\\n\\n" + user exactly as the OpenAI-compatible backends join them."""
    tx = tx_block if tx_block is not None else transcript_block(rec, checklist)
    return system_prompt(), tx + "\n\n" + prompts.criterion_block(checklist, cr)


def target_text(t):
    """Assistant target: compact JSON in schema key order (reasoning first = think-then-answer)."""
    out = {k: t[k] for k in SCHEMA_KEYS if k in t and t[k] not in (None,)}
    out.setdefault("evidence", [])
    return json.dumps(out, ensure_ascii=False)


def parse_answer(text):
    """Parse a model completion → dict or None (tolerant: first {...} block)."""
    if not isinstance(text, str):
        return None
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        js = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    return js if isinstance(js, dict) else None


# ---------------------------------------------------------------- helpers

def norm_lang(rec):
    lang = str(rec.get("language") or "").upper().strip()
    lang = {"KK": "KZ", "KAZ": "KZ", "QAZ": "KZ", "RUS": "RU", "ENG": "EN"}.get(lang, lang)
    if lang in LANGS:
        return lang
    det = data.detect_language([t.get("text") for t in rec.get("transcript") or []])
    det = str(det or "").upper()
    return {"KK": "KZ"}.get(det, det) if det else "RU"


def norm_value(cr, v):
    sv = ckl.strict_value(cr, v) if v is not None else None
    return None if sv is None else str(sv)


def _crit_map(cl):
    return {c["id"]: c for c in cl["criteria"]}


def _ev(evidence):
    return [{"turn": e["turn"], "quote": e["quote"]} for e in evidence or []
            if isinstance(e, dict) and isinstance(e.get("turn"), int) and e.get("quote")]


def _example(rec, cl, cr, value, source, target, origin, weight=None, extra=None):
    t = dict(target)
    t["value"] = value
    t.setdefault("confidence", 0.9)
    ex = {"conversation_id": rec["conversation_id"], "dialog_hash": rec.get("hash") or data.dialog_hash(rec),
          "checklist_id": cl["id"], "criterion_id": cr["id"], "value": value, "language": norm_lang(rec),
          "kind": rec.get("kind"), "source": source, "origin": origin,
          "weight": float(weight if weight is not None else DEFAULT_WEIGHTS[source]),
          "criterion_hash": ckl.criterion_hash(cl, cr), "target": t, "_rec": rec, "_cl": cl}
    ex.update(extra or {})
    return ex


# ---------------------------------------------------------------- sources

def from_labels(path, recs):
    """`python -m cqa label` output: {cid: {"expected": {crit: value}, "notes": {crit: text}, "labeller"}}."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for cid, lab in json.loads(path.read_text(encoding="utf-8")).items():
        rec = recs.get(cid)
        if rec is None:
            continue
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        for crid, v in (lab.get("expected") or {}).items():
            cr = _crit_map(cl).get(crid)
            val = cr and norm_value(cr, v)
            if val is None:
                continue
            note = (lab.get("notes") or {}).get(crid, "")
            out.append(_example(rec, cl, cr, val, "human", {"reasoning": note, "justification": note},
                                f"label:{lab.get('labeller', '')}"))
    return out


def from_overrides(group, recs):
    """results/overrides/<group>.json: {cid: {crit: {value, justification, reviewer}}}."""
    from .. import store
    p = store.overrides_path(group)
    if not p.exists():
        return []
    out = []
    for cid, crits in json.loads(p.read_text(encoding="utf-8")).items():
        rec = recs.get(cid)
        if rec is None:
            continue
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        for crid, o in (crits or {}).items():
            cr = _crit_map(cl).get(crid)
            val = cr and norm_value(cr, (o or {}).get("value"))
            if val is None:
                continue
            j = (o or {}).get("justification", "")
            out.append(_example(rec, cl, cr, val, "human", {"reasoning": j, "justification": j},
                                f"override:{(o or {}).get('reviewer', '')}"))
    return out


def from_appeals(group, recs):
    """Resolved appeals. 'changed' ones are also overrides (deduped later); 'confirmed' ones are a human
    confirmation of the AI value, so the AI justification/evidence become the target."""
    from .. import appeals
    out = []
    for a in appeals.load(group):
        res = a.get("resolution") or {}
        rec = recs.get(a.get("conversation_id"))
        if a.get("status") != "resolved" or rec is None:
            continue
        try:
            cl = ckl.load(a.get("checklist_id") or ckl.pick_for_channel(rec["kind"]))
        except (ValueError, FileNotFoundError):
            continue
        cr = _crit_map(cl).get(a.get("criterion_id"))
        val = cr and norm_value(cr, res.get("value"))
        if val is None:
            continue
        if res.get("decision") == "confirmed":
            tgt = {"reasoning": a.get("ai_justification", ""), "justification": a.get("ai_justification", ""),
                   "evidence": _ev(a.get("ai_evidence")), "missing_where": a.get("missing_where", "")}
        else:
            tgt = {"reasoning": res.get("reason", ""), "justification": res.get("reason", "")}
        out.append(_example(rec, cl, cr, val, "human", tgt, f"appeal:{a.get('id')}:{res.get('decision')}"))
    return out


def from_gold(gold_dir=None):
    from ..gold import GOLD_DIR, load_gold
    gold_dir = Path(gold_dir or GOLD_DIR)
    if "holdout" in gold_dir.name:
        raise ValueError("gold_holdout is the evaluation holdout — never train on it")
    out = []
    for i, g in enumerate(load_gold(gold_dir)):
        rec = data.normalize(g["dialog"], i)
        rec["hash"] = data.dialog_hash(rec)
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        rat = g.get("rationale") or {}
        for crid, v in (g.get("expected") or {}).items():
            cr = _crit_map(cl).get(crid)
            val = cr and norm_value(cr, v)
            if val is None:
                continue
            note = rat.get(crid, "") if isinstance(rat, dict) else ""
            out.append(_example(rec, cl, cr, val, "gold", {"reasoning": note, "justification": note}, f"gold:{g['file']}"))
    return out


def jury_agreed(r):
    votes = [str(v) for v in (r.get("votes") or []) if v is not None]
    return (len(votes) >= 2 and len(set(votes)) == 1) or r.get("decided_by") == "consensus"


def from_jury(path, recs, min_conf=0.9, require_agreement=True):
    """Distillation: teacher = cloud-jury result. Keep a verdict only if the jurors agreed OR confidence >= min_conf
    (require_agreement=False keeps everything that has a valid value — not recommended). Skips failed criteria and
    verdicts with an evidence warning."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    from ..store import upgrade_turns
    doc = upgrade_turns(doc)
    pv = ((doc.get("meta") or {}).get("reproducibility") or {}).get("prompt_version")
    out, skipped = [], Counter()
    for d in doc.get("dialogs") or []:
        rec = recs.get(d.get("conversation_id"))
        if rec is None:
            skipped["no_dialog"] += 1
            continue
        try:
            cl = ckl.load(d.get("checklist_id") or ckl.pick_for_channel(rec["kind"]))
        except (ValueError, FileNotFoundError):
            skipped["no_checklist"] += 1
            continue
        cmap = _crit_map(cl)
        for r in d.get("criteria") or []:
            cr = cmap.get(r.get("criterion_id"))
            val = cr and norm_value(cr, r.get("value"))
            if val is None or r.get("error"):
                skipped["invalid"] += 1
                continue
            conf = float(r.get("confidence") or 0)
            if require_agreement and not (jury_agreed(r) or conf >= min_conf):
                skipped["low_agreement"] += 1
                continue
            if r.get("evidence_warning"):
                skipped["evidence_warning"] += 1
                continue
            tgt = {"reasoning": r.get("reasoning") or r.get("justification", ""), "justification": r.get("justification", ""),
                   "evidence": _ev(r.get("evidence")), "missing_where": r.get("missing_where", ""),
                   "na_reason": r.get("na_reason", ""), "confidence": round(min(max(conf, 0.0), 1.0), 2)}
            if cr["id"] == "04" and r.get("client_requests"):
                tgt["client_requests"] = r["client_requests"]
            stale = bool(r.get("criterion_hash")) and r["criterion_hash"] != ckl.criterion_hash(cl, cr)
            out.append(_example(rec, cl, cr, val, "distill", tgt, f"jury:{Path(path).name}",
                                extra={"teacher_prompt_version": pv, "stale_rubric": stale,
                                       "teacher_agreed": jury_agreed(r)}))
    return out, dict(skipped)


def latest_jury_file(results_dir=None):
    """Newest results/dev_jury_v*.json produced with the current PROMPT_VERSION (falls back to the newest)."""
    files = sorted(Path(results_dir or config.RESULTS_DIR).glob("dev_jury_v*.json"),
                   key=lambda p: int("".join(ch for ch in p.stem.split("_v")[-1] if ch.isdigit()) or 0))
    for p in reversed(files):
        try:
            pv = json.loads(p.read_text(encoding="utf-8"))["meta"]["reproducibility"]["prompt_version"]
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if pv == prompts.PROMPT_VERSION:
            return p
    return files[-1] if files else None


# ---------------------------------------------------------------- dedupe / split / balance

def dedupe(examples):
    """One example per (dialog content, checklist, criterion); the highest-priority source wins
    (human > gold > distill), then the first seen."""
    best = {}
    for ex in examples:
        k = (ex["dialog_hash"], ex["checklist_id"], ex["criterion_id"])
        if k not in best or PRIORITY[ex["source"]] < PRIORITY[best[k]["source"]]:
            best[k] = ex
    return list(best.values())


def _h(seed, s):
    return hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()


def conversation_groups(examples):
    """Union conversation ids that share a transcript hash → one split unit (no leakage via duplicates)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_hash = {}
    for ex in examples:
        cid = "cid:" + ex["conversation_id"]
        find(cid)
        h = by_hash.setdefault(ex["dialog_hash"], cid)
        parent[find(cid)] = find(h)
    return {ex["conversation_id"]: find("cid:" + ex["conversation_id"]) for ex in examples}


def split(examples, val_frac=0.15, seed=13):
    """Stratified by (language, source family) at the conversation-group level. Returns (train, val)."""
    grp = conversation_groups(examples)
    strata = defaultdict(dict)
    for ex in examples:
        g = grp[ex["conversation_id"]]
        strata[(ex["language"], "synthetic" if ex["source"] == "gold" else "real")].setdefault(g, None)
    val_groups = set()
    for key in sorted(strata):
        gs = sorted(strata[key], key=lambda g: _h(seed, g))
        n = len(gs)
        k = int(round(n * val_frac))
        if val_frac > 0 and k == 0 and n >= 3:
            k = 1
        val_groups.update(gs[:k])
    train = [ex for ex in examples if grp[ex["conversation_id"]] not in val_groups]
    val = [ex for ex in examples if grp[ex["conversation_id"]] in val_groups]
    return train, val


def expand_weights(examples):
    """Sample weight → integer repetition (trl's SFTTrainer has no per-example weights; repetition is the
    portable equivalent). Fractional parts are kept deterministically by hash."""
    out = []
    for ex in examples:
        w = max(ex["weight"], 0.0)
        n = int(w) + (1 if (int(_h("w", ex["dialog_hash"] + ex["criterion_id"])[:8], 16) / 0xFFFFFFFF) < (w - int(w)) else 0)
        for i in range(max(n, 1)):
            out.append(dict(ex, rep=i))
    return out


def balance_languages(examples, targets=None, seed=13):
    """Upsample (duplicate) languages below targets[lang] * count(largest language). Train split only."""
    targets = {**DEFAULT_LANG_TARGETS, **(targets or {})}
    by = defaultdict(list)
    for ex in examples:
        by[ex["language"]].append(ex)
    if not by:
        return []
    top = max(len(v) for v in by.values())
    rng = random.Random(seed)
    out = list(examples)
    for lang, items in sorted(by.items()):
        need = int(round(targets.get(lang, 0) * top)) - len(items)
        pool = sorted(items, key=lambda e: _h(seed, e["dialog_hash"] + e["criterion_id"] + str(e.get("rep", 0))))
        for i in range(max(need, 0)):
            out.append(dict(pool[i % len(pool)], upsampled=True))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- render / stats / build

def render(ex, tx_cache=None):
    rec, cl = ex["_rec"], ex["_cl"]
    cr = _crit_map(cl)[ex["criterion_id"]]
    key = (ex["dialog_hash"], cl["id"])
    if tx_cache is not None and key not in tx_cache:
        tx_cache[key] = transcript_block(rec, cl)
    sysm, user = build_prompt(rec, cl, cr, tx_cache[key] if tx_cache is not None else None)
    meta = {k: v for k, v in ex.items() if not k.startswith("_") and k != "target"}
    return {"id": f"{ex['conversation_id']}:{cl['id']}:{ex['criterion_id']}", "messages": [
        {"role": "system", "content": sysm}, {"role": "user", "content": user},
        {"role": "assistant", "content": target_text(ex["target"])}], "meta": meta}


def stats(rows):
    def cnt(key):
        return dict(sorted(Counter(str(r["meta"].get(key)) for r in rows).items()))
    per_crit = defaultdict(Counter)
    for r in rows:
        per_crit[r["meta"]["criterion_id"]][r["meta"]["value"]] += 1
    chars = sum(len(m["content"]) for r in rows for m in r["messages"])
    return {"rows": len(rows), "conversations": len({r["meta"]["conversation_id"] for r in rows}),
            "by_source": cnt("source"), "by_language": cnt("language"), "by_criterion": cnt("criterion_id"),
            "by_criterion_value": {k: dict(sorted(v.items())) for k, v in sorted(per_crit.items())},
            "upsampled": sum(1 for r in rows if r["meta"].get("upsampled")),
            "stale_rubric": sum(1 for r in rows if r["meta"].get("stale_rubric")),
            "chars": chars, "est_tokens_chars_div_3_5": int(chars / 3.5)}


def build(out_dir=None, group="DEV", gold_dir=None, labels=None, jury=None, use_gold=True, use_distill=True,
          use_human=True, min_conf=0.9, val_frac=0.15, seed=13, lang_targets=None, weights=None, augment=None,
          recs=None, write=True):
    """Collect → dedupe → split → (train: augment, weight repetition, language balance) → JSONL + stats."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if recs is None:
        try:
            recs = data.load_group(group) if (config.DATA_DIR / f"{group}.json").exists() else []
        except Exception:   # noqa: BLE001 — offline / no data: build from gold only
            recs = []
    recs = {r["conversation_id"]: r for r in recs}
    collected, report = [], {"sources": {}, "skipped": {}}
    if use_human:
        lab_path = Path(labels) if labels else config.ROOT / "examples" / "dev_labels.json"
        h = from_labels(lab_path, recs) + from_overrides(group, recs) + from_appeals(group, recs)
        collected += h
        report["sources"]["human"] = len(h)
    if use_gold:
        g = from_gold(gold_dir)
        collected += g
        report["sources"]["gold"] = len(g)
    if use_distill:
        files = [Path(j) for j in (jury if isinstance(jury, list | tuple) else [jury])] if jury else \
            [p for p in [latest_jury_file()] if p]
        for f in files:
            d, sk = from_jury(f, recs, min_conf=min_conf)
            collected += d
            report["sources"][f"distill:{f.name}"] = len(d)
            report["skipped"][f.name] = sk
    for ex in collected:
        ex["weight"] = weights[ex["source"]]
    uniq = dedupe(collected)
    report["dedupe"] = {"collected": len(collected), "unique": len(uniq)}
    train, val = split(uniq, val_frac=val_frac, seed=seed)
    if augment:
        train = train + augment(train)
    train = balance_languages(expand_weights(train), lang_targets, seed=seed)
    cache = {}
    tr_rows, va_rows = [render(e, cache) for e in train], [render(e, cache) for e in val]
    report["train"], report["val"] = stats(tr_rows), stats(va_rows)
    report["leakage_conversations"] = sorted({r["meta"]["conversation_id"] for r in tr_rows}
                                             & {r["meta"]["conversation_id"] for r in va_rows})
    report["prompt_version"] = prompts.PROMPT_VERSION
    report["format"] = "openai-json-mode (serve with CQA_OPENAI_TOOLS=0)"
    if write:
        out = Path(out_dir or default_out_dir())
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", tr_rows), ("val", va_rows)):
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        report["out_dir"] = str(out)
    return report, tr_rows, va_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the CQA SFT dataset (JSONL, chat format) + stats.")
    ap.add_argument("--out", default=None, help="output dir (default data/finetune, git-ignored)")
    ap.add_argument("--group", default="DEV")
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--labels", default=None, help="cqa label output (default examples/dev_labels.json)")
    ap.add_argument("--jury", action="append", default=None, help="jury run(s) to distil (default: newest dev_jury_v*.json "
                    "with the current prompt version)")
    ap.add_argument("--no-gold", action="store_true")
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--kz-target", type=float, default=DEFAULT_LANG_TARGETS["KZ"])
    ap.add_argument("--en-target", type=float, default=DEFAULT_LANG_TARGETS["EN"])
    ap.add_argument("--augment", action="store_true", help="name/number swaps (+ back-translation if CQA_FT_TRANSLATOR set)")
    a = ap.parse_args(argv)
    aug = None
    if a.augment:
        from .augment import Augmenter
        aug = Augmenter.from_env().augment_examples
    rep, _, _ = build(a.out, a.group, a.gold_dir, a.labels, a.jury, not a.no_gold, not a.no_distill, not a.no_human,
                      a.min_conf, a.val_frac, a.seed, {"KZ": a.kz_target, "EN": a.en_target}, augment=aug)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


if __name__ == "__main__":
    main()

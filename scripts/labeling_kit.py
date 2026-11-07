#!/usr/bin/env python
"""Human-validation kit: stratified 50-dialog DEV sample, sealed 15-dialog held-out, per-annotator sessions.

    python scripts/labeling_kit.py                 # writes results/human/{sample,heldout_sealed,session_A,session_B}.json
    python -m cqa label --annotator A --sample results/human/session_A.json

The kit stores conversation ids only (no dialog text, no model answers). Labels are written by the human
labeller to data/human/labels_<A>.json (git-ignored: notes may quote clients). Nothing here invents labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20261108
N_SAMPLE, N_HELDOUT = 50, 15
OUT = ROOT / "results" / "human"
CODES = ["contact", "discovery", "identification", "consultation", "empathy", "dialogue", "closure", "critical"]


def system_scores(path):
    """{cid: (total, gross)} from a v1 results.json or a store-format run (used only to stratify, never shown)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in d.get("results") or d.get("dialogs") or []:
        if "score" in r:
            out[r["conversation_id"]] = (r["score"].get("final_score"), bool(r["score"].get("critical_flag")))
        else:
            out[r["conversation_id"]] = (r.get("total"), bool(r.get("gross_error")))
    return out


def score_bin(t):
    return "na" if t is None else "low" if t < 70 else "mid" if t < 90 else "high"


def stratum(rec):
    return f"{rec['kind']}-{rec.get('language') or '?'}"


def pick_sample(recs, scores, n=N_SAMPLE, seed=SEED):
    """All calls + all gross-error dialogs, then fill each language proportionally, round-robin over score bins."""
    rnd = random.Random(seed)
    ids = [r["conversation_id"] for r in recs]
    lang = {r["conversation_id"]: r.get("language") or "?" for r in recs}
    must = [r["conversation_id"] for r in recs if r["kind"] == "call" or scores.get(r["conversation_id"], (None, False))[1]]
    chosen = list(dict.fromkeys(must))[:n]
    langs = sorted(set(lang.values()))
    quota = {lg: round(n * sum(lang[i] == lg for i in ids) / len(ids)) for lg in langs}
    while sum(quota.values()) != n:                       # fix rounding drift on the biggest language
        big = max(quota, key=lambda k: quota[k])
        quota[big] += 1 if sum(quota.values()) < n else -1
    for lg in langs:
        have = sum(lang[i] == lg for i in chosen)
        bins = {}
        for i in ids:
            if lang[i] == lg and i not in chosen:
                bins.setdefault(score_bin(scores.get(i, (None,))[0]), []).append(i)
        for b in bins.values():
            rnd.shuffle(b)
        order = sorted(bins)
        while have < quota[lg] and any(bins.values()):
            for b in order:
                if bins[b] and have < quota[lg]:
                    chosen.append(bins[b].pop())
                    have += 1
    rest = [i for i in ids if i not in chosen]
    rnd.shuffle(rest)
    chosen += rest[: n - len(chosen)]
    return sorted(chosen[:n])


def pick_heldout(sample, recs, scores, k=N_HELDOUT, seed=SEED + 1):
    """Language-stratified held-out that contains >=1 call and >=1 gross-error dialog."""
    rnd = random.Random(seed)
    by = {r["conversation_id"]: r for r in recs}
    pool = list(sample)
    rnd.shuffle(pool)
    out = []
    for cond in (lambda i: by[i]["kind"] == "call", lambda i: scores.get(i, (None, False))[1]):
        x = next((i for i in pool if cond(i) and i not in out), None)
        if x:
            out.append(x)
    langs = sorted({by[i].get("language") for i in sample})
    quota = {lg: max(1, round(k * sum(by[i].get("language") == lg for i in sample) / len(sample))) for lg in langs}
    for lg in langs:
        for i in pool:
            if len(out) >= k or sum(by[j].get("language") == lg for j in out) >= quota[lg]:
                break
            if by[i].get("language") == lg and i not in out:
                out.append(i)
    for i in pool:
        if len(out) >= k:
            break
        if i not in out:
            out.append(i)
    return sorted(out[:k])


def ids_hash(ids):
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def build(recs, scores, out_dir=OUT, annotators=("A", "B")):
    from cqa import checklist as ckl
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = pick_sample(recs, scores)
    held = pick_heldout(sample, recs, scores)
    by = {r["conversation_id"]: r for r in recs}
    strata = {}
    for i in sample:
        k = f"{stratum(by[i])}/{score_bin(scores.get(i, (None,))[0])}"
        strata[k] = strata.get(k, 0) + 1
    cls = {}
    for kind in sorted({by[i]["kind"] for i in sample}):
        cl = ckl.load(ckl.pick_for_channel(kind))
        blob = json.dumps(cl, ensure_ascii=False, sort_keys=True, default=str).encode()
        cls[kind] = {"id": cl["id"], "sha256": hashlib.sha256(blob).hexdigest()[:12]}
    (out_dir / "sample.json").write_text(json.dumps({
        "group": "DEV", "seed": SEED, "n": len(sample), "ids": sample,
        "gross_in_sample": sorted(i for i in sample if scores.get(i, (None, False))[1]),
        "calls_in_sample": sorted(i for i in sample if by[i]["kind"] == "call"),
        "strata": dict(sorted(strata.items())), "heldout_sha256": ids_hash(held), "checklists": cls,
        "note": "15 of these ids are a sealed held-out (heldout_sealed.json): reports exclude them until --final."},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "heldout_sealed.json").write_text(json.dumps({
        "sealed": True, "n": len(held), "ids": held, "sha256": ids_hash(held),
        "rule": "Do not use these dialogs to tune prompts/checklists; reported only with agreement.py --final."},
        indent=1), encoding="utf-8")
    for a in annotators:
        order = list(sample)
        random.Random(f"{SEED}-{a}").shuffle(order)          # different order per annotator → less order bias
        (out_dir / f"session_{a}.json").write_text(json.dumps({
            "annotator": a, "group": "DEV", "ids": order, "labels_out": f"data/human/labels_{a}.json",
            "show_model_answers": False}, indent=1), encoding="utf-8")
    return sample, held


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="DEV")
    p.add_argument("--system", default=str(ROOT / "results" / "dev" / "results.json"),
                   help="system run used ONLY to stratify by score/gross (never shown to annotators)")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    from cqa import data
    recs = data.load_group(a.group)
    sample, held = build(recs, system_scores(a.system), a.out)
    print(f"sample {len(sample)} dialogs, sealed held-out {len(held)} → {a.out}")
    print("next: python -m cqa label --annotator A --sample results/human/session_A.json")


if __name__ == "__main__":
    main()

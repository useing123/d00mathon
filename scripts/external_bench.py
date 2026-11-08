#!/usr/bin/env python
"""External validation on a public, human-labelled customer-support dataset (MAIA-DQE).

    python scripts/external_bench.py prepare            # download MAIA-DQE, fixed-seed stratified sample (n=80, seed=42)
    python scripts/external_bench.py run                # system on THEIR rubric (ext_maia_dqe) + on freedom_chat_v1
    python scripts/external_bench.py metrics            # agreement vs human labels -> results/external/*.json + table

The evaluated input file (data/external/maia_sample.json, git-ignored) contains transcripts ONLY — the human labels are
kept in results/external/maia_sample_labels.json (ids + labels, committed) and are never shown to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cqa import ext_data  # noqa: E402

OUT = ROOT / "results" / "external"
INPUT = ROOT / "data" / "external" / "maia_sample.json"
LABELS = OUT / "maia_sample_labels.json"
RUNS = {"ext": ("ext_maia_dqe", OUT / "maia_ext_run.json"), "freedom": ("freedom_chat_v1", OUT / "maia_freedom_run.json")}
REPS, SEED = 2000, 7
# Published inter-annotator agreement (MAIA-DQE paper, Table 2: 3 annotators, 10 PT_PT dialogues, observed % full agreement).
IAA = {"TS": "30.0% full / 60.0% partial (Likert 1-5); 80% full when binarised (paper §3.3)",
       "DC": "90.0% full", "POL": "98.36% full (turn level)", "SENS": "92.12% full (turn level)"}


# ---------------- statistics ----------------
def kappa(a, b):
    n = len(a)
    if not n:
        return None
    cats = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (1.0 if po == 1 else 0.0) if pe >= 1 else (po - pe) / (1 - pe)


def qwk(a, b, scale):
    n, k = len(a), len(scale)
    if not n:
        return None
    ix = {v: i for i, v in enumerate(scale)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[ix[x]][ix[y]] += 1 / n
    pa = [sum(r) for r in obs]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return (1.0 if num == 0 else 0.0) if den == 0 else 1 - num / den


def _ranks(x):
    s = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and x[s[j + 1]] == x[s[i]]:
            j += 1
        for t in range(i, j + 1):
            r[s[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return None if va == 0 or vb == 0 else cov / (va * vb) ** 0.5


def pr(sys_v, hum_v, neg):
    """Precision / recall of the system for the (rare) negative class `neg`."""
    tp = sum(s == neg and h == neg for s, h in zip(sys_v, hum_v, strict=True))
    ps, ph = sum(s == neg for s in sys_v), sum(h == neg for h in hum_v)
    return (tp / ps if ps else None), (tp / ph if ph else None)


def boot(fn, pairs, reps=REPS, seed=SEED):
    if len(pairs) < 2:
        return None
    rnd = random.Random(seed)
    vals = []
    for _ in range(reps):
        smp = [pairs[rnd.randrange(len(pairs))] for _ in pairs]
        v = fn([p[0] for p in smp], [p[1] for p in smp])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 3), round(vals[int(0.975 * (len(vals) - 1))], 3)]


def eb_round(x):
    return None if x is None else round(x, 3)


def acc(a, b):
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None


def stat_block(pairs, kind, scale=None, neg=None):
    """pairs: [(system, human)]. kind: binary | scale."""
    s, h = [p[0] for p in pairs], [p[1] for p in pairs]
    out = {"n": len(pairs), "accuracy": acc(s, h), "accuracy_ci": boot(acc, pairs), "kappa": kappa(s, h),
           "kappa_ci": boot(kappa, pairs), "human_dist": {str(v): h.count(v) for v in sorted(set(h), key=str)},
           "system_dist": {str(v): s.count(v) for v in sorted(set(s), key=str)}}
    if kind == "scale":
        out.update({"qwk": qwk(s, h, scale), "qwk_ci": boot(lambda a, b: qwk(a, b, scale), pairs),
                    "spearman": spearman(s, h), "spearman_ci": boot(spearman, pairs),
                    "within1": acc([abs(x - y) <= 1 for x, y in pairs], [True] * len(pairs)),
                    "mae": sum(abs(x - y) for x, y in pairs) / len(pairs)})
    if neg is not None:
        p, r = pr(s, h, neg)
        out.update({"neg_class": neg, "neg_precision": p, "neg_recall": r,
                    "neg_precision_ci": boot(lambda a, b: pr(a, b, neg)[0], pairs),
                    "neg_recall_ci": boot(lambda a, b: pr(a, b, neg)[1], pairs)})
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


# ---------------- steps ----------------
def prepare(n, seed):
    recs = ext_data.load_maia()
    smp = ext_data.stratified_sample(recs, n=n, seed=seed)
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    blind = [{k: v for k, v in r.items() if k != "human_labels"} for r in smp]
    INPUT.write_text(json.dumps(blind, ensure_ascii=False, indent=1), encoding="utf-8")
    LABELS.write_text(json.dumps({
        "dataset": "MAIA-DQE (Mendonça et al., GEM 2023)", "source": "https://github.com/johndmendonca/MAIA-DQE",
        "license": ext_data.MAIA_LICENSE, "note": "ids + human labels only; transcripts are not redistributed",
        "sample": {"n": len(smp), "seed": seed, "pool": len(recs), "rule": (ext_data.stratified_sample.__doc__ or "").strip()},
        "labels": {r["conversation_id"]: {**r["human_labels"], "stratum": ext_data.maia_stratum(r["human_labels"]),
                                          "n_turns": len(r["transcript"])} for r in smp}}, indent=1), encoding="utf-8")
    print(f"sample {len(smp)} / {len(recs)} -> {INPUT} ; labels -> {LABELS}")


def run(which, preset, concurrency, model=None):
    env = {k: v for k, v in os.environ.items() if k not in ("CQA_MODEL", "CQA_ENSEMBLE", "CQA_JURORS", "CQA_JUDGE")}
    if model:   # single-model override (explicit CQA_MODEL wins over the non-ensemble preset)
        env["CQA_MODEL"], env["CQA_ENSEMBLE"] = model, "0"
    for w in which:
        cid, out = RUNS[w]
        tag = "" if preset == "jury" or model else f"_{preset}"
        raw = ROOT / "data" / "external" / f"run_{w}{tag}.json"
        out = out.with_name(out.stem + tag + ".json")
        cmd = [sys.executable, "-m", "cqa", "evaluate", "--input", str(INPUT), "--group", f"ext_maia_{w}",
               "--checklist", cid, "--out", str(raw), "--concurrency", str(concurrency)]
        if model:
            pass
        elif preset in ("econom", "balanced", "max", "jury", "cascade", "hybrid"):
            cmd += ["--preset", preset]
        else:   # self-hosted presets (local, cascade-local) are selected via CQA_PRESET (see cqa/presets.py)
            env["CQA_PRESET"] = preset
            cmd += ["--fresh"]   # the per-criterion cache is not keyed by model: never reuse cloud verdicts here
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        doc = json.loads(raw.read_text(encoding="utf-8"))
        meta = {k: doc["meta"].get(k) for k in ("cost_usd", "usage", "model", "elapsed_s", "errors")}
        slim = {"checklist": cid, "preset": None if model else preset, "model_override": model, "meta": meta,
                "dialogs": {d["conversation_id"]: {c["criterion_id"]: {"value": c.get("value"), "is_na": c.get("is_na"),
                                                                     "justification": c.get("justification"),
                                                                     "votes": c.get("votes"), "decided_by": c.get("decided_by")}
                                                    for c in d["criteria"]} for d in doc["dialogs"]}}
        out.write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{w}: cost ${doc['meta'].get('cost_usd')} -> {out}")


def _val(c):
    if c is None or c.get("is_na") or c.get("value") in (None, "NA"):
        return "NA"
    v = c["value"]
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


SPEC = {"TS": ("task_success", "scale", None), "DC": ("dropped", "binary", "NO"),
        "POL": ("agent_polite", "binary", "NO"), "SENS": ("agent_sensible", "binary", "NO")}


def ext_metrics(lab, ext, disagreements=None):
    """System on THEIR rubric vs human labels. Dialogs the system failed on (NA/missing) are counted in n_missing."""
    S = ext["dialogs"]
    ids = list(lab)
    yn = {1: "YES", 0: "NO"}
    out = {}
    for k, (hk, kind, neg) in SPEC.items():
        pairs, missing = [], 0
        for i in ids:
            h = lab[i][hk]
            h = h if k == "TS" else (yn[1 - h] if k == "DC" else yn[h])
            s = _val((S.get(i) or {}).get(k))
            if s == "NA" or h is None:
                missing += 1
                continue
            pairs.append((s, h))
            if s != h and disagreements is not None:
                disagreements.append({"id": i, "criterion": k, "human": h, "system": s,
                                      "system_rationale": (S[i][k].get("justification") or "")[:400]})
        out[k] = {**stat_block(pairs, kind, scale=[1, 2, 3, 4, 5], neg=neg), "n_missing": missing}
    tsb = [(_val(S[i]["TS"]) >= 4, lab[i]["task_success"] >= 4) for i in ids
           if i in S and isinstance(_val(S[i].get("TS")), int)]
    out["TS_binary_ge4"] = stat_block(tsb, "binary", neg=False)
    out["cost_usd"] = ext["meta"]["cost_usd"]
    out["setup"] = ext.get("model_override") or ext.get("preset")
    out["model"] = ext["meta"].get("model")
    return out


def metrics():
    lab = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    res = {"ext": {}, "freedom": {}, "iaa_ceiling": IAA}
    disagreements = []
    if RUNS["ext"][1].exists():
        res["ext"] = ext_metrics(lab, json.loads(RUNS["ext"][1].read_text(encoding="utf-8")), disagreements)
    for f in sorted(OUT.glob("maia_ext_run_*.json")):   # other setups (self-hosted: local, cascade-local, ...)
        res["ext_" + f.stem.removeprefix("maia_ext_run_")] = ext_metrics(lab, json.loads(f.read_text(encoding="utf-8")))
    fr = json.loads(RUNS["freedom"][1].read_text(encoding="utf-8")) if RUNS["freedom"][1].exists() else None
    if fr:
        F = fr["dialogs"]
        ids = [i for i in lab if i in F]
        # 04 (0..20) vs Task Success (1..5): rank correlation + "resolved" (04>=15 vs TS>=4)
        p04 = [(_val(F[i]["04"]), lab[i]["task_success"]) for i in ids if _val(F[i].get("04")) != "NA"]
        res["freedom"]["04_vs_TS_spearman"] = {"n": len(p04), "spearman": eb_round(spearman([a for a, _ in p04], [b for _, b in p04])),
                                               "spearman_ci": boot(spearman, p04),
                                               "n_na": sum(_val(F[i].get("04")) == "NA" for i in ids)}
        res["freedom"]["04ge15_vs_TSge4"] = stat_block([(s >= 15, h >= 4) for s, h in p04], "binary", neg=False)
        # 05 (politeness+empathy) / 08 (gross) vs agent impoliteness
        p05 = [(_val(F[i]["05"]), "YES" if lab[i]["agent_polite"] else "NO") for i in ids]
        res["freedom"]["05_vs_POL"] = stat_block(p05, "binary", neg="NO")
        p08 = [("NO" if _val(F[i]["08"]) == "YES" else "YES", "YES" if lab[i]["agent_polite"] else "NO") for i in ids]
        res["freedom"]["08_vs_POL(08=YES means impolite)"] = stat_block(p08, "binary", neg="NO")
        # 07 NA (dialog cut off) vs Dropped
        p07 = [(_val(F[i]["07"]) == "NA", bool(lab[i]["dropped"])) for i in ids]
        res["freedom"]["07NA_vs_DC"] = stat_block(p07, "binary", neg=True)
        res["freedom"]["cost_usd"] = fr["meta"]["cost_usd"]
    rnd = random.Random(SEED)
    res["disagreements_total"] = len(disagreements)
    res["disagreement_sample"] = sorted(rnd.sample(disagreements, min(10, len(disagreements))), key=lambda d: d["criterion"])
    (OUT / "maia_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "maia_disagreements_all.json").write_text(json.dumps(disagreements, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "disagreement_sample"}, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    r = sub.add_parser("run")
    r.add_argument("--which", default="ext,freedom")
    r.add_argument("--preset", default="jury")
    r.add_argument("--concurrency", type=int, default=6)
    r.add_argument("--model", default=None, help="single model instead of the preset (e.g. claude-haiku-5-5)")
    sub.add_parser("metrics")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.n, a.seed)
    elif a.cmd == "run":
        run(a.which.split(","), a.preset, a.concurrency, a.model)
    else:
        metrics()


if __name__ == "__main__":
    main()

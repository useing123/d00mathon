#!/usr/bin/env python3
"""Build EXPERIMENTS.md (+ docs/img/pareto.svg) from the raw result files.

Deterministic: every number is re-computed from per-dialog predictions in results/ against the labels in
examples/gold*/ (or, where no per-dialog file exists, read from a named summary file). Bootstrap uses a fixed
seed. Missing files are skipped, so the script runs on a fresh clone (results/ is mostly git-ignored) and can be
re-run whenever new runs land (e.g. results/kz_models/*_holdout.json is auto-discovered).

    python scripts/build_experiments.py [--out EXPERIMENTS.md] [--svg docs/img/pareto.svg] [--reps 10000]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SEED = 20261108
CRIT = ["01", "02", "03", "04", "05", "06", "07", "08"]
JEV_USD = None  # filled from results/jev/holdout.json

# GPU metadata for local runs whose result files do not record it. Source noted per entry.
GPU_META = {
    "q38_27b_fp8": ("Qwen3.8-27B-FP8 (local, vLLM)", "1× H100 SXM", 3.99, ".cache/runpod_pod.json at run time"),
    "issai_9b_kk": ("ISSAI Qwen3.5-9B-Kazakh (local, vLLM)", None, None, "GPU not recorded in result file"),
}


# ----------------------------------------------------------------------------------------------- loading
def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_gold(d):
    out = {}
    for p in sorted((ROOT / "examples" / d).glob("*.json")):
        g = jload(p)
        out[g["dialog"]["conversation_id"]] = g
    return out


GOLD = {"gold": load_gold("gold"), "holdout": load_gold("gold_holdout")}


def _norm_dialog(cid, x):
    """Accept a per-dialog record in any known shape -> {conversation_id, criteria[{criterion_id,value}], ...}."""
    if not isinstance(x, dict):
        return None
    d = dict(x)
    d.setdefault("conversation_id", cid)
    cr = d.get("criteria")
    if cr is None:
        cr = d.get("pred") or d.get("predicted") or d.get("values")
    if isinstance(cr, dict):  # {"01": "YES", ...} or {"01": {"value": ..}}
        cr = [{"criterion_id": k, "value": (v.get("value") if isinstance(v, dict) else v)} for k, v in cr.items()]
    if not isinstance(cr, list):
        return None
    d["criteria"] = cr
    if d.get("total") is None:
        d["total"] = d.get("pred_total", d.get("score"))
    if "gross_error" not in d:
        d["gross_error"] = d.get("pred_gross", d.get("gross"))
    return d


def dialogs_of(obj):
    """Per-dialog records from: a list; dialogs_detail / dialogs / per_dialog (list or {cid: record})."""
    if isinstance(obj, list):
        src = obj
    elif isinstance(obj, dict):
        src = next((obj[k] for k in ("per_dialog", "dialogs_detail", "dialogs")
                    if isinstance(obj.get(k), (list, dict)) and obj.get(k)), None)
    else:
        src = None
    if src is None:
        return None
    items = src.items() if isinstance(src, dict) else ((None, x) for x in src)
    out = [_norm_dialog(c, x) for c, x in items]
    return [x for x in out if x and x.get("conversation_id")] or None


def _n(v):
    return str(v).upper() if v is not None else None


def score(path, suite):
    """Per-dialog re-computation against the gold labels. Returns None if the file is missing/unusable."""
    ds = dialogs_of(jload(path)) if path else None
    gold = GOLD[suite]
    if not ds:
        return None
    by = {d["conversation_id"]: d for d in ds if d.get("conversation_id") in gold}
    if not by:
        return None
    per = {}
    for cid, g in gold.items():
        d = by.get(cid)
        if d is None:
            continue
        pred = {c["criterion_id"]: c.get("value") for c in d.get("criteria", [])}
        labels = {k: (_n(pred.get(k)) == _n(ev), ev, pred.get(k)) for k, ev in g["expected"].items()}
        pt = d.get("total") if d.get("total") is not None else d.get("provisional_total")
        ae = abs(pt - g["expected_total"]) if pt is not None and g.get("expected_total") is not None else None
        per[cid] = {"labels": labels, "ae": ae, "eg": bool(g.get("expected_gross_error")),
                    "pg": bool(d.get("gross_error"))}
    return {"path": str(path), "suite": suite, "per": per}


def agg(run):
    per = run["per"]
    n = sum(len(x["labels"]) for x in per.values())
    m = sum(ok for x in per.values() for ok, _, _ in x["labels"].values())
    aes = [x["ae"] for x in per.values() if x["ae"] is not None]
    tp = sum(x["eg"] and x["pg"] for x in per.values())
    fp = sum(x["pg"] and not x["eg"] for x in per.values())
    fn = sum(x["eg"] and not x["pg"] for x in per.values())
    crit = {k: [0, 0] for k in CRIT}
    na = {"NA→app": 0, "app→NA": 0}
    for x in per.values():
        for k, (ok, ev, pv) in x["labels"].items():
            crit[k][0] += ok
            crit[k][1] += 1
            if _n(ev) == "NA" and _n(pv) != "NA":
                na["NA→app"] += 1
            if _n(ev) != "NA" and _n(pv) == "NA":
                na["app→NA"] += 1
    return {"dialogs": len(per), "n": n, "match": m, "acc": m / n if n else None,
            "mae": sum(aes) / len(aes) if aes else None,
            "P": tp / (tp + fp) if tp + fp else None, "R": tp / (tp + fn) if tp + fn else None,
            "crit": crit, "na": na}


# ----------------------------------------------------------------------------------------------- stats
def dialog_acc(runs):
    """cid -> mean (over runs) share of correct labels; restricted to dialogs present in every run."""
    common = set.intersection(*[set(r["per"]) for r in runs])
    return {c: statistics.mean(sum(ok for ok, _, _ in r["per"][c]["labels"].values()) / len(r["per"][c]["labels"])
                               for r in runs) for c in sorted(common)}


def boot_ci(vals, reps, seed=SEED):
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return (None, None)
    s = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return s[int(0.025 * reps)], s[int(0.975 * reps) - 1]


def paired(runs_a, runs_b, reps):
    a, b = dialog_acc(runs_a), dialog_acc(runs_b)
    ids = sorted(set(a) & set(b))
    diff = [a[i] - b[i] for i in ids]
    rng = random.Random(SEED)
    n = len(diff)
    bs = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo, hi = bs[int(0.025 * reps)], bs[int(0.975 * reps) - 1]
    p_boot = min(1.0, 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / reps)
    # exact McNemar on labels, first run of each
    ra, rb = runs_a[0]["per"], runs_b[0]["per"]
    bb = cc = 0
    for i in ids:
        for k in ra[i]["labels"]:
            x, y = ra[i]["labels"][k][0], rb[i]["labels"].get(k, (False,))[0]
            bb += x and not y
            cc += y and not x
    k, m = min(bb, cc), bb + cc
    p_mc = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    return {"n": n, "delta": sum(diff) / n if n else 0, "lo": lo, "hi": hi, "p_boot": p_boot,
            "b": bb, "c": cc, "p_mcnemar": p_mc}


# ----------------------------------------------------------------------------------------------- registry
def bench_cost(cfg):
    usd = dl = wall = 0.0
    for p in glob.glob(str(R / "bench" / f"{glob.escape(cfg)}__*.json")):
        if p.endswith(".dialogs.json"):
            continue
        s = jload(p) or {}
        if s.get("suite") not in ("gold", "holdout") or not s.get("dialogs"):
            continue
        usd += s.get("usd") or 0
        dl += s["dialogs"]
        wall += s.get("wall_s") or 0
    return (usd / dl, wall / dl) if dl else (None, None)


def family_of(cfg):
    if cfg.startswith("jury"):
        return "cloud jury"
    return "cloud single"


def build_registry():
    global JEV_USD
    regs = []
    bench = R / "bench"
    cfgs = sorted({Path(p).name.split("__")[0] for p in glob.glob(str(bench / "*__*.dialogs.json"))})
    for c in cfgs:
        ho = sorted(glob.glob(str(bench / f"{glob.escape(c)}__holdout*.dialogs.json")))
        go = sorted(glob.glob(str(bench / f"{glob.escape(c)}__gold.dialogs.json")))
        usd, sec = bench_cost(c)
        regs.append(dict(name=c, family=family_of(c) + (" (prompt " + c.split("@")[1] + ")" if "@" in c else ""),
                         holdout=ho, gold=go, usd=usd, sec=sec, gpu="—", src=ho[0] if ho else (go[0] if go else "")))
    js = jload(R / "jev" / "holdout.json") or {}
    JEV_USD = js.get("usd_per_dialog")
    for sub, nm in (("", "Jev 1.14 alone (full rules)"), ("truncated700/", "Jev 1.14 alone (rules ≤700 chars)")):
        ho = sorted(glob.glob(str(R / "jev" / sub / "holdout*.dialogs.json")))
        go = sorted(glob.glob(str(R / "jev" / sub / "gold*.dialogs.json")))
        ss = [jload(p.replace(".dialogs.json", ".json")) or {} for p in ho]
        usd = statistics.mean([s["usd_per_dialog"] for s in ss if s.get("usd_per_dialog")] or [0]) or None
        sec = statistics.mean([s["wall_s"] / s["dialogs"] for s in ss if s.get("wall_s")] or [0]) or None
        regs.append(dict(name=nm, family="decision model (API)", holdout=ho, gold=go, usd=usd, sec=sec, gpu="—",
                         src=ho[0] if ho else ""))

    def oss(path, name, family, gold=None, gpu="—", gpu_hr=None):
        d = jload(path)
        if not d:
            return
        cloud = (d.get("routing") or {}).get("cloud_usd") or 0
        dl = d.get("dialogs") or 10
        usd = cloud / dl
        if (d.get("settings") or {}).get("first") == "jev" and JEV_USD:
            usd += JEV_USD
        if gpu_hr:
            usd += gpu_hr * d.get("wall_s", 0) / 3600 / dl
        elif gpu != "—":
            usd = None  # local GPU price unknown -> do not report a fake $0
        ts = (d.get("routing") or {}).get("tier_share") or {}
        regs.append(dict(name=name, family=family, holdout=[str(path)], gold=[gold] if gold else [], usd=usd,
                         sec=d.get("sec_per_dialog"), gpu=gpu, src=str(path),
                         tiers="/".join(f"{100 * ts.get(k, 0):.0f}" for k in "123") if ts else ""))

    oss(R / "oss" / "jev3_holdout.json", "cascade Jev → haiku → sonnet (preset `cascade`)", "cascade")
    oss(R / "oss" / "jev_haiku_holdout.json", "cascade Jev → haiku", "cascade")
    oss(R / "oss" / "q8_3_holdout.json", "cascade Qwen3-8B-AWQ → haiku → sonnet", "cascade (local tier 1)",
        gpu="1× A40", gpu_hr=0.59)
    for p in sorted(glob.glob(str(R / "kz_models" / "*.json"))):
        d = jload(p)
        if not isinstance(d, dict) or not dialogs_of(d):
            continue
        if "__" in Path(p).stem or d.get("method"):  # shared schema: <method>__<model>.json
            stem = Path(p).stem
            meth = d.get("method") or stem.split("__")[0]
            mod = d.get("model") or (stem.split("__", 1)[1] if "__" in stem else "?")
            suite = d.get("suite") if d.get("suite") in ("gold", "holdout") else "holdout"
            dl = d.get("dialogs") if isinstance(d.get("dialogs"), int) else len(dialogs_of(d))
            usd = d.get("usd_per_dialog")
            if usd is None and d.get("price_hr") and d.get("wall_s"):
                usd = d["price_hr"] * d["wall_s"] / 3600 / dl
            if usd is None and d.get("usd") is not None:
                usd = d["usd"] / dl
            sec = d.get("sec_per_dialog") or (d["wall_s"] / dl if d.get("wall_s") else None)
            regs.append(dict(name=d.get("label") or f"{mod} [{meth}]", family=d.get("family") or f"open/decision ({meth})",
                             holdout=[p] if suite == "holdout" else [], gold=[p] if suite == "gold" else [],
                             usd=usd, sec=sec, gpu=d.get("gpu") or "n/r", src=p))
            continue
        tag = d.get("tag") or Path(p).stem.replace("_holdout", "")
        meta = d.get("meta") or {}
        nm, gpu, hr, _ = GPU_META.get(tag, (None, None, None, None))
        nm = meta.get("label") or nm or f"{tag} (local, {(d.get('settings') or {}).get('local_model', '?')})"
        gpu = meta.get("gpu") or gpu or "n/r"
        hr = meta.get("price_hr") or hr
        esc = (d.get("routing") or {}).get("escalated_share")
        fam = "local OSS" if not esc else "cascade (local tier 1)"
        oss(p, nm, fam, gpu=gpu, gpu_hr=hr)
    # older local single-model runs
    q32 = jload(R / "bench_qwen3-32b-awq.json") or {}
    dv = q32.get("dev") or {}
    regs.append(dict(name="Qwen3-32B-AWQ alone (local, vLLM)", family="local OSS",
                     holdout=[str(R / "bench_qwen3-32b-awq_gold_holdout.json")],
                     gold=[str(R / "bench_qwen3-32b-awq_gold.json")], usd=dv.get("gpu_usd_per_dialog"),
                     sec=dv.get("sec_per_dialog"), gpu="1× A40", src=str(R / "bench_qwen3-32b-awq_gold_holdout.json"),
                     note="$/s from its DEV run"))
    regs.append(dict(name="Qwen3-8B-AWQ alone (local, vLLM)", family="local OSS",
                     holdout=[str(R / "bench_oss_qwen3-8b_gold_holdout.json")],
                     gold=[str(R / "bench_oss_qwen3-8b_gold.json")], usd=0.00085, sec=5.2, gpu="1× A40",
                     src=str(R / "bench_oss_qwen3-8b_gold_holdout.json"), note="$/s from docs/OSS_ROUTER.md"))
    fs = jload(R / "finetune" / "summary.json") or {}
    for k, nm in (("base", "Qwen3-8B bf16 base (JSON mode)"), ("lora", "Qwen3-8B + QLoRA adapter (16 steps)")):
        w = (((fs.get("results") or {}).get("holdout") or {}).get(k) or {}).get("wall_s")
        regs.append(dict(name=nm, family="fine-tune", holdout=[str(R / "finetune" / f"holdout_{k}.json")],
                         gold=[str(R / "finetune" / f"gold_{k}.json")],
                         usd=0.27 * w / 3600 / 10 if w else None, sec=w / 10 if w else None, gpu="1× RTX A5000",
                         src=str(R / "finetune" / f"holdout_{k}.json"), note="base+adapter served concurrently"))
    for k, nm in (("cascade", "cascade Jev→haiku→sonnet, KZ signals OFF (hybrid_kz run)"),
                  ("hybrid", "hybrid: KZ parser signals ON → cascade")):
        p = R / "hybrid_kz" / f"{k}_gold_holdout.json"
        regs.append(dict(name=nm, family="cascade", holdout=[str(p)], gold=[str(R / "hybrid_kz" / f"{k}_gold.json")],
                         usd=None, sec=None, gpu="—", src=str(p), note="cost not recorded (≈ cascade)"))
    # internal jury code-version runs (same preset, evolving prompt/code) — used for variance + ablations
    for f, nm in JURY_VERSIONS:
        regs.append(dict(name=nm, family="jury code version", holdout=[str(R / f)], gold=[], usd=None, sec=None,
                         gpu="—", src=str(R / f)))
    merged = {}
    for r in regs:
        m = merged.get(r["name"])
        if m is None:
            merged[r["name"]] = r
        else:
            m["holdout"] += r["holdout"]
            m["gold"] += r["gold"]
            for k in ("usd", "sec"):
                m[k] = m.get(k) if m.get(k) is not None else r.get(k)
    regs = list(merged.values())
    out = []
    for r in regs:
        r["H"] = [x for x in (score(p, "holdout") for p in r["holdout"]) if x]
        r["G"] = [x for x in (score(p, "gold") for p in r["gold"]) if x]
        if r["H"] or r["G"]:
            out.append(r)
    return out


JURY_VERSIONS = [
    ("gold_holdout_report.json", "jury v8-prompt, default (EVALUATION held-out)"),
    ("gold_holdout_confirm_report.json", "jury v8-prompt, --confirm 03,04"),
    ("gold_holdout_v9.json", "jury prompt v9 run a"),
    ("gold_holdout_v9b.json", "jury prompt v9 run b"),
    ("gold_holdout_final.json", "jury --final (sonnet+opus, median-of-3 per juror)"),
    ("gold_holdout_v10.json", "jury prompt v10 (blind judge)"),
    ("gold_holdout_v11.json", "jury v6 code (120 s wait, single judge sample)"),
    ("gold_holdout_v12.json", "jury v7 code (180 s, median-of-3 judge, rubric alignment)"),
]


# ----------------------------------------------------------------------------------------------- formatting
def f3(x):
    return "—" if x is None else f"{x:.3f}"


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def money(x):
    if x is None:
        return "—"
    return f"${x:.5f}" if x < 0.01 else f"${x:.4f}" if x < 1 else f"${x:.2f}"


def link(p):
    if not p:
        return "—"
    rel = str(Path(p).resolve().relative_to(ROOT))
    return f"[{Path(rel).name}]({quote(rel)})"


def summarize(r, reps):
    if not r["H"]:
        r["acc"] = None
        return r
    ags = [agg(x) for x in r["H"]]
    r["runs"] = [a["acc"] for a in ags]
    r["acc"] = statistics.mean(r["runs"])
    r["ci"] = boot_ci(list(dialog_acc(r["H"]).values()), reps)
    r["mae"] = statistics.mean([a["mae"] for a in ags if a["mae"] is not None] or [float("nan")])
    r["P"], r["R"] = ags[0]["P"], ags[0]["R"]
    crit = {k: [sum(a["crit"][k][0] for a in ags), sum(a["crit"][k][1] for a in ags)] for k in CRIT}
    r["crit"] = {k: (m / n if n else None) for k, (m, n) in crit.items()}
    r["na"] = ags[0]["na"]
    r["gacc"] = statistics.mean([agg(x)["acc"] for x in r["G"]]) if r["G"] else None
    # KZ subset of held-out (dialog ids containing -kz-)
    kz = [agg({"per": {c: v for c, v in x["per"].items() if "-kz-" in c}}) for x in r["H"]]
    r["kz"] = statistics.mean([a["acc"] for a in kz if a["acc"] is not None] or [float("nan")])
    return r


# ----------------------------------------------------------------------------------------------- DEV vs jury
def dev_agreement(reps):
    ref = dialogs_of(jload(R / "dev_jury_v7.json"))
    if not ref:
        return []
    refm = {d["conversation_id"]: {c["criterion_id"]: _n(c.get("value")) for c in d["criteria"]} for d in ref}
    reft = {d["conversation_id"]: d.get("total") for d in ref}
    rows = []
    cand = [("Jev → haiku → sonnet cascade", R / "oss" / "jev3_dev.dialogs.json"),
            ("gpt-6-luna", R / "bench" / "gpt-6-luna__dev.dialogs.json"),
            ("haiku", R / "bench" / "haiku__dev.dialogs.json"),
            ("gpt-5.3-codex (DEV-20)", R / "bench" / "gpt-5.3-codex__dev20.dialogs.json"),
            ("Qwen3-32B-AWQ local", R / "bench_qwen3-32b-awq_dev.json"),
            ("jury v6 code (same preset)", R / "dev_jury_v6.json"),
            ("jury v5 code (same preset)", R / "dev_jury_v5.json"),
            ("jury v4 code (same preset)", R / "dev_jury_v4.json")]
    for nm, p in cand:
        ds = dialogs_of(jload(p))
        if not ds:
            continue
        per, aes = {}, []
        for d in ds:
            cid = d["conversation_id"]
            if cid not in refm:
                continue
            pr = {c["criterion_id"]: _n(c.get("value")) for c in d.get("criteria", [])}
            per[cid] = sum(pr.get(k) == v for k, v in refm[cid].items()) / len(refm[cid])
            if d.get("total") is not None and reft.get(cid) is not None:
                aes.append(abs(d["total"] - reft[cid]))
        if not per:
            continue
        v = list(per.values())
        rows.append((nm, len(v), sum(v) / len(v), boot_ci(v, reps), sum(aes) / len(aes) if aes else None, p))
    return rows


# ----------------------------------------------------------------------------------------------- errors
STEP = {"04": 5, "06": 3}


def classify(k, ev, pv):
    e, p = _n(ev), _n(pv)
    if p is None:
        return "failed / missing"
    if (e == "NA") != (p == "NA"):
        return "NA vs applicable"
    if k in STEP:
        try:
            return "scale: adjacent step" if abs(int(e) - int(p)) <= STEP[k] else "scale: ≥2 steps"
        except ValueError:
            pass
    return "binary flip"


def excerpt(g, k, maxc=170):
    tr = g["dialog"]["transcript"]
    ops = [t for t in tr if t.get("speaker") == "operator"]
    pick = ops[-1:] if k == "07" else ops[:1] if k == "01" else sorted(ops, key=lambda t: -len(t["text"]))[:2]
    cl = [t for t in tr if t.get("speaker") == "client"][:1]
    out = []
    for t in cl + pick:
        s = t["text"].replace("\n", " ")
        out.append(f"> **{t['speaker']}** [{t.get('timestamp', '')}]: {s[:maxc]}{'…' if len(s) > maxc else ''}")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- SVG
def pareto_svg(points, path):
    W, H, L, B, T, RM = 860, 520, 70, 60, 30, 230
    xs = [math.log10(p[1]) for p in points]
    ys = [p[2] for p in points]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
    y0, y1 = math.floor(min(ys) * 20) / 20, 1.0
    X = lambda v: L + (math.log10(v) - x0) / (x1 - x0) * (W - L - RM)  # noqa: E731
    Y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)  # noqa: E731
    front = pareto_front(points)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for e in range(x0, x1 + 1):
        x = X(10 ** e)
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="#ddd"/>')
        o.append(f'<text x="{x:.1f}" y="{H - B + 16}" text-anchor="middle">${10 ** e:g}</text>')
    y = y0
    while y <= y1 + 1e-9:
        o.append(f'<line x1="{L}" y1="{Y(y):.1f}" x2="{W - RM}" y2="{Y(y):.1f}" stroke="#ddd"/>')
        o.append(f'<text x="{L - 6}" y="{Y(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
        y += 0.05
    o.append(f'<text x="{(L + W - RM) / 2}" y="{H - 15}" text-anchor="middle">$ per dialog (log scale)</text>')
    o.append(f'<text x="18" y="{(T + H - B) / 2}" transform="rotate(-90 18 {(T + H - B) / 2})" '
             'text-anchor="middle">held-out exact agreement (80 labels)</text>')
    fp = sorted(front, key=lambda p: p[1])
    o.append('<polyline fill="none" stroke="#d33" stroke-width="1.5" points="'
             + " ".join(f"{X(p[1]):.1f},{Y(p[2]):.1f}" for p in fp) + '"/>')
    for i, (nm, c, a, fam) in enumerate(sorted(points, key=lambda p: p[1])):
        on = (nm, c, a, fam) in front
        o.append(f'<circle cx="{X(c):.1f}" cy="{Y(a):.1f}" r="{5 if on else 3.5}" '
                 f'fill="{"#d33" if on else "#579"}"><title>{nm}: {a:.3f} @ ${c:.5f}</title></circle>')
        o.append(f'<text x="{X(c) + 6:.1f}" y="{Y(a) - 4 - (i % 3) * 9:.1f}" font-size="9">{i + 1}</text>')
    for i, (nm, c, a, fam) in enumerate(sorted(points, key=lambda p: p[1])):
        o.append(f'<text x="{W - RM + 10}" y="{T + 10 + i * 13}" font-size="9" '
                 f'fill="{"#d33" if (nm, c, a, fam) in front else "#333"}">{i + 1}. '
                 f'{nm.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:40]}</text>')
    o.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(o) + "\n", encoding="utf-8")


def pareto_front(points):
    front, best = [], -1
    for p in sorted(points, key=lambda p: (p[1], -p[2])):
        if p[2] > best + 1e-12:
            front.append(p)
            best = p[2]
    return front


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "EXPERIMENTS.md"))
    ap.add_argument("--svg", default=str(ROOT / "docs" / "img" / "pareto.svg"))
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    reps = a.reps
    regs = [summarize(r, reps) for r in build_registry()]
    by = {r["name"]: r for r in regs}
    ho = sorted([r for r in regs if r["acc"] is not None], key=lambda r: (-r["acc"], r.get("mae") or 99))
    L = []
    w = L.append

    def get(name):
        return by.get(name)

    jury = get("jury-haiku+sonnet>opus")
    casc = get("cascade Jev → haiku → sonnet (preset `cascade`)")
    jev = get("Jev 1.14 alone (full rules)")
    q38 = next((r for r in regs if r["name"].startswith("Qwen3.8-27B")), None)
    issai = next((r for r in regs if r["name"].startswith("ISSAI")), None)
    ftb, ftl = get("Qwen3-8B bf16 base (JSON mode)"), get("Qwen3-8B + QLoRA adapter (16 steps)")
    hyb_on = get("hybrid: KZ parser signals ON → cascade")
    hyb_off = get("cascade Jev→haiku→sonnet, KZ signals OFF (hybrid_kz run)")
    codex = get("gpt-5.3-codex")
    pairs = []

    def P(nmA, A, nmB, B, why):
        if A and B and A["H"] and B["H"]:
            pairs.append((nmA, nmB, paired(A["H"], B["H"], reps), why, A, B))
            return pairs[-1][2]
        return None

    p_cj = P("cascade Jev→haiku→sonnet", casc, "jury haiku+sonnet→opus (2 runs)", jury, "main cost claim")
    p_cc = P("gpt-5.3-codex (2 runs)", codex, "jury haiku+sonnet→opus (2 runs)", jury, "best single model vs jury")
    p_qj = P("Qwen3.8-27B-FP8 local", q38, "Jev 1.14 (3 runs)", jev, "local vs decision model")
    p_qjury = P("Qwen3.8-27B-FP8 local", q38, "jury haiku+sonnet→opus (2 runs)", jury, "local vs jury")
    p_iq = P("ISSAI 9B Kazakh", issai, "Qwen3.8-27B-FP8", q38, "KZ-specialised vs general")
    p_ft = P("QLoRA adapter", ftl, "Qwen3-8B base", ftb, "fine-tune effect")
    p_hy = P("hybrid (KZ signals ON)", hyb_on, "cascade (signals OFF)", hyb_off, "parser signals effect")
    P("Jev 1.14 (3 runs)", jev, "jury haiku+sonnet→opus (2 runs)", jury, "decision model vs jury")
    P("jury v7 code", get("jury v7 code (180 s, median-of-3 judge, rubric alignment)"), "jury v6 code",
      get("jury v6 code (120 s wait, single judge sample)"), "v7 bundle")
    P("cascade Qwen3-8B tier 1", get("cascade Qwen3-8B-AWQ → haiku → sonnet"), "cascade Jev tier 1", casc,
      "tier-1 choice")
    P("Jev full rules (3 runs)", jev, "Jev rules ≤700 (3 runs)", get("Jev 1.14 alone (rules ≤700 chars)"),
      "rule truncation")

    def sig(p):
        if p is None:
            return "n/a"
        return "significant (p<0.05)" if max(p["p_boot"], p["p_mcnemar"]) < 0.05 else "**not significant**"

    def acc_ci(r):
        if not r or r.get("acc") is None:
            return "n/a"
        return f"{r['acc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"

    devrows = dev_agreement(reps)
    dev_casc = next((x for x in devrows if x[0].startswith("Jev")), None)
    svg = Path(a.svg)
    pts = [(r["name"], r["usd"], r["acc"], r["family"]) for r in ho
           if r.get("usd") and r["family"] != "jury code version"]
    if pts:
        pareto_svg(pts, svg)
    front = {p[0] for p in pareto_front(pts)}

    # ------------------------------------------------------------------ header + executive summary
    w("# EXPERIMENTS — CQA scoring of Freedom Broker support dialogs\n")
    w("_Generated by `python scripts/build_experiments.py` from the raw files in `results/` and the labels in "
      "`examples/gold*/`. Do not edit by hand; re-run the script when new runs land. Bootstrap: "
      f"{reps:,} resamples of dialogs, seed {SEED}._\n")
    w("## 1. Executive summary\n")
    w("All accuracies are exact agreement with **our own synthetic labels** on the 10-dialog held-out set "
      "(80 labels, 1 label = 1.25 pp). Brackets = 95 % bootstrap CI over dialogs. With n = 10 almost no pairwise "
      "difference among the top configurations is statistically significant.\n")
    ex = []
    if casc and jury and p_cj:
        ratio = jury["usd"] / casc["usd"] if casc.get("usd") and jury.get("usd") else None
        dv = (f"; on 65 real DEV dialogs it agrees with the v7 jury on {dev_casc[2]:.3f} of labels"
              if dev_casc else "")
        ex.append(f"**Cascade ≈ jury at a fraction of the cost.** Jev→haiku→sonnet: {acc_ci(casc)} at "
                  f"{money(casc['usd'])}/dialog vs the Claude jury {acc_ci(jury)} at {money(jury['usd'])}/dialog "
                  f"(≈{ratio:.0f}× cheaper on held-out; ≈5–17× incl. DEV, docs/OSS_ROUTER.md){dv}. "
                  f"Paired Δ = {p_cj['delta']:+.3f} [{p_cj['lo']:+.3f}, {p_cj['hi']:+.3f}], {sig(p_cj)}.")
    if codex and p_cc:
        ex.append(f"**Best single cloud model:** gpt-5.3-codex {acc_ci(codex)} (runs "
                  f"{', '.join(f'{x:.3f}' for x in codex['runs'])}), MAE {f2(codex['mae'])}, "
                  f"{money(codex['usd'])}/dialog. Vs jury: Δ = {p_cc['delta']:+.3f}, {sig(p_cc)}.")
    if q38 and p_qj and p_qjury:
        ex.append(f"**A local open-weight model reaches the jury band with no cloud call.** Qwen3.8-27B-FP8 on "
                  f"{q38['gpu']}: {acc_ci(q38)}, MAE {f2(q38['mae'])}, {f2(q38.get('sec'))} s/dialog, "
                  f"≈{money(q38['usd'])}/dialog GPU time. Vs jury Δ = {p_qjury['delta']:+.3f} ({sig(p_qjury)}); "
                  f"vs Jev Δ = {p_qj['delta']:+.3f} ({sig(p_qj)}). Single run.")
    if issai and q38 and p_iq:
        ex.append(f"**Kazakh-specialised ≠ better here.** ISSAI 9B Kazakh {acc_ci(issai)} vs general Qwen3.8-27B "
                  f"{acc_ci(q38)} (Δ = {p_iq['delta']:+.3f}, {sig(p_iq)}); the KZ subset is only 2 dialogs "
                  f"(ISSAI {f3(issai['kz'])} vs {f3(q38['kz'])}), and the models also differ in size (9B vs 27B).")
    if ftb and ftl and p_ft:
        ex.append(f"**Fine-tune after 0.32 epoch: MAE ↓, agreement =.** QLoRA Qwen3-8B {f3(ftl['acc'])} vs base "
                  f"{f3(ftb['acc'])} (Δ = {p_ft['delta']:+.3f}, {sig(p_ft)}); total MAE {f2(ftb['mae'])} → "
                  f"{f2(ftl['mae'])}. Still far below Jev 1.14 ({f3(jev['acc']) if jev else 'n/a'}).")
    if hyb_on and hyb_off and p_hy:
        ex.append(f"**Kazakh parser signals: no measurable gain.** Hybrid {f3(hyb_on['acc'])} vs cascade "
                  f"{f3(hyb_off['acc'])} on held-out (Δ = {p_hy['delta']:+.3f}, {sig(p_hy)}); KZ subset "
                  f"{f3(hyb_on['kz'])} vs {f3(hyb_off['kz'])}.")
    ex.append("**Noise floor:** repeated held-out runs of the same configuration differ by 0–2 labels "
              "(0–2.5 pp, §3.2), the same size as most claimed differences.")
    for i, e in enumerate(ex, 1):
        w(f"{i}. {e}")
    w("")

    # ------------------------------------------------------------------ protocol
    ng, nh = len(GOLD["gold"]), len(GOLD["holdout"])
    lang = lambda s: ", ".join(f"{L_} {sum(f'-{L_.lower()}-' in c for c in GOLD[s])}" for L_ in ("RU", "KZ", "EN"))  # noqa: E731
    w("## 2. Evaluation protocol\n")
    w("| set | dialogs | labels | languages | labelled by | used for |")
    w("|---|---|---|---|---|---|")
    w(f"| tuned gold `examples/gold/` | {ng} | {8 * ng} | {lang('gold')} | us (rule authors), **synthetic** | "
      "prompt/rule development, threshold tuning, prompt-variant selection |")
    w(f"| held-out `examples/gold_holdout/` | {nh} | {8 * nh} | {lang('holdout')} | us, **synthetic**, written "
      "after tuning | reporting only; model selection (so no longer pristine) |")
    w("| DEV `results/dev_jury_v7.json` | 65 | 520 | real RU/KZ/EN chats + calls | **no human labels** | "
      "agreement with our own jury (consistency, not accuracy) |\n")
    w("Rules:")
    w("- Prompts, rubric text, router thresholds and prompt variants are tuned **only on the tuned gold**; "
      "held-out is run once per configuration, decisions taken before looking (docs/MODELS.md protocol).")
    w("- Leakage controls: `tests/test_models_bench.py` checks few-shot variants for verbatim held-out lines; the "
      "fine-tune dataset builder refuses `examples/gold_holdout` and splits by conversation "
      "(`leakage_conversations = []`); the tuned gold **is** in the fine-tune train set, so gold numbers for the "
      "adapter only measure fit.")
    w("- Known contamination: one general 04/03 rule was added after a held-out mismatch (hold-02), the Jev "
      "rule-truncation choice was made after seeing both sets, and held-out was used to rank models "
      "(docs/HISTORY.md, docs/JEV.md, docs/MODELS.md).\n")
    w("| metric | definition |")
    w("|---|---|")
    w("| per-criterion agreement | exact match of value (YES/NO/NA or scale step) per criterion |")
    w("| overall agreement | matched labels / all labels (8 × dialogs); runs averaged when repeated |")
    w("| total MAE | mean |predicted 0–100 total − expected_total| per dialog (scoring in code) |")
    w("| gross P / R | precision / recall of the criterion-08 gross-error flag (held-out has 1 positive) |")
    w("| NA confusion | expected NA predicted applicable, and vice versa |")
    w("| $/dialog | provider-reported tokens × `cqa/pricing.py`; local = GPU $/h × wall ÷ dialogs |")
    w("| s/dialog | wall-clock ÷ dialogs at the run's concurrency (8–16), not single-call latency |\n")

    # ------------------------------------------------------------------ statistics
    w("## 3. Statistics\n")
    w("### 3.1 Paired comparisons (held-out, per-dialog)\n")
    w("Δ = mean per-dialog agreement A − B (runs averaged per dialog), 95 % paired-bootstrap CI and two-sided "
      "bootstrap p; McNemar exact test on the 80 labels of the first run of each (b = A right/B wrong, "
      "c = reverse; treats labels as independent, which is optimistic).\n")
    w("| A | B | purpose | dialogs | Δ agreement [95 % CI] | p (boot) | b / c | p (McNemar) | verdict |")
    w("|---|---|---|---|---|---|---|---|---|")
    for nmA, nmB, p, why, _a, _b in pairs:
        w(f"| {nmA} | {nmB} | {why} | {p['n']} | {p['delta']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | "
          f"{p['p_boot']:.3f} | {p['b']} / {p['c']} | {p['p_mcnemar']:.3f} | {sig(p)} |")
    w("")
    w("### 3.2 Run-to-run variance (same configuration, repeated held-out runs)\n")
    w("| configuration | runs | per-run agreement | range (labels of 80) |")
    w("|---|---|---|---|")
    for r in ho:
        if len(r.get("runs", [])) > 1:
            rng_ = (max(r["runs"]) - min(r["runs"])) * 80
            w(f"| {r['name']} | {len(r['runs'])} | {' / '.join(f'{x:.3f}' for x in r['runs'])} | {rng_:.0f} |")
    jv = [r for r in regs if r["family"] == "jury code version" and r["acc"] is not None]
    if jv:
        w(f"| jury haiku+sonnet→opus across {len(jv)} code/prompt versions (§6) | {len(jv)} | "
          f"{min(r['acc'] for r in jv):.3f}–{max(r['acc'] for r in jv):.3f} | "
          f"{(max(r['acc'] for r in jv) - min(r['acc'] for r in jv)) * 80:.0f} |")
    w("\nRule of thumb used throughout: **±2 labels of 80 (±0.025) is noise.** Providers' temperature is not "
      "settable for some models (Anthropic endpoint rejects it; reasoning models), so repeats are genuinely "
      "stochastic.\n")
    w("### 3.3 Interval width\n")
    if ho:
        widths = [r["ci"][1] - r["ci"][0] for r in ho]
        w(f"Median 95 % CI width on held-out: **{statistics.median(widths):.3f}** "
          f"(range {min(widths):.3f}–{max(widths):.3f}). Configurations whose CIs overlap this much cannot be "
          "ranked from this set; the leaderboard order is descriptive.\n")

    # ------------------------------------------------------------------ leaderboard
    w("## 4. Master leaderboard\n")
    w("Every configuration with per-dialog results on disk. Sorted by held-out agreement (mean over runs). "
      "Tuned-gold is optimistic (development set). ★ = on the Pareto frontier (accuracy vs $/dialog). "
      "Cascade tier shares = % of criteria decided by tier 1/2/3.\n")
    w("| # | configuration | family | held-out [95 % CI] | runs | tuned gold | MAE (held-out) | gross P/R | "
      "NA→app / app→NA | $/dialog | s/dialog | GPU | tiers | source |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ho, 1):
        star = " ★" if r["name"] in front else ""
        pr = f"{f2(r['P'])}/{f2(r['R'])}"
        w(f"| {i} | {r['name']}{star} | {r['family']} | {acc_ci(r)} | {len(r['runs'])} | {f3(r['gacc'])} | "
          f"{f2(r['mae'])} | {pr} | {r['na']['NA→app']}/{r['na']['app→NA']} | {money(r.get('usd'))} | "
          f"{f2(r.get('sec'))} | {r.get('gpu', '—')} | {r.get('tiers', '')} | {link(r['src'])} |")
    w("")
    gonly = [r for r in regs if r["acc"] is None and r["G"]]
    if gonly:
        w("Tuned-gold-only runs (prompt variants not promoted to held-out): " + "; ".join(
            f"{r['name']} {statistics.mean(agg(x)['acc'] for x in r['G']):.3f}" for r in gonly) + ".\n")
    if pts:
        rel = Path(a.svg).resolve()
        try:
            rel = rel.relative_to(ROOT)
        except ValueError:
            pass
        w(f"![Pareto: held-out agreement vs $/dialog]({rel})\n")
        w("Pareto frontier (cheapest-first): " + " → ".join(
            f"{p[0]} ({p[2]:.3f}, {money(p[1])})" for p in sorted(pareto_front(pts), key=lambda p: p[1])) + ".\n")
    if devrows:
        w("### 4.1 DEV (65 real dialogs): agreement with the v7 jury — consistency, NOT accuracy\n")
        w("| configuration | dialogs | agreement with jury v7 [95 % CI] | total MAE vs jury | source |")
        w("|---|---|---|---|---|")
        for nm, n, ag, ci, mae, p in devrows:
            w(f"| {nm} | {n} | {ag:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {f2(mae)} | {link(p)} |")
        w("\nOlder jury code versions vs v7 measure prompt/rubric drift of the reference itself. Older runs were "
          "scored against older references in docs/MODELS.md (v4/v5); here everything is re-computed against v7, "
          "so numbers differ from those docs.\n")

    # ------------------------------------------------------------------ per criterion
    w("## 5. Per-criterion results and error analysis\n")
    top = [r for r in ho if r["family"] != "jury code version"][:12]
    w("### 5.1 Held-out agreement by criterion (top configurations; **bold** < 0.90)\n")
    w("| configuration | " + " | ".join(CRIT) + " |")
    w("|---|" + "---|" * 8)
    for r in top:
        w(f"| {r['name']} | " + " | ".join(
            (f"**{r['crit'][k]:.2f}**" if r["crit"][k] is not None and r["crit"][k] < 0.9 else f2(r["crit"][k]))
            for k in CRIT) + " |")
    allc = {k: [0, 0] for k in CRIT}
    kinds, pairs_ct = {}, {}
    for r in ho:
        if r["family"] == "jury code version":
            continue
        for run in r["H"]:
            for cid, x in run["per"].items():
                for k, (ok, ev, pv) in x["labels"].items():
                    allc[k][1] += 1
                    if not ok:
                        allc[k][0] += 1
                        kinds.setdefault(k, {}).setdefault(classify(k, ev, pv), 0)
                        kinds[k][classify(k, ev, pv)] += 1
                        pairs_ct[(cid, k)] = pairs_ct.get((cid, k), 0) + 1
    nruns = sum(len(r["H"]) for r in ho if r["family"] != "jury code version")
    w(f"\n### 5.2 Where all configurations fail (pooled over {nruns} held-out runs)\n")
    w("| criterion | error rate | NA vs applicable | scale: adjacent step | scale: ≥2 steps | binary flip | failed |")
    w("|---|---|---|---|---|---|---|")
    for k in sorted(CRIT, key=lambda k: -allc[k][0] / max(1, allc[k][1])):
        kk = kinds.get(k, {})
        w(f"| {k} | {allc[k][0] / max(1, allc[k][1]):.3f} | {kk.get('NA vs applicable', 0)} | "
          f"{kk.get('scale: adjacent step', 0)} | {kk.get('scale: ≥2 steps', 0)} | {kk.get('binary flip', 0)} | "
          f"{kk.get('failed / missing', 0)} |")
    w("\nMost frequently missed (dialog, criterion) pairs (share of runs that missed it):\n")
    w("| dialog | criterion | expected | missed in | gold rationale |")
    w("|---|---|---|---|---|")
    hard = sorted(pairs_ct.items(), key=lambda x: (-x[1], x[0]))[:8]
    for (cid, k), c in hard:
        g = GOLD["holdout"][cid]
        w(f"| {cid} | {k} | {g['expected'][k]} | {c}/{nruns} | {(g.get('rationale') or {}).get(k, '')} |")
    w("\nTypical failure modes, quoted from the (synthetic, safe-to-quote) held-out dialogs:\n")
    for (cid, k), c in hard[:2]:
        g = GOLD["holdout"][cid]
        w(f"**{cid} — criterion {k}** (expected `{g['expected'][k]}`, missed by {c}/{nruns} runs; "
          f"title: “{g.get('title', '')}”)\n")
        w(excerpt(g, k))
        rat = (g.get("rationale") or {}).get(k)
        w(f"\nLabel rationale: {rat or '(none recorded)'}. Models disagree on how much of a partial or "
          "self-contradicting answer still earns credit — the boundary is defined by our synthetic rubric "
          "reading, so part of this 'error' is label ambiguity.\n")
    w("Patterns: 02 errors are almost all **NA vs applicable** (was clarification needed at all?); 04/06 errors "
      "are mostly **one scale step** apart (15 vs 20, 7 vs 10); 07 confuses a bare farewell with a closing "
      "summary/offer. These are the criteria the jury flags `needs_review` most often on DEV "
      "(04 ×31, 07 ×23, 06 ×17, 02 ×8 of 95 flags, docs/EVALUATION.md).\n")

    # ------------------------------------------------------------------ ablations
    w("## 6. Ablations and negative results\n")
    w("Each row is a single change; Δ on held-out unless stated. Negative and null results are kept on purpose.\n")
    w("| change | A | B | Δ (labels of 80) | paired p (boot) | reading | source |")
    w("|---|---|---|---|---|---|---|")

    def abl(label, A, B, reading, src=None):
        if not (A and B and A.get("acc") is not None and B.get("acc") is not None):
            return
        p = paired(A["H"], B["H"], reps)
        w(f"| {label} | {A['name']} {A['acc']:.3f} | {B['name']} {B['acc']:.3f} | "
          f"{(A['acc'] - B['acc']) * 80:+.1f} | {p['p_boot']:.2f} | {reading} | {link(src or A['src'])} |")

    abl("prompt v1 few-shot (luna)", get("gpt-6-luna@v1"), get("gpt-6-luna"), "within noise; kept for `econom`")
    abl("prompt v2 step-by-step (luna)", get("gpt-6-luna@v2"), get("gpt-6-luna"), "no gain")
    abl("prompt v2 step-by-step (haiku)", get("haiku@v2"), get("haiku"), "no gain")
    abl("KZ parser signals ON", hyb_on, hyb_off, "null result (§1)")
    abl("--confirm 03,04 (judge re-checks consensus)", get("jury v8-prompt, --confirm 03,04"),
        get("jury v8-prompt, default (EVALUATION held-out)"), "no 03/04 verdict changed; +8–22 % cost")
    abl("--final profile (sonnet+opus, median-of-3 per juror)",
        get("jury --final (sonnet+opus, median-of-3 per juror)"), get("jury prompt v9 run b"),
        "not better at ~2–4× cost; opt-in")
    abl("v7 bundle: 120→180 s wait + median-of-3 judge + rubric alignment",
        get("jury v7 code (180 s, median-of-3 judge, rubric alignment)"),
        get("jury v6 code (120 s wait, single judge sample)"), "+1 label; bundled, cannot attribute")
    abl("Jev full rules vs rules ≤700 chars", jev, get("Jev 1.14 alone (rules ≤700 chars)"),
        "largest single effect; chosen after seeing held-out")
    abl("cascade tier 1 = Qwen3-8B instead of Jev", get("cascade Qwen3-8B-AWQ → haiku → sonnet"), casc,
        "**failed**: 65 % escalated, lower accuracy")
    abl("cascade: add sonnet tier 3", casc, get("cascade Jev → haiku"), "small gain, +cost")
    abl("QLoRA 16 steps (0.32 epoch)", ftl, ftb, "agreement tie, MAE ↓")
    tb = jload(R / "jev" / "tiebreak.json") or {}
    for k, v in tb.items():
        w(f"| Jev as tie-breaker ({k}) | judge correct {v['judge_correct']}/{v['disagreements']} | "
          f"Jev correct {v['jev_correct']}/{v['disagreements']} | {v['jev_correct'] - v['judge_correct']:+d} cases | "
          f"— | {'no gain' if v['jev_correct'] <= v['judge_correct'] else 'gain only vs weak judge setup'} | "
          f"{link(R / 'jev' / 'tiebreak.json')} |")
    hs = jload(R / "hybrid_kz" / "summary.json") or {}
    if hs.get("holdout"):
        h = hs["holdout"]
        w(f"| KZ signals for Jev alone | jev+signals {h['jev_signals']['agree']:.3f} | jev plain "
          f"{h['jev_plain']['agree']:.3f} | {h['jev_signals']['correct'] - h['jev_plain']['correct']:+d} | — | "
          f"tuned gold: {hs['gold']['jev_signals']['correct'] - hs['gold']['jev_plain']['correct']:+d} labels "
          f"(opposite sign) | {link(R / 'hybrid_kz' / 'summary.json')} |")
    tj = jload(R / "oss" / "tune_jev_all.json")
    if tj and casc:
        best = max(x["agreement"] for x in tj)
        nb = sum(x["agreement"] == best for x in tj)
        w(f"| router threshold tuning (replay on tuned gold) | best replay {best:.3f} | held-out {casc['acc']:.3f} | "
          f"{(casc['acc'] - best) * 80:+.1f} | — | {nb}/{len(tj)} grid points tie at the optimum → overfit "
          f"risk | {link(R / 'oss' / 'tune_jev_all.json')} |")
    w("\nThe v7 jury bundle (median-of-3 judge sampling on 04/06, 180 s wait threshold, rubric alignment) shipped "
      "as one change; on DEV it changed 34/520 verdicts (docs/EVALUATION.md) while held-out moved by one label, "
      "so its accuracy effect is unidentified. Within that run, a single judge sample would have differed from the "
      "median in ≈3 of 29 scale cases.\n")

    # ------------------------------------------------------------------ threats
    w("## 7. Threats to validity\n")
    for t in [
        "**Synthetic labels written by the rule authors.** Gold measures how faithfully a model applies *our* "
        "reading of the checklist, not agreement with Freedom Broker auditors. No human labels exist yet.",
        f"**Tiny n.** Held-out = {nh} dialogs / {8 * nh} labels; 1 gross-error positive (P/R of 1.0 means 1/1). "
        "CIs (§3.3) are wider than every gap among the top ~10 configurations.",
        "**Held-out contamination.** One rule changed after a held-out miss; held-out used for model selection "
        "and the Jev rule-length choice; tuned gold is in the fine-tune train set.",
        "**Prompt/code drift between runs.** Bench matrix = one frozen snapshot (prompt v9); jury versions v8–v10 "
        "prompt and v6/v7 code differ; local Qwen3-32B ran on an earlier prompt state; Jev/cascade/kz_models runs "
        "used the code of their day. Cross-family comparisons mix versions.",
        "**Provider non-determinism.** Temperature is not settable on the Anthropic endpoint or reasoning models; "
        "repeat runs differ by 0–2 labels. Most configurations have a single run.",
        "**DEV agreement ≠ accuracy.** It measures closeness to our own jury; a model that disagrees may be right; "
        "the 08 rule wording was influenced by a DEV dialog.",
        "**Cost assumptions.** $ uses list prices on 2026-11-07 (promo prices for Gemini); GPU $ uses RunPod "
        "secure hourly rates × wall time at high concurrency, excluding idle/boot time.",
        "**Selection effects in the matrix.** Prompt variants searched only on the cheapest models; DEV runs "
        "only for cheap models (budget).",
    ]:
        w(f"- {t}")
    w("")

    # ------------------------------------------------------------------ reproducibility
    w("## 8. Reproducibility\n")
    w("| experiment | command | models / hardware | output |")
    w("|---|---|---|---|")
    for row in [
        ("multi-vendor matrix", "`python scripts/bench_matrix.py snapshot && python scripts/bench_matrix.py run "
         "--suites gold,holdout && python scripts/bench_matrix.py matrix`",
         "claude-haiku/sonnet/opus/fable, gpt-5.3-codex, gpt-6-luna, gpt-5.4-mini, gemini-4-argon, "
         "gemini-3.8-flash; prompt v9 snapshot", "`results/bench/`, `results/bench_matrix.json`"),
        ("jury on gold / held-out", "`python -m cqa gold [--dir examples/gold_holdout] [--confirm 03,04]`",
         "haiku-5-5 + sonnet-5-5, judge opus-5-5", "`results/gold*_*.json`"),
        ("Jev 1.14", "`python scripts/bench_jev.py` (OpenRouter alpha Decisions API)",
         "typesafe/jev-1.14-20261014", "`results/jev/`"),
        ("cascade", "`python scripts/bench_oss.py cascade --suite holdout --first jev` (thresholds via "
         "`bench_oss.py tune` on tuned gold)", "Jev → haiku → sonnet; T1 0.7, HARD_CONF 0.9, T2 0.7",
         "`results/oss/`"),
        ("local OSS single model", "`python scripts/runpod_vllm.py up --model <repo>` then "
         "`python results/kz_models/run_local_only.py <served> <base_url> <tag>`",
         "vLLM `vllm/vllm-openai:latest`, thinking off; A40 (Qwen3-8B/32B), H100 SXM (Qwen3.8-27B-FP8)",
         "`results/kz_models/`, `results/bench_*`"),
        ("hybrid KZ signals", "`python scripts/eval_hybrid_kz.py`", "CQA_KZ_SIGNALS=1 + cascade",
         "`results/hybrid_kz/`"),
        ("fine-tune", "`python scripts/finetune_dataset.py && bash scripts/finetune_runpod.sh` "
         "(`--max-steps 16`)", "Qwen/Qwen3-8B, QLoRA r16 α32 nf4; RTX A5000; torch 2.14.0, transformers 5.20.2, "
         "peft 0.21.2, trl 1.17.1, vLLM 0.33.0", "`results/finetune/`"),
        ("this report", "`python scripts/build_experiments.py`", f"stdlib only; bootstrap seed {SEED}",
         "`EXPERIMENTS.md`, `docs/img/pareto.svg`"),
    ]:
        w("| " + " | ".join(row) + " |")
    w("\nSeeds: LLM sampling is provider-default (not seedable for most providers); bootstrap seed fixed above. "
      "vLLM image tag was `latest` (exact version not recorded except the fine-tune run, 0.33.0); LitJev / other "
      "decision-model runtime versions are not recorded in the result files.\n")
    w("Recorded spend: multi-vendor matrix $40.13 (docs/MODELS.md); jury DEV v7 $10.28; Qwen3-32B pod $0.28; "
      "fine-tune pod ≈$0.22; Jev ≈$0.07; hybrid ≈$0.19 (docs/*).\n")
    w("### What we would do next\n")
    for t in [
        "**Human labels**: `python -m cqa label --group DEV -n 50` by ≥2 annotators; report Cohen's κ per "
        "criterion (inter-annotator ceiling) before any model claim; then re-run §4 on those labels.",
        "**Bigger held-out**: ≥50 dialogs (CI half-width ≈ ±0.02) written by someone who did not write the rules.",
        "**Full fine-tune epoch(s)** on the ungated Apache-2.0 `issai/Qwen3.5-9B-Kazakh` base and on "
        "Qwen3.8-27B, with human rows weighted 3×; evaluate only on held-out + human DEV.",
        "**Calibration**: reliability diagrams / ECE of Jev and local logprob confidences; re-tune the router on a "
        "larger set with cross-validation instead of a 16-dialog replay.",
        "**Repeat every top-10 configuration ≥3×** to separate model from sampling variance.",
    ]:
        w(f"- {t}")
    w("")
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out} ({len(ho)} held-out configurations, {len(pairs)} paired tests)"
          + (f" and {a.svg}" if pts else ""))


if __name__ == "__main__":
    main()

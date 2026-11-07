#!/usr/bin/env python3
# ruff: noqa: E402
"""KZ / open-model / decision-model experiments → ONE shared result schema (docs/KZ_MODELS.md).

    # generative open model on vLLM, tier 1 only (no escalation), same prompts as bench_oss cascade --first local
    python scripts/bench_kz.py local --served qwen3.8-27b-fp8 --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.8-27B-FP8
    # Jev-compatible decision endpoint (Jev, LitJev, scripts/decision_server.py) through cqa/jev.py's question builder
    python scripts/bench_kz.py decision --method litjev --model Qwen/Qwen3.8-27B-FP8 \
        --endpoint http://127.0.0.1:8001/v1/systemone --suite holdout|gold
    # cascade: a saved decision run as tier 1 → haiku → sonnet (router thresholds via flags)
    python scripts/bench_kz.py cascade --method litjev --model Qwen/Qwen3.8-27B-FP8 --t1 0.7 --hard-conf 0.9 --t2 0.7
    # offline threshold tuning on the TUNED gold only (tier 1 = saved gold decision run; haiku/sonnet = cached samples)
    python scripts/bench_kz.py tune --method litjev --model Qwen/Qwen3.8-27B-FP8
    python scripts/bench_kz.py convert <bench_oss holdout json> --method vllm-generate --model M   # old → schema
    python scripts/bench_kz.py table

Schema (results/kz_models/<method>__<model>.json): {method, model, gpu, split, n_dialogs, per_dialog:[{id, lang,
per_criterion:{01..08:{pred,gold}}, total_pred, total_gold}], per_criterion_agreement, overall_agreement, total_mae,
gross:{p,r}, subsets:{KZ,RU,EN}, s_per_dialog, usd_per_dialog, escalation_share?}. Raw dialog results (for cascade
memo / re-scoring) go to results/kz_models/raw/.
"""
import os as _os
import sys

sys.path = [p for p in sys.path if _os.path.abspath(p or ".") != _os.path.dirname(_os.path.abspath(__file__))]
import argparse  # noqa: E402
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))   # appended: scripts/ must not shadow stdlib modules
OUT = ROOT / "results" / "kz_models"
RAW = OUT / "raw"
SUITES = {"gold": "examples/gold", "holdout": "examples/gold_holdout"}
GPU_USD = {"A40": 0.59}


def slug(model):
    return model.replace("/", "_")


def lang_of(cid):
    return next((L for L in ("kz", "ru", "en") if f"-{L}-" in cid), "?")


def load(suite):
    from cqa import data
    from cqa.gold import load_gold
    gold = load_gold(ROOT / SUITES[suite])
    return gold, [data.normalize(g["dialog"], i) for i, g in enumerate(gold)]


def schema(method, model, gpu, suite, gold, res, s_per_dialog, usd_per_dialog, **extra):
    from cqa.gold import compute_metrics
    m = compute_metrics(gold, res)
    by = {d["conversation_id"]: d for d in res}
    per = []
    for g in gold:
        cid = g["dialog"]["conversation_id"]
        d = by.get(cid) or {"criteria": []}
        pred = {c["criterion_id"]: c.get("value") for c in d["criteria"]}
        per.append({"id": cid, "lang": lang_of(cid),
                    "per_criterion": {k: {"pred": pred.get(k), "gold": v} for k, v in sorted(g["expected"].items())},
                    "total_pred": d.get("total"), "total_gold": g.get("expected_total"),
                    "gross_pred": d.get("gross_error"), "gross_gold": g.get("expected_gross_error"),
                    "status": d.get("status")})
    subsets = {}
    for L in ("kz", "ru", "en"):
        sub = [g for g in gold if lang_of(g["dialog"]["conversation_id"]) == L]
        if sub:
            subsets[L.upper()] = compute_metrics(sub, res)["overall_agreement"]
    return {"method": method, "model": model, "gpu": gpu, "split": suite, "n_dialogs": len(gold), "per_dialog": per,
            "per_criterion_agreement": {k: v["agreement"] for k, v in m["per_criterion"].items()},
            "overall_agreement": m["overall_agreement"], "total_mae": m["total_mae"],
            "gross": {"p": m["gross"]["precision"], "r": m["gross"]["recall"]}, "subsets": subsets,
            "s_per_dialog": s_per_dialog, "usd_per_dialog": usd_per_dialog,
            "incomplete": m["incomplete"], **extra}


def save(d, suite, raw=None):
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{d['method']}__{slug(d['model'])}" + ("" if suite == "holdout" else f"__{suite}")
    (OUT / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    if raw is not None:
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f"{name}.dialogs.json").write_text(json.dumps(raw, ensure_ascii=False))
    print(name, {k: d.get(k) for k in ("overall_agreement", "total_mae", "gross", "subsets", "s_per_dialog",
                                        "usd_per_dialog", "incomplete", "escalation_share")})


def gpu_usd(gpu, s):
    return round(GPU_USD.get(gpu, 0) * s / 3600, 5) if s is not None else None


def decision(a):
    os.environ["CQA_JEV_BASE_URL"] = a.endpoint
    os.environ["CQA_JEV_MODEL"] = a.served or a.model
    from cqa import checklist as ckl, jev
    cl = ckl.load("freedom_chat_v1")
    gold, recs = load(a.suite)
    t = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda r: jev.evaluate(r, cl), recs))
    wall = time.time() - t
    errs = [d.get("error") for d in res if d.get("error")]
    if errs:
        print("errors:", errs[:3])
    s = round(wall / len(recs), 2)
    d = schema(a.method, a.model, a.gpu, a.suite, gold, res, s, gpu_usd(a.gpu, s),
               endpoint=a.endpoint, concurrency=a.concurrency, errors=errs[:5])
    save(d, a.suite, res)


def local(a):
    """Generative open model: bench_oss cascade --first local with escalation disabled (tier 1 only)."""
    from cqa import router
    router.tier1_reasons = lambda *x, **k: []
    import bench_oss
    tag = f"kzrun_{slug(a.model)}_{a.suite}"
    sys.argv = ["bench_oss.py", "cascade", "--suite", a.suite, "--first", "local", "--local-model", a.served,
                "--base-url", a.base_url, "--no-logprobs", "--concurrency", str(a.concurrency), "--tag", tag]
    bench_oss.OUT = RAW
    bench_oss.main()
    convert_file(RAW / f"{tag}_{a.suite}.json", a.method, a.model, a.gpu, a.suite)


def convert_file(f, method, model, gpu, suite="holdout"):
    b = json.load(open(f))
    gold, _ = load(suite)
    s = b["sec_per_dialog"]
    d = schema(method, model, gpu, suite, gold, b["dialogs_detail"], s, gpu_usd(gpu, s), source=str(Path(f).name),
               errors=b.get("errors", [])[:5])
    save(d, suite)


def _ev_settings(a):
    return dict(first="jev", t1=a.t1, hard=a.hard, hard_conf=a.hard_conf, t2=a.t2, haiku_low=a.haiku_low,
                judge=a.judge, logprobs=False)


def cascade(a):
    """Tier 1 = the saved decision run (no re-run); escalations go to haiku/sonnet via the router (disk-cached)."""
    from cqa import config, router
    config.PROVIDER = "anthropic"
    name = f"{a.method}__{slug(a.model)}" + ("" if a.suite == "holdout" else f"__{a.suite}")
    memo = {d["conversation_id"]: d for d in json.load(open(RAW / f"{name}.dialogs.json"))}
    base = json.load(open(OUT / f"{name}.json"))
    ev = router.CascadeEvaluator(use_cache=True, **_ev_settings(a))
    ev.jev_memo.update(memo)
    gold, recs = load(a.suite)
    t = time.time()
    res, meta = ev.evaluate_many(recs, concurrency=a.concurrency)
    wall = time.time() - t
    r = meta["routing"]
    s1 = base["s_per_dialog"]
    cloud = round(r["cloud_usd"] / len(recs), 5)
    d = schema(f"cascade-{a.method}", a.model, base["gpu"], a.suite, gold, res,
               round(s1 + wall / len(recs), 2), round((base["usd_per_dialog"] or 0) + cloud, 5),
               escalation_share=r["escalated_share"], tier_share=r["tier_share"], cloud_usd_per_dialog=cloud,
               tier1_s_per_dialog=s1, settings=r["settings"], reasons=r["reasons"])
    save(d, a.suite, res)


def tune(a):
    """Grid on TUNED gold only. Tier 1 = saved gold decision run; haiku/sonnet = all-escalate samples recorded by
    bench_oss (results/oss/jev_all_gold.json — haiku/sonnet calls do not depend on the tier-1 model)."""
    import itertools

    import bench_oss

    from cqa import router
    gold, _ = load("gold")
    S = bench_oss._samples("jev_all", "gold")
    t1run = json.load(open(RAW / f"{a.method}__{slug(a.model)}__gold.dialogs.json"))
    for dl in t1run:
        for c in dl["criteria"]:
            k = (dl["conversation_id"], c["criterion_id"])
            if k in S:
                S[k] = {**S[k], "local_value": c.get("value"), "local_conf": c.get("confidence"),
                        "local_error": None if c.get("value") is not None else "no value", "local_evidence_ok": True}
    rows = []
    for t1, hc, t2, hl, judge in itertools.product([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99], [0.9, 0.95, 0.99, 1.01],
                                                   [0.0, 0.7, 0.8, 0.9, 1.01], [0.0, 0.5, 0.7],
                                                   ["claude-sonnet-5-5", ""]):
        if not judge and (t2 or hl):
            continue
        s = router.settings(t1=t1, hard_conf=hc, hard="04,05,06", t2=t2, haiku_low=hl, judge=judge, first="jev")
        agr, tiers = bench_oss.replay(S, gold, s)
        rows.append({"t1": t1, "hard_conf": hc, "t2": t2, "haiku_low": hl, "judge": bool(judge), "agreement": agr,
                     "tiers": tiers, "cost_proxy": round(tiers[2] + 15 * tiers[3], 3)})
    rows.sort(key=lambda r: (-r["agreement"], r["cost_proxy"]))
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"tune_{a.method}__{slug(a.model)}.json").write_text(json.dumps(rows, indent=1))
    lo = max(rows, key=lambda r: (r["tiers"][1], r["agreement"]))
    print("least escalation:", lo)
    for r in rows[:5]:
        print(r)
    print("CHOICE:", next(r for r in rows if r["judge"]))


def table(_a=None):
    rows = []
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        if d.get("split") != "holdout" or "per_dialog" not in d:
            continue
        rows.append(d)
    rows.sort(key=lambda d: -(d["overall_agreement"] or 0))
    print("| method | model | GPU | agr. | MAE | gross P/R | KZ | RU | EN | s/dialog | $/dialog | escal. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in rows:
        sb, g = d["subsets"], d["gross"]
        print(f"| {d['method']} | {d['model']} | {d['gpu']} | {d['overall_agreement']} | {d['total_mae']} | "
              f"{g['p']} / {g['r']} | {sb.get('KZ')} | {sb.get('RU')} | {sb.get('EN')} | {d['s_per_dialog']} | "
              f"{d['usd_per_dialog']} | {d.get('escalation_share', '–')} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["local", "decision", "cascade", "tune", "convert", "table"])
    p.add_argument("file", nargs="?")
    p.add_argument("--method", default="vllm-generate")
    p.add_argument("--model", default="")
    p.add_argument("--served")
    p.add_argument("--gpu", default="A40")
    p.add_argument("--suite", default="holdout", choices=list(SUITES))
    p.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/decisions")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--hard", default="04,05,06")
    p.add_argument("--hard-conf", type=float, default=0.9)
    p.add_argument("--t2", type=float, default=0.7)
    p.add_argument("--haiku-low", type=float, default=0.0)
    p.add_argument("--judge", default="claude-sonnet-5-5")
    a = p.parse_args()
    if a.cmd == "convert":
        return convert_file(a.file, a.method, a.model, a.gpu)
    {"local": local, "decision": decision, "cascade": cascade, "tune": tune, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()

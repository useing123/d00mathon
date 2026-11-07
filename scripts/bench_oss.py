#!/usr/bin/env python3
"""Open-weight model sweep on ONE RunPod pod + cascade-router benchmark (docs/OSS_ROUTER.md).

    # swap the model served by the running pod (podEditJob keeps the GPU reservation, restarts vLLM)
    python scripts/bench_oss.py swap --model Qwen/Qwen3-14B-AWQ --served qwen3-14b-awq [--extra ...]
    # per-model benchmark = scripts/bench_models.py run (identical prompts/checklist/scoring for every model)
    # cascade on a suite (gold | holdout | dev); --all-escalate records Haiku for EVERY criterion (tuning data)
    python scripts/bench_oss.py cascade --suite gold --first local|jev --local-model qwen3-14b-awq \
        [--conf 0.9 --hard 04,05,06 --hard-conf 0.99] [--tiebreak claude-sonnet-5-5] [--all-escalate] --tag T
    # offline threshold grid on the TUNED gold set only (replays .cache/router samples, no API calls)
    python scripts/bench_oss.py tune --tag T
    python scripts/bench_oss.py table       # results/oss_router_table.json + markdown

Outputs: results/oss/<tag>_<suite>.json (gold metrics or DEV agreement with the cloud jury + meta.routing).
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "oss"
SUITES = {"gold": "examples/gold", "holdout": "examples/gold_holdout", "dev": "data/DEV.json"}


def newest_jury():
    f = sorted(glob.glob(str(ROOT / "results/dev_jury_v*.json")), key=lambda f: int(re.search(r"_v(\d+)", f).group(1)))
    return f[-1]


def swap(a):
    import runpod_vllm as rp
    pid = json.loads(rp.STATE.read_text())["id"]
    from cqa.router import vllm_lora_args
    ns = argparse.Namespace(model=a.model, served_name=a.served, max_len=a.max_len, no_think=not a.think,
                            extra=a.extra + vllm_lora_args())   # CQA_LOCAL_ADAPTER → --enable-lora --lora-modules
    rp.gql("mutation($in: PodEditJobInput!){ podEditJob(input:$in){ id } }",
           {"in": {"podId": pid, "dockerArgs": rp.vllm_args(ns), "imageName": a.image, "containerDiskInGb": 60}})
    st = json.loads(rp.STATE.read_text())
    st["model"] = a.served
    rp.STATE.write_text(json.dumps(st))
    print("swapped", pid, "→", a.served)


def _records(suite):
    from cqa import data
    from cqa.gold import load_gold
    if suite == "dev":
        return None, data.load_file(str(ROOT / SUITES["dev"]))
    gold = load_gold(ROOT / SUITES[suite])
    return gold, [data.normalize(g["dialog"], i) for i, g in enumerate(gold)]


def _jev_memo(suite):
    """Re-use the Jev runs already made by the Jev agent (results/jev/<suite>.dialogs.json, run 1) — no new spend."""
    f = ROOT / "results" / "jev" / f"{suite}.dialogs.json"
    return {d["conversation_id"]: d for d in json.load(open(f))} if f.exists() else {}


def cascade(a):
    from cqa import config
    os.environ.setdefault("CQA_OPENAI_API_KEY", _secret("CQA_LOCAL_LLM_TOKEN") or "none")
    config.OPENAI_API_KEY = os.environ["CQA_OPENAI_API_KEY"]
    if a.base_url:
        config.OPENAI_BASE_URL = a.base_url
    config.PROVIDER = "anthropic"
    from cqa import router
    over = dict(first=a.first, local_model=a.local_model, t1=a.t1, hard=a.hard, hard_conf=a.hard_conf, t2=a.t2,
                haiku_low=a.haiku_low, judge=a.judge, mode=a.mode, local_timeout=a.local_timeout,
                force="3" if a.all_escalate else None, logprobs=not a.no_logprobs)
    ev = router.CascadeEvaluator(use_cache=True, **over)
    if a.first == "jev" and not a.live_jev:
        ev.jev_memo.update(_jev_memo(a.suite))
    gold, recs = _records(a.suite)
    if a.limit:
        recs = recs[: a.limit]
    t = time.time()
    res, meta = ev.evaluate_many(recs, concurrency=a.concurrency)
    wall = time.time() - t
    out = {"tag": a.tag, "suite": a.suite, "settings": meta["routing"]["settings"], "routing": meta["routing"],
           "wall_s": round(wall, 1), "dialogs": len(recs), "sec_per_dialog": round(wall / len(recs), 2),
           "incomplete": meta["incomplete_dialogs"], "errors": meta["errors"][:20]}
    if gold is not None:
        from cqa.gold import compute_metrics
        m = compute_metrics(gold, res)
        out["metrics"] = {k: m[k] for k in ("overall_agreement", "total_mae", "gross", "na_confusion", "mismatches")}
        out["metrics"]["per_criterion"] = {k: v["agreement"] for k, v in m["per_criterion"].items()}
    else:
        from bench_models import compare
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = OUT / f"{a.tag}_dev.dialogs.json"
        tmp.write_text(json.dumps({"dialogs": res}, ensure_ascii=False))
        out["vs_cloud"] = compare(tmp, newest_jury())
        out["cloud_file"] = Path(newest_jury()).name
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}_{a.suite}.json").write_text(json.dumps({**out, "dialogs_detail": res if gold is not None else None},
                                                           ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"} | {"metrics": {
        k: out.get("metrics", {}).get(k) for k in ("overall_agreement", "total_mae", "per_criterion")}},
        ensure_ascii=False, indent=1)[:3000])


def _secret(k):
    import bench_models
    return bench_models.secret(k)


def _samples(tag, suite):
    """(cid, crit) -> {local: {...}, escalate: {...}} from an --all-escalate run file."""
    d = json.load(open(OUT / f"{tag}_{suite}.json"))
    S = {}
    for dl in d["dialogs_detail"]:
        for c in dl["criteria"]:
            r = c.get("routing") or {}
            S[(dl["conversation_id"], c["criterion_id"])] = r
    return S


def replay(S, gold, s):
    """Route every gold (dialog, criterion) under settings ``s`` from an --all-escalate run (all tiers recorded)."""
    from cqa import router
    match = n = 0
    tiers = {1: 0, 2: 0, 3: 0}
    for g in gold:
        cid = g["dialog"]["conversation_id"]
        for k, exp in g["expected"].items():
            r = S.get((cid, k))
            if r is None:
                continue
            local = {"value": r["local_value"], "confidence": r["local_conf"], "error": r.get("local_error")}
            if not r.get("local_evidence_ok", True) and s["first"] != "jev":
                local["rejected_quotes"] = [1]
            why1 = router.tier1_reasons({"id": k, "gross": k == "08"}, local, s)
            if s["first"] == "jev":
                why1 = [w for w in why1 if w != "evidence_failed"]
            v, t = r["local_value"], 1
            if why1:
                h, so = r.get("haiku") or {}, r.get("sonnet") or {}
                why2 = router.tier2_reasons(local, h, s) if s["judge"] else []
                v, t = (so.get("value"), 3) if why2 and so.get("value") is not None else (h.get("value"), 2)
            tiers[t] += 1
            n += 1
            match += str(v) == str(exp)
    return round(match / n, 4), {k: round(c / n, 3) for k, c in tiers.items()}


def tune(a):
    """Grid over T1, HARD_CONF, T2, HAIKU_LOW on the TUNED gold set only. Objective: max agreement, then lowest
    cost proxy (share tier2 + 15 x share tier3 — Sonnet ≈15x Haiku per call). Policy: 04/05/06 escalate unless
    p >= HARD_CONF with HARD_CONF >= 0.9 (spec)."""
    from cqa import router
    from cqa.gold import load_gold
    gold = load_gold(ROOT / SUITES["gold"])
    S = _samples(a.tag, "gold")
    rows = []
    for t1, hc, t2, hl, judge in itertools.product([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
                                                   [0.9, 0.95, 0.99, 0.999, 1.01], [0.0, 0.7, 0.8, 0.9, 0.95, 1.01],
                                                   [0.0, 0.5, 0.7, 0.8], ["claude-sonnet-5-5", ""]):
        if not judge and (t2 or hl):
            continue
        s = router.settings(t1=t1, hard_conf=hc, hard="04,05,06", t2=t2, haiku_low=hl, judge=judge, first=a.first)
        agr, tiers = replay(S, gold, s)
        rows.append({"t1": t1, "hard_conf": hc, "t2": t2, "haiku_low": hl, "judge": bool(judge), "agreement": agr,
                     "tiers": tiers, "cost_proxy": round(tiers[2] + 15 * tiers[3], 3)})
    rows.sort(key=lambda r: (-r["agreement"], r["cost_proxy"]))
    (OUT / f"tune_{a.tag}.json").write_text(json.dumps(rows, indent=1))
    for r in rows[:8]:
        print(r)
    print("CHOICE (3 tiers):", next(r for r in rows if r["judge"]))
    print("CHOICE (2 tiers):", next(r for r in rows if not r["judge"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["swap", "cascade", "tune"])
    p.add_argument("--model")
    p.add_argument("--served")
    p.add_argument("--image", default="vllm/vllm-openai:latest")
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--think", action="store_true")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--suite", default="gold", choices=list(SUITES))
    p.add_argument("--first", default="local", choices=["local", "jev"])
    p.add_argument("--live-jev", action="store_true", help="call Jev instead of re-using results/jev runs")
    p.add_argument("--local-model", default="qwen3-14b-awq")
    p.add_argument("--base-url")
    p.add_argument("--t1", type=float)
    p.add_argument("--t2", type=float)
    p.add_argument("--haiku-low", type=float)
    p.add_argument("--judge", help="tier-3 model; '' disables")
    p.add_argument("--no-logprobs", action="store_true")
    p.add_argument("--hard")
    p.add_argument("--hard-conf", type=float)
    p.add_argument("--mode")
    p.add_argument("--local-timeout", type=float)
    p.add_argument("--all-escalate", action="store_true")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="cascade")
    a = p.parse_args()
    {"swap": swap, "cascade": cascade, "tune": tune}[a.cmd](a)


if __name__ == "__main__":
    main()

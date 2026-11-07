"""Offline replay of the evidence step (cqa/evidence_step.py) on saved runs — verdicts are taken from the files
and never re-decided, so "verdicts changed" is checked by construction + an explicit equality assert.

  python scripts/measure_evidence_step.py   →  results/oss/evidence_step.json
Sources: cascade Jev→haiku→sonnet = results/oss/jev3_{holdout,dev}; local decision cascade (Qwen3.8-27B vllm-choice,
held-out, per-criterion tiers not saved → every disputed verdict is cited = upper bound on cost);
jury = results/gold_holdout_final.json (held-out), results/dev_jury_v7.json (DEV).
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))   # append: scripts/numbers.py would shadow stdlib
from cqa import (  # noqa: E402
    checklist as ckl,  # noqa: E402
    config,
    data,
    evidence_step as es,  # noqa: E402
    prompts,
)
from cqa.llm import LLM, USAGE_BY_MODEL, cost_usd  # noqa: E402

R = ROOT / "results"


def _crit(dialogs):
    return {(d["conversation_id"], c["criterion_id"]): c for d in dialogs for c in d["criteria"]}


def _cov(items):
    n = len(items)
    return round(sum(bool(c.get("evidence")) for c in items) / n, 3) if n else None


def main():
    from bench_oss import _records  # imported late: it puts scripts/ (numbers.py) first on sys.path
    config.PROVIDER = "anthropic"
    llm = LLM(provider="anthropic", model="claude-haiku-5-5")
    recs = {s: {r["conversation_id"]: r for r in _records(s)[1]} for s in ("holdout", "dev")}
    srcs = {
        ("cascade", "holdout"): _crit(json.load(open(R / "oss/jev3_holdout.json"))["dialogs_detail"]),
        ("cascade", "dev"): _crit(json.load(open(R / "oss/jev3_dev.dialogs.json"))["dialogs"]),
        ("jury", "holdout"): _crit(json.load(open(R / "gold_holdout_final.json"))["dialogs"]),
        ("jury", "dev"): _crit(json.load(open(R / "dev_jury_v7.json"))["dialogs"]),
    }
    kz = json.load(open(R / "kz_models/cascade-vllm-choice__Qwen_Qwen3.8-27B-FP8.json"))
    srcs[("cascade-local", "holdout")] = {(d["id"], k): {"criterion_id": k, "value": str(v["pred"]), "evidence": [],
                                                       "routing": {"tier": 1}}
                                          for d in kz["per_dialog"] for k, v in d["per_criterion"].items()}
    out = {}
    for (mode, suite), crit in srcs.items():
        before = {k: c["value"] for k, c in crit.items()}
        ev0 = {k: bool(c.get("evidence")) for k, c in crit.items()}
        row = {"criteria": len(crit), "coverage_before": _cov(list(crit.values()))}
        if mode != "jury":
            jobs = []
            for (cid, k), c in crit.items():
                rec = recs[suite][cid]
                ck = ckl.load(ckl.pick_for_channel(rec["kind"]))
                cr = next(x for x in ck["criteria"] if x["id"] == k)
                if c.get("routing", {}).get("tier") == 1 and not c.get("evidence") and es.wanted(cr, c["value"], "disputed"):
                    jobs.append((rec, ck, cr, c))
            u0 = {m: dict(u) for m, u in USAGE_BY_MODEL.items()}
            t = time.time()

            def run(j):
                rec, ck, cr, c = j
                a = ck["assumptions"]
                tx = prompts.transcript_block(rec, data.format_transcript(rec, a.get("ignore_internal_hold_turns", True)),
                                              data.format_facts(data.compute_facts(rec, a.get("long_wait_seconds"))))
                es.apply(c, llm, rec, tx, ck, cr, m="disputed")
                return c["routing"]["evidence_step"]

            by_rec = {}
            for j in jobs:
                by_rec.setdefault(j[0]["conversation_id"], []).append(j)
            with ThreadPoolExecutor(16) as px:   # one dialog's criteria serially → prompt-cache hits on the transcript
                steps = [s for g in px.map(lambda g: [run(j) for j in g], by_rec.values()) for s in g]
            wall = time.time() - t
            d_usage = {m: {k: u.get(k, 0) - u0.get(m, {}).get(k, 0) for k in u} for m, u in USAGE_BY_MODEL.items()}
            usd = sum(cost_usd(u, m) for m, u in d_usage.items())
            nd = len({k[0] for k in crit})
            calls = [s for s in steps if not s.get("cached")]
            assert {k: c["value"] for k, c in crit.items()} == before, "verdict changed"
            disp = [c for c in crit.values() if c.get("routing", {}).get("evidence_step")]
            row.update(coverage_after=_cov(list(crit.values())), cited_calls=len(jobs),
                       step_ok=sum(s["status"] == "ok" for s in steps), step_missing=sum(s["status"] != "ok" for s in steps),
                       errors=sum(bool(s.get("error")) for s in steps), cached=len(steps) - len(calls),
                       disputed_tier1_coverage_after=_cov(disp),
                       usd_per_dialog=round(usd / nd, 5), usd_total=round(usd, 4),
                       added_s_per_dialog_serial=round(sum(s["s"] or 0 for s in calls) / nd, 2) if calls else None,
                       wall_s=round(wall, 1), verdicts_changed=0)
        # disputed-type verdicts (any tier) with a quote, for every mode
        dz, dz0 = [], []
        for (cid, k), c in crit.items():
            rec = recs[suite][cid]
            cr = next(x for x in ckl.load(ckl.pick_for_channel(rec["kind"]))["criteria"] if x["id"] == k)
            if es.is_disputed(cr, c["value"]):
                dz.append(c)
                dz0.append(ev0[(cid, k)])
        row["disputed_share"] = round(len(dz) / len(crit), 3)
        row["disputed_coverage_before"] = round(sum(dz0) / len(dz0), 3) if dz0 else None
        row["disputed_coverage_after"] = _cov(dz)
        out[f"{mode}/{suite}"] = row
        print(f"{mode}/{suite}", json.dumps(row), flush=True)
    (R / "oss/evidence_step.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

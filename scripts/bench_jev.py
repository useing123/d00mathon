"""Benchmark TypeSafe Jev on the gold sets + jury tie-breaker analysis.

    .venv/bin/python scripts/bench_jev.py run [--repeats 1] [--min-budget 0.05]
    .venv/bin/python scripts/bench_jev.py tiebreak      # uses results/jev/*.dialogs.json + results/bench jury runs

Same metrics code as `python -m cqa gold` (cqa.gold.compute_metrics). Writes results/jev/.
Stops if the OpenRouter key's limit_remaining drops below --min-budget.
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cqa import checklist as ckl, data, gold, jev  # noqa: E402

OUT = ROOT / "results" / "jev"
SUITES = {"gold": ROOT / "examples" / "gold", "holdout": ROOT / "examples" / "gold_holdout"}
COMPARE = ["jury-haiku+sonnet>opus", "gpt-5.3-codex", "sonnet", "gpt-6-luna", "gpt-6-luna@v1"]


def run_suite(name, tag, min_budget):
    items = gold.load_gold(SUITES[name])
    cl = ckl.load("freedom_chat_v1")
    rem = jev.limit_remaining()
    if rem is not None and rem < min_budget:
        sys.exit(f"limit_remaining {rem} < {min_budget}; stop")
    recs = [data.normalize(g["dialog"], i) for i, g in enumerate(items)]
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda r: jev.evaluate(r, cl), recs))
    wall = time.time() - t0
    m = gold.compute_metrics(items, results)
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in results)
    lat = [r["latency_s"] for r in results]
    summary = {"config": "jev-1.14", "suite": name, "tag": tag, "dialogs": m["dialogs"],
               "overall_agreement": m["overall_agreement"],
               "per_criterion": {k: v["agreement"] for k, v in m["per_criterion"].items()},
               "total_mae": m["total_mae"], "gross": m["gross"], "na_confusion": m["na_confusion"],
               "incomplete": len(m["incomplete"]), "usd": round(cost, 6), "usd_per_dialog": round(cost / len(recs), 6),
               "wall_s": round(wall, 1), "latency_median_s": round(statistics.median(lat), 2),
               "latency_max_s": max(lat), "calls_per_dialog": 1,
               "mismatches": [f"{x['conversation_id']}/{x['criterion_id']}: exp {x['expected']} got {x['predicted']}"
                              for x in m["mismatches"]]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}{tag}.dialogs.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (OUT / f"{name}{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def tiebreak():
    """Where the jury's jurors disagreed, compare the judge's final value vs Jev's choice against gold."""
    report = {}
    for jury in sorted({p.name.split("__")[0] for p in (ROOT / "results/bench").glob("jury-*__gold.dialogs.json")}):
        rows, judge_ok, jev_ok = [], 0, 0
        for suite in SUITES:
            jp = ROOT / "results/bench" / f"{jury}__{suite}.dialogs.json"
            vp = OUT / f"{suite}.dialogs.json"
            if not jp.exists() or not vp.exists():
                continue
            exp = {g["dialog"]["conversation_id"]: g["expected"] for g in gold.load_gold(SUITES[suite])}
            jv = {d["conversation_id"]: {c["criterion_id"]: c for c in d["criteria"]} for d in json.loads(vp.read_text())}
            for d in json.loads(jp.read_text()):
                for c in d["criteria"]:
                    votes = [str(v).upper() for v in (c.get("votes") or [])]
                    if len(set(votes)) < 2:
                        continue
                    cid, k = d["conversation_id"], c["criterion_id"]
                    e = str(exp.get(cid, {}).get(k)).upper()
                    jvv = jv.get(cid, {}).get(k, {})
                    a, b = str(c.get("value")).upper() == e, str(jvv.get("value")).upper() == e
                    judge_ok += a
                    jev_ok += b
                    rows.append({"dialog": cid, "criterion": k, "votes": votes, "gold": e, "judge": c.get("value"),
                                 "jev": jvv.get("value"), "jev_conf": jvv.get("confidence")})
        report[jury] = {"disagreements": len(rows), "judge_correct": judge_ok, "jev_correct": jev_ok, "cases": rows}
    (OUT / "tiebreak.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for k, v in report.items():
        print(f"{k}: disagreements={v['disagreements']} judge_correct={v['judge_correct']} jev_correct={v['jev_correct']}")
    return report


def compare():
    m = json.loads((ROOT / "results/bench_matrix.json").read_text())
    rows = {r["config"]: r for r in m["rows"]}
    print(f"{'config':26} {'gold':>6} {'hold':>6} {'MAE g/h':>12} {'$/dlg':>9}")
    for c in COMPARE:
        r = rows.get(c)
        if r:
            print(f"{c:26} {r['gold']['overall_agreement']:>6} {r['holdout']['overall_agreement']:>6} "
                  f"{r['gold']['total_mae']:>5}/{r['holdout']['total_mae']:<6} {r.get('usd_per_dialog')}")
    for s in SUITES:
        p = OUT / f"{s}.json"
        if p.exists():
            j = json.loads(p.read_text())
            print(f"jev-1.14 {s:17} {j['overall_agreement']:>6} MAE {j['total_mae']} ${j['usd_per_dialog']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tiebreak", "compare"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--min-budget", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd == "run":
        before = jev.limit_remaining()
        for i in range(a.repeats):
            for s in SUITES:
                r = run_suite(s, "" if i == 0 else f"_r{i + 1}", a.min_budget)
                print(json.dumps({k: r[k] for k in ("suite", "tag", "overall_agreement", "per_criterion", "total_mae",
                                                    "gross", "usd_per_dialog", "latency_median_s")}, ensure_ascii=False))
        time.sleep(5)
        print("limit_remaining before/after:", before, jev.limit_remaining())
    elif a.cmd == "tiebreak":
        tiebreak()
    else:
        compare()

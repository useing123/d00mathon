#!/usr/bin/env python
"""Run-to-run variance of the jury on the same dialogs (cache OFF): per-criterion flip rate and total-score std.

    python -m cqa evaluate --group DEV --ids <ids> --fresh --out results/variance/run1.json   # ×3
    python scripts/variance.py results/variance/run*.json --out results/variance/summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

CRITERIA = ["01", "02", "03", "04", "05", "06", "07", "08"]


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {r["conversation_id"]: {"values": {c["criterion_id"]: c.get("value") for c in r["criteria"]},
                                   "total": r.get("total") if r.get("total") is not None else r.get("provisional_total"),
                                   "gross": bool(r.get("gross_error"))} for r in d["dialogs"]}, d.get("meta", {})


def summarize(runs):
    common = sorted(set.intersection(*(set(r) for r in runs)))
    per = {}
    for k in CRITERIA:
        vals = [[r[i]["values"].get(k) for r in runs] for i in common]
        flips = [i for i, v in zip(common, vals, strict=True) if len({str(x) for x in v}) > 1]
        pair = [str(a) != str(b) for v in vals for a, b in itertools.combinations(v, 2)]
        per[k] = {"n": len(common), "flip_rate": round(len(flips) / len(common), 3) if common else None,
                  "pairwise_disagreement": round(sum(pair) / len(pair), 3) if pair else None, "flipped_dialogs": flips}
    stds, ranges = {}, []
    for i in common:
        t = [r[i]["total"] for r in runs]
        if all(x is not None for x in t):
            stds[i] = round(statistics.stdev(t), 2)
            ranges.append(max(t) - min(t))
    cells = sum(p["n"] for p in per.values())
    flipped = sum(len(p["flipped_dialogs"]) for p in per.values())
    return {"runs": len(runs), "dialogs": len(common),
            "flip_rate_all_criteria": round(flipped / cells, 3) if cells else None,
            "per_criterion": per,
            "total_std_mean": round(statistics.mean(stds.values()), 2) if stds else None,
            "total_std_median": round(statistics.median(stds.values()), 2) if stds else None,
            "total_std_max": max(stds.values()) if stds else None,
            "total_range_mean": round(statistics.mean(ranges), 2) if ranges else None,
            "dialogs_with_any_flip": sum(any(i in p["flipped_dialogs"] for p in per.values()) for i in common),
            "gross_flips": [i for i in common if len({r[i]["gross"] for r in runs}) > 1],
            "total_std_per_dialog": stds}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/variance/summary.json")
    a = p.parse_args(argv)
    loaded = [load(x) for x in a.runs]
    runs = [x[0] for x in loaded]
    out = {"files": a.runs, "cost_usd": [m.get("cost_usd") for _, m in loaded],
           "model": [m.get("model") for _, m in loaded], "dialogs_per_run": [len(r) for r in runs],
           "all_runs": summarize(runs),
           "pairwise": {f"{a.runs[i]} vs {a.runs[j]}": summarize([runs[i], runs[j]])["flip_rate_all_criteria"]
                        for i, j in itertools.combinations(range(len(runs)), 2)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    s = out["all_runs"]
    print(json.dumps({k: v for k, v in s.items() if k not in ("per_criterion", "total_std_per_dialog")}, ensure_ascii=False))
    print({k: v["flip_rate"] for k, v in s["per_criterion"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

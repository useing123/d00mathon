#!/usr/bin/env python
"""Inter-annotator and system–human agreement on the human-validation sample.

    python scripts/agreement.py                         # A vs B vs system, held-out excluded
    python scripts/agreement.py --final                 # include the sealed held-out (end of project only)

Inputs: data/human/labels_A.json, data/human/labels_B.json (from `cqa label --annotator`),
optional data/human/consensus.json (adjudicated; same format), system run (v1 results.json or a store run).
Without an adjudicated file the consensus is the A==B subset; disagreements go to results/human/adjudicate.json.
Outputs: results/human/agreement.json, results/human/disagreements.json, docs/HUMAN_VALIDATION.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CRITERIA = ["01", "02", "03", "04", "05", "06", "07", "08"]
SCALES = {"04": [0, 5, 10, 15, 20], "06": [0, 4, 7, 10]}
WEIGHTS = {"01": 10, "02": 15, "03": 20, "04": 20, "05": 15, "06": 15, "07": 5, "08": 0}
CODES = dict(zip(["contact", "discovery", "identification", "consultation", "empathy", "dialogue", "closure", "critical"],
                 CRITERIA, strict=True))
REPS, SEED = 2000, 7


# ---------- value normalisation ----------
def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "YES" if v else "NO"
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().upper()
    return int(s) if s.lstrip("-").isdigit() else s


def total(vals):
    """Same rule as cqa.scoring: NA dropped, weights rescaled to 100; 08 does not add points."""
    earned, mx = Decimal(0), 0
    for k, w in WEIGHTS.items():
        v = vals.get(k)
        if k == "08" or v in (None, "NA") or not w:
            continue
        mx += w
        if k in SCALES:
            earned += Decimal(w) * Decimal(v) / max(SCALES[k])
        elif v == "YES":
            earned += w
    if not mx:
        return None
    return float((Decimal(earned) * 100 / Decimal(str(mx))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def load_labels(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {cid: {k: norm(v) for k, v in lab.get("expected", {}).items()} for cid, lab in d.items() if lab.get("expected")}


def load_system(path):
    """{cid: {"values": {...}, "rationale": {...}, "total": x, "gross": bool}} from v1 or store format."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    if "results" in d:                        # v1 submission: null = NA, codes are names
        for r in d["results"]:
            vals, rat = {}, {}
            for c in r.get("criteria", []):
                k = CODES.get(c["criterion_id"], c["criterion_id"])
                vals[k] = "NA" if c.get("value") is None else norm(c["value"])
                rat[k] = c.get("rationale", "")
            sc = r.get("score") or {}
            out[r["conversation_id"]] = {"values": vals, "rationale": rat, "total": sc.get("final_score"),
                                         "gross": bool(sc.get("critical_flag"))}
    else:
        for r in d.get("dialogs", []):
            vals = {c["criterion_id"]: norm(c.get("value")) for c in r["criteria"]}
            rat = {c["criterion_id"]: c.get("justification", "") for c in r["criteria"]}
            t = r.get("total") if r.get("total") is not None else r.get("provisional_total")
            out[r["conversation_id"]] = {"values": vals, "rationale": rat, "total": t, "gross": bool(r.get("gross_error"))}
    return out


# ---------- agreement statistics ----------
def cohen_kappa(a, b):
    n = len(a)
    if not n:
        return None
    cats = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def qw_kappa(a, b, scale):
    """Quadratic-weighted κ on ordinal values (pairs with NA are dropped by the caller)."""
    n = len(a)
    if not n:
        return None
    idx = {v: i for i, v in enumerate(scale)}
    k = len(scale)
    ai, bi = [idx[x] for x in a], [idx[y] for y in b]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(ai, bi, strict=True):
        obs[x][y] += 1 / n
    pa = [sum(r) for r in obs]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1 - num / den


def crit_stat(k, a, b):
    """a, b: aligned value lists for criterion k → dict of point estimates."""
    out = {"n": len(a), "agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None, "kappa": cohen_kappa(a, b)}
    if k in SCALES:
        pairs = [(x, y) for x, y in zip(a, b, strict=True) if isinstance(x, int) and isinstance(y, int)]
        out["n_scored"] = len(pairs)
        out["qw_kappa"] = qw_kappa([p[0] for p in pairs], [p[1] for p in pairs], SCALES[k]) if pairs else None
        out["mae"] = sum(abs(x - y) for x, y in pairs) / len(pairs) if pairs else None
    return out


def boot(fn, items, reps=REPS, seed=SEED):
    """Percentile 95% CI of fn(resampled items) over dialogs."""
    if len(items) < 2:
        return None
    rnd = random.Random(seed)
    vals = []
    for _ in range(reps):
        v = fn([items[rnd.randrange(len(items))] for _ in items])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 3), round(vals[int(0.975 * (len(vals) - 1))], 3)]


def r3(x):
    return None if x is None else round(x, 3)


def compare(X, Y, ids, reps=REPS, gross_x=None, gross_y=None, tot_x=None, tot_y=None):
    """X, Y: {cid: {crit: value}}. Per-criterion stats + CIs, total MAE, gross agreement."""
    ids = [i for i in ids if i in X and i in Y]
    per = {}
    for k in CRITERIA:
        rows = [(X[i][k], Y[i][k]) for i in ids if X[i].get(k) is not None and Y[i].get(k) is not None]
        if not rows:
            continue
        s = crit_stat(k, [r[0] for r in rows], [r[1] for r in rows])
        main = "qw_kappa" if k in SCALES else "kappa"
        s["ci"] = {m: boot(lambda rs, m=m, k=k: crit_stat(k, [r[0] for r in rs], [r[1] for r in rs]).get(m), rows, reps)
                   for m in ([main, "agreement", "mae"] if k in SCALES else [main, "agreement"])}
        per[k] = {m: (r3(v) if not isinstance(v, (int, dict)) or isinstance(v, bool) else v) for m, v in s.items()}
    tx = tot_x or {i: total(X[i]) for i in ids}
    ty = tot_y or {i: total(Y[i]) for i in ids}
    diffs = [abs(tx[i] - ty[i]) for i in ids if tx.get(i) is not None and ty.get(i) is not None]
    gx = gross_x or {i: X[i].get("08") == "YES" for i in ids}
    gy = gross_y or {i: Y[i].get("08") == "YES" for i in ids}
    gpairs = [(gx[i], gy[i]) for i in ids]
    return {"dialogs": len(ids), "per_criterion": per,
            "total_mae": r3(sum(diffs) / len(diffs)) if diffs else None,
            "total_mae_ci": boot(lambda d: sum(d) / len(d), diffs, reps),
            "gross": {"agreement": r3(sum(a == b for a, b in gpairs) / len(gpairs)) if gpairs else None,
                      "kappa": r3(cohen_kappa([a for a, _ in gpairs], [b for _, b in gpairs])),
                      "both": sum(a and b for a, b in gpairs), "only_first": sum(a and not b for a, b in gpairs),
                      "only_second": sum(b and not a for a, b in gpairs)}}


def gross_pr(system, human, ids):
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in ids:
        p, h = system[i], human[i]
        c["tp" if p and h else "fp" if p else "fn" if h else "tn"] += 1
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    return {**c, "precision": r3(prec), "recall": r3(rec)}


def build_consensus(A, B, adjudicated=None):
    """Adjudicated labels win; otherwise only criteria where A == B. Returns (consensus, to_adjudicate)."""
    cons, todo = {}, []
    for i in sorted(set(A) & set(B)):
        if adjudicated and i in adjudicated:
            cons[i] = dict(adjudicated[i])
            continue
        cons[i] = {}
        for k in CRITERIA:
            a, b = A[i].get(k), B[i].get(k)
            if a is not None and a == b:
                cons[i][k] = a
            elif a is not None and b is not None:
                todo.append({"conversation_id": i, "criterion_id": k, "A": a, "B": b})
    return cons, todo


def run(A, B, system, ids, adjudicated=None, reps=REPS):
    cons, todo = build_consensus(A, B, adjudicated)
    S = {i: system[i]["values"] for i in system}
    full = [i for i in cons if all(k in cons[i] for k in CRITERIA)]
    out = {"dialogs": {"A": len(set(A) & set(ids)), "B": len(set(B) & set(ids)), "both": len(set(A) & set(B) & set(ids)),
                       "consensus_complete": len(set(full) & set(ids))},
           "inter_annotator": compare(A, B, ids, reps),
           "system_vs_A": compare(S, A, ids, reps, tot_x={i: system[i]["total"] for i in system}),
           "system_vs_B": compare(S, B, ids, reps, tot_x={i: system[i]["total"] for i in system})}
    # consensus: per-criterion on any agreed cell; totals/gross only on dialogs with all 8 agreed (or adjudicated)
    out["system_vs_consensus"] = compare(S, cons, ids, reps)
    tot_ids = [i for i in full if i in ids and i in system]
    out["system_vs_consensus"]["total_mae"] = r3(sum(abs(system[i]["total"] - total(cons[i])) for i in tot_ids
                                                     if system[i]["total"] is not None) / len(tot_ids)) if tot_ids else None
    gids = [i for i in ids if i in system and i in cons and cons[i].get("08") is not None]
    out["gross_pr_vs_consensus"] = gross_pr({i: system[i]["gross"] for i in gids}, {i: cons[i]["08"] == "YES" for i in gids}, gids)
    for who, H in (("A", A), ("B", B)):
        g = [i for i in ids if i in system and i in H and H[i].get("08") is not None]
        out[f"gross_pr_vs_{who}"] = gross_pr({i: system[i]["gross"] for i in g}, {i: H[i]["08"] == "YES" for i in g}, g)
    dis = []
    for i in sorted(ids):
        if i not in system or i not in cons:
            continue
        for k in CRITERIA:
            h, s = cons[i].get(k), S[i].get(k)
            if h is not None and s != h:
                dis.append({"conversation_id": i, "criterion_id": k, "human": h, "system": s,
                            "A": A[i].get(k), "B": B[i].get(k), "system_rationale": system[i]["rationale"].get(k, "")})
    return out, dis, [t for t in todo if t["conversation_id"] in ids]


# ---------- report ----------
def fmt(x, ci=None):
    if x is None:
        return "—"
    return f"{x:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def markdown(res, dis, todo, meta):
    L = ["# Human validation", "",
         f"Generated {meta['generated']} by `python scripts/agreement.py{' --final' if meta['final'] else ''}`. "
         f"Labels: two independent annotators (A, B) on the stratified DEV sample (`results/human/sample.json`); "
         f"{'sealed held-out INCLUDED' if meta['final'] else 'sealed held-out (15 dialogs) EXCLUDED'}. "
         f"System run: `{meta['system']}`. 95% CIs: dialog-level percentile bootstrap ({meta['reps']} resamples).", "",
         f"Dialogs labelled: A={res['dialogs']['A']}, B={res['dialogs']['B']}, both={res['dialogs']['both']}, "
         f"consensus complete={res['dialogs']['consensus_complete']}.", ""]
    for key, title in (("inter_annotator", "Inter-annotator agreement (A vs B)"), ("system_vs_A", "System vs A"),
                       ("system_vs_B", "System vs B"), ("system_vs_consensus", "System vs consensus")):
        r = res[key]
        L += [f"## {title}", "", "| criterion | n | raw agreement | κ (Cohen; quadratic-weighted for 04/06) | MAE (04/06) |",
              "|---|---|---|---|---|"]
        for k, s in r["per_criterion"].items():
            main = "qw_kappa" if k in SCALES else "kappa"
            L.append(f"| {k} | {s['n']} | {fmt(s['agreement'], s['ci'].get('agreement'))} | {fmt(s.get(main), s['ci'].get(main))} | "
                     f"{fmt(s.get('mae'), s['ci'].get('mae')) if k in SCALES else ''} |")
        g = r["gross"]
        L += ["", f"Total-score MAE: {fmt(r['total_mae'], r.get('total_mae_ci'))} points. Gross-error (08) agreement "
              f"{fmt(g['agreement'])}, κ {fmt(g['kappa'])} (both={g['both']}, only first={g['only_first']}, "
              f"only second={g['only_second']}).", ""]
    L += ["## Gross-error detection (system) on real dialogs", "", "| reference | tp | fp | fn | tn | precision | recall |",
          "|---|---|---|---|---|---|---|"]
    for ref in ("consensus", "A", "B"):
        g = res[f"gross_pr_vs_{ref}"]
        L.append(f"| {ref} | {g['tp']} | {g['fp']} | {g['fn']} | {g['tn']} | {fmt(g['precision'])} | {fmt(g['recall'])} |")
    L += ["", f"## Disagreements system vs consensus ({len(dis)})", "",
          "Full list with the system's rationale: `results/human/disagreements.json`.", "",
          "| dialog | criterion | human | system | system rationale |", "|---|---|---|---|---|"]
    for d in dis[:60]:
        rat = (d["system_rationale"] or "").replace("|", "/").replace("\n", " ")[:160]
        L.append(f"| {d['conversation_id']} | {d['criterion_id']} | {d['human']} | {d['system']} | {rat} |")
    L += ["", f"Cells still awaiting adjudication (A ≠ B): {len(todo)} — see `results/human/adjudicate.json`.", ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="data/human/labels_A.json")
    p.add_argument("--b", default="data/human/labels_B.json")
    p.add_argument("--consensus", default="data/human/consensus.json", help="adjudicated labels (optional)")
    p.add_argument("--system", default="results/dev/results.json")
    p.add_argument("--sample", default="results/human/sample.json")
    p.add_argument("--heldout", default="results/human/heldout_sealed.json")
    p.add_argument("--final", action="store_true", help="include the sealed held-out (end of project only)")
    p.add_argument("--out-dir", default="results/human")
    p.add_argument("--md", default="docs/HUMAN_VALIDATION.md")
    p.add_argument("--reps", type=int, default=REPS)
    a = p.parse_args(argv)
    for f in (a.a, a.b):
        if not Path(f).exists():
            print(f"no human labels at {f} — run `python -m cqa label --annotator A|B --sample results/human/session_A|B.json` "
                  f"first (labels are never generated automatically)")
            return 2
    A, B = load_labels(a.a), load_labels(a.b)
    adj = load_labels(a.consensus) if Path(a.consensus).exists() else None
    system = load_system(a.system)
    ids = json.loads(Path(a.sample).read_text())["ids"] if Path(a.sample).exists() else sorted(set(A) | set(B))
    held = set(json.loads(Path(a.heldout).read_text())["ids"]) if Path(a.heldout).exists() else set()
    if not a.final:
        ids = [i for i in ids if i not in held]
    res, dis, todo = run(A, B, system, ids, adj, a.reps)
    meta = {"generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "final": a.final, "system": a.system,
            "reps": a.reps, "heldout_excluded": not a.final, "n_ids": len(ids), "adjudicated": bool(adj)}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps({"meta": meta, **res}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "disagreements.json").write_text(json.dumps(dis, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "adjudicate.json").write_text(json.dumps(todo, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.md).write_text(markdown(res, dis, todo, meta), encoding="utf-8")
    ia = res["inter_annotator"]
    print(f"A–B: total MAE {ia['total_mae']}; system–consensus total MAE {res['system_vs_consensus']['total_mae']}; "
          f"{len(dis)} system disagreements; {len(todo)} cells to adjudicate → {out}, {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

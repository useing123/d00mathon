"""Gold regression set: hand-labelled synthetic dialogs + agreement metrics.

Each examples/gold/*.json holds {"dialog": <record>, "expected": {"01": "YES", ..., "04": 15, ...},
"expected_total", "rationale"}. `python -m cqa gold [--mock]` evaluates them and reports:
per-criterion exact-match agreement, overall agreement, total MAE, gross-flag precision/recall,
NA-vs-applicable confusion, list of disagreements. Writes results/gold_report.json.
"""
import json
from collections import Counter
from pathlib import Path

from . import config, data
from .evaluator import Evaluator

GOLD_DIR = Path(__file__).resolve().parent.parent / "examples" / "gold"


def load_gold(gold_dir=None):
    items = []
    for p in sorted(Path(gold_dir or GOLD_DIR).glob("*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        g["file"] = p.name
        items.append(g)
    return items


def _norm(v):
    return str(v).upper() if v is not None else None


def compute_metrics(gold, results):
    """Pure function: gold items + dialog results (same conversation ids) -> metrics dict."""
    by = {d["conversation_id"]: d for d in results}
    per = {}
    na_conf = Counter()   # (expected_is_na, predicted_is_na)
    gross = Counter()
    abs_err, mismatches = [], []
    total_match = total_n = 0
    for g in gold:
        cid = g["dialog"]["conversation_id"]
        d = by.get(cid)
        if d is None:
            continue
        pred = {c["criterion_id"]: c.get("value") for c in d["criteria"]}
        for k, ev in g["expected"].items():
            pv = pred.get(k)
            ok = _norm(pv) == _norm(ev)
            s = per.setdefault(k, {"n": 0, "match": 0})
            s["n"] += 1
            s["match"] += ok
            total_n += 1
            total_match += ok
            if not ok:
                mismatches.append({"conversation_id": cid, "criterion_id": k, "expected": ev, "predicted": pv,
                                   "rationale": g.get("rationale", {}).get(k, "")})
            na_conf[(f"exp_{'NA' if _norm(ev) == 'NA' else 'applicable'}",
                     f"pred_{'NA' if _norm(pv) == 'NA' else ('FAILED' if pv is None else 'applicable')}")] += 1
        eg, pg = bool(g.get("expected_gross_error")), bool(d.get("gross_error"))
        gross["tp" if eg and pg else "fp" if pg else "fn" if eg else "tn"] += 1
        pt = d.get("total") if d.get("total") is not None else d.get("provisional_total")
        if pt is not None and g.get("expected_total") is not None:
            abs_err.append(abs(pt - g["expected_total"]))
    for s in per.values():
        s["agreement"] = round(s["match"] / s["n"], 3) if s["n"] else None
    prec = gross["tp"] / (gross["tp"] + gross["fp"]) if (gross["tp"] + gross["fp"]) else None
    rec = gross["tp"] / (gross["tp"] + gross["fn"]) if (gross["tp"] + gross["fn"]) else None
    return {
        "dialogs": len([g for g in gold if g["dialog"]["conversation_id"] in by]),
        "per_criterion": dict(sorted(per.items())),
        "overall_agreement": round(total_match / total_n, 3) if total_n else None,
        "total_mae": round(sum(abs_err) / len(abs_err), 2) if abs_err else None,
        "gross": {**dict(gross), "precision": prec, "recall": rec},
        "na_confusion": {f"{a}/{b}": n for (a, b), n in sorted(na_conf.items())},
        "incomplete": [d["conversation_id"] for d in results if d.get("status") == "incomplete"],
        "mismatches": mismatches,
    }


def load_labels(path, group):
    """Human labels for real dialogs (`python -m cqa label`) → gold items (dialogs are loaded from the group)."""
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = {r["conversation_id"]: r for r in data.load_group(group)}
    items = []
    for cid, lab in labels.items():
        if cid not in recs or not lab.get("expected"):
            continue
        rec = recs[cid]
        from . import checklist as ckl, scoring
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        t = scoring.compute_total(cl["criteria"], lab["expected"], cl["assumptions"]["rounding"], cl["assumptions"]["decimals"])
        items.append({"dialog": {k: v for k, v in rec.items() if k in ("conversation_id", "channel", "language", "transcript")},
                      "expected": lab["expected"], "expected_total": t["total"], "expected_gross_error": t["gross_error"],
                      "rationale": lab.get("notes", {}), "file": f"labels:{cid}"})
    return items


def run_gold(mock=False, gold_dir=None, out=None, use_cache=True, labels=None, group="DEV"):
    if mock:
        config.PROVIDER = "mock"
    gold = load_labels(labels, group) if labels else load_gold(gold_dir)
    recs = [data.normalize(g["dialog"], i) for i, g in enumerate(gold)]
    results, meta = Evaluator(use_cache=use_cache).evaluate_many(recs)
    m = compute_metrics(gold, results)
    m["meta"] = {k: meta.get(k) for k in ("provider", "model", "cost_usd", "elapsed_s", "errors")}
    m["source"] = f"labels:{labels}" if labels else str(gold_dir or GOLD_DIR)
    out = Path(out or (config.RESULTS_DIR / "gold_report.json"))
    from .util import atomic_write
    atomic_write(out, json.dumps({"metrics": m, "dialogs": results}, ensure_ascii=False, indent=1))
    m["report"] = str(out)
    return m


def format_report(m):
    lines = [f"Gold set: {m['dialogs']} dialogs | overall agreement {m['overall_agreement']} | total MAE {m['total_mae']}"]
    for k, s in m["per_criterion"].items():
        lines.append(f"  {k}: {s['match']}/{s['n']} = {s['agreement']}")
    g = m["gross"]
    lines.append(f"  08 gross flag: precision {g['precision']} recall {g['recall']} (tp={g.get('tp', 0)} fp={g.get('fp', 0)} "
                 f"fn={g.get('fn', 0)})")
    lines.append("  NA confusion: " + ", ".join(f"{k}={v}" for k, v in m["na_confusion"].items()))
    for x in m["mismatches"]:
        lines.append(f"  ✗ {x['conversation_id']} {x['criterion_id']}: expected {x['expected']} got {x['predicted']}")
    return "\n".join(lines)


def label_dialog(rec, checklist, existing=None, ask=input, say=print):
    """Interactive labelling of one dialog against the checklist. Returns {"expected": {...}, "notes": {...}}."""
    say(data.format_transcript(rec))
    say("")
    exp, notes = {}, {}
    from .checklist import strict_value
    for cr in checklist["criteria"]:
        allowed = ([str(x) for x in cr["scale"]] if cr.get("type") == "scale" else ["YES", "NO"]) + (["NA"] if cr.get("allow_na") else [])
        prev = (existing or {}).get("expected", {}).get(cr["id"])
        while True:
            v = ask(f"{cr['id']} {cr['name']} [{'/'.join(allowed)}]{f' (Enter = {prev})' if prev is not None else ''}: ").strip()
            if not v and prev is not None:
                v = str(prev)
            sv = strict_value(cr, v)
            if sv is not None:
                exp[cr["id"]] = sv
                break
            say(f"  invalid; allowed: {', '.join(allowed)}")
        n = ask("   note (optional): ").strip()
        if n:
            notes[cr["id"]] = n
    return {"expected": exp, "notes": notes}

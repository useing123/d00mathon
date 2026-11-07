"""Hybrid (KZ signals ON) vs cascade (OFF) on gold + held-out, overall and KZ subset; Jev alone = routing.local_value.
Reads results/hybrid_kz/{cascade,hybrid}_{gold,gold_holdout}.json produced by `python -m cqa gold --preset ...`."""
import json
from pathlib import Path

from cqa import data, rules_kz

R = Path("results/hybrid_kz")


def gold(d):
    out = {}
    for p in sorted(Path(d).glob("*.json")):
        g = json.loads(p.read_text())
        rec = data.normalize(g["dialog"], 0)
        kz = (rec.get("language") or "").upper() == "KZ" or rules_kz.lang_of(
            [t.get("text") or "" for t in rec["transcript"] if t["speaker"] == "client"]) in ("KZ", "MIXED")
        out[rec["conversation_id"]] = (g["expected"], kz)
    return out


def agree(G, res, field, kz_only=False):
    n = ok = 0
    for d in res["dialogs"]:
        exp, kz = G[d["conversation_id"]]
        if kz_only and not kz:
            continue
        for c in d["criteria"]:
            v = c["routing"].get("local_value") if field == "jev" else c["value"]
            n += 1
            ok += str(v).upper() == str(exp[c["criterion_id"]]).upper()
    return {"labels": n, "agree": round(ok / n, 4) if n else None, "correct": ok}


out = {}
for s, d in (("gold", "examples/gold"), ("holdout", "examples/gold_holdout")):
    G = gold(d)
    for p in ("cascade", "hybrid"):
        res = json.loads((R / f"{p}_gold{'_holdout' if s == 'holdout' else ''}.json").read_text())
        for sub in (False, True):
            key = f"{s}{'_kz' if sub else ''}"
            out.setdefault(key, {})[p] = agree(G, res, "value", sub)
            out[key][f"jev_{'signals' if p == 'hybrid' else 'plain'}"] = agree(G, res, "jev", sub)
print(json.dumps(out, indent=1))
(R / "summary.json").write_text(json.dumps(out, indent=1))

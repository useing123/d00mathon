"""Advanced router layer (docs/ROUTER.md): calibrated tier-1 confidence + per-criterion cost-aware policy.

Pure, offline, dependency-free. Three pieces:

* calibration  — per-criterion temperature scaling of the tier-1 (Jev) confidence, shrunk towards a global
                 temperature (16 labels per criterion is too few for isotonic per criterion); global isotonic
                 (PAV) is provided for comparison.
* metrics      — risk–coverage curve, AURC, ECE, reliability bins.
* policy       — per criterion: escalate tier 1 → Haiku when calibrated p < theta_k; then Haiku's answer is kept
                 ("none"), replaced by Sonnet when Haiku disagrees with tier 1 ("disagree") or always ("always").
                 Fail-closed rules are not tunable: tier-1 error/invalid → Haiku, 08 = YES → Haiku.

Live hook: ``CQA_ROUTER_POLICY=results/router_policy/policy_<budget>.json`` makes cqa.router use ``patch()``
(replaces its ``tier1_reasons`` / ``tier2_reasons``). Unset → the cascade is unchanged.
"""
from __future__ import annotations

import json
import math

EPS = 1e-4
THETAS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]   # 1.01 = always escalate
S2_MODES = ["none", "disagree", "always"]


# ------------------------------------------------------------------------------------------ calibration
def _logit(p):
    p = min(max(float(p), EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def temp_scale(p, t):
    return _sig(_logit(p) / t)


def nll(pairs, t):
    s = 0.0
    for p, y in pairs:
        q = min(max(temp_scale(p, t), 1e-6), 1 - 1e-6)
        s -= math.log(q if y else 1 - q)
    return s


T_GRID = [round(0.5 * 1.1 ** i, 4) for i in range(60)]   # 0.5 … ~150


def fit_temperature(pairs, prior_t=None, prior_w=0.0):
    """argmin NLL over T (grid); optional shrinkage: + prior_w * (log T - log prior_t)^2."""
    if not pairs:
        return prior_t or 1.0

    def obj(t):
        return nll(pairs, t) + (prior_w * (math.log(t) - math.log(prior_t)) ** 2 if prior_t else 0.0)
    return min(T_GRID, key=obj)


def fit_calibration(items, prior_w=4.0):
    """items: [{k, p, y}] → {"global_T", "T": {k: T_k}}. Per-criterion T is shrunk to the global T."""
    pairs = [(i["p"], i["y"]) for i in items]
    g = fit_temperature(pairs)
    per = {}
    for k in sorted({i["k"] for i in items}):
        per[k] = fit_temperature([(i["p"], i["y"]) for i in items if i["k"] == k], prior_t=g, prior_w=prior_w)
    return {"method": "temperature", "global_T": g, "T": per, "prior_w": prior_w}


def calibrate(p, k, cal):
    if not cal or p is None:
        return p
    return temp_scale(p, cal["T"].get(k, cal["global_T"]))


def isotonic_fit(pairs):
    """Pool-adjacent-violators on (p, y) → list of (upper_p, value) steps (non-decreasing)."""
    blocks = []
    for p, y in sorted(pairs):
        blocks.append([p, float(y), 1])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            b = blocks.pop()
            blocks[-1][0] = b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    return [(b[0], b[1] / b[2]) for b in blocks]


def isotonic_apply(steps, p):
    for up, v in steps:
        if p <= up:
            return v
    return steps[-1][1] if steps else p


# ------------------------------------------------------------------------------------------ metrics
def risk_coverage(pairs):
    """pairs (confidence, correct) → [(coverage, risk)] keeping the most confident first; ties enter together."""
    pts, n = [], len(pairs)
    srt = sorted(pairs, key=lambda x: -x[0])
    err = kept = 0
    i = 0
    while i < n:
        j = i
        while j < n and srt[j][0] == srt[i][0]:
            err += not srt[j][1]
            kept += 1
            j += 1
        pts.append((kept / n, err / kept))
        i = j
    return pts


def aurc(pairs):
    """Area under the risk–coverage curve (per-sample, ties averaged): mean over k of risk@top-k."""
    n = len(pairs)
    if not n:
        return None
    rc = risk_coverage(pairs)
    s, prev = 0.0, 0
    for cov, risk in rc:
        kept = round(cov * n)
        s += risk * (kept - prev)
        prev = kept
    return s / n


def reliability(pairs, bins=10):
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if sel:
            out.append({"lo": lo, "hi": hi, "n": len(sel), "conf": sum(p for p, _ in sel) / len(sel),
                        "acc": sum(y for _, y in sel) / len(sel)})
    return out


def ece(pairs, bins=10):
    n = len(pairs)
    return sum(b["n"] / n * abs(b["acc"] - b["conf"]) for b in reliability(pairs, bins)) if n else None


# ------------------------------------------------------------------------------------------ policy
def _same(a, b):
    return a is not None and b is not None and str(a).upper() == str(b).upper()


def route_item(it, rule, cal=None):
    """One (dialog, criterion) under a per-criterion rule {theta, s2}. ``it`` holds tier answers:
    jev_v/jev_p, haiku_v, sonnet_v (+ optional qwen_v; rule["agree"] escalates when Jev ≠ Qwen).
    Returns (value, tier, reasons)."""
    why = []
    if it.get("jev_v") is None:
        why.append("local_error")
    else:
        p = calibrate(it.get("jev_p") or 0.0, it["k"], cal)
        if p < rule["theta"]:
            why.append("low_conf")
        if it["k"] == "08" and str(it["jev_v"]).upper() == "YES":
            why.append("gross_yes")
        if rule.get("agree") and it.get("qwen_v") is not None and not _same(it["jev_v"], it["qwen_v"]):
            why.append("tier1_disagree")
    if not why:
        return it["jev_v"], 1, why
    s2 = rule.get("s2", "none")
    if it.get("haiku_v") is None or s2 == "always" or (s2 == "disagree" and not _same(it["haiku_v"], it["jev_v"])):
        if it.get("sonnet_v") is not None:
            return it["sonnet_v"], 3, why
    return it["haiku_v"], 2, why


def run_policy(items, policy, cal=None, prices=None, n_dialogs=None):
    """Apply {k: rule} to items → summary {acc, correct, n, tiers, usd_per_dialog, per_dialog, per_item}."""
    prices = prices or {}
    tiers, calls, ok, per_item = {1: 0, 2: 0, 3: 0}, {"haiku": 0, "sonnet": 0}, 0, []
    for it in items:
        v, t, why = route_item(it, policy[it["k"]], cal)
        tiers[t] += 1
        calls["haiku"] += t >= 2
        calls["sonnet"] += t == 3
        good = _same(v, it["exp"])
        ok += good
        per_item.append({"cid": it["cid"], "k": it["k"], "ok": good, "exp": it["exp"], "pred": v, "tier": t,
                         "why": why})
    nd = n_dialogs or len({it["cid"] for it in items}) or 1
    usd = prices.get("tier1_dialog", 0.0) + (calls["haiku"] * prices.get("haiku_call", 0.0)
                                            + calls["sonnet"] * prices.get("sonnet_call", 0.0)) / nd
    n = len(items) or 1
    return {"acc": ok / n, "correct": ok, "n": len(items), "tiers": {k: v / n for k, v in tiers.items()},
            "calls": calls, "usd_per_dialog": usd, "per_item": per_item}


def options(agree=False):
    return [{"theta": t, "s2": s, **({"agree": True} if agree else {})} for t in THETAS for s in S2_MODES]


def pareto_policies(items, cal, prices, agree=False):
    """Exact Pareto frontier (cost, correct) of per-criterion policies: per-criterion option tables combined
    with dominance pruning (a multiple-choice knapsack). Returns [(usd_per_dialog, correct, policy)]."""
    nd = len({it["cid"] for it in items}) or 1
    front = [(prices.get("tier1_dialog", 0.0), 0, {})]
    for k in sorted({it["k"] for it in items}):
        sub = [it for it in items if it["k"] == k]
        opts = []
        for r in options(agree):
            s = run_policy(sub, {k: r}, cal, {"haiku_call": prices["haiku_call"], "sonnet_call": prices["sonnet_call"]}, nd)
            opts.append((s["usd_per_dialog"], s["correct"], r))
        opts = _prune([(c, a, r) for c, a, r in opts])
        nxt = [(c0 + c, a0 + a, {**p0, k: r}) for c0, a0, p0 in front for c, a, r in opts]
        front = _prune(nxt)
    return front


def _prune(pts):
    """Keep non-dominated (lower cost, higher correct); among exact ties keep the first (= more conservative
    option order: lower theta, then none < disagree < always)."""
    pts = sorted(pts, key=lambda x: (round(x[0], 9), -x[1]))
    out, best = [], -1
    for c, a, r in pts:
        if a > best:
            out.append((c, a, r))
            best = a
    return out


def best_under_budget(front, budget):
    ok = [f for f in front if f[0] <= budget + 1e-12]
    return max(ok, key=lambda f: (f[1], -f[0])) if ok else None


# ------------------------------------------------------------------------------------------ live hook
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def patch(router_globals, path):
    """Replace cqa.router.tier1_reasons / tier2_reasons by the per-criterion policy in ``path``. The current
    criterion id is tracked per thread (tier2_reasons has no criterion argument)."""
    import threading
    pol = load(path)
    rules, cal = pol["policy"], pol.get("calibration")
    orig1 = router_globals["tier1_reasons"]
    cur = threading.local()

    def tier1_reasons(cr, local, s):
        cur.k = cr["id"]
        why = [w for w in orig1(cr, local, s) if w not in ("low_conf", "hard_criterion")]
        rule = rules.get(cr["id"])
        if rule and local and not local.get("error") and local.get("value") is not None:
            p = calibrate(float(local.get("confidence") or 0), cr["id"], cal)
            if p < rule["theta"]:
                why.append("low_conf_calibrated")
        return sorted(set(why))

    def tier2_reasons(local, haiku, s):
        return policy_tier2(rules.get(getattr(cur, "k", None)), local, haiku)

    router_globals["tier1_reasons"] = tier1_reasons
    router_globals["tier2_reasons"] = tier2_reasons
    return pol


def policy_tier2(rule, local, haiku):
    """Sonnet reasons for a criterion rule (used by the live hook)."""
    if haiku is None or haiku.get("error") or haiku.get("value") is None:
        return ["haiku_error"]
    s2 = (rule or {}).get("s2", "none")
    if s2 == "always":
        return ["policy_always"]
    if s2 == "disagree" and local and not _same(local.get("value"), haiku.get("value")):
        return ["tier1_haiku_disagree"]
    return []


# ------------------------------------------------------------------------------------------ EXPERIMENTS.md
REGISTER = ("per-criterion policy, budget $0.001/dialog", "per-criterion policy, budget $0.0025/dialog",
            "cascade + Sonnet on any Jev≠Haiku disagreement", "policy $0.004 + escalate on Jev≠Qwen3.8-27B")


def registry_entries(root):
    """Rows for scripts/build_experiments.py (its build_registry schema) from results/router_policy/study.json."""
    from pathlib import Path
    p = Path(root) / "results" / "router_policy" / "study.json"
    if not p.exists():
        return []
    out = []
    for r in json.loads(p.read_text(encoding="utf-8")).get("policies", []):
        if r["name"] not in REGISTER:
            continue
        h = r["holdout"]
        out.append(dict(name=f"router policy: {r['name']}", family="router policy (offline replay)",
                        holdout=[str(Path(root) / r["file"])], gold=[], usd=h["usd_per_dialog"], sec=None, gpu="—",
                        src=str(Path(root) / r["file"]),
                        tiers="/".join(f"{100 * h['tiers'][k]:.0f}" for k in ("1", "2", "3")),
                        note="offline replay, docs/ROUTER.md"))
    return out

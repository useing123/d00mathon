#!/usr/bin/env python3
"""Offline advanced-router study (docs/ROUTER.md). NO API calls, NO GPU: replays cached tier outputs.

    python scripts/router_policy.py            # fit on tuned gold, evaluate every policy ONCE on held-out

Sources (all already on disk):
  tuned gold (16 dialogs)  results/oss/jev_all_gold.json — --all-escalate run: Jev, Haiku AND Sonnet for all 128
  held-out (10 dialogs)    Jev: results/jev/holdout.dialogs.json (run 1 = what the cascade used)
                           Haiku/Sonnet: .cache/router (exact cascade prompt) where cached, else the single-model
                           benchmark run with the same prompts (results/bench/{haiku,sonnet}__holdout.dialogs.json)
                           Qwen3.8-27B: results/kz_models/q38_27b_fp8_holdout.json (all 80; no gold run exists)
Outputs: results/router_policy/{study.json, policy_<budget>.json, heldout__<policy>.dialogs.json},
         docs/img/{risk_coverage,reliability,router_pareto}.svg
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cqa import routing_policy as rp  # noqa: E402

OUT = ROOT / "results" / "router_policy"
IMG = ROOT / "docs" / "img"
CRIT = ["01", "02", "03", "04", "05", "06", "07", "08"]
BUDGETS = [0.001, 0.0015, 0.0025, 0.004, 0.008]
REPS = 10000


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def prices():
    g = jl(ROOT / "results/oss/jev_all_gold.json")
    n = g["routing"]["criteria"]   # force=3: Haiku and Sonnet were called for every criterion
    u = g["routing"]["cloud_usd_by_model"]
    return {"tier1_dialog": jl(ROOT / "results/jev/holdout.json")["usd_per_dialog"],
            "haiku_call": u["claude-haiku-5-5"] / n, "sonnet_call": u["claude-sonnet-5-5"] / n,
            "qwen_gpu_dialog": 0.0294,   # Qwen3.8-27B-FP8 on 1x H100 SXM, EXPERIMENTS.md row 5
            "source": "Haiku/Sonnet $/call = cloud_usd_by_model / 128 calls in jev_all_gold.json; "
                      "Jev $/dialog = results/jev/holdout.json; Qwen GPU $/dialog = EXPERIMENTS.md"}


def gold_labels(d):
    out = {}
    for p in sorted((ROOT / "examples" / d).glob("*.json")):
        g = jl(p)
        out[g["dialog"]["conversation_id"]] = g
    return out


def _dialogs(p):
    d = jl(p)
    if isinstance(d, dict):
        d = d["dialogs_detail"] if isinstance(d.get("dialogs_detail"), list) else d["dialogs"]
    return {x["conversation_id"]: {c["criterion_id"]: c for c in x["criteria"]} for x in d}


def load_gold_items():
    G = gold_labels("gold")
    d = jl(ROOT / "results/oss/jev_all_gold.json")
    items = []
    for dl in d["dialogs_detail"]:
        for c in dl["criteria"]:
            r, cid = c["routing"], dl["conversation_id"]
            h, s = r.get("haiku") or {}, r.get("sonnet") or {}
            items.append({"cid": cid, "k": c["criterion_id"], "exp": G[cid]["expected"][c["criterion_id"]],
                          "jev_v": r["local_value"], "jev_p": r["local_conf"], "haiku_v": h.get("value"),
                          "haiku_p": h.get("confidence"), "sonnet_v": s.get("value"), "sonnet_p": s.get("confidence"),
                          "src": "cascade-cache"})
    return items


def _router_cache(which, model, rec, cl, cr):
    from cqa import checklist as ckl
    key = hashlib.sha1(f"anthropic:{model}:LLM:{ckl.criterion_hash(cl, cr)}:{rec['hash']}".encode()).hexdigest()[:20]
    p = ROOT / ".cache" / "router" / f"{which}_{key}.json"
    return jl(p) if p.exists() else None


def load_holdout_items():
    from cqa import checklist as ckl, data
    G = gold_labels("gold_holdout")
    jev = _dialogs(ROOT / "results/jev/holdout.dialogs.json")
    hb = _dialogs(ROOT / "results/bench/haiku__holdout.dialogs.json")
    sb = _dialogs(ROOT / "results/bench/sonnet__holdout.dialogs.json")
    qf = ROOT / "results/kz_models/q38_27b_fp8_holdout.json"
    qw = _dialogs(qf) if qf.exists() else {}
    items, src = [], {"haiku_cache": 0, "haiku_bench": 0, "sonnet_cache": 0, "sonnet_bench": 0}
    for i, (cid, g) in enumerate(sorted(G.items())):
        rec = data.normalize(g["dialog"], i)
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        for cr in cl["criteria"]:
            k = cr["id"]
            j = jev[cid][k]
            h = _router_cache("escalate", "claude-haiku-5-5", rec, cl, cr)
            hs = "cache" if h else "bench"
            src["haiku_" + hs] += 1
            h = h or hb[cid][k]
            s = _router_cache("judge", "claude-sonnet-5-5", rec, cl, cr)
            ss = "cache" if s else "bench"
            src["sonnet_" + ss] += 1
            s = s or sb[cid][k]
            items.append({"cid": cid, "k": k, "exp": g["expected"][k], "jev_v": j.get("value"),
                          "jev_p": j.get("confidence"), "haiku_v": h.get("value"), "haiku_p": h.get("confidence"),
                          "sonnet_v": s.get("value"), "sonnet_p": s.get("confidence"),
                          "qwen_v": (qw.get(cid, {}).get(k) or {}).get("value"), "h_src": hs, "s_src": ss})
    return items, src


def cascade_replay(items, pr, t1=0.7, hard_conf=0.9, t2=0.7):
    """The current cascade (preset `cascade`) replayed with cqa.router's own reason functions."""
    from cqa import router
    s = router.settings(first="jev", t1=t1, hard_conf=hard_conf, hard="04,05,06", t2=t2, haiku_low=0.0,
                        judge="claude-sonnet-5-5")
    per, calls = [], {"haiku": 0, "sonnet": 0}
    for it in items:
        local = {"value": it["jev_v"], "confidence": it["jev_p"]}
        why1 = [w for w in router.tier1_reasons({"id": it["k"], "gross": it["k"] == "08"}, local, s)
                if w != "evidence_failed"]
        v, t = it["jev_v"], 1
        if why1:
            h = {"value": it["haiku_v"], "confidence": it["haiku_p"]}
            v, t = (it["sonnet_v"], 3) if router.tier2_reasons(local, h, s) else (it["haiku_v"], 2)
        calls["haiku"] += t >= 2
        calls["sonnet"] += t == 3
        per.append({"cid": it["cid"], "k": it["k"], "ok": rp._same(v, it["exp"]), "exp": it["exp"], "pred": v,
                    "tier": t, "why": why1})
    nd = len({i["cid"] for i in items})
    n = len(per)
    return {"acc": sum(p["ok"] for p in per) / n, "correct": sum(p["ok"] for p in per), "n": n,
            "tiers": {t: sum(p["tier"] == t for p in per) / n for t in (1, 2, 3)}, "calls": calls,
            "usd_per_dialog": pr["tier1_dialog"] + (calls["haiku"] * pr["haiku_call"]
                                                     + calls["sonnet"] * pr["sonnet_call"]) / nd,
            "per_item": per}


def _total(cid, crits):
    """Total + gross flag with the same deterministic scoring as the evaluator (for MAE / gross P/R columns)."""
    from cqa import checklist as ckl, data, scoring
    g = gold_labels("gold_holdout")[cid]
    cl = ckl.load(ckl.pick_for_channel(data.normalize(g["dialog"], 0)["kind"]))
    vals = {}
    for c in crits:
        v = c["value"]
        vals[c["criterion_id"]] = str(v).upper() if str(v).upper() in ("YES", "NO", "NA") else int(v)
    try:
        t = scoring.compute_total(cl["criteria"], vals)
    except (ValueError, TypeError):
        return {}
    return {"total": t["total"], "gross_error": t["gross_error"]}


# ------------------------------------------------------------------------------------------------ stats
def to_run(per_item):
    run = {}
    for p in per_item:
        run.setdefault(p["cid"], {"labels": {}})["labels"][p["k"]] = (p["ok"], p["exp"], p["pred"])
    return {"per": run}


def ci_and_delta(res, ref):
    import build_experiments as be
    run, rr = to_run(res["per_item"]), to_run(ref["per_item"])
    acc = be.dialog_acc([run])
    lo, hi = be.boot_ci(list(acc.values()), REPS)
    d = be.paired([run], [rr], REPS)
    return {"ci": [lo, hi], "vs_cascade": {k: d[k] for k in ("delta", "lo", "hi", "p_boot", "b", "c", "p_mcnemar")}}


# ------------------------------------------------------------------------------------------------ svg
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _frame(W, H, L, B, T, R, xlab, ylab, xt, yt, X, Y, title):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{(L + W - R) / 2}" y="16" text-anchor="middle" font-size="13">{title}</text>']
    for v, lab in xt:
        o.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{H - B}" stroke="#ddd"/>')
        o.append(f'<text x="{X(v):.1f}" y="{H - B + 15}" text-anchor="middle">{lab}</text>')
    for v, lab in yt:
        o.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke="#ddd"/>')
        o.append(f'<text x="{L - 5}" y="{Y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle">{xlab}</text>')
    o.append(f'<text x="16" y="{(T + H - B) / 2}" transform="rotate(-90 16 {(T + H - B) / 2})" '
             f'text-anchor="middle">{ylab}</text>')
    return o


def _save(o, name):
    o.append("</svg>")
    IMG.mkdir(parents=True, exist_ok=True)
    (IMG / name).write_text("\n".join(o) + "\n", encoding="utf-8")


def svg_risk_coverage(curves, name="risk_coverage.svg"):
    W, H, L, B, T, R = 760, 440, 60, 50, 30, 190
    ymax = 0.5
    X = lambda v: L + v * (W - L - R)  # noqa: E731
    Y = lambda v: T + (ymax - min(v, ymax)) / ymax * (H - T - B)  # noqa: E731
    o = _frame(W, H, L, B, T, R, "coverage (share of tier-1 answers kept, most confident first)",
               "risk = error rate of kept answers", [(i / 10, f"{i / 10:.1f}") for i in range(11)],
               [(i / 10, f"{i / 10:.1f}") for i in range(6)], X, Y, "Risk–coverage of Jev confidence (tuned gold)")
    for i, (lab, pts, a) in enumerate(curves):
        c = "#000" if lab == "all" else COL[i % len(COL)]
        w = 2.5 if lab == "all" else 1.2
        path = [(0.0, pts[0][1])] + pts if pts else []
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="{w}" points="'
                 + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in path) + '"/>')
        o.append(f'<text x="{W - R + 10}" y="{T + 12 + i * 15}" fill="{c}">{lab}: AURC {a:.3f}</text>')
    _save(o, name)


def svg_reliability(before, after, ece_b, ece_a, name="reliability.svg"):
    W, H, L, B, T, R = 560, 440, 60, 50, 30, 30
    X = lambda v: L + v * (W - L - R)  # noqa: E731
    Y = lambda v: T + (1 - v) * (H - T - B)  # noqa: E731
    o = _frame(W, H, L, B, T, R, "predicted confidence", "observed accuracy",
               [(i / 10, f"{i / 10:.1f}") for i in range(11)], [(i / 10, f"{i / 10:.1f}") for i in range(11)], X, Y,
               "Reliability of Jev confidence (tuned gold, 10 bins)")
    o.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#999" stroke-dasharray="4"/>')
    for bins, c, dx in ((before, "#d62728", -3), (after, "#1f77b4", 3)):
        o.append(f'<polyline fill="none" stroke="{c}" points="'
                 + " ".join(f"{X(b['conf']):.1f},{Y(b['acc']):.1f}" for b in bins) + '"/>')
        for b in bins:
            r = 2 + min(8, b["n"] ** 0.5)
            o.append(f'<circle cx="{X(b["conf"]) + dx:.1f}" cy="{Y(b["acc"]):.1f}" r="{r:.1f}" fill="{c}" '
                     f'fill-opacity="0.5"><title>n={b["n"]} conf={b["conf"]:.3f} acc={b["acc"]:.3f}</title></circle>')
    o.append(f'<text x="{L + 10}" y="{T + 15}" fill="#d62728">raw — ECE {ece_b:.3f}</text>')
    o.append(f'<text x="{L + 10}" y="{T + 30}" fill="#1f77b4">temperature-calibrated — ECE {ece_a:.3f}</text>')
    _save(o, name)


def svg_pareto(front_pts, held_pts, name="router_pareto.svg"):
    import math
    W, H, L, B, T, R = 820, 460, 60, 50, 30, 250
    xs = [c for c, _ in front_pts] + [c for _, c, _ in held_pts]
    x0, x1 = math.log10(min(xs) * 0.8), math.log10(max(xs) * 1.25)
    ys = [a for _, a in front_pts] + [a for _, _, a in held_pts]
    y0 = math.floor(min(ys) * 20) / 20
    X = lambda v: L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    Y = lambda v: T + (1 - v) / (1 - y0) * (H - T - B)  # noqa: E731
    xt = [(v, f"${v:g}") for v in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05) if x0 <= math.log10(v) <= x1]
    yt, y = [], y0
    while y <= 1.0001:
        yt.append((y, f"{y:.2f}"))
        y += 0.05
    o = _frame(W, H, L, B, T, R, "$ per dialog (log)", "exact agreement", xt, yt, X, Y,
               "Per-criterion policies: Pareto on tuned gold (line) vs held-out (points)")
    o.append('<polyline fill="none" stroke="#2ca02c" stroke-width="1.5" points="'
             + " ".join(f"{X(c):.1f},{Y(a):.1f}" for c, a in front_pts) + '"/>')
    o.append(f'<text x="{W - R + 10}" y="{T + 12}" fill="#2ca02c">— gold Pareto (in-sample)</text>')
    for i, (lab, c, a) in enumerate(held_pts):
        col = "#d62728" if "cascade" in lab else "#1f77b4"
        o.append(f'<circle cx="{X(c):.1f}" cy="{Y(a):.1f}" r="4" fill="{col}"><title>{lab}: {a:.3f} @ ${c:.5f}'
                 f'</title></circle><text x="{X(c) + 5:.1f}" y="{Y(a) - 5:.1f}" font-size="9">{i + 1}</text>')
        o.append(f'<text x="{W - R + 10}" y="{T + 30 + i * 13}" font-size="9" fill="{col}">{i + 1}. {lab[:44]}</text>')
    _save(o, name)


# ------------------------------------------------------------------------------------------------ main
def pairs_of(items, key="jev_p", cal=None):
    return [(rp.calibrate(i[key] or 0.0, i["k"], cal) if cal else float(i[key] or 0.0), rp._same(i["jev_v"], i["exp"]))
            for i in items if i["jev_v"] is not None]


def main():
    pr = prices()
    gold = load_gold_items()
    held, src = load_holdout_items()
    for it in gold + held:
        it["y"], it["p"] = rp._same(it["jev_v"], it["exp"]), float(it["jev_p"] or 0.0)

    # 1. selective prediction
    rc = {}
    curves = []
    for lab in ["all"] + CRIT:
        sub = gold if lab == "all" else [i for i in gold if i["k"] == lab]
        pp = pairs_of(sub)
        rc[lab] = {"aurc_gold": rp.aurc(pp), "aurc_holdout": rp.aurc(pairs_of(
            held if lab == "all" else [i for i in held if i["k"] == lab])),
            "acc_tier1_gold": sum(y for _, y in pp) / len(pp), "curve_gold": rp.risk_coverage(pp)}
        # oracle (perfect ranking) AURC for reference
        n, e = len(pp), sum(not y for _, y in pp)
        rc[lab]["aurc_oracle_gold"] = sum(max(0, k - (n - e)) / k for k in range(1, n + 1)) / n
        curves.append((lab, rc[lab]["curve_gold"], rc[lab]["aurc_gold"]))
    svg_risk_coverage(curves)

    # 2. calibration (fit on tuned gold only)
    cal = rp.fit_calibration(gold)
    iso = rp.isotonic_fit([(i["p"], i["y"]) for i in gold])
    calib = {"temperature": cal, "ece": {}}
    for nm, its in (("gold", gold), ("holdout", held)):
        raw = pairs_of(its)
        tc = pairs_of(its, cal=cal)
        isp = [(rp.isotonic_apply(iso, p), y) for p, y in raw]
        calib["ece"][nm] = {"raw": rp.ece(raw), "temperature": rp.ece(tc), "isotonic_global": rp.ece(isp)}
        calib["ece"][nm]["per_criterion"] = {k: {"raw": rp.ece(pairs_of([i for i in its if i["k"] == k])),
                                                 "temperature": rp.ece(pairs_of([i for i in its if i["k"] == k], cal=cal))}
                                             for k in CRIT}
    svg_reliability(rp.reliability(pairs_of(gold)), rp.reliability(pairs_of(gold, cal=cal)),
                    calib["ece"]["gold"]["raw"], calib["ece"]["gold"]["temperature"])

    # 3. cost-aware per-criterion policies (tuned gold only)
    front = rp.pareto_policies(gold, cal, pr)
    chosen = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for b in BUDGETS:
        f = rp.best_under_budget(front, b)
        if not f:
            continue
        g = rp.run_policy(gold, f[2], cal, pr)
        chosen[b] = f[2]
        (OUT / f"policy_{b:g}.json").write_text(json.dumps({
            "budget_usd_per_dialog": b, "fitted_on": "examples/gold (tuned, 16 dialogs)",
            "gold": {"acc": g["acc"], "usd_per_dialog": g["usd_per_dialog"], "tiers": g["tiers"]},
            "policy": f[2], "calibration": cal, "prices": pr,
            "semantics": "escalate tier1→Haiku if calibrated p < theta (plus fail-closed: tier-1 error, 08=YES); "
                         "s2: none = keep Haiku, disagree = Sonnet if Haiku≠tier1, always = Sonnet"}, indent=1))
    # single global threshold on calibrated p (+ global s2) — 2 parameters instead of 16
    glob_rows = []
    for o in rp.options():
        pol = {k: o for k in CRIT}
        g = rp.run_policy(gold, pol, cal, pr)
        glob_rows.append((g["usd_per_dialog"], g["correct"], o))
    glob_front = rp._prune(glob_rows)

    # 4/5. held-out: every policy ONCE
    casc = cascade_replay(held, pr)
    casc_gold = cascade_replay(gold, pr)
    pols: list = [("current cascade (replay; t1 .7, hard .9, t2 .7)", None, casc, 0.0)]
    pols.append(("Jev only (fail-closed: 08=YES→Haiku)", {k: {"theta": 0.0, "s2": "none"} for k in CRIT}, None, 0.0))
    pols.append(("Haiku on every criterion", {k: {"theta": 1.01, "s2": "none"} for k in CRIT}, None, 0.0))
    for b, pol in chosen.items():
        pols.append((f"per-criterion policy, budget ${b:g}/dialog", pol, None, 0.0))
    gb = rp.best_under_budget(glob_front, 0.004)
    pols.append((f"global calibrated θ={gb[2]['theta']}, s2={gb[2]['s2']} (budget $0.004)",
                 {k: gb[2] for k in CRIT}, None, 0.0))
    # agreement routing: cascade thresholds but Sonnet decides whenever Jev and Haiku disagree (any confidence)
    pols.append(("cascade + Sonnet on any Jev≠Haiku disagreement", "casc_dis", None, 0.0))
    # agreement with a second cheap tier-1 (Qwen3.8-27B): untuned rule, held-out only (no gold run cached)
    if any(i.get("qwen_v") is not None for i in held):
        b0 = 0.004 if 0.004 in chosen else max(chosen)
        pols.append((f"policy ${b0:g} + escalate on Jev≠Qwen3.8-27B", {k: {**chosen[b0][k], "agree": True} for k in CRIT},
                     None, pr["qwen_gpu_dialog"]))
    rows = []
    for name, pol, res, extra in pols:
        if res is None and pol == "casc_dis":
            res = cascade_replay(held, pr, t2=1.01)
            gres = cascade_replay(gold, pr, t2=1.01)
        elif res is None:
            res = rp.run_policy(held, pol, cal, pr)
            gres = rp.run_policy(gold, pol, cal, pr)
        else:
            gres = casc_gold
        st = ci_and_delta(res, casc)
        src_of = {(i["cid"], i["k"]): i for i in held}
        st["bench_proxy_answers_used"] = sum(
            (p["tier"] == 2 and src_of[(p["cid"], p["k"])]["h_src"] == "bench")
            or (p["tier"] == 3 and src_of[(p["cid"], p["k"])]["s_src"] == "bench") for p in res["per_item"])
        row = {"name": name, "policy": pol if isinstance(pol, dict) else None,
               "gold": {"acc": gres["acc"], "usd_per_dialog": gres["usd_per_dialog"] + extra},
               "holdout": {"acc": res["acc"], "correct": res["correct"], "n": res["n"], **st,
                           "usd_per_dialog": res["usd_per_dialog"] + extra,
                           "tiers": res["tiers"], "calls": res["calls"]}}
        slug = "".join(ch if ch.isascii() and ch.isalnum() else "_" for ch in name.lower())[:60].strip("_")
        dl = {}
        for p in res["per_item"]:
            dl.setdefault(p["cid"], []).append({"criterion_id": p["k"], "value": p["pred"], "tier": p["tier"],
                                                "reasons": p["why"]})
        f = OUT / f"heldout__{slug}.dialogs.json"
        f.write_text(json.dumps({"dialogs": [{"conversation_id": c, "criteria": v, **_total(c, v)} for c, v in dl.items()]},
                                ensure_ascii=False, indent=0))
        row["file"] = str(f.relative_to(ROOT))
        rows.append(row)
        print(f"{name:60s} gold {gres['acc']:.3f} | held-out {res['acc']:.3f} [{st['ci'][0]:.3f},{st['ci'][1]:.3f}] "
              f"Δ {st['vs_cascade']['delta']:+.3f} [{st['vs_cascade']['lo']:+.3f},{st['vs_cascade']['hi']:+.3f}] "
              f"${res['usd_per_dialog'] + extra:.5f} proxy {st['bench_proxy_answers_used']} "
              f"tiers {[round(v, 3) for v in res['tiers'].values()]}")
    svg_pareto([(c, a / len(gold)) for c, a, _ in front], [(r["name"], r["holdout"]["usd_per_dialog"],
                                                             r["holdout"]["acc"]) for r in rows])
    study = {"prices": pr, "holdout_sources": src, "risk_coverage": {k: {kk: vv for kk, vv in v.items()
                                                                         if kk != "curve_gold"} for k, v in rc.items()},
             "calibration": calib, "gold_pareto": [{"usd_per_dialog": c, "acc": a / len(gold), "policy": p}
                                                   for c, a, p in front],
             "global_theta_pareto": [{"usd_per_dialog": c, "acc": a / len(gold), "rule": p} for c, a, p in glob_front],
             "policies": rows, "references": {"cascade_reported": 0.963, "jury_reported": 0.95}}
    (OUT / "study.json").write_text(json.dumps(study, indent=1, ensure_ascii=False))
    print("AURC", {k: round(v["aurc_gold"], 4) for k, v in rc.items()})
    print("ECE", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_criterion"}
                             for k, v in calib["ece"].items()}), "T", cal["global_T"], cal["T"])
    print("sources", src)


if __name__ == "__main__":
    main()

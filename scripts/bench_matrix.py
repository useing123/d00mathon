#!/usr/bin/env python3
"""Cost / accuracy / latency matrix across model setups (Anthropic, OpenAI, Gemini, local) — same pipeline for all.

    python scripts/bench_matrix.py snapshot              # freeze cqa/ + checklists/ + gold sets into a scratch dir
    python scripts/bench_matrix.py run --suites gold,holdout [--configs a,b] [--budget 40] [--parallel 4]
    python scripts/bench_matrix.py run --suites dev --configs haiku,sonnet ...
    python scripts/bench_matrix.py matrix                # aggregate -> results/bench_matrix.json (+ local vLLM runs)

Every configuration is evaluated by the SAME code, checklists, prompts, evidence validation and scoring: only
CQA_MODEL / CQA_JURORS / CQA_JUDGE differ (cqa.llm.cloud_route picks the vendor from the model id). Runs execute
from a frozen snapshot of the working tree so concurrent edits cannot leak into some configurations but not others;
each result records the checklist/criterion hashes and prompt version, and `matrix` refuses to mix snapshots.
Per-criterion cache is OFF (fresh calls). Cost = provider-reported token usage × cqa/pricing.py (assumed list prices).
Keys come from the environment or ~/.hermes/.env and are never printed.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv").exists() else sys.executable
SNAP = Path(os.environ.get("CQA_BENCH_SNAPSHOT", Path.home() / ".hermes" / "cache" / "scratch" / "cqa_bench_snapshot"))
OUT = ROOT / "results" / "bench"
KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY_QUORUM", "GOOGLE_API_KEY")

HK, SN, OP, FB = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"
CONFIGS = {   # name -> (jurors, judge); a single juror and no judge = single-model mode
    "haiku": ([HK], None), "sonnet": ([SN], None), "opus": ([OP], None), "fable": ([FB], None),
    "gpt-6-luna": (["gpt-6-luna"], None), "gpt-5.4-mini": (["gpt-5.4-mini"], None),
    "gpt-5.3-codex": (["gpt-5.3-codex"], None),
    "gemini-4-argon": (["gemini-4-argon"], None), "gemini-3.8-flash": (["gemini-3.8-flash"], None),
    "jury-haiku+sonnet>opus": ([HK, SN], OP),
    "jury-sonnet+opus>opus": ([SN, OP], OP),
    "jury-sonnet+luna>opus": ([SN, "gpt-6-luna"], OP),
}
SUITES = {"gold": "examples/gold", "holdout": "examples/gold_holdout", "dev": "data/DEV.json",
          "dev20": "data/DEV.json"}   # dev20 = first 20 DEV dialogs (budget-limited DEV check for expensive configs)


def _secrets():
    env = {}
    f = Path.home() / ".hermes" / ".env"
    for line in (f.read_text().splitlines() if f.exists() else []):
        k, _, v = line.partition("=")
        if k.strip() in KEYS and v:
            env[k.strip()] = v.strip().strip("'\"")
    return {k: os.environ.get(k) or env.get(k) for k in KEYS if os.environ.get(k) or env.get(k)}


def snapshot(_a=None):
    if SNAP.exists():
        shutil.rmtree(SNAP)
    for d in ("cqa", "checklists", "examples"):
        shutil.copytree(ROOT / d, SNAP / d, ignore=shutil.ignore_patterns("__pycache__", "static"))
    (SNAP / "data").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "DEV.json", SNAP / "data" / "DEV.json")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "cqa", "checklists"], cwd=ROOT, capture_output=True, text=True).stdout
    (SNAP / "SNAPSHOT.json").write_text(json.dumps({"git_head": git, "dirty_files": dirty.split("\n")[:40],
                                                    "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    print("snapshot ->", SNAP)


# ---------------------------------------------------------------- one config × one suite (runs inside SNAP)
def _val(c):
    if c.get("is_na") or str(c.get("value")).upper() == "NA":
        return "NA"
    return None if c.get("value") is None else str(c.get("value")).upper()


def compare_to_reference(results, ref_dialogs):
    """Agreement with a reference run (our own jury — NOT ground truth)."""
    ref = {d["conversation_id"]: d for d in ref_dialogs}
    per, na, gross, mae = {}, {}, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, []
    for d in results:
        r = ref.get(d["conversation_id"])
        if not r:
            continue
        pc = {c["criterion_id"]: c for c in d["criteria"]}
        for c in r["criteria"]:
            k = c["criterion_id"]
            ev, pv = _val(c), _val(pc.get(k, {}))
            s = per.setdefault(k, {"n": 0, "match": 0})
            s["n"] += 1
            s["match"] += ev == pv
            key = f"exp_{'NA' if ev == 'NA' else 'applicable'}/pred_{'NA' if pv == 'NA' else 'FAILED' if pv is None else 'applicable'}"
            na[key] = na.get(key, 0) + 1
        eg, pg = bool(r.get("gross_error")), bool(d.get("gross_error"))
        gross["tp" if eg and pg else "fp" if pg else "fn" if eg else "tn"] += 1
        pt = d.get("total") if d.get("total") is not None else d.get("provisional_total")
        if pt is not None and r.get("total") is not None:
            mae.append(abs(pt - r["total"]))
    for s in per.values():
        s["agreement"] = round(s["match"] / s["n"], 3)
    n, m = sum(s["n"] for s in per.values()), sum(s["match"] for s in per.values())
    tp, fp, fn = gross["tp"], gross["fp"], gross["fn"]
    return {"dialogs": len([d for d in results if d["conversation_id"] in ref]), "per_criterion": dict(sorted(per.items())),
            "overall_agreement": round(m / n, 3) if n else None, "total_mae": round(sum(mae) / len(mae), 2) if mae else None,
            "gross": {**gross, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None},
            "na_confusion": dict(sorted(na.items())),
            "incomplete": [d["conversation_id"] for d in results if d.get("status") == "incomplete"]}


def split_config(name):
    """'<config>@<prompt variant>' -> (jurors, judge, variant); no suffix = base prompts (v0)."""
    base, _, variant = name.partition("@")
    jurors, judge = CONFIGS[base]
    return jurors, judge, variant or "v0"


def one(a):
    jurors, judge, variant = split_config(a.config)
    os.environ.update(_secrets())
    os.environ.update(CQA_PROVIDER="anthropic", CQA_ENSEMBLE="1" if judge else "0", CQA_MODEL=jurors[0],
                      CQA_JURORS=",".join(jurors), CQA_JUDGE=judge or jurors[0], CQA_CONFIRM="", CQA_FALLBACK_MODELS="",
                      CQA_CONCURRENCY=str(a.concurrency), CQA_PROMPT_VARIANT=variant)
    sys.path.insert(0, str(SNAP))
    from cqa import data, gold, llm, prompts
    from cqa.evaluator import Evaluator
    assert Path(llm.__file__).is_relative_to(SNAP), "must run from the snapshot"
    t0 = time.time()
    if a.suite in ("dev", "dev20"):
        recs = data.load_file(str(SNAP / SUITES["dev"]))[: 20 if a.suite == "dev20" else None]
        results, meta = Evaluator(use_cache=False).evaluate_many(recs)
        refp = sorted(glob.glob(str(ROOT / "results/dev_jury_v*.json")), key=lambda f: int(f.rsplit("_v", 1)[1].split(".")[0]))[-1]
        metrics = compare_to_reference(results, json.load(open(refp))["dialogs"])
        metrics["reference"] = Path(refp).name
    else:
        items = gold.load_gold(SNAP / SUITES[a.suite])
        recs = [data.normalize(g["dialog"], i) for i, g in enumerate(items)]
        results, meta = Evaluator(use_cache=False).evaluate_many(recs)
        metrics = gold.compute_metrics(items, results)
    wall = time.time() - t0
    n = len(results)
    by_model = {m: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in u.items()} for m, u in llm.USAGE_BY_MODEL.items()}
    usd = sum(u["usd"] for u in llm.USAGE_BY_MODEL.values())
    toks = sum(u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"] + u["output_tokens"]
               for u in llm.USAGE_BY_MODEL.values())
    hashes = {cid: c.get("version_hash") for cid, c in (meta.get("reproducibility", {}).get("checklists") or {}).items()}
    doc = {"config": a.config, "suite": a.suite, "jurors": jurors, "judge": judge, "prompt_variant": variant, "dialogs": n,
           "wall_s": round(wall, 1), "sec_per_dialog": round(wall / max(n, 1), 2), "concurrency": a.concurrency,
           "usd": round(usd, 4), "usd_per_dialog": round(usd / max(n, 1), 5), "tokens_per_dialog": round(toks / max(n, 1)),
           "calls_per_dialog": round(sum(u["calls"] for u in by_model.values()) / max(n, 1), 2),
           "usage_by_model": by_model, "errors": len(meta.get("errors") or []), "error_samples": (meta.get("errors") or [])[:5],
           "metrics": {k: v for k, v in metrics.items() if k != "mismatches"},
           "mismatches": metrics.get("mismatches", []),
           "prompt_version": prompts.PROMPT_VERSION, "checklist_hashes": hashes,
           "snapshot": json.loads((SNAP / "SNAPSHOT.json").read_text()),
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.config}__{a.suite}" + (f"_r{a.rep}" if a.rep > 1 else "")
    doc["rep"] = a.rep
    (OUT / f"{tag}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (OUT / f"{tag}.dialogs.json").write_text(json.dumps(results, ensure_ascii=False))
    print(json.dumps({k: doc[k] for k in ("config", "suite", "usd", "wall_s", "errors")} |
                     {"agreement": doc["metrics"].get("overall_agreement")}))


# ---------------------------------------------------------------- orchestration with a hard budget
def spent():
    return sum(json.load(open(f))["usd"] for f in glob.glob(str(OUT / "*__*.json")) if not f.endswith(".dialogs.json"))


def run(a):
    names = a.configs.split(",") if a.configs else list(CONFIGS)
    sfx = f"_r{a.rep}" if a.rep > 1 else ""
    jobs = [(c, s) for s in a.suites.split(",") for c in names if a.force or not (OUT / f"{c}__{s}{sfx}.json").exists()]

    def go(job):
        c, s = job
        sp = spent()
        if sp >= a.budget:
            print(f"SKIP {c}/{s}: budget reached (${sp:.2f} >= ${a.budget})", flush=True)
            return
        log = OUT / f"{c}__{s}{sfx}.log"
        OUT.mkdir(parents=True, exist_ok=True)
        t = time.time()
        r = subprocess.run([PY, __file__, "one", "--config", c, "--suite", s, "--concurrency", str(a.concurrency), "--rep", str(a.rep)],
                           cwd=SNAP, capture_output=True, text=True)
        log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
        print(f"[{time.time() - t:.0f}s rc={r.returncode}] {c}/{s} {r.stdout.strip()[-300:]}  total spent ${spent():.2f}", flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(go, jobs))


# ---------------------------------------------------------------- aggregate
def pareto(points):
    """points: [(name, cost, acc)] -> names on the frontier (no other point is cheaper-or-equal AND more accurate)."""
    front = []
    for n, c, s in points:
        if not any((c2 <= c and s2 > s) or (c2 < c and s2 >= s) for _, c2, s2 in points):
            front.append(n)
    return front


def matrix(a):
    from cqa import pricing  # repo copy (same table as the snapshot)
    rows = {}
    for f in sorted(glob.glob(str(OUT / "*__*.json"))):
        if f.endswith(".dialogs.json"):
            continue
        d = json.load(open(f))
        row = rows.setdefault(d["config"], {"config": d["config"], "jurors": d["jurors"], "judge": d["judge"],
                                            "vendor": "mixed" if d["judge"] else _vendor(d["jurors"][0]), "reps": {}})
        if d.get("rep", 1) > 1:
            row["reps"].setdefault(d["suite"], []).append(d)
        else:
            row[d["suite"]] = d
    # checklist version hashes include the model id, so compare the snapshot + base prompt version instead
    snaps = {(r[s]["snapshot"]["taken_at"], r[s]["prompt_version"].split("+")[0]) for r in rows.values() for s in SUITES if s in r}
    out = []
    for r in rows.values():
        row = {k: r[k] for k in ("config", "jurors", "judge", "vendor")}
        row["prompt_variant"] = r["config"].partition("@")[2] or "v0"
        for s in SUITES:
            if s not in r:
                row[s] = None
                continue
            d = r[s]
            row[s] = {"dialogs": d["dialogs"], "overall_agreement": d["metrics"]["overall_agreement"],
                      "per_criterion": {k: v["agreement"] for k, v in d["metrics"]["per_criterion"].items()},
                      "total_mae": d["metrics"]["total_mae"], "gross": d["metrics"]["gross"],
                      "na_confusion": d["metrics"]["na_confusion"], "incomplete": len(d["metrics"]["incomplete"]),
                      "usd": d["usd"], "usd_per_dialog": d["usd_per_dialog"], "wall_s": d["wall_s"],
                      "sec_per_dialog": d["sec_per_dialog"], "tokens_per_dialog": d["tokens_per_dialog"],
                      "calls_per_dialog": d["calls_per_dialog"], "errors": d["errors"], "usage_by_model": d["usage_by_model"]}
            if s != "dev":
                row[s]["mismatches"] = [f"{m['conversation_id']}/{m['criterion_id']}: exp {m['expected']} got {m['predicted']}"
                                        for m in d["mismatches"]]
            reps = [d] + r["reps"].get(s, [])
            if len(reps) > 1:   # repeated runs: report every run and the mean (run-to-run noise)
                accs = [x["metrics"]["overall_agreement"] for x in reps]
                maes = [x["metrics"]["total_mae"] for x in reps if x["metrics"]["total_mae"] is not None]
                row[s]["runs"] = [{"overall_agreement": x["metrics"]["overall_agreement"], "total_mae": x["metrics"]["total_mae"],
                                   "gross_recall": x["metrics"]["gross"]["recall"], "usd": x["usd"]} for x in reps]
                row[s]["overall_agreement_mean"] = round(sum(accs) / len(accs), 4)
                row[s]["total_mae_mean"] = round(sum(maes) / len(maes), 2) if maes else None
        g, h = row.get("gold"), row.get("holdout")
        if g and h:   # pooled synthetic agreement over 26 dialogs × 8 criteria; cost per dialog over both suites
            n = g["dialogs"] + h["dialogs"]
            row["synthetic_agreement"] = round((g["overall_agreement"] * g["dialogs"] + h["overall_agreement"] * h["dialogs"]) / n, 3)
            row["usd_per_dialog"] = round((g["usd"] + h["usd"]) / n, 5)
            row["sec_per_dialog"] = round((g["wall_s"] + h["wall_s"]) / n, 2)
        if h:
            row["holdout_score"] = h.get("overall_agreement_mean", h["overall_agreement"])
        dv = OUT / f"{row['config']}__dev.dialogs.json"
        if dv.exists() and row.get("dev20") is None:   # same 20-dialog subset as dev20 runs, computed offline (no API calls)
            sub = json.load(open(dv))[:20]
            ref = sorted(glob.glob(str(ROOT / "results/dev_jury_v*.json")), key=lambda f: int(f.rsplit("_v", 1)[1].split(".")[0]))[-1]
            m20 = compare_to_reference(sub, json.load(open(ref))["dialogs"])
            row["dev20"] = {"dialogs": m20["dialogs"], "overall_agreement": m20["overall_agreement"], "total_mae": m20["total_mae"],
                            "gross": m20["gross"], "derived_from": "first 20 dialogs of the full DEV run"}
        out.append(row)
    out += _local_rows()
    pts = [(r["config"], r["usd_per_dialog"], r["holdout_score"]) for r in out if r.get("holdout_score") is not None
           and r.get("usd_per_dialog") is not None]
    front = pareto(pts)
    for r in out:
        r["pareto"] = r["config"] in front
    spend = {}
    for f in glob.glob(str(OUT / "*__*.json")):
        if f.endswith(".dialogs.json"):
            continue
        for m, u in json.load(open(f))["usage_by_model"].items():
            spend[_vendor(m)] = round(spend.get(_vendor(m), 0) + u["usd"], 4)
    local = []
    for f in sorted(glob.glob(str(ROOT / "results/bench_local*.json"))):
        try:
            local.append({"file": Path(f).name, **json.load(open(f))})
        except Exception as e:   # noqa: BLE001 — report, don't crash
            local.append({"file": Path(f).name, "error": str(e)})
    doc = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "snapshot": json.loads((SNAP / "SNAPSHOT.json").read_text()),
           "same_snapshot_and_base_prompts": len(snaps) == 1, "prices_as_of": pricing.PRICES_AS_OF,
           "prices_usd_per_mtok": {k: dict(zip(("input", "output", "cache_read", "cache_write"), v, strict=True))
                                   for k, v in pricing.PRICES.items()},
           "price_notes": pricing.PRICE_NOTES, "spend_usd_by_vendor": spend, "spend_usd_total": round(sum(spend.values()), 2),
           "pareto_axis": "holdout_score (held-out gold, mean over repeated runs) vs usd_per_dialog (gold+holdout)",
           "pareto": front, "rows": sorted(out, key=lambda r: -(r.get("holdout_score") or 0)),
           "selection": select_winner(out),
           "local": local, "presets": PRESETS_DOC}
    (ROOT / "results" / "bench_matrix.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"wrote results/bench_matrix.json: {len(out)} configs, frontier={front}, spend={spend}")


def _local_rows():
    """Local vLLM runs from scripts/bench_models.py (results/bench_<tag>.json) — another agent's run, read, not rerun.
    Marked external_run: different snapshot / prompt state, so they are shown but excluded from the winner selection."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/bench_*.json"))):
        name = Path(f).name
        if name == "bench_matrix.json" or name.count("_") > 1 and not name.endswith("summary.json") and "_gold" in name:
            continue
        try:
            d = json.load(open(f))
        except Exception:   # noqa: BLE001
            continue
        if not isinstance(d, dict) or "gpu_price_hr" not in d or "dev" not in d:
            continue
        tag = name[len("bench_"):-len(".json")]
        row = {"config": f"local:{d['model']}", "jurors": [d["model"]], "judge": None, "vendor": "local", "prompt_variant": "v0",
               "external_run": True, "source_files": [name], "gpu_price_hr": d["gpu_price_hr"], "reps": {}}
        secs = 0.0
        for suite, fn in (("gold", f"bench_{tag}_gold.json"), ("holdout", f"bench_{tag}_gold_holdout.json")):
            p2 = ROOT / "results" / fn
            if not p2.exists():
                continue
            m = json.load(open(p2))["metrics"]
            el = (m.get("meta") or {}).get("elapsed_s") or 0
            secs += el
            row[suite] = {"dialogs": m["dialogs"], "overall_agreement": m["overall_agreement"],
                          "per_criterion": {k: v["agreement"] for k, v in m["per_criterion"].items()}, "total_mae": m["total_mae"],
                          "gross": m["gross"], "na_confusion": m["na_confusion"], "incomplete": len(m["incomplete"]),
                          "wall_s": el, "sec_per_dialog": round(el / max(m["dialogs"], 1), 2), "tokens_per_dialog": None,
                          "usd": round(d["gpu_price_hr"] * el / 3600, 4)}
            row["source_files"].append(fn)
        if row.get("gold") and row.get("holdout"):
            n = row["gold"]["dialogs"] + row["holdout"]["dialogs"]
            row["holdout_score"] = row["holdout"]["overall_agreement"]
            row["usd_per_dialog"] = round(d["gpu_price_hr"] * secs / 3600 / n, 5)
            row["sec_per_dialog"] = round(secs / n, 2)
        dv = d.get("dev") or {}
        if dv.get("vs_cloud"):
            row["dev"] = {"dialogs": dv["dialogs"], "overall_agreement": dv["vs_cloud"]["overall_exact"],
                          "total_mae": dv["vs_cloud"]["total_mae"], "reference": dv.get("cloud_file"),
                          "usd_per_dialog": dv.get("gpu_usd_per_dialog"), "sec_per_dialog": dv.get("sec_per_dialog")}
        rows.append(row)
    return rows


def select_winner(rows):
    """User-defined rule: max held-out exact agreement (mean of repeats); tie-break lower total MAE, higher gross
    recall, lower $/dialog. Only configs that ran the held-out set qualify."""
    def key(r):
        h = r["holdout"]
        return (-r["holdout_score"], h.get("total_mae_mean", h["total_mae"]) or 0, -(h["gross"]["recall"] or 0),
                r.get("usd_per_dialog") or 0)
    ranked = sorted([r for r in rows if r.get("holdout_score") is not None and r.get("usd_per_dialog") is not None
                     and not r.get("external_run")], key=key)
    return {"rule": "max holdout exact agreement (mean of repeats); ties: lower MAE, higher gross recall, lower $/dialog",
            "ranking": [{"config": r["config"], "holdout_score": r["holdout_score"],
                         "runs": len(r["holdout"].get("runs") or [1]), "usd_per_dialog": r["usd_per_dialog"]} for r in ranked],
            "winner": ranked[0]["config"] if ranked else None, "runner_up": ranked[1]["config"] if len(ranked) > 1 else None}


def _vendor(m):
    return "anthropic" if m.startswith("claude") else "openai" if m.startswith("gpt") else "google" if m.startswith("gemini") else "local"


PRESETS_DOC = {}   # filled from cqa.presets at aggregation time (kept in one place)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    o = sub.add_parser("one")
    o.add_argument("--config", required=True, help="config name, optionally @v1|@v2|@v3 prompt variant")
    o.add_argument("--suite", required=True, choices=list(SUITES))
    o.add_argument("--concurrency", type=int, default=16)
    o.add_argument("--rep", type=int, default=1, help="repeat index (>1 writes <config>__<suite>_r<rep>.json)")
    r = sub.add_parser("run")
    r.add_argument("--suites", default="gold,holdout")
    r.add_argument("--configs")
    r.add_argument("--budget", type=float, default=40.0)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--force", action="store_true")
    r.add_argument("--rep", type=int, default=1)
    sub.add_parser("matrix")
    a = p.parse_args()
    if a.cmd == "matrix":
        sys.path.insert(0, str(ROOT))
        try:
            from cqa.presets import PRESETS
            PRESETS_DOC.update(PRESETS)
        except ImportError:
            pass
    {"snapshot": snapshot, "one": one, "run": run, "matrix": matrix}[a.cmd](a)


if __name__ == "__main__":
    main()

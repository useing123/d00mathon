#!/usr/bin/env python3
"""Benchmark a single OpenAI-compatible model (no jury) with the SAME pipeline as the cloud path.

    python scripts/bench_models.py run --base-url URL --model NAME [--tag qwen3-32b] [--concurrency 12] [--price-hr 0.59]
    python scripts/bench_models.py compare results/bench_<tag>_dev.json results/dev_jury_v4.json

`run` executes `cqa gold` (examples/gold and examples/gold_holdout if present) and `cqa evaluate` on
data/DEV.json via subprocess with CQA_PROVIDER=openai, CQA_ENSEMBLE=0 — checklists, prompts, evidence
validation and scoring are untouched. It then compares DEV results with the cloud jury run and writes
results/bench_<tag>.json. API key: CQA_OPENAI_API_KEY or CQA_LOCAL_LLM_TOKEN (env or ~/.hermes/.env).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv").exists() else sys.executable


def secret(key):
    if os.environ.get(key):
        return os.environ[key]
    f = Path.home() / ".hermes" / ".env"
    for line in (f.read_text().splitlines() if f.exists() else []):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip("'\"")


def _val(c):
    return "NA" if c.get("is_na") else c.get("value")


def compare(local_path, cloud_path):
    L = {d["conversation_id"]: d for d in json.load(open(local_path))["dialogs"]}
    C = {d["conversation_id"]: d for d in json.load(open(cloud_path))["dialogs"]}
    ids = sorted(set(L) & set(C))
    per, mae, gross_agree, n_tot = {}, [], 0, 0
    for i in ids:
        lc = {c["criterion_id"]: c for c in L[i]["criteria"]}
        for c in C[i]["criteria"]:
            k = c["criterion_id"]
            p = per.setdefault(k, [0, 0])
            if k in lc:
                p[0] += 1
                p[1] += _val(lc[k]) == _val(c)
        lt, ct = L[i].get("total"), C[i].get("total")
        if lt is not None and ct is not None:
            mae.append(abs(float(lt) - float(ct)))
        gross_agree += bool(L[i].get("gross_error")) == bool(C[i].get("gross_error"))
        n_tot += 1
    tot = sum(p[0] for p in per.values())
    m = sum(p[1] for p in per.values())
    return {"dialogs_compared": len(ids),
            "per_criterion_exact": {k: round(v[1] / v[0], 3) for k, v in sorted(per.items()) if v[0]},
            "overall_exact": round(m / tot, 3) if tot else None,
            "total_mae": round(sum(mae) / len(mae), 2) if mae else None,
            "gross_flag_agreement": round(gross_agree / n_tot, 3) if n_tot else None,
            "gross_local": sum(bool(L[i].get("gross_error")) for i in ids),
            "gross_cloud": sum(bool(C[i].get("gross_error")) for i in ids)}


def sh(args, env, log):
    t = time.time()
    r = subprocess.run([PY, "-m", "cqa", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print(f"[{time.time() - t:.0f}s rc={r.returncode}] cqa {' '.join(args)}  (log {log})", flush=True)
    return time.time() - t, r


def run(a):
    env = dict(os.environ, CQA_PROVIDER="openai", CQA_ENSEMBLE="0", CQA_OPENAI_JURY="0", CQA_MODEL=a.model,
               CQA_OPENAI_BASE_URL=a.base_url, CQA_CONCURRENCY=str(a.concurrency),
               CQA_OPENAI_API_KEY=secret("CQA_OPENAI_API_KEY") or secret("CQA_LOCAL_LLM_TOKEN") or "none")
    out = {"model": a.model, "base_url": a.base_url.split("//")[-1].split(".")[0] + "…", "gpu_price_hr": a.price_hr,
           "concurrency": a.concurrency, "gold": {}}
    R = ROOT / "results"
    for name, d in (("gold", "examples/gold"), ("gold_holdout", "examples/gold_holdout")):
        if not (ROOT / d).exists() or "gold" in a.skip:
            continue
        rep = R / f"bench_{a.tag}_{name}.json"
        sec, r = sh(["gold", "--fresh", "--dir", d, "--out", str(rep)], env, R / f"bench_{a.tag}_{name}.log")
        try:
            out["gold"][name] = {**json.load(open(rep))["metrics"], "elapsed_s": round(sec, 1)}
            out["gold"][name].pop("dialogs_detail", None)
        except Exception as e:
            out["gold"][name] = {"error": str(e)[:300], "stderr_tail": r.stderr[-600:]}
    if "dev" not in a.skip:
        dev = R / f"bench_{a.tag}_dev.json"
        sec, r = sh(["evaluate", "--input", "data/DEV.json", "--fresh", "--concurrency", str(a.concurrency),
                     "--out", str(dev)], env, R / f"bench_{a.tag}_dev.log")
        try:
            d = json.load(open(dev))
            n = len(d["dialogs"])
            meta = d.get("meta", {})
            out["dev"] = {"dialogs": n, "elapsed_s": round(sec, 1), "sec_per_dialog": round(sec / max(n, 1), 2),
                          "gpu_usd_per_dialog": round(a.price_hr * sec / 3600 / max(n, 1), 5),
                          "incomplete": len(meta.get("incomplete_dialogs") or []),
                          "errors": len(meta.get("errors") or []), "usage": meta.get("usage"),
                          "vs_cloud": compare(dev, a.cloud), "cloud_file": Path(a.cloud).name}
        except Exception as e:
            out["dev"] = {"error": str(e)[:300], "stderr_tail": r.stderr[-600:]}
    (R / f"bench_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "compare"])
    p.add_argument("files", nargs="*")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--tag", default="local")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--price-hr", type=float, default=0.0)
    p.add_argument("--skip", default="")
    newest = sorted(glob.glob(str(ROOT / "results/dev_jury_v*.json")),
                    key=lambda f: int(re.search(r"_v(\d+)", f).group(1)))
    p.add_argument("--cloud", default=newest[-1] if newest else None)
    a = p.parse_args()
    if a.cmd == "compare":
        print(json.dumps(compare(*a.files), indent=1))
    else:
        run(a)


if __name__ == "__main__":
    main()

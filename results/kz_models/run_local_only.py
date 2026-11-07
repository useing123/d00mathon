"""Local-only (tier 1, no escalation) held-out run via scripts/bench_oss.py cascade, KZ subset added.
usage: .venv/bin/python results/kz_models/run_local_only.py <served-name> <base_url> <tag>"""
import json, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cqa import router
router.tier1_reasons = lambda *a, **k: []          # never escalate: measure the open model alone
import bench_oss
served, url, tag = sys.argv[1:4]
sys.argv = ["bench_oss.py", "cascade", "--suite", "holdout", "--first", "local", "--local-model", served,
            "--base-url", url, "--no-logprobs", "--concurrency", "8", "--tag", tag]
bench_oss.OUT = ROOT / "results" / "kz_models"
bench_oss.main()
d = json.load(open(bench_oss.OUT / f"{tag}_holdout.json"))
gold, recs = bench_oss._records("holdout")
KZ = re.compile(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")
def text(r): return json.dumps(r, ensure_ascii=False)
kz_ids = {r["conversation_id"] for r in recs if len(KZ.findall(text(r))) > 20}
from cqa.gold import compute_metrics
sub_res = [x for x in d["dialogs_detail"] if x["conversation_id"] in kz_ids]
try:
    sub_gold = {k: v for k, v in gold.items() if k in kz_ids} if isinstance(gold, dict) else [g for g in gold if g.get("conversation_id") in kz_ids]
    m = compute_metrics(sub_gold, sub_res)
    kz = {"n": len(kz_ids), "agreement": m["overall_agreement"], "mae": m["total_mae"]}
except Exception as e:
    kz = {"n": len(kz_ids), "error": str(e)[:200]}
d["kz_subset"] = kz
(bench_oss.OUT / f"{tag}_holdout.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
print("KZ", kz, "gross", d["metrics"]["gross"])

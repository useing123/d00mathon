"""Per-language subset agreement from saved held-out runs (dialog id contains -kz-/-ru-/-en- or by script)."""
import json, re, sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
import bench_oss
from cqa.gold import compute_metrics
gold, recs = bench_oss._records("holdout")
KZ = re.compile(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")
lang = {}
for r in recs:
    i = r["conversation_id"]; lang[i] = next(L for L in ("kz", "ru", "en") if f"-{L}-" in i)
for f in sorted(glob.glob(str(ROOT/"results/kz_models/*_holdout.json"))):
    d = json.load(open(f)); out = {}
    for L in ("kz", "ru", "en"):
        ids = {k for k, v in lang.items() if v == L}
        res = [x for x in d["dialogs_detail"] if x["conversation_id"] in ids]
        try: m = compute_metrics(gold, res); out[L] = (len(ids), m["overall_agreement"], m["total_mae"])
        except Exception as e: out[L] = (len(ids), "err " + str(e)[:80])
    print(Path(f).name, out)

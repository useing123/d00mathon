"""CLI: python -m cqa evaluate --group DEV --out results/dev.json"""
import argparse
import json
import sys
import time
from collections import Counter

from . import config, data, store
from .evaluator import Evaluator, rescore


def _progress(done, total):
    sys.stderr.write(f"\r  {done}/{total} criterion evaluations")
    sys.stderr.flush()


def cmd_evaluate(a):
    if a.input:
        recs = data.load_file(a.input)
        group = a.group or a.input.rsplit("/", 1)[-1].split(".")[0]
    else:
        recs = data.load_group(a.group, refresh=a.refresh)
        group = a.group
    if a.ids:
        want = set(a.ids.split(","))
        recs = [r for r in recs if r["conversation_id"] in want]
    if a.limit:
        recs = recs[: a.limit]
    print(f"Evaluating {len(recs)} dialogs (group={group}, provider={config.PROVIDER}, model={config.MODEL}, cache={'off' if a.fresh else 'on'})", file=sys.stderr)
    ev = Evaluator(use_cache=not a.fresh, samples=a.samples)
    t0 = time.time()
    dialogs, meta = ev.evaluate_many(recs, checklist_override=a.checklist, progress=_progress, concurrency=a.concurrency)
    print(file=sys.stderr)
    out = a.out or str(store.results_path(group))
    doc = store.save_results(group, dialogs, meta, out)
    csv_path = a.csv or out.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(store.to_csv(doc, "ai"))
    tot = [d["total"] for d in dialogs if d["total"] is not None]
    print(json.dumps({"dialogs": len(dialogs), "elapsed_s": round(time.time() - t0, 1), "cost_usd": meta["cost_usd"],
                      "usage": meta["usage"], "errors": meta["errors"][:10],
                      "mean_total": round(sum(tot) / len(tot), 2) if tot else None,
                      "gross_flags": sum(d["gross_error"] for d in dialogs), "json": out, "csv": csv_path}, ensure_ascii=False, indent=1))


def cmd_rescore(a):
    doc = store.load_results(a.group)
    doc = rescore(doc)
    store.save_results(a.group, doc["dialogs"], doc["meta"])
    print("rescored", len(doc["dialogs"]))


def cmd_stats(a):
    doc = store.load_results(a.group)
    ds = doc["dialogs"]
    per = {}
    for d in ds:
        for r in d["criteria"]:
            per.setdefault(f"{r['criterion_id']} {r['name']}", Counter())[str(r["value"])] += 1
    for k, c in per.items():
        print(k, dict(sorted(c.items())))
    tot = sorted(d["total"] for d in ds if d["total"] is not None)
    bins = Counter(min(int(t // 10) * 10, 90) for t in tot)
    if not tot:
        print("no totals"); return
    print("totals: n=%d min=%.2f median=%.2f max=%.2f mean=%.2f" % (len(tot), tot[0], tot[len(tot)//2], tot[-1], sum(tot)/len(tot)))
    print("histogram:", {f"{k}-{k+10}": bins[k] for k in sorted(bins)})
    print("gross:", [d["conversation_id"] for d in ds if d["gross_error"]])
    dis = [(d["conversation_id"], r["criterion_id"], r.get("votes")) for d in ds for r in d["criteria"] if r.get("self_consistency_disagreement")]
    print("self-consistency disagreements:", len(dis))
    for x in dis:
        print("  ", x)
    rej = sum(len(r.get("rejected_quotes") or []) for d in ds for r in d["criteria"])
    print("rejected (hallucinated) quotes kept out:", rej)


def main():
    p = argparse.ArgumentParser(prog="cqa")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="evaluate a group (fetched via CQA_API_URL) or a local JSON file")
    e.add_argument("--group")
    e.add_argument("--input", help="local JSON file instead of fetching")
    e.add_argument("--out")
    e.add_argument("--csv")
    e.add_argument("--checklist", help="force a checklist id (default: auto by channel)")
    e.add_argument("--ids", help="comma-separated conversation ids")
    e.add_argument("--limit", type=int)
    e.add_argument("--samples", type=int, help="self-consistency samples (default from checklist)")
    e.add_argument("--concurrency", type=int)
    e.add_argument("--refresh", action="store_true", help="re-fetch group even if data/<group>.json exists")
    e.add_argument("--fresh", action="store_true", help="ignore per-criterion cache (re-run everything)")
    e.add_argument("--reeval", action="store_true", help="re-evaluate with current checklist: only changed criteria hit the LLM (default behaviour with cache)")
    r = sub.add_parser("rescore", help="recompute totals after reweighting (no LLM)")
    r.add_argument("--group", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--group", required=True)
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8090)
    a = p.parse_args()
    if a.cmd == "evaluate":
        if not a.group and not a.input:
            p.error("--group or --input required")
        cmd_evaluate(a)
    elif a.cmd == "rescore":
        cmd_rescore(a)
    elif a.cmd == "stats":
        cmd_stats(a)
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("cqa.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()

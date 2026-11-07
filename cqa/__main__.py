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
    """Evaluate a fetched group or a local JSON file; writes results JSON + CSV."""
    if a.final:
        config.apply_final_profile()
    if a.confirm:
        config.CONFIRM = a.confirm
    if a.mock:
        config.PROVIDER = "mock"
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
    ev = Evaluator(use_cache=not a.fresh, samples=a.samples)
    models = (f"jury[{'+'.join(config.JURORS)}] judge={config.JUDGE}" if ev.ensemble else ev.llm.model)
    print(f"Evaluating {len(recs)} dialogs (group={group}, provider={config.PROVIDER}, models={models}, "
          f"confirm={sorted(config.confirm_criteria()) or '-'}, cache={'off' if a.fresh else 'on'})", file=sys.stderr)
    for w in ev.preflight():
        print("WARNING: " + w, file=sys.stderr)
    t0 = time.time()
    dialogs, meta = ev.evaluate_many(recs, checklist_override=a.checklist, progress=_progress, concurrency=a.concurrency)
    print(file=sys.stderr)
    out = a.out or str(store.results_path(group))
    doc = store.save_results(group, dialogs, meta, out)
    csv_path = a.csv or out.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(store.to_csv(doc, "ai"))
    if a.organizer:
        with open(a.organizer, "w", encoding="utf-8") as f:
            f.write(store.to_organizer_csv(doc, fill_provisional=a.fill_provisional))
        with open(a.organizer.rsplit(".", 1)[0] + ".diagnostics.csv", "w", encoding="utf-8") as f:
            f.write(store.to_diagnostics_csv(doc))
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
        print("no totals")
        return
    print(f"totals: n={len(tot)} min={tot[0]:.2f} median={tot[len(tot) // 2]:.2f} max={tot[-1]:.2f} mean={sum(tot) / len(tot):.2f}")
    print("histogram:", {f"{k}-{k+10}": bins[k] for k in sorted(bins)})
    print("gross:", [d["conversation_id"] for d in ds if d["gross_error"]])
    dis = [(d["conversation_id"], r["criterion_id"], r.get("votes"))
           for d in ds for r in d["criteria"] if r.get("self_consistency_disagreement")]
    print("self-consistency disagreements:", len(dis))
    for x in dis:
        print("  ", x)
    rej = sum(len(r.get("rejected_quotes") or []) for d in ds for r in d["criteria"])
    print("rejected (hallucinated) quotes kept out:", rej)


def cmd_gold(a):
    if a.final:
        config.apply_final_profile()
    if a.confirm:
        config.CONFIRM = a.confirm
    from .gold import format_report, run_gold
    m = run_gold(mock=a.mock, gold_dir=a.dir, out=a.out, use_cache=not a.fresh, labels=a.labels, group=a.group)
    print(format_report(m))
    print(json.dumps({"report": m["report"], "meta": m["meta"]}, ensure_ascii=False))


def cmd_label(a):
    """Human labelling of real dialogs -> examples/dev_labels.json (git-ignored). Labels are never invented by the tool."""
    from pathlib import Path

    from . import checklist as ckl
    from .gold import label_dialog
    from .util import atomic_write
    path = Path(a.out)
    labels = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    recs = data.load_group(a.group)
    ids = a.ids.split(",") if a.ids else [r["conversation_id"] for r in recs if r["conversation_id"] not in labels][: a.n]
    for cid in ids:
        rec = next((r for r in recs if r["conversation_id"] == cid), None)
        if rec is None:
            print(f"unknown dialog {cid}")
            continue
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        print(f"\n===== {cid} ({rec['kind']}, {rec.get('language')}) — label by the organizer rubric =====")
        lab = label_dialog(rec, cl, labels.get(cid))
        lab["labeller"] = a.labeller
        labels[cid] = lab
        atomic_write(path, json.dumps(labels, ensure_ascii=False, indent=1))   # saved after every dialog
        print(f"saved {cid} → {path} ({len(labels)} labelled)")


def cmd_export(a):
    if a.run:
        from .util import safe_child
        doc = json.loads(safe_child(config.RESULTS_DIR, a.run, ".json").read_text(encoding="utf-8"))
    else:
        doc = store.load_results(a.group)
    if not doc:
        sys.exit(f"no results for group {a.group}")
    txt = store.to_organizer_csv(doc, fill_provisional=a.fill_provisional) if a.format == "organizer" else store.to_csv(doc, a.format)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(a.out)
    else:
        print(txt, end="")


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
    e.add_argument("--final", action="store_true",
                   help="control-run profile: jurors sonnet+opus, judge re-checks unanimous 02/04/06, median of 3 samples on 04/06")
    e.add_argument("--organizer", help="also write the submission CSV (conversation_id, 01..08, total, gross_error) here "
                                       "+ <name>.diagnostics.csv")
    e.add_argument("--fill-provisional", action="store_true",
                   help="opt-in: in the submission CSV only, use provisional_total for dialogs still incomplete after rescue")
    e.add_argument("--confirm", help="criteria where the judge re-checks unanimous jury verdicts, e.g. 03,04 (control runs)")
    e.add_argument("--mock", action="store_true", help="offline deterministic heuristic backend (no API key; same as CQA_PROVIDER=mock)")
    r = sub.add_parser("rescore", help="recompute totals after reweighting (no LLM)")
    r.add_argument("--group", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--group", required=True)
    gd = sub.add_parser("gold", help="evaluate the hand-labelled gold set (examples/gold) and report agreement")
    gd.add_argument("--mock", action="store_true")
    gd.add_argument("--confirm", help="criteria where the judge re-checks unanimous jury verdicts, e.g. 03,04")
    gd.add_argument("--final", action="store_true", help="control-run profile (see evaluate --final)")
    gd.add_argument("--fresh", action="store_true")
    gd.add_argument("--out")
    gd.add_argument("--dir", help="directory with gold *.json (default examples/gold; held-out: examples/gold_holdout)")
    gd.add_argument("--labels", help="human labels JSON from `cqa label` (e.g. examples/dev_labels.json)")
    gd.add_argument("--group", default="DEV", help="group whose dialogs the --labels refer to")
    ex = sub.add_parser("export", help="export results: organizer submission CSV or ai/human/both CSV")
    ex.add_argument("--group", required=True)
    ex.add_argument("--run")
    ex.add_argument("--format", default="organizer", choices=["organizer", "ai", "human", "both"])
    ex.add_argument("--out")
    ex.add_argument("--fill-provisional", action="store_true")
    lb = sub.add_parser("label", help="label real dialogs by hand (stored in a git-ignored JSON)")
    lb.add_argument("--group", default="DEV")
    lb.add_argument("--ids", help="comma-separated conversation ids (default: next unlabelled)")
    lb.add_argument("-n", type=int, default=5, help="how many unlabelled dialogs to label")
    lb.add_argument("--out", default="examples/dev_labels.json")
    lb.add_argument("--labeller", default="")
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8090)
    for sp in (e, gd):
        sp.add_argument("--preset", choices=["econom", "balanced", "max", "jury"],
                        help="model setup from the benchmark matrix (docs/MODELS.md); default: max")
    a = p.parse_args()
    if getattr(a, "preset", None):
        from .presets import apply
        apply(a.preset, vars(config))
    if a.cmd == "evaluate":
        if not a.group and not a.input:
            p.error("--group or --input required")
        cmd_evaluate(a)
    elif a.cmd == "rescore":
        cmd_rescore(a)
    elif a.cmd == "stats":
        cmd_stats(a)
    elif a.cmd == "gold":
        cmd_gold(a)
    elif a.cmd == "export":
        cmd_export(a)
    elif a.cmd == "label":
        cmd_label(a)
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("cqa.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()

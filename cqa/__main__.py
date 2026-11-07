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


def _getkey(prompt):
    """One keystroke (no Enter) on a TTY; falls back to a line read for pipes/tests."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.strip()[:1] or "\n"
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(ch if ch.isprintable() else "")
    return "\n" if ch in "\r\n" else ch


def fast_label_dialog(rec, checklist, existing=None, key=_getkey, say=print, ask=input):
    """One keystroke per criterion: y/n (x = NA), scale → digit index; Enter keeps the previous label,
    ? shows the rules, c adds a comment, b goes back. The model's answers are never shown."""
    from .checklist import render
    say(data.format_transcript(rec))
    crs, exp, notes = checklist["criteria"], dict((existing or {}).get("expected", {})), dict((existing or {}).get("notes", {}))
    i = 0
    while i < len(crs):
        cr = crs[i]
        if cr.get("type") == "scale":
            opts = {str(k): v for k, v in enumerate(cr["scale"])}
        else:
            opts = {"y": "YES", "n": "NO"}
        if cr.get("allow_na"):
            opts["x"] = "NA"
        prev = exp.get(cr["id"])
        say(f"\n── {cr['id']} {cr['name']}{' [ГРУБАЯ ОШИБКА: y = ошибка есть]' if cr.get('gross') else ''}")
        say(render(cr.get("rubric", ""), checklist).strip())
        keys = " ".join(f"{k}={v}" for k, v in opts.items())
        k = key(f"[{keys} | ?=правила c=комментарий b=назад]{f' Enter={prev}' if prev is not None else ''}: ").lower()
        if k == "?":
            say(render(cr.get("rules", "") or "(нет доп. правил)", checklist).strip())
        elif k == "c":
            notes[cr["id"]] = ask("   комментарий: ").strip()
        elif k == "b":
            i = max(0, i - 1)
        elif k == "\n" and prev is not None:
            i += 1
        elif k in opts:
            exp[cr["id"]] = opts[k]
            i += 1
        else:
            say("  неверная клавиша")
    return {"expected": exp, "notes": {k: v for k, v in notes.items() if v}}


def cmd_label(a):
    """Human labelling of real dialogs -> examples/dev_labels.json or data/human/labels_<annotator>.json (git-ignored).
    Labels are never invented by the tool."""
    from pathlib import Path

    from . import checklist as ckl
    from .gold import label_dialog
    from .util import atomic_write
    session = json.loads(Path(a.sample).read_text(encoding="utf-8")) if a.sample else {}
    who = a.annotator or a.labeller or session.get("annotator", "")
    out = a.out or session.get("labels_out") or (f"data/human/labels_{who}.json" if a.annotator else "examples/dev_labels.json")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    recs = data.load_group(session.get("group") or a.group)
    if a.ids:
        ids = a.ids.split(",")
    elif session:
        ids = [i for i in session["ids"] if i not in labels]
        print(f"annotator {who}: {len(session['ids']) - len(ids)}/{len(session['ids'])} done, {len(ids)} left → {path}")
    else:
        ids = [r["conversation_id"] for r in recs if r["conversation_id"] not in labels][: a.n]
    fast = bool(session or a.annotator) and not a.slow
    for n, cid in enumerate(ids, 1):
        rec = next((r for r in recs if r["conversation_id"] == cid), None)
        if rec is None:
            print(f"unknown dialog {cid}")
            continue
        cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
        print(f"\n===== [{n}/{len(ids)}] {cid} ({rec['kind']}, {rec.get('language')}) — label by the organizer rubric =====")
        t0 = time.time()
        try:
            lab = (fast_label_dialog if fast else label_dialog)(rec, cl, labels.get(cid))
        except (EOFError, KeyboardInterrupt):
            print("\nstopped; progress is saved — rerun the same command to continue")
            return
        lab["labeller"] = who
        lab["seconds"] = round(time.time() - t0, 1)
        labels[cid] = lab
        atomic_write(path, json.dumps(labels, ensure_ascii=False, indent=1))   # saved after every dialog
        print(f"saved {cid} → {path} ({len(labels)} labelled, {lab['seconds']} s)")


def cmd_export(a):
    if a.format == "v1":
        return cmd_export_v1(a)
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


def _group_ids(group):
    """conversation ids of the group in API order (data/<group>.json), or None if the data file is absent."""
    from .util import safe_child
    try:
        p = safe_child(config.DATA_DIR, group, ".json")
    except ValueError:
        return None
    return [r["conversation_id"] for r in data.load_file(p)] if p.exists() else None


def cmd_export_v1(a):
    """Submission JSON 1.0 -> results/dev/results.json (DEV) or results/prod/results.json (control group)."""
    from pathlib import Path

    from . import submission
    from .util import atomic_write
    doc = store.load_run(a.group, a.run)
    if not doc:
        sys.exit(f"no results for group {a.group}" + (f" / run {a.run}" if a.run else ""))
    ids = _group_ids(a.group)
    obj = submission.to_v1(doc, ids=ids)
    out = Path(a.out) if a.out else submission.out_path(a.group)
    errs, warns = submission.validate(obj, prod=out.parent.name == "prod", expected_ids=ids, custom_ok=True)
    for w in warns[:10]:
        print("WARNING: " + w, file=sys.stderr)
    if errs:
        sys.exit("export failed validation:\n  " + "\n  ".join(errs[:20]))
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
    n_err = sum(r["status"] == "error" for r in obj["results"])
    print(f"{out}: {len(obj['results'])} records ({n_err} error), schema {obj['schema_version']}, model {obj['model']}, valid")


def cmd_validate(a):
    from . import submission
    try:
        obj = json.loads(open(a.file, encoding="utf-8").read())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        sys.exit(f"{a.file}: not valid UTF-8 JSON: {e}")
    prod = a.prod or "/prod/" in a.file.replace("\\", "/")
    errs, warns = submission.validate(obj, prod=prod, expected_ids=_group_ids(a.group) if a.group else None,
                                      custom_ok=a.allow_custom)
    for w in warns[:20]:
        print("WARNING: " + w)
    print("\n".join("ERROR: " + e for e in errs) if errs else
          f"{a.file}: valid JSON 1.0 ({len(obj['results'])} records{', prod: exactly 10' if prod else ''})")
    sys.exit(1 if errs else 0)


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
    ex = sub.add_parser("export", help="export results: v1 submission JSON, organizer CSV or ai/human/both CSV")
    ex.add_argument("--group", required=True)
    ex.add_argument("--run", help="results/<run>.json (default: results/<group>.json)")
    ex.add_argument("--format", default="organizer", choices=["v1", "organizer", "ai", "human", "both"],
                    help="v1 = submission JSON 1.0 (default out: results/dev/results.json for DEV, else results/prod/results.json)")
    va = sub.add_parser("validate", help="validate a v1.0 results.json")
    va.add_argument("file")
    va.add_argument("--prod", action="store_true", help="enforce exactly 10 records (auto for .../prod/...)")
    va.add_argument("--group", help="also check ids == data/<group>.json ids")
    va.add_argument("--allow-custom", action="store_true", help="replaced/custom criterion codes are warnings")
    ex.add_argument("--out")
    ex.add_argument("--fill-provisional", action="store_true")
    lb = sub.add_parser("label", help="label real dialogs by hand (stored in a git-ignored JSON)")
    lb.add_argument("--group", default="DEV")
    lb.add_argument("--ids", help="comma-separated conversation ids (default: next unlabelled)")
    lb.add_argument("-n", type=int, default=5, help="how many unlabelled dialogs to label")
    lb.add_argument("--out", help="labels JSON (default: session labels_out, data/human/labels_<A>.json or examples/dev_labels.json)")
    lb.add_argument("--labeller", default="")
    lb.add_argument("--annotator", help="annotator id (A/B); enables the one-keystroke mode")
    lb.add_argument("--sample", help="session file from scripts/labeling_kit.py (results/human/session_A.json)")
    lb.add_argument("--slow", action="store_true", help="old typed-answer mode (YES/NO + note per criterion)")
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8090)
    for sp in (e, gd):
        sp.add_argument("--preset", choices=["econom", "balanced", "max", "jury", "cascade", "hybrid"],
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
    elif a.cmd == "validate":
        cmd_validate(a)
    elif a.cmd == "label":
        cmd_label(a)
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("cqa.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()

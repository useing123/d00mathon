"""Result storage and export. AI results: results/<group>.json; human overrides: results/overrides/<group>.json."""
import csv
import io
import json
import time

from . import checklist as ckl, config
from .evaluator import apply_overrides


def results_path(group):
    return config.RESULTS_DIR / f"{group.lower()}.json"


def overrides_path(group):
    return config.RESULTS_DIR / "overrides" / f"{group.lower()}.json"


def load_results(group):
    p = results_path(group)
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    ov = load_overrides(group)
    for d in doc["dialogs"]:
        apply_overrides(d, ckl.load(d["checklist_id"]), ov.get(d["conversation_id"], {}))
    return doc


def save_results(group, dialogs, meta, out=None):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    doc = {"group": group, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "meta": meta, "dialogs": dialogs}
    p = out or results_path(group)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return doc


def load_overrides(group):
    p = overrides_path(group)
    return json.loads(p.read_text()) if p.exists() else {}


def save_override(group, conv_id, crit_id, value, justification, reviewer=""):
    ov = load_overrides(group)
    d = ov.setdefault(conv_id, {})
    if value in (None, "", "AI"):
        d.pop(crit_id, None)
    else:
        d[crit_id] = {"value": value, "justification": justification, "reviewer": reviewer,
                      "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not d:
        ov.pop(conv_id, None)
    p = overrides_path(group)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ov, ensure_ascii=False, indent=1))
    return ov


def to_csv(doc, which="ai"):
    """Flat CSV: one row per dialog, one column per criterion + total + gross. which = ai | human | both"""
    crit_ids = []
    for d in doc["dialogs"]:
        for r in d["criteria"]:
            if r["criterion_id"] not in crit_ids:
                crit_ids.append(r["criterion_id"])
    buf = io.StringIO()
    w = csv.writer(buf)
    head = ["conversation_id", "channel", "language", "checklist_id"]
    if which in ("ai", "both"):
        head += [f"c{c}" for c in crit_ids] + ["total", "gross_error"]
    if which in ("human", "both"):
        head += [f"human_c{c}" for c in crit_ids] + ["human_total", "human_gross_error", "reviewed"]
    w.writerow(head)
    for d in doc["dialogs"]:
        row = [d["conversation_id"], d.get("channel"), d.get("language"), d.get("checklist_id")]
        vals = {r["criterion_id"]: r.get("value") for r in d["criteria"]}
        if which in ("ai", "both"):
            row += [vals.get(c, "") for c in crit_ids] + [d.get("total"), "YES" if d.get("gross_error") else "NO"]
        if which in ("human", "both"):
            h = d.get("human", {})
            row += [h.get("values", {}).get(c, "") for c in crit_ids] + [h.get("total"),
                    "YES" if h.get("gross_error") else "NO", "YES" if h.get("reviewed") else "NO"]
        w.writerow(row)
    return buf.getvalue()

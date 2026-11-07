"""Result storage and export. AI results: results/<group>.json; human overrides: results/overrides/<group>.json."""
import csv
import io
import json
import time

from . import checklist as ckl, config
from .evaluator import apply_overrides
from .util import atomic_write


def results_path(group):
    from .util import safe_child
    return safe_child(config.RESULTS_DIR, str(group).lower(), ".json")


def overrides_path(group):
    from .util import safe_child
    return safe_child(config.RESULTS_DIR / "overrides", str(group).lower(), ".json")


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
    atomic_write(p, json.dumps(doc, ensure_ascii=False, indent=1))
    return doc


def load_overrides(group):
    p = overrides_path(group)
    return json.loads(p.read_text()) if p.exists() else {}


def save_override(group, conv_id, crit_id, value, justification, reviewer="", checklist_id=None):
    """Validate (if checklist_id given) and store a human override. Raises ValueError on invalid values."""
    if value not in (None, "", "AI") and checklist_id:
        from .evaluator import validate_override
        cl = ckl.load(checklist_id)
        cr = next((c for c in cl["criteria"] if c["id"] == crit_id), None)
        if cr is None:
            raise ValueError(f"unknown criterion {crit_id!r}")
        value = validate_override(cr, value)
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
    atomic_write(p, json.dumps(ov, ensure_ascii=False, indent=1))
    return ov


def _cell(v):
    """Neutralise spreadsheet formula injection (=, +, -, @, tab, CR at the start of a text cell)."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def to_csv(doc, which="ai"):
    """Flat CSV: one row per dialog, one column per criterion + total + gross. which = ai | human | both"""
    crit_ids = []
    for d in doc["dialogs"]:
        for r in d["criteria"]:
            if r["criterion_id"] not in crit_ids:
                crit_ids.append(r["criterion_id"])
    buf = io.StringIO()
    w = csv.writer(buf)
    head = ["conversation_id", "channel", "language", "checklist_id", "status", "failed_criteria", "needs_review"]
    if which in ("ai", "both"):
        head += [f"c{c}" for c in crit_ids] + ["total", "provisional_total", "gross_error"]
    if which in ("human", "both"):
        head += [f"human_c{c}" for c in crit_ids] + ["human_total", "human_gross_error", "reviewed"]
    w.writerow(head)
    for d in doc["dialogs"]:
        row = [d["conversation_id"], d.get("channel"), d.get("language"), d.get("checklist_id"),
               d.get("status", "complete"), ";".join(d.get("failed_criteria") or []), ";".join(d.get("needs_review") or [])]
        vals = {r["criterion_id"]: r.get("value") for r in d["criteria"]}
        if which in ("ai", "both"):
            row += [vals.get(c, "") for c in crit_ids] + [d.get("total"), d.get("provisional_total"),
                                                          "YES" if d.get("gross_error") else "NO"]
        if which in ("human", "both"):
            h = d.get("human", {})
            row += [h.get("values", {}).get(c, "") for c in crit_ids] + [h.get("total"),
                    "YES" if h.get("gross_error") else "NO", "YES" if h.get("reviewed") else "NO"]
        w.writerow([_cell(x) for x in row])
    return buf.getvalue()


def to_organizer_csv(doc, fill_provisional=False):
    """Submission file: ONLY scored columns — conversation_id, one column per criterion id (YES/NO/NA or scale
    step), total (2 decimals), gross_error (YES/NO). The exact organizer spec is not published; this format is our
    assumption (docs/API.md). Incomplete dialogs keep an empty total unless ``fill_provisional`` (opt-in) is set;
    diagnostics (status, failed criteria) go to ``to_diagnostics_csv``."""
    crit_ids = _crit_ids(doc)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["conversation_id"] + crit_ids + ["total", "gross_error"])
    for d in doc["dialogs"]:
        vals = {r["criterion_id"]: r.get("value") for r in d["criteria"]}
        tot = d.get("total")
        if tot is None and fill_provisional:
            tot = d.get("provisional_total")
        w.writerow([_cell(x) for x in [d["conversation_id"]] + [("" if vals.get(c) is None else vals.get(c)) for c in crit_ids]
                    + ["" if tot is None else f"{tot:.2f}", "YES" if d.get("gross_error") else "NO"]])
    return buf.getvalue()


def to_diagnostics_csv(doc):
    """Companion file for the submission: per-dialog status, failed / needs-review criteria, provisional total."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["conversation_id", "status", "failed_criteria", "needs_review", "provisional_total", "warnings"])
    for d in doc["dialogs"]:
        w.writerow([_cell(x) for x in [d["conversation_id"], d.get("status", "complete"), ";".join(d.get("failed_criteria") or []),
                                       ";".join(d.get("needs_review") or []), d.get("provisional_total"),
                                       " | ".join(d.get("warnings") or [])]])
    return buf.getvalue()


def _crit_ids(doc):
    ids = []
    for d in doc["dialogs"]:
        for r in d["criteria"]:
            if r["criterion_id"] not in ids:
                ids.append(r["criterion_id"])
    return ids

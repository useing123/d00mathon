"""Submission format v1.0 (results/<group>/results.json).

ALL format details that depend on the organizer Student Guide appendix live in THIS file (``SPEC``). Items marked
ASSUMED were inferred from the guide's description and must be confirmed against the appendix text; adjust here
only — the CLI (`python -m cqa export --format v1`), the API (/api/export/<group>/results.json), the JSON schema
(docs/results.schema.json, regenerate with `python -m cqa.submission --schema`) and the tests all read SPEC.
"""
import json
import time

from . import checklist as ckl, config

SPEC = {
    "format_version": "1.0",
    # internal criterion id -> submission code  (ASSUMED: codes and their order)
    "criteria": {
        "01": "contact",
        "02": "discovery",
        "03": "identification",
        "04": "consultation",
        "05": "empathy",
        "06": "dialogue_management",
        "07": "closing",
        "08": "critical",
    },
    "binary_values": {"YES": True, "NO": False, "NA": None},   # ASSUMED: YES/NO/NA -> true/false/null
    "scale_value": "step",            # ASSUMED: scale criteria -> the numeric step (e.g. 15), NA -> null
    "turn_base": 1,                   # evidence_turns are 1-based (same as data.TURN_BASE)
    "score_decimals": 2,              # final_score rounded to 2 decimals (ASSUMED: null when not computable)
    "use_human_overrides": True,      # ASSUMED: reviewer overrides / resolved appeals replace AI values
    "out_path": "results/{group}/results.json",   # results/dev/results.json, results/prod/results.json
}
ASSUMED = [
    "criterion codes contact/discovery/identification/consultation/empathy/dialogue_management/closing/critical "
    "for internal ids 01..08",
    "YES→true, NO→false, NA→null; scale criteria → numeric step (NA→null)",
    "document layout: top-level {format_version, group, generated_at, source_run, turn_numbering, dialogs[]}; "
    "per dialog {conversation_id, checklist_id, final_score, critical_flag, criteria{code: {value, evidence_turns}}}",
    "final_score = total on the 0..100 scale (NA criteria excluded, weights renormalised), null if incomplete",
    "critical_flag = criterion 08 (gross error) == YES",
    "human overrides and resolved appeals take precedence over AI values",
]


def _crit_value(cr, v):
    if v is None:
        return None
    if cr.get("type") == "scale":
        return None if v == "NA" else int(v)
    return SPEC["binary_values"][str(v)]


def to_v1(doc, source_run=""):
    """Convert a results doc (with overrides applied, see store.load_results) to the v1.0 submission dict."""
    from .data import TURN_BASE
    assert SPEC["turn_base"] == TURN_BASE
    codes = SPEC["criteria"]
    out = []
    for d in doc["dialogs"]:
        cl = ckl.load(d["checklist_id"])
        defs = {c["id"]: c for c in cl["criteria"]}
        h = d.get("human") or {}
        use_h = SPEC["use_human_overrides"] and h.get("reviewed")
        vals = h.get("values", {}) if use_h else {}
        crits = {}
        for r in d["criteria"]:
            cid = r["criterion_id"]
            if cid not in codes:
                raise ValueError(f"criterion {cid!r} has no submission code in submission.SPEC['criteria']")
            v = vals.get(cid, r.get("value"))
            crits[codes[cid]] = {"value": _crit_value(defs.get(cid, r), v),
                                 "evidence_turns": sorted({int(e["turn"]) for e in r.get("evidence") or []
                                                           if isinstance(e.get("turn"), int)})}
        total = h.get("total") if use_h else d.get("total")
        gross = h.get("gross_error") if use_h else d.get("gross_error")
        out.append({"conversation_id": d["conversation_id"], "checklist_id": d["checklist_id"],
                    "final_score": None if total is None else round(float(total), SPEC["score_decimals"]),
                    "critical_flag": bool(gross), "criteria": crits})
    return {"format_version": SPEC["format_version"], "group": doc.get("group"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_run": source_run or doc.get("run", ""),
            "turn_numbering": "1-based", "dialogs": out}


def schema():
    """JSON schema of the v1.0 document, generated from SPEC (docs/results.schema.json)."""
    crit = {"type": "object", "additionalProperties": False, "required": ["value", "evidence_turns"],
            "properties": {"value": {"type": ["boolean", "integer", "null"]},
                           "evidence_turns": {"type": "array", "items": {"type": "integer", "minimum": SPEC["turn_base"]},
                                              "uniqueItems": True}}}
    codes = list(SPEC["criteria"].values())
    dialog = {"type": "object", "additionalProperties": False,
              "required": ["conversation_id", "checklist_id", "final_score", "critical_flag", "criteria"],
              "properties": {"conversation_id": {"type": "string", "minLength": 1}, "checklist_id": {"type": "string"},
                             "final_score": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
                             "critical_flag": {"type": "boolean"},
                             "criteria": {"type": "object", "additionalProperties": False, "required": codes,
                                          "properties": {c: {"$ref": "#/$defs/criterion"} for c in codes}}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "cqa-results-v1.0",
            "title": "CQA submission results.json v1.0 (see cqa/submission.py SPEC; ASSUMED items pending the guide)",
            "type": "object", "additionalProperties": False,
            "required": ["format_version", "group", "generated_at", "source_run", "turn_numbering", "dialogs"],
            "properties": {"format_version": {"const": SPEC["format_version"]}, "group": {"type": "string"},
                           "generated_at": {"type": "string"}, "source_run": {"type": "string"},
                           "turn_numbering": {"const": "1-based"}, "dialogs": {"type": "array", "items": {"$ref": "#/$defs/dialog"}}},
            "$defs": {"criterion": crit, "dialog": dialog}}


def validate(obj):
    """Validate against schema() plus semantic checks; returns a list of problems (empty = valid)."""
    errs = []
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(schema())
        errs += [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(obj)]
    except ImportError:   # minimal fallback when jsonschema is not installed
        if obj.get("format_version") != SPEC["format_version"]:
            errs.append("format_version mismatch")
    seen = set()
    for i, d in enumerate(obj.get("dialogs") or []):
        cid = d.get("conversation_id")
        if cid in seen:
            errs.append(f"dialogs/{i}: duplicate conversation_id {cid}")
        seen.add(cid)
        crit = (d.get("criteria") or {}).get(SPEC["criteria"]["08"], {})
        if crit and d.get("critical_flag") != (crit.get("value") is True):
            errs.append(f"dialogs/{i}: critical_flag disagrees with criterion {SPEC['criteria']['08']}")
    return errs


def out_path(group):
    from .util import safe_child
    return safe_child(config.RESULTS_DIR, str(group).lower(), "") / "results.json"


if __name__ == "__main__":
    import sys
    if "--schema" in sys.argv:
        print(json.dumps(schema(), ensure_ascii=False, indent=1))

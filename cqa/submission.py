"""Submission format JSON 1.0 (organizer Student Guide appendix «Как сдать результаты»).

All format rules live in THIS file (SPEC + to_v1 + validate); the CLI (`python -m cqa export --format v1`,
`python -m cqa validate <file>`), the API (/api/export/<group>/results.json), docs/results.schema.json
(`python -m cqa.submission --schema`) and the tests read them from here.

Rules (from the appendix):
- top level {"schema_version":"1.0","team_name","model":{"mode":local|external_api|hybrid,"name"},"results":[...]}
- ok record {"conversation_id","status":"ok","criteria":[8 x {"criterion_id","value","rationale","evidence_turns"}],
  "score":{"final_score":0..100 (2 decimals),"critical_flag":== value of "critical"}}
- error record {"conversation_id","status":"error","criteria":[],"score":null,"error":"<reason>"}; never a zero score
- evidence_turns: 1-based index into the ORIGINAL transcript array (client, bot, operator and system events);
  data.normalize keeps every turn, in order (no drop/merge), so data.TURN_BASE numbering == original numbering
- one record per dialog, no duplicate ids, ids exactly as in the API; prod has exactly 10 records
"""
import json
import re

from . import checklist as ckl, config

SPEC = {
    "schema_version": "1.0",
    # internal checklist id -> official criterion code. A checklist criterion may set `code:` explicitly;
    # custom criteria without a mapping export with their configured id.
    "criteria": {"01": "contact", "02": "discovery", "03": "identification", "04": "consultation",
                 "05": "empathy", "06": "dialogue", "07": "closure", "08": "critical"},
    # allowed values per official code (null = N/A); scale values are the raw step before weighting
    "domains": {"contact": [True, False], "discovery": [True, False, None], "identification": [True, False, None],
                "consultation": [0, 5, 10, 15, 20, None], "empathy": [True, False], "dialogue": [0, 4, 7, 10],
                "closure": [True, False, None], "critical": [True, False]},
    "binary_values": {"YES": True, "NO": False, "NA": None},
    "turn_base": 1,
    "score_decimals": 2,
    "prod_records": 10,
    "modes": ["local", "external_api", "hybrid"],
    "use_human_overrides": True,   # reviewer overrides / resolved appeals replace AI values (our decision)
}
CODES = list(SPEC["criteria"].values())


def team_name():
    return config.get("CQA_TEAM_NAME", "казачи капай") or "казачи капай"


def criterion_code(cr):
    return str(cr.get("code") or SPEC["criteria"].get(str(cr["id"])) or cr["id"])


def model_info(meta):
    """{"mode","name"} from the run meta: actual model ids, comma-separated."""
    meta = meta or {}
    rep = meta.get("reproducibility") or {}
    names = list(dict.fromkeys((rep.get("jurors") or []) + ([rep["judge"]] if rep.get("judge") else [])))
    if not names:
        names = [m for m in re.split(r"[\s\[\]+=,]+", str(meta.get("model") or "")) if m and m not in ("jury", "judge")]
    prov = str(meta.get("provider") or config.PROVIDER)
    if prov == "mock":
        mode = "local"
    elif prov == "openai":
        base = str(config.OPENAI_BASE_URL)
        mode = "local" if re.search(r"127\.0\.0\.1|localhost|runpod|0\.0\.0\.0", base) else "external_api"
    else:
        mode = "external_api"
    mode = config.get("CQA_MODEL_MODE", mode) or mode
    return {"mode": mode, "name": ", ".join(names) or "unknown"}


def _value(cr, v):
    if v is None or v == "NA":
        return None
    if cr.get("type") == "scale":
        return int(v)
    return SPEC["binary_values"][str(v)]


def _error(cid, reason):
    return {"conversation_id": cid, "status": "error", "criteria": [], "score": None, "error": reason}


def _record(d):
    cid = d["conversation_id"]
    h = d.get("human") or {}
    use_h = bool(SPEC["use_human_overrides"] and h.get("reviewed"))
    incomplete = (h.get("status") == "incomplete") if use_h else (d.get("status") == "incomplete")
    if incomplete:
        failed = d.get("failed_criteria") or [r["criterion_id"] for r in d["criteria"] if r.get("value") is None]
        errs = "; ".join(f"{r['criterion_id']}: {r.get('error')}" for r in d["criteria"] if r.get("error"))
        return _error(cid, f"evaluation failed for criteria {', '.join(failed) or '?'}" + (f" ({errs[:300]})" if errs else ""))
    total = h.get("total") if use_h else d.get("total")
    if total is None:
        return _error(cid, "no applicable weighted criteria (final score undefined)")
    vals = (h.get("values") or {}) if use_h else {}
    defs = {c["id"]: c for c in ckl.load(d["checklist_id"])["criteria"]}
    crits, critical = [], None
    for r in d["criteria"]:
        cr = defs.get(r["criterion_id"], r)
        code = criterion_code(cr)
        val = _value(cr, vals.get(r["criterion_id"], r.get("value")))
        ov = (h.get("overrides") or {}).get(r["criterion_id"]) if use_h else None
        rat = (ov or {}).get("justification") or r.get("justification") or r.get("na_reason") or r.get("missing_where") or ""
        crits.append({"criterion_id": code, "value": val, "rationale": str(rat).strip(),
                      "evidence_turns": sorted({int(e["turn"]) for e in r.get("evidence") or [] if isinstance(e.get("turn"), int)})})
        if cr.get("gross") or code == "critical":
            critical = val
    gross = h.get("gross_error") if use_h else d.get("gross_error")
    flag = bool(critical) if critical is not None else bool(gross)
    return {"conversation_id": cid, "status": "ok", "criteria": crits,
            "score": {"final_score": round(float(total), SPEC["score_decimals"]), "critical_flag": flag}}


def to_v1(doc, ids=None, meta=None):
    """doc: results doc with overrides applied. ids: the group's conversation ids in API order; dialogs without
    a result become error records (never dropped)."""
    by = {d["conversation_id"]: d for d in doc.get("dialogs") or []}
    order = list(ids) if ids else list(by)
    out = []
    for cid in order:
        d = by.get(cid)
        if d is None:
            out.append(_error(cid, "dialog was not evaluated (no result in the run)"))
            continue
        try:
            out.append(_record(d))
        except Exception as e:   # malformed result -> explicit error record, not a silent drop
            out.append(_error(cid, f"export failed: {e}"))
    return {"schema_version": SPEC["schema_version"], "team_name": team_name(),
            "model": model_info(meta if meta is not None else doc.get("meta")), "results": out}


def schema():
    crit = {"type": "object", "additionalProperties": False, "required": ["criterion_id", "value", "rationale", "evidence_turns"],
            "properties": {"criterion_id": {"type": "string", "minLength": 1},
                           "value": {"type": ["boolean", "integer", "null"]}, "rationale": {"type": "string"},
                           "evidence_turns": {"type": "array", "items": {"type": "integer", "minimum": SPEC["turn_base"]}}}}
    ok = {"type": "object", "additionalProperties": False, "required": ["conversation_id", "status", "criteria", "score"],
          "properties": {"conversation_id": {"type": "string", "minLength": 1}, "status": {"const": "ok"},
                         "criteria": {"type": "array", "items": {"$ref": "#/$defs/criterion"}, "minItems": 1},
                         "score": {"type": "object", "additionalProperties": False, "required": ["final_score", "critical_flag"],
                                   "properties": {"final_score": {"type": "number", "minimum": 0, "maximum": 100},
                                                  "critical_flag": {"type": "boolean"}}}}}
    err = {"type": "object", "additionalProperties": False, "required": ["conversation_id", "status", "criteria", "score", "error"],
           "properties": {"conversation_id": {"type": "string", "minLength": 1}, "status": {"const": "error"},
                          "criteria": {"type": "array", "maxItems": 0}, "score": {"type": "null"},
                          "error": {"type": "string", "minLength": 1}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "cqa-results-1.0",
            "title": "CQA results.json, format JSON 1.0 (organizer Student Guide appendix)",
            "type": "object", "additionalProperties": False, "required": ["schema_version", "team_name", "model", "results"],
            "properties": {"schema_version": {"const": "1.0"}, "team_name": {"type": "string", "minLength": 1},
                           "model": {"type": "object", "additionalProperties": False, "required": ["mode", "name"],
                                     "properties": {"mode": {"enum": SPEC["modes"]}, "name": {"type": "string", "minLength": 1}}},
                           "results": {"type": "array", "items": {"oneOf": [{"$ref": "#/$defs/ok"}, {"$ref": "#/$defs/error"}]}}},
            "$defs": {"criterion": crit, "ok": ok, "error": err}}


def _same(a, b):   # True == 1 in Python; keep JSON types distinct
    return type(a) is type(b) and a == b


def validate(obj, prod=False, expected_ids=None, custom_ok=False, custom_checklist=False):
    """Return (errors, warnings): schema + semantic rules of the appendix. prod=True enforces exactly 10 records.
    custom_ok: a replaced/custom criterion code is a warning instead of an error (control-step criterion swap)."""
    errs, warns = [], []
    try:
        import jsonschema
        for e in jsonschema.Draft202012Validator(schema()).iter_errors(obj):
            errs.append(f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:200]}")
    except ImportError:
        warns.append("jsonschema not installed: structural schema check skipped")
    if not isinstance(obj, dict):
        return errs or ["root must be an object"], warns
    res = obj.get("results") if isinstance(obj.get("results"), list) else []
    ids = [r.get("conversation_id") for r in res if isinstance(r, dict)]
    dup = sorted({i for i in ids if ids.count(i) > 1}, key=str)
    if dup:
        errs.append(f"duplicate conversation_id: {dup}")
    if prod and len(res) != SPEC["prod_records"]:
        errs.append(f"prod must contain exactly {SPEC['prod_records']} records, got {len(res)}")
    if expected_ids is not None:
        miss, extra = set(expected_ids) - set(ids), set(ids) - set(expected_ids)
        if miss:
            errs.append(f"missing records for API ids: {sorted(miss)[:10]}")
        if extra:
            errs.append(f"ids not in the API set: {sorted(extra)[:10]}")
    for i, r in enumerate(res):
        if not isinstance(r, dict) or r.get("status") != "ok":
            continue
        where = f"results/{i} ({r.get('conversation_id')})"
        crits = [c for c in r.get("criteria") or [] if isinstance(c, dict)]
        codes = [c.get("criterion_id") for c in crits]
        if len(set(codes)) != len(codes):
            errs.append(f"{where}: criterion repeated: {codes}")
        if set(codes) != set(CODES):
            msg = f"{where}: criteria {sorted(map(str, codes))} != official 8 {sorted(CODES)}"
            (warns if custom_checklist or (custom_ok and "critical" in codes) else errs).append(msg)
        for c in crits:
            dom = SPEC["domains"].get(c.get("criterion_id"))
            if dom is not None and not any(_same(c.get("value"), x) for x in dom):
                errs.append(f"{where}: {c.get('criterion_id')}={c.get('value')!r} not in {dom}")
            if not str(c.get("rationale") or "").strip():
                warns.append(f"{where}: {c.get('criterion_id')} has an empty rationale")
        crit = next((c for c in crits if c.get("criterion_id") == "critical"), None)
        sc = r.get("score") if isinstance(r.get("score"), dict) else {}
        if crit is not None and sc.get("critical_flag") is not crit.get("value"):
            errs.append(f"{where}: score.critical_flag must equal value of criterion critical")
        fs = sc.get("final_score")
        if isinstance(fs, (int, float)) and round(fs, 2) != fs:
            errs.append(f"{where}: final_score must be rounded to 2 decimals")
    return errs, warns


def out_path(group, prod=None):
    """results/dev/results.json for DEV, results/prod/results.json for any other (control) group."""
    if prod is None:
        prod = str(group).upper() != "DEV"
    return config.ROOT / "results" / ("prod" if prod else "dev") / "results.json"


if __name__ == "__main__":
    import sys
    if "--schema" in sys.argv:
        print(json.dumps(schema(), ensure_ascii=False, indent=1))

"""Checklists: YAML storage, validation, per-criterion hashing, channel auto-pick."""
import hashlib
import json
import math
import re

import yaml

from . import config

DEFAULT_ASSUMPTIONS = {"rounding": "half_up", "decimals": 2, "first_reply": "first_human_operator_turn",
                       "ignore_internal_hold_turns": True, "self_consistency_samples": 3,
                       "confidence_threshold": 0.8}


def list_checklists():
    out = []
    for p in sorted(config.CHECKLIST_DIR.glob("*.yaml")):
        c = load(p.stem)
        out.append({"id": c["id"], "name": c.get("name", c["id"]), "channel": c.get("channel", "chat"),
                    "criteria": len(c["criteria"])})
    return out


ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")


def load(cid):
    if not ID_RE.fullmatch(str(cid)) or ".." in str(cid):
        raise ValueError(f"invalid checklist id {cid!r}")
    base = config.CHECKLIST_DIR.resolve()
    p = (base / f"{cid}.yaml").resolve()
    if p.parent != base:
        raise ValueError(f"invalid checklist id {cid!r}")
    return validate(_resolve(yaml.safe_load(p.read_text(encoding="utf-8")), cid))


def _resolve(c, cid, depth=0):
    """`extends: <base id>` inherits everything from the base; `criteria_overrides` patch inherited criteria by id."""
    base_id = c.pop("extends", None)
    if not base_id:
        c.pop("criteria_overrides", None)
        return c
    if depth > 3 or base_id == cid:
        raise ValueError(f"checklist {cid!r}: bad extends chain")
    if not ID_RE.fullmatch(str(base_id)):
        raise ValueError(f"invalid base checklist id {base_id!r}")
    base = _resolve(yaml.safe_load((config.CHECKLIST_DIR / f"{base_id}.yaml").read_text(encoding="utf-8")), base_id, depth + 1)
    over = {o["id"]: o for o in c.pop("criteria_overrides", []) or []}
    merged = {**base, **{k: v for k, v in c.items() if k != "criteria"}}
    crits = [dict(x) for x in (c.get("criteria") or base.get("criteria") or [])]
    unknown = set(over) - {x["id"] for x in crits}
    if unknown:
        raise ValueError(f"checklist {cid!r}: overrides for unknown criteria {sorted(unknown)}")
    merged["criteria"] = [{**x, **over.get(x["id"], {})} for x in crits]
    merged["assumptions"] = {**base.get("assumptions", {}), **c.get("assumptions", {})}
    return merged


def validate(c):
    if not isinstance(c, dict) or "id" not in c or not c.get("criteria"):
        raise ValueError("checklist needs id and criteria[]")
    c.setdefault("channel", "chat")
    a = dict(DEFAULT_ASSUMPTIONS)
    a.update(c.get("assumptions") or {})
    c["assumptions"] = a
    ids = set()
    for cr in c["criteria"]:
        cr["id"] = str(cr["id"])
        if cr["id"] in ids:
            raise ValueError(f"duplicate criterion id {cr['id']}")
        ids.add(cr["id"])
        cr.setdefault("type", "binary")
        cr.setdefault("weight", 0)
        try:
            w = float(cr["weight"])
        except (TypeError, ValueError):
            raise ValueError(f"criterion {cr['id']}: weight must be a number") from None
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"criterion {cr['id']}: weight must be a finite non-negative number")
        cr.setdefault("allow_na", False)
        cr.setdefault("gross", False)
        cr.setdefault("rules", "")
        if cr["type"] == "scale":
            if not cr.get("scale"):
                raise ValueError(f"criterion {cr['id']}: scale type needs scale: [..]")
            for x in cr["scale"]:
                if isinstance(x, str) and re.fullmatch(r"\s*-?\d+\s*", x):
                    continue   # integer given as text (e.g. from the UI editor)
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or int(x) != x:
                    raise ValueError(f"criterion {cr['id']}: scale steps must be integers (got {x!r})")
            raw = [int(x) for x in cr["scale"]]
            if len(set(raw)) != len(raw):
                raise ValueError(f"criterion {cr['id']}: scale steps must be unique")
            if any(x < 0 for x in raw):
                raise ValueError(f"criterion {cr['id']}: scale steps must be non-negative")
            cr["scale"] = sorted(raw)
            if cr["scale"][-1] <= 0:
                raise ValueError(f"criterion {cr['id']}: top scale step must be > 0")
        elif cr["type"] != "binary":
            raise ValueError(f"criterion {cr['id']}: type must be binary|scale")
    return c


def strict_value(cr, v):
    """Canonical value if v is EXACTLY an allowed value for cr, else None. No truncation (4.5 is not 4)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        if not v.is_integer():
            return None
        v = int(v)
    s = str(v).strip().upper()
    if s == "NA":
        return "NA" if cr.get("allow_na") else None
    if cr.get("type", "binary") == "scale":
        if not re.fullmatch(r"-?\d+", s):
            return None
        n = int(s)
        return n if n in [int(x) for x in cr["scale"]] else None
    return s if s in ("YES", "NO") else None


def save(c):
    c = validate(c)
    if not all(ch.isalnum() or ch in "_-." for ch in c["id"]):
        raise ValueError("bad checklist id")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", c["id"]) or ".." in c["id"]:
        raise ValueError("bad checklist id")
    from .util import atomic_write
    path = config.CHECKLIST_DIR / f"{c['id']}.yaml"
    out = c
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if raw.get("extends"):   # keep inheritance: store only what differs from the base checklist
            out = _diff_from_base(c, raw["extends"])
    atomic_write(path, yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=1000))
    return c


def _diff_from_base(c, base_id):
    base = load(base_id)
    out = {k: v for k, v in c.items() if k not in ("criteria", "assumptions") and base.get(k) != v}
    out["id"] = c["id"]
    out["extends"] = base_id
    ass = {k: v for k, v in c.get("assumptions", {}).items() if base["assumptions"].get(k) != v}
    if ass:
        out["assumptions"] = ass
    bcr = {x["id"]: x for x in base["criteria"]}
    if [x["id"] for x in c["criteria"]] != [x["id"] for x in base["criteria"]]:
        out["criteria"] = c["criteria"]          # structure changed (added/removed/reordered) → store criteria in full
    else:
        ov = [{"id": x["id"], **{k: v for k, v in x.items() if k != "id" and bcr[x["id"]].get(k) != v}}
              for x in c["criteria"]]
        ov = [o for o in ov if len(o) > 1]
        if ov:
            out["criteria_overrides"] = ov
    return out


def criterion_hash(checklist, cr):
    """Hash of everything that changes the LLM's judgement for this criterion (weight excluded:
    reweighting only changes scoring, not the per-criterion value)."""
    payload = {k: v for k, v in cr.items() if k != "weight"}
    payload["_desc"] = checklist.get("description", "")
    payload["_assumptions"] = {k: v for k, v in checklist["assumptions"].items() if k not in ("rounding", "decimals")}
    # model / jury identity is NOT part of this hash: it is added to the cache key in Evaluator.evaluate_criterion
    from .prompts import PROMPT_VERSION, SYSTEM, TOOL
    payload["_pv"] = PROMPT_VERSION
    # hash the actual prompt content too, so an edit without a version bump cannot reuse stale cache entries
    payload["_prompt"] = hashlib.sha256((SYSTEM + json.dumps(TOOL, ensure_ascii=False, sort_keys=True)).encode()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def pick_for_channel(kind, chat_id=None, call_id=None):
    chat_id = chat_id or config.get("CQA_CHECKLIST_CHAT", "freedom_chat_v1")
    call_id = call_id or config.get("CQA_CHECKLIST_CALL", "freedom_call_v1")
    return call_id if kind == "call" else chat_id

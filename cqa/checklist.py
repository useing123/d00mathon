"""Checklists: YAML storage, validation, per-criterion hashing, channel auto-pick."""
import hashlib
import json

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


def load(cid):
    p = config.CHECKLIST_DIR / f"{cid}.yaml"
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    return validate(c)


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
        cr.setdefault("allow_na", False)
        cr.setdefault("gross", False)
        cr.setdefault("rules", "")
        if cr["type"] == "scale":
            if not cr.get("scale"):
                raise ValueError(f"criterion {cr['id']}: scale type needs scale: [..]")
            cr["scale"] = sorted(int(x) for x in cr["scale"])
        elif cr["type"] != "binary":
            raise ValueError(f"criterion {cr['id']}: type must be binary|scale")
    return c


def save(c):
    c = validate(c)
    if not all(ch.isalnum() or ch in "_-." for ch in c["id"]):
        raise ValueError("bad checklist id")
    (config.CHECKLIST_DIR / f"{c['id']}.yaml").write_text(
        yaml.safe_dump(c, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    return c


def criterion_hash(checklist, cr):
    """Hash of everything that changes the LLM's judgement for this criterion (weight excluded:
    reweighting only changes scoring, not the per-criterion value)."""
    payload = {k: v for k, v in cr.items() if k != "weight"}
    payload["_desc"] = checklist.get("description", "")
    payload["_assumptions"] = {k: v for k, v in checklist["assumptions"].items() if k not in ("rounding", "decimals")}
    payload["_model"] = f"{config.PROVIDER}:{config.MODEL}"
    from .prompts import PROMPT_VERSION
    payload["_pv"] = PROMPT_VERSION
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def pick_for_channel(kind, chat_id=None, call_id=None):
    chat_id = chat_id or config.get("CQA_CHECKLIST_CHAT", "freedom_chat_v1")
    call_id = call_id or config.get("CQA_CHECKLIST_CALL", "freedom_call_v1")
    return call_id if kind == "call" else chat_id

"""Two-step appeals (guide items 05–06): a user disputes one criterion (status 'open', mandatory comment), a reviewer
CONFIRMS the AI value or CHANGES it (mandatory reason) -> status 'resolved'. A change is stored through the normal
override storage (store.save_override), so totals are recomputed exactly like a manual review. Storage:
results/appeals/<group>.json, atomic writes, append-only audit trail per appeal."""
import json
import threading
import time
import uuid

from . import config, store
from .util import atomic_write, safe_child

_LOCK = threading.Lock()
STATUSES = ("open", "resolved")
DECISIONS = ("confirmed", "changed")


def _path(group):
    return safe_child(config.RESULTS_DIR / "appeals", str(group).lower(), ".json")


def load(group):
    p = _path(group)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _save(group, items):
    p = _path(group)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, json.dumps(items, ensure_ascii=False, indent=1))


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def list_appeals(group, status=None):
    items = load(group)
    if status:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        items = [a for a in items if a["status"] == status]
    return items


def create(group, dialog, criterion_id, comment, author="", run=""):
    """dialog = the result dict (with overrides applied). Raises ValueError on bad input."""
    comment = (comment or "").strip()
    if not comment:
        raise ValueError("comment is required")
    r = next((c for c in dialog["criteria"] if c["criterion_id"] == criterion_id), None)
    if r is None:
        raise ValueError(f"unknown criterion {criterion_id!r}")
    conv = dialog["conversation_id"]
    ov = (dialog.get("human") or {}).get("overrides", {}).get(criterion_id)
    with _LOCK:
        items = load(group)
        if any(a for a in items if a["status"] == "open" and a["conversation_id"] == conv and a["criterion_id"] == criterion_id):
            raise ValueError("an open appeal for this criterion already exists")
        a = {"id": uuid.uuid4().hex[:10], "group": group, "run": run, "conversation_id": conv, "checklist_id": dialog["checklist_id"],
             "criterion_id": criterion_id, "criterion_name": r.get("name", ""), "status": "open",
             "ai_value": r.get("value"), "value_at_appeal": ov["value"] if ov else r.get("value"),
             "ai_justification": r.get("justification", ""), "ai_evidence": r.get("evidence") or [],
             "missing_where": r.get("missing_where", ""), "total_at_appeal": _total(dialog),
             "comment": comment, "author": author, "created_at": _now(), "resolution": None,
             "audit": [{"at": _now(), "who": author, "action": "opened", "comment": comment}]}
        items.append(a)
        _save(group, items)
    return a


def _total(d):
    h = d.get("human") or {}
    return h["total"] if h.get("reviewed") else d.get("total")


def resolve(group, appeal_id, decision, reason, reviewer="", value=None):
    """decision: 'confirm' | 'change' (also accepts 'confirmed'/'changed'). Change -> override via store."""
    decision = {"confirm": "confirmed", "change": "changed"}.get(decision, decision)
    if decision not in DECISIONS:
        raise ValueError("decision must be confirm or change")
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("reason is required")
    with _LOCK:
        items = load(group)
        a = next((x for x in items if x["id"] == appeal_id), None)
        if a is None:
            raise KeyError(appeal_id)
        if a["status"] != "open":
            raise ValueError("appeal is already resolved")
        new_value = a["value_at_appeal"]
        if decision == "changed":
            if value in (None, ""):
                raise ValueError("value is required for change")
            ov = store.save_override(group, a["conversation_id"], a["criterion_id"], value,
                                     f"Апелляция {a['id']}: {reason}", reviewer or "appeal", checklist_id=a["checklist_id"])
            new_value = ov[a["conversation_id"]][a["criterion_id"]]["value"]
            if str(new_value) == str(a["value_at_appeal"]):
                decision = "confirmed"   # "changed" to the same value is a confirmation
        a["status"] = "resolved"
        a["resolution"] = {"decision": decision, "value": new_value, "previous_value": a["value_at_appeal"],
                           "reason": reason, "reviewer": reviewer, "at": _now()}
        a["audit"].append({"at": _now(), "who": reviewer, "action": decision, "reason": reason,
                           "from": a["value_at_appeal"], "to": new_value})
        _save(group, items)
    return a


def delete(group, appeal_id):
    """Remove an appeal record (admin/test cleanup). Does not undo an override."""
    with _LOCK:
        items = load(group)
        rest = [x for x in items if x["id"] != appeal_id]
        if len(rest) == len(items):
            raise KeyError(appeal_id)
        _save(group, rest)

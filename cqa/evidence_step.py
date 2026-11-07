"""Evidence step for the cascade: a cheap "cite, don't decide" call for non-escalated tier-1 verdicts.

Tier 1 (Jev / a local decision model) returns a value without quotes. For verdicts that are likely to be disputed
(NO, NA, scale < max; optionally 08=YES) a small model (default claude-haiku-5-5) is shown the FIXED decision and
asked only to point at the turn(s) that support it. Quotes go through the same substring check as every other
path (``evaluator.validate_evidence``). The value is never changed here.

Settings (env):
  CQA_EVIDENCE_STEP      off | disputed | all     (default ``disputed`` inside the cascade)
  CQA_EVIDENCE_MODEL     model id                 (default claude-haiku-5-5)
  CQA_EVIDENCE_08_YES    1 = treat 08=YES as disputed (default 1; it is escalated by default anyway)
  CQA_EVIDENCE_ESCALATE  1 = when no quote verifies, escalate the criterion to the cascade's tier 2 (default 0)
"""
import hashlib
import json
import os
import time

from . import checklist as ckl, config, evaluator as _ev, prompts

MODES = ("off", "disputed", "all")
CACHE = config.ROOT / ".cache" / "router"

TOOL = {
    "name": "cite_evidence",
    "description": "Cite verbatim transcript quotes that support an already-made decision on one criterion.",
    "input_schema": {"type": "object", "properties": {
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "turn": {"type": "integer"}, "quote": {"type": "string"}}, "required": ["turn", "quote"]},
            "description": "1–3 ДОСЛОВНЫЕ цитаты (подстрока реплики) с номером реплики."},
        "rationale": {"type": "string", "description": "1–2 предложения по-русски: как цитаты подтверждают решение."},
    }, "required": ["evidence", "rationale"]},
}


def mode(default="disputed"):
    m = (os.environ.get("CQA_EVIDENCE_STEP") or default).strip().lower()
    return m if m in MODES else default


def _flag(name, default):
    return (os.environ.get(name) or default).strip() not in ("0", "", "false", "no")


def is_disputed(cr, value):
    """Verdicts that cost the operator points or flag a gross violation."""
    v = str(value)
    if v in ("NO", "NA"):
        return True
    if cr.get("type") == "scale" and v.lstrip("-").isdigit():
        return int(v) < max(int(x) for x in cr["scale"])
    return v == "YES" and cr["id"] == "08" and _flag("CQA_EVIDENCE_08_YES", "1")


def wanted(cr, value, m):
    return m == "all" or (m == "disputed" and is_disputed(cr, value))


def prompt(checklist, cr, value):
    hint = {"NA": "реплику(и), из которых видно, почему критерий неприменим",
            "NO": "реплику(и), где оператор нарушил требование, или реплику, где ожидаемое действие должно было быть, "
                  "но его нет (процитируй её как есть)"}.get(str(value), "реплику(и), на которых основано это решение")
    return (prompts.criterion_block(checklist, cr).rsplit("\n\nОцени ТОЛЬКО", 1)[0]
            + f"\n\nРЕШЕНИЕ УЖЕ ПРИНЯТО: value = {value}. Не пересматривай его и не оценивай заново.\n"
            f"Твоя задача — только найти {hint}. Цитаты — дословные фрагменты (5–25 слов) одной реплики, "
            "с её номером [N]. Если подтверждающей реплики нет — верни пустой evidence. Затем вызови cite_evidence.")


def cite(llm, rec, tx, checklist, cr, value, use_cache=True):
    key = hashlib.sha1(f"evstep:{llm.model}:{ckl.criterion_hash(checklist, cr)}:{rec['hash']}:{value}".encode()).hexdigest()[:20]
    p = CACHE / f"evstep_{key}.json"
    if use_cache and p.exists():
        return {**json.loads(p.read_text()), "cached": True}
    t = time.time()
    try:
        out = llm.call(prompts.SYSTEM, tx, prompt(checklist, cr, value), TOOL, 0.0)
        valid, invalid = _ev.validate_evidence(rec, out.get("evidence"))
        o = {"evidence": valid, "rejected_quotes": invalid, "rationale": str(out.get("rationale") or "")[:600]}
    except Exception as e:   # noqa: BLE001
        o = {"error": str(e)[:300], "evidence": [], "rejected_quotes": []}
    o["elapsed_s"] = round(time.time() - t, 2)
    if use_cache and not o.get("error"):
        CACHE.mkdir(parents=True, exist_ok=True)
        _ev.store_atomic(p, json.dumps(o, ensure_ascii=False))
    return o


def apply(res, llm, rec, tx, checklist, cr, m=None, use_cache=True):
    """Attach verified evidence to a finalised tier-1 result ``res`` (in place). Returns True if the caller
    should escalate (no verified quote and CQA_EVIDENCE_ESCALATE=1). Never changes ``res['value']``."""
    m = m or mode()
    r = res.setdefault("routing", {})
    if m == "off" or res.get("evidence") or not wanted(cr, res.get("value"), m):
        r["evidence_step"] = {"status": "skipped"}
        return False
    o = cite(llm, rec, tx, checklist, cr, res["value"], use_cache)
    ok = bool(o["evidence"])
    r["evidence_step"] = {"status": "ok" if ok else "evidence_missing", "model": llm.model,
                          "s": o.get("elapsed_s"), "cached": bool(o.get("cached")), "error": o.get("error"),
                          "rejected": len(o.get("rejected_quotes") or [])}
    if ok:
        res["evidence"] = o["evidence"]
        res["evidence_rationale"] = o.get("rationale", "")
        if o.get("rationale"):
            res["justification"] = f"{o['rationale']} [{res.get('justification', '')}]".strip()
    else:
        res["evidence_missing"] = True
    return not ok and _flag("CQA_EVIDENCE_ESCALATE", "0")

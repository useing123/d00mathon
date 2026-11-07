"""TypeSafe Jev (``typesafe/jev-1.14``) decision-model adapter via the OpenRouter Decisions API.

One HTTP call per dialog: every checklist criterion becomes one Jev question
(binary → ``choice`` yes/no[/na]; scale → ``score`` over the rubric steps in ascending order,
plus a ``noul`` applicability question when NA is allowed; gross → ``choice`` yes/no).
Jev returns probabilities but no evidence quotes, so results carry ``evidence=[]`` and are
meant as a decision/routing layer or cheap pre-pass, not a standalone evidence-backed scorer.
See docs/JEV.md.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import data, scoring

# Endpoint is configurable so the SAME question builder feeds Jev (OpenRouter) and Jev-compatible self-hosted
# decision servers (LitJev, qwen-jev-like / openjev wrappers in scripts/decision_server.py): docs/KZ_MODELS.md.
MODEL = os.environ.get("CQA_JEV_MODEL", "typesafe/jev-1.14")
URL = os.environ.get("CQA_JEV_BASE_URL", "https://openrouter.ai/api/alpha/decisions")
KEY_URL = "https://openrouter.ai/api/v1/key"
MAX_STATE_CHARS = 12000
RULES_CHARS = int(os.environ.get("CQA_JEV_RULES_CHARS", "0"))   # 0 = full rules; a 700-char cut lost ~9 pts

ROLE = ("Задача: оценить работу ЧЕЛОВЕКА-оператора (OPERATOR) службы поддержки Freedom Broker в этом чате по чек-листу. "
        "Реплики BOT и SYSTEM не оцениваются: приветствие/прощание бота не засчитывается оператору. "
        "Блок ФАКТЫ вычислен кодом (номера реплик, паузы, язык) — доверяй ему.")


def api_key():
    k = os.environ.get("CQA_JEV_API_KEY") or (None if "openrouter.ai" in URL else "none")
    if k:
        return k.strip()
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k.strip()
    env = Path(os.environ.get("CQA_JEV_ENV", "/root/gtm_work/.env"))
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("OPENROUTER_API_KEY not set")


def _post(url, body=None, key=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key or api_key()}",
                                          "Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def limit_remaining(key=None):
    return _post(KEY_URL, key=key)["data"].get("limit_remaining")


def _scale_levels(cr):
    """Ascending (value, description) pairs parsed from rubric lines like '20 — ...'."""
    desc = {}
    for line in (cr.get("rubric") or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[—-]\s*(.+)", line)
        if m:
            desc[int(m.group(1))] = m.group(2).strip()
    return [(v, desc.get(v, str(v))) for v in sorted(int(x) for x in cr["scale"])]


def _opt_desc(cr, label):
    for line in (cr.get("rubric") or "").splitlines():
        if line.strip().startswith(label + " "):
            return line.strip()[len(label):].strip(" —-")
    return label


def _instr(cr):
    rules = " ".join((cr.get("rules") or "").split())
    if RULES_CHARS and len(rules) > RULES_CHARS:
        rules = rules[:RULES_CHARS].rsplit(" ", 1)[0] + " …"
    return f"Критерий {cr['id']} «{cr['name']}». {' '.join(cr.get('rubric', '').split())} Правила: {rules}"


def build_questions(checklist):
    qs = {}
    for cr in checklist["criteria"]:
        cid = cr["id"]
        if cr.get("type") == "scale":
            qs[f"c{cid}"] = {"type": "score", "instructions": _instr(cr),
                             "criteria": [f"{v} — {d}" for v, d in _scale_levels(cr)]}
            if cr.get("allow_na"):
                qs[f"c{cid}_na"] = {"type": "noul", "instructions": f"Критерий {cid} «{cr['name']}»: неприменим (NA)? "
                                    + _opt_desc(cr, "NA"),
                                    "criteria": {"true": "NA — неприменим", "false": "Применим, нужно ставить оценку"}}
        else:
            opts = {"yes": _opt_desc(cr, "YES"), "no": _opt_desc(cr, "NO")}
            if cr.get("allow_na"):
                opts["na"] = _opt_desc(cr, "NA")
            qs[f"c{cid}"] = {"type": "choice", "instructions": _instr(cr), "criteria": opts}
    return qs


def build_state(rec, checklist=None):
    from . import checklist as ckl
    lw = ckl.long_wait_seconds(checklist) if checklist else None
    facts = data.format_facts(data.compute_facts(rec, lw))
    tr = data.format_transcript(rec)
    if len(tr) > MAX_STATE_CHARS:   # keep start (greeting/request) and end (closing)
        h = MAX_STATE_CHARS * 2 // 3
        tr = tr[:h] + "\n[… середина диалога опущена …]\n" + tr[-(MAX_STATE_CHARS - h):]
    return f"{ROLE}\n\nФАКТЫ:\n{facts}\n\nДИАЛОГ (номер реплики, время, роль):\n{tr}"


def parse_answers(checklist, answers):
    """Jev answers → our criterion dicts + values {id: YES/NO/NA/int}."""
    out, values = [], {}
    for cr in checklist["criteria"]:
        cid = cr["id"]
        a = answers.get(f"c{cid}") or {}
        val, conf = None, None
        if cr.get("type") == "scale":
            levels = _scale_levels(cr)
            na = answers.get(f"c{cid}_na")
            if na is not None and float(na.get("noul", 0)) > 0.5:
                val, conf = "NA", float(na["noul"])
            elif "score" in a:   # Jev's "score" is an expectation (e.g. 2.4); take the most probable level
                pr = a.get("probabilities") or {}
                i = int(max(pr, key=lambda k: pr[k])) if pr else int(round(float(a["score"])))
                i = max(0, min(i, len(levels) - 1))
                val = levels[i][0]
                conf = float(pr.get(str(i), 0)) or None
        elif "choice" in a:
            val = a["choice"].upper()
            conf = (a.get("probabilities") or {}).get(a["choice"], a.get("confidence"))
        values[cid] = val
        out.append({"criterion_id": cid, "name": cr["name"], "type": cr.get("type", "binary"), "value": val,
                    "is_na": val == "NA", "confidence": conf, "evidence": [],
                    "justification": f"decision model (Jev) p={conf:.2f}" if conf is not None else "decision model (Jev)",
                    "jev_raw": a, "needs_review": conf is None or conf < 0.7})
    return out, values


def evaluate(rec, checklist, key=None, retries=2):
    body = {"model": MODEL, "state": build_state(rec, checklist), "questions": build_questions(checklist)}
    t0, err = time.time(), None
    for attempt in range(retries + 1):
        try:
            resp = _post(URL, body, key=key)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            err = getattr(e, "read", lambda: b"")()[:300] or str(e)
            time.sleep(2 * (attempt + 1))
    else:
        return {"conversation_id": rec["conversation_id"], "status": "incomplete", "criteria": [],
                "error": str(err), "latency_s": round(time.time() - t0, 2)}
    answers = resp.get("answers") or resp.get("decisions") or resp
    crit, values = parse_answers(checklist, answers)
    t = scoring.compute_total(checklist["criteria"], values, checklist["assumptions"].get("rounding", "half_up"),
                              checklist["assumptions"].get("decimals", 2))
    return {"conversation_id": rec["conversation_id"], "checklist_id": checklist["id"], "criteria": crit,
            "total": t["total"], "gross_error": t["gross_error"],
            "status": "ok" if all(v is not None for v in values.values()) else "incomplete",
            "latency_s": round(time.time() - t0, 2), "usage": resp.get("usage"), "provider": "jev", "model": MODEL,
            "endpoint": URL}

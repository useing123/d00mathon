"""Loading, fetching and normalising dialog records."""
import hashlib
import json
import os
import re
from datetime import datetime

import httpx

from . import config

HOLD_START = ("инициирована консультация", "клиент находится на удержании", "consultation initiated", "is on hold",
              "placed on hold", "client on hold", "customer on hold",
              "клиент поставлен на удержание", "кеңес басталды", "күту режимінде")
HOLD_END = ("клиент больше не находится на удержании", "no longer on hold", "removed from hold",
            "снят с удержания", "күту режимінен шығарылды")


def _records(obj):
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], dict) and "records" in obj["data"]:
            return obj["data"]["records"]
        if "records" in obj:
            return obj["records"]
        if "transcript" in obj:
            return [obj]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unrecognised JSON: expected data.records[], records[], list or a single record")


def load_json(obj):
    return [normalize(r, i) for i, r in enumerate(_records(obj))]


def load_file(path):
    with open(path, encoding="utf-8") as f:
        return load_json(json.load(f))


def fetch_group(group_id, save=True):
    from .util import safe_child
    safe_child(config.DATA_DIR, group_id, ".json")   # validates the id before it reaches a URL or a path
    url = config.get("CQA_API_URL")
    if not url:
        raise RuntimeError("CQA_API_URL is not set (put it in .env)")
    r = httpx.get(url, params={"group_id": group_id}, timeout=180) if "?" not in url else \
        httpx.get(url + "&group_id=" + group_id, timeout=180)
    r.raise_for_status()
    obj = r.json()
    if save:
        from .util import atomic_write
        atomic_write(safe_child(config.DATA_DIR, group_id, ".json"), json.dumps(obj, ensure_ascii=False))
    return load_json(obj)


def load_group(group_id, refresh=False):
    from .util import safe_child
    p = safe_child(config.DATA_DIR, group_id, ".json")
    if p.exists() and not refresh:
        return load_file(p)
    return fetch_group(group_id)


def parse_paste(text, channel="agent chat", language="RU"):
    """Paste format: one turn per line, 'speaker: text' or '[mm:ss] speaker: text'."""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ts = ""
        if line.startswith("[") and "]" in line:
            ts, line = line[1:line.index("]")], line[line.index("]") + 1:].strip()
        sp, _, tx = line.partition(":")
        if normalize_speaker(sp) is None or len(sp) > 40:
            sp, tx = "client", line
        else:
            sp = sp.strip()
        turns.append({"timestamp": ts, "speaker": sp, "text": tx.strip()})
    return normalize({"conversation_id": "pasted-" + hashlib.md5(text.encode()).hexdigest()[:8],
                      "channel": channel, "language": language, "transcript": turns}, 0)


SPEAKER_ALIASES = {
    "operator": ("operator", "agent", "manager", "specialist", "support", "оператор", "агент", "менеджер",
                 "специалист", "консультант", "маман"),
    "client": ("client", "customer", "user", "caller", "клиент", "пользователь", "абонент", "клиент(ка)"),
    "bot": ("bot", "assistant", "ivr", "aim", "virtual assistant", "chatbot", "бот", "ассистент"),
    "system": ("system", "система", "sys", "event"),
}
_ALIAS = {a: canon for canon, al in SPEAKER_ALIASES.items() for a in al}


def normalize_speaker(sp):
    """Map a raw speaker label to operator|client|bot|system, or None if unknown."""
    s = str(sp or "").strip().lower().replace("_", " ")
    if s in _ALIAS:
        return _ALIAS[s]
    head = s.split()[0] if s.split() else ""
    return _ALIAS.get(head)


def parse_timestamp(ts):
    """Return ("offset", seconds) for mm:ss / hh:mm:ss / plain seconds, ("abs", datetime) for ISO, or (None, None)."""
    ts = str(ts or "").strip()
    if not ts:
        return None, None
    if re.fullmatch(r"\d+(?:\.\d+)?", ts):
        return "offset", float(ts)
    if re.fullmatch(r"\d{1,3}:\d{1,2}(?::\d{1,2})?(?:\.\d+)?", ts):
        return "offset", _mmss(ts)
    dt = _parse_iso(ts)
    if dt is not None:
        return "abs", dt
    return None, None


def _parse_iso(ts):
    try:
        s = str(ts).strip().replace("Z", "+00:00").replace(" ", "T", 1)
        m = re.match(r"^(.*?\d)\.(\d+)(.*)$", s)
        if m:  # trim >6-digit fractions (e.g. .1234567) that datetime cannot parse
            s = f"{m.group(1)}.{m.group(2)[:6]}{m.group(3)}"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _mmss(ts):
    try:
        parts = [float(x) for x in ts.split(":")]
        sec = 0
        for p in parts:
            sec = sec * 60 + p
        return float(sec)
    except Exception:
        return None


# ---- turn numbering: the ONE conversion point. Every user-facing turn number (prompt labels, evidence,
# facts, UI, API, exports) is 1-based; Python list positions are 0-based. Use turn_pos()/turn_at() only.
TURN_BASE = 1


def turn_pos(n):
    """User-facing turn number -> list position (or -1 if not an int)."""
    try:
        return int(n) - TURN_BASE
    except (TypeError, ValueError):
        return -1


def turn_at(rec, n):
    """Turn dict for a user-facing turn number, or None if out of range."""
    i = turn_pos(n)
    tr = rec.get("transcript") or []
    return tr[i] if 0 <= i < len(tr) else None


def normalize(rec, i):
    rec = dict(rec)
    rec.setdefault("conversation_id", f"dialog-{i+1}")
    ch = (rec.get("channel") or "agent chat").lower()
    rec["kind"] = "call" if ("call" in ch or "voice" in ch or "phone" in ch) else "chat"
    turns = []
    first = None
    internal = False
    warnings = []
    unknown = {}
    for idx, t in enumerate(rec.get("transcript") or []):
        t = dict(t)
        t["index"] = idx + TURN_BASE
        raw_sp = t.get("speaker")
        sp = normalize_speaker(raw_sp)
        if sp is None:
            unknown[str(raw_sp)] = unknown.get(str(raw_sp), 0) + 1
            sp = "unknown"
        if raw_sp != sp:
            t["speaker_raw"] = raw_sp
        t["speaker"] = sp
        kind, val = parse_timestamp(t.get("timestamp"))
        off = None
        if kind == "offset":
            off = val
        elif kind == "abs":
            if first is None:
                first = val
            try:
                off = (val - first).total_seconds()
            except TypeError:  # naive vs aware mix
                off = (val.replace(tzinfo=None) - first.replace(tzinfo=None)).total_seconds()
        t["offset"] = off
        low = (t.get("text") or "").lower()
        if sp == "system" and any(k in low for k in HOLD_END):   # check END first: "no longer on hold" ⊃ "on hold"
            internal = False
            t["hold_event"] = "end"
        elif sp == "system" and any(k in low for k in HOLD_START):
            internal = True
            t["hold_event"] = "start"
        t["internal"] = bool(internal and sp == "operator")
        turns.append(t)
    if unknown:
        warnings.append("unknown speaker roles (kept as UNKNOWN, not credited to operator): " +
                        ", ".join(f"{k!r}×{v}" for k, v in unknown.items()))
    rec["transcript"] = turns
    rec["warnings"] = warnings
    rec["hash"] = dialog_hash(rec)
    return rec


def dialog_hash(rec):
    core = [(t.get("speaker"), t.get("text"), str(t.get("timestamp") or "")) for t in rec.get("transcript") or []]
    return hashlib.sha256(json.dumps([rec.get("channel"), rec.get("language"), core], ensure_ascii=False).encode()).hexdigest()[:16]


def fmt_offset(sec):
    if sec is None:
        return "?"
    sec = int(sec)
    return f"{sec//60:02d}:{sec%60:02d}"


def format_transcript(rec, mark_internal=True):
    lines = []
    for t in rec["transcript"]:
        sp = t.get("speaker", "?").upper()
        if mark_internal and t.get("internal"):
            sp += " [ВНУТРЕННИЕ ПЕРЕГОВОРЫ НА УДЕРЖАНИИ — клиент не слышит]"
        lines.append(f"[{t['index']}] ({fmt_offset(t.get('offset'))}) {sp}: {t.get('text','')}")
    return "\n".join(lines)


KZ_LETTERS = set("әғқңөұүһі")
# anchored wait-warning intents (not bare "момент"/"минут": «в данный момент», «в течение 5 минут» are not warnings)
WAIT_RE = re.compile(
    r"(?:одну|пару|несколько)\s+(?:минут[уы]?|секунд[уы]?)|минут(?:оч)?ку|секунд(?:оч)?ку|\bподожд[иеё]|ожидайте,?\s+(?:я\s+)?(?:уточн|провер)"
    r"|(?:сейчас|я)\s+(?:уточню|проверю|посмотрю|узнаю|запрошу)|(?:уточню|проверю)\s+(?:информацию|у коллег|в системе|и вернусь)"
    r"|мне\s+нужно\s+(?:время|уточнить|проверить)|оставайтесь\s+на\s+линии|не\s+кладите\s+трубку"
    r"|күте\s+тұрыңыз|күтіңіз|бір\s+сәт|қазір\s+(?:қараймын|қарайын|тексеремін|нақтылаймын)"
    r"|let\s+me\s+(?:check|look|verify|find)|(?:please\s+)?(?:hold\s+on|wait\s+a\s+(?:moment|minute|second))|one\s+(?:moment|sec|second|minute)"
    r"|i(?:'ll| will)\s+(?:check|look into|verify)", re.I)
ACK_ONLY = ("спасибо", "благодарю", "понял", "поняла", "понятно", "ок", "ok", "okay", "хорошо", "ясно", "ага",
            "рақмет", "рахмет", "рахмет сізге", "түсіндім", "жарайды", "thanks", "thank you", "got it", "great")
AUTH_RE = re.compile(
    r"(?<![а-яёa-z])(?:клиент|пользователь|вы)\s+(?:успешно\s+)?(?:авторизован[аоы]?|верифицирован[аоы]?|идентифицирован[аоы]?)"
    r"|(?:client|customer|user)\s+(?:is\s+|has\s+been\s+)?(?:successfully\s+)?(?:authenticated|verified)"
    r"|клиент\s+сәйкестендірілді", re.I)
AUTH_NEG_RE = re.compile(r"\bне\s+(?:авторизован|верифицирован|идентифицирован|прош[её]л)|not\s+(?:authenticated|verified)"
                         r"|для\s+авторизации|to\s+authenticate|failed", re.I)


def says_authenticated(text):
    """Affirmative, non-negated authentication statement (e.g. «Клиент авторизован»); negated/conditional → False."""
    t = text or ""
    return bool(AUTH_RE.search(t)) and not AUTH_NEG_RE.search(t)
KZ_WORDS = ("сәлем", "салем", "салеметсиз", "саламатсыз", "комектес", "акша", "картамен", "шотты",
             "рахмет", "рақмет", "жоқ", "иә", "калай", "қалай", "керек", "бар ма", "қайда", "кайда")
END_MARKERS = ("завершил", "прерван", "отключ", "аяқтады", "ended", "left the chat", "hung up", "disconnected", "closed")
TRIVIAL = {"оператор", "operator", "agent", "агент", "менеджер", "маман", "человек", "1", "2", "0"}


def _msg_language(tx):
    low = (tx or "").lower()
    if low.strip(" .!?") in TRIVIAL:
        return None
    kz_letters = sum(ch in KZ_LETTERS for ch in low)
    kz_words = sum(w in low for w in KZ_WORDS)
    ru = sum("а" <= ch <= "я" or ch == "ё" for ch in low)
    en = sum("a" <= ch <= "z" for ch in low)
    cyr = ru + kz_letters
    if kz_words or (kz_letters >= 2 and kz_letters >= 0.03 * max(cyr, 1)):   # a Kazakh name in Russian text is not enough
        return "KZ"
    if cyr == 0 and en == 0:
        return None
    return "RU" if cyr >= en else "EN"


def detect_language(texts):
    """Per-message script heuristic, majority by characters; None if no signal. Kazakh-specific letters must be a
    noticeable share of the Cyrillic text (or a common Kazakh word must occur)."""
    weights = {}
    for tx in texts:
        lang = _msg_language(tx)
        if lang:
            weights[lang] = weights.get(lang, 0) + len(tx or "")
    if not weights:
        return None
    return max(sorted(weights), key=lambda k: weights[k])


GROSS_SIGNALS = re.compile(
    r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b|(?:^|\s)@[a-z0-9_]{4,}|telegram|телеграм|whatsapp|ватсап|вотсап"
    r"|на\s+(?:мою|мой)\s+(?:карту|счёт|счет|номер)|to\s+my\s+(?:personal\s+)?(?:card|account)|лично\s+(?:за|мне)"
    r"|туп(?:ой|ая|ые)|идиот|дура[кч]|stupid|idiot|ақымақ|түсінбейтін\s+адам", re.I)


def default_long_wait_seconds():
    """Wait threshold from the default chat checklist YAML (`assumptions.long_wait_seconds`) — the single source."""
    from . import checklist
    return checklist.long_wait_seconds(checklist.load(checklist.pick_for_channel("chat")))


def compute_facts(rec, long_wait_s=None):
    """Deterministic facts the LLM should not have to derive (offsets arithmetic, roles, language, ending).
    `long_wait_s` comes from the checklist (`assumptions.long_wait_seconds`); None → the default chat checklist."""
    if long_wait_s is None:
        long_wait_s = default_long_wait_seconds()
    tr = rec["transcript"]
    ops = [t for t in tr if t["speaker"] == "operator" and not t.get("internal")]
    first_op = ops[0]["index"] if ops else None
    first_block = []   # consecutive operator messages before the next client message = one "first reply"
    if first_op is not None:
        for t in tr[turn_pos(first_op):]:
            if t["speaker"] == "client":
                break
            if t["speaker"] == "operator" and not t.get("internal"):
                first_block.append(t)
    client_txt = [t.get("text", "") for t in tr if t["speaker"] == "client"]
    per_msg = {detect_language([x]) for x in client_txt} - {None}
    facts = {"first_operator_turn": first_op, "first_operator_turns": [t["index"] for t in first_block],
             "client_language_guess": detect_language(client_txt),
             "client_mixed_languages": sorted(per_msg) if len(per_msg) > 1 else [],
             "language_metadata": rec.get("language"),
             "system_says_client_authenticated": any(
                 t["speaker"] in ("system", "bot") and says_authenticated(t.get("text")) for t in tr)}
    if first_op is not None:
        facts["first_operator_reply_language_guess"] = detect_language([t.get("text", "") for t in first_block])

    def is_ack(t):
        return (t.get("text") or "").lower().strip(" .!?,)(") in ACK_ONLY

    # longest wait for the operator after the operator's first reply (queue time before it is not scored);
    # a gap that starts from a pure acknowledgement ("спасибо") is not a wait for an answer
    best = None
    long_waits = []
    if first_op is not None:
        pending = None              # first unanswered, non-acknowledgement client turn after an operator turn
        since_client = []           # operator turns since the last client turn (wait warning may be in any of them)
        last_op = None
        ack_seen = False
        for t in tr:
            if t["index"] <= first_op:
                if t["speaker"] == "operator":
                    last_op = t
                    since_client.append(t)
                continue
            if t["speaker"] == "client":
                if pending is None:
                    since_client = []      # a new request: earlier warnings no longer apply
                    if not is_ack(t):
                        pending = t
                # a follow-up ping ("жду", "алло?") while already waiting keeps the operator's earlier warning
                ack_seen = ack_seen or is_ack(t)
            elif t["speaker"] == "operator" and not t.get("internal"):
                start = pending or last_op
                if start is not None and start.get("offset") is not None and t.get("offset") is not None and \
                        not (pending is None and ack_seen):
                    gap = t["offset"] - start["offset"]
                    recent = since_client or ([last_op] if last_op else [])
                    w = {"seconds": round(gap), "from_turn": start["index"], "to_turn": t["index"],
                         "operator_warned_before": bool(any(WAIT_RE.search(o.get("text") or "") for o in recent)),
                         "kind": "reply_to_client" if pending is not None else "between_operator_messages"}
                    if gap > long_wait_s:
                        long_waits.append(w)
                    if best is None or gap > best["seconds"]:
                        best = w
                pending, last_op, ack_seen = None, t, False
                since_client.append(t)
    facts["longest_wait_after_first_reply"] = best
    facts["long_waits"] = long_waits
    facts["long_wait_seconds"] = long_wait_s
    facts["gross_signal_turns"] = [t["index"] for t in tr if t["speaker"] == "operator" and not t.get("internal")
                                   and GROSS_SIGNALS.search(t.get("text") or "")]
    holds = []
    for t in tr:
        if t.get("hold_event") == "start":
            prev_op = next((p for p in reversed(tr[:turn_pos(t["index"])]) if p["speaker"] == "operator"), None)
            holds.append({"start_turn": t["index"], "operator_warned_before": bool(
                prev_op and WAIT_RE.search(prev_op.get("text") or ""))})
        elif t.get("hold_event") == "end" and holds and "end_turn" not in holds[-1]:
            holds[-1]["end_turn"] = t["index"]
            s0, s1 = tr[turn_pos(holds[-1]["start_turn"])].get("offset"), t.get("offset")
            if s0 is not None and s1 is not None:
                holds[-1]["seconds"] = round(s1 - s0)
    facts["holds"] = holds
    non_sys = [t for t in tr if t["speaker"] in ("operator", "client", "bot")]
    end_sys = [t["index"] for t in tr if t["speaker"] == "system" and any(m in (t.get("text") or "").lower() for m in END_MARKERS)]
    facts["ending"] = {"last_speaker": non_sys[-1]["speaker"] if non_sys else None,
                       "last_operator_turn": ops[-1]["index"] if ops else None,
                       "system_end_event_turn": end_sys[-1] if end_sys else None}
    if os.environ.get("CQA_KZ_SIGNALS") == "1":   # preset hybrid; off → byte-identical FACTS (docs/HYBRID_KZ.md)
        from . import rules_kz
        facts["kz_signals"] = rules_kz.signals(rec)
    return facts


def format_facts(f):
    lines = []
    if f["first_operator_turn"] is not None:
        tb = f.get("first_operator_turns") or [f["first_operator_turn"]]
        lines.append(f"- Первая реплика человека-оператора: №{f['first_operator_turn']}"
                     + (f" (вместе с подряд идущими №{', №'.join(map(str, tb[1:]))} до ответа клиента)" if len(tb) > 1 else ""))
    else:
        lines.append("- Человек-оператор в диалоге не отвечал.")
    if f.get("client_mixed_languages"):
        lines.append(f"- Клиент смешивает языки: {', '.join(f['client_mixed_languages'])}.")
    if f.get("system_says_client_authenticated"):
        lines.append("- Система/бот сообщает, что клиент авторизован/верифицирован.")
    lines.append(f"- Язык клиента (эвристика по алфавиту): {f['client_language_guess'] or 'не определён'}; "
                 f"язык в метаданных: {f['language_metadata']}"
                 + (f"; язык первой реплики оператора: {f.get('first_operator_reply_language_guess') or 'не определён'}"
                    if f["first_operator_turn"] is not None else ""))
    w = f["longest_wait_after_first_reply"]
    if w:
        kind = ("ответ на сообщение клиента" if w.get("kind") == "reply_to_client"
                else "пауза между сообщениями оператора без реплики клиента")
        lines.append(f"- Самое долгое ожидание ответа оператора после его первой реплики: {w['seconds']} с "
                     f"(реплики №{w['from_turn']} → №{w['to_turn']}; {kind}); оператор предупреждал об ожидании перед этим: "
                     f"{'да' if w['operator_warned_before'] else 'нет'}.")
    lw = f.get("long_waits") or []
    unwarned = [x for x in lw if not x["operator_warned_before"]]
    if lw:
        def d(x):
            k = "ответ клиенту" if x["kind"] == "reply_to_client" else "пауза между сообщениями оператора"
            return f"№{x['from_turn']}→№{x['to_turn']} {x['seconds']} с ({k})"
        lines.append(f"- Ожиданий дольше {f['long_wait_seconds']} с после первой реплики: {len(lw)}; "
                     f"из них БЕЗ предупреждения: {len(unwarned)}"
                     + (": " + "; ".join(d(x) for x in unwarned) if unwarned else "") + ".")
    else:
        lines.append(f"- Ожиданий дольше {f['long_wait_seconds']} с после первой реплики оператора нет.")
    if f.get("gross_signal_turns"):
        lines.append("- Реплики оператора с признаками возможной грубой ошибки (номер карты, мессенджер, «на мою карту», "
                     f"оскорбительная лексика) — проверь по тексту: №{', №'.join(map(str, f['gross_signal_turns']))}.")
    for h in f["holds"]:
        lines.append(f"- Удержание (системное событие) с реплики №{h['start_turn']}"
                     + (f" до №{h['end_turn']}" if "end_turn" in h else "")
                     + (f", {h['seconds']} с" if "seconds" in h else "")
                     + f"; оператор предупредил перед удержанием: {'да' if h['operator_warned_before'] else 'нет'}.")
    e = f["ending"]
    lines.append(f"- Последний говорящий (без системы): {e['last_speaker']}; последняя реплика оператора: №{e['last_operator_turn']}; "
                 "системное событие завершения: "
                 f"{('№' + str(e['system_end_event_turn'])) if e['system_end_event_turn'] is not None else 'нет'}.")
    if f.get("kz_signals"):
        from . import rules_kz
        lines.append(rules_kz.format_signals(f["kz_signals"]))
    return "\n".join(lines)


SPEECH_WPS = 2.5   # assumed speaking rate (words/s) used only to estimate whether a turn was cut into (no audio)


def call_metrics(rec):
    """Informational, audio-free call metrics from turn start times (NOT scored). Turn durations are unknown,
    so a turn's length is approximated by the time until the next turn starts."""
    tr = [t for t in rec["transcript"] if t["speaker"] in ("operator", "client") and not t.get("internal")
          and t.get("offset") is not None]
    if rec.get("kind") != "call" or len(tr) < 2:
        return None
    op_words = op_time = 0.0
    changes = interrupts = 0
    latencies = []
    for a, b in zip(tr, tr[1:], strict=False):
        gap = max(0.0, b["offset"] - a["offset"])
        words = len((a.get("text") or "").split())
        if a["speaker"] == "operator" and gap > 0:
            op_words += words
            op_time += gap
        if a["speaker"] != b["speaker"]:
            changes += 1
            if gap < words / SPEECH_WPS:          # next speaker started before this turn could have been said
                interrupts += 1
            if a["speaker"] == "client" and b["speaker"] == "operator":
                latencies.append(gap)
    lat = sorted(latencies)

    def q(p):
        return round(lat[min(len(lat) - 1, int(p * (len(lat) - 1) + 0.5))], 1) if lat else None
    return {"operator_wpm": round(op_words / op_time * 60, 1) if op_time else None,
            "speaker_changes": changes,
            "overlap_rate": round(interrupts / changes, 3) if changes else None,
            "operator_response_s": {"median": q(0.5), "p90": q(0.9), "max": round(lat[-1], 1) if lat else None,
                                    "n": len(lat)},
            "note": "informational, estimated from turn start times only; not scored"}

"""Loading, fetching and normalising dialog records."""
import hashlib
import json
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
    url = config.get("CQA_API_URL")
    if not url:
        raise RuntimeError("CQA_API_URL is not set (put it in .env)")
    r = httpx.get(url + ("&" if "?" in url else "?") + "group_id=" + group_id, timeout=180)
    r.raise_for_status()
    obj = r.json()
    if save:
        config.DATA_DIR.mkdir(exist_ok=True)
        (config.DATA_DIR / f"{group_id}.json").write_text(json.dumps(obj, ensure_ascii=False))
    return load_json(obj)


def load_group(group_id, refresh=False):
    p = config.DATA_DIR / f"{group_id}.json"
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
        t["index"] = idx
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
WAIT_PHRASES = ("минут", "подожд", "ожидайт", "уточню", "проверю", "секунд", "момент", "одну минуту", "оставайтесь на линии",
                "сейчас посмотрю", "сейчас проверю", "запрошу", "уточняю", "проверяю",
                "күте тұр", "күтіңіз", "тексеріп", "нақтылап", "бір сәт", "қазір қарайын",
                "wait", "moment", "hold on", "let me check", "let me look", "checking", "one sec")
ACK_ONLY = ("спасибо", "благодарю", "понял", "поняла", "понятно", "ок", "ok", "okay", "хорошо", "ясно", "ага",
            "рақмет", "рахмет", "рахмет сізге", "түсіндім", "жарайды", "thanks", "thank you", "got it", "great")
AUTH_MARKERS = ("авторизован", "верифицирован", "идентифицирован", "authenticated", "verified client", "client verified",
                "сәйкестендірілді", "авторизацияланған")
KZ_WORDS = ("сәлем", "салем", "рахмет", "рақмет", "жоқ", "иә", "калай", "қалай", "керек", "бар ма", "қайда", "кайда")
END_MARKERS = ("завершил", "прерван", "отключ", "аяқтады", "ended", "left the chat", "hung up", "disconnected", "closed")
TRIVIAL = {"оператор", "operator", "agent", "агент", "менеджер", "маман", "человек", "1", "2", "0"}


def detect_language(texts):
    """Script heuristic: Kazakh-specific letters → KZ, other Cyrillic → RU, Latin → EN; None if no signal."""
    kz = ru = en = 0
    for tx in texts:
        low = tx.lower()
        if low.strip(" .!?") in TRIVIAL:
            continue
        kz += sum(ch in KZ_LETTERS for ch in low) + 2 * sum(w in low for w in KZ_WORDS)
        ru += sum("а" <= ch <= "я" or ch == "ё" for ch in low)
        en += sum("a" <= ch <= "z" for ch in low)
    if kz >= 2:
        return "KZ"
    if ru == 0 and en == 0:
        return None
    return "RU" if ru >= en else "EN"


def compute_facts(rec):
    """Deterministic facts the LLM should not have to derive (offsets arithmetic, roles, language, ending)."""
    tr = rec["transcript"]
    ops = [t for t in tr if t["speaker"] == "operator" and not t.get("internal")]
    first_op = ops[0]["index"] if ops else None
    first_block = []   # consecutive operator messages before the next client message = one "first reply"
    if first_op is not None:
        for t in tr[first_op:]:
            if t["speaker"] == "client":
                break
            if t["speaker"] == "operator" and not t.get("internal"):
                first_block.append(t)
    facts = {"first_operator_turn": first_op, "first_operator_turns": [t["index"] for t in first_block],
             "client_language_guess": detect_language([t.get("text", "") for t in tr if t["speaker"] == "client"]),
             "language_metadata": rec.get("language"),
             "system_says_client_authenticated": any(
                 t["speaker"] in ("system", "bot") and any(m in (t.get("text") or "").lower() for m in AUTH_MARKERS) for t in tr)}
    if first_op is not None:
        facts["first_operator_reply_language_guess"] = detect_language([t.get("text", "") for t in first_block])

    def is_ack(t):
        return (t.get("text") or "").lower().strip(" .!?,)(") in ACK_ONLY

    # longest wait for the operator after the operator's first reply (queue time before it is not scored);
    # a gap that starts from a pure acknowledgement ("спасибо") is not a wait for an answer
    best = None
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
                if pending is None and not is_ack(t):
                    pending = t
                ack_seen = ack_seen or is_ack(t)
                since_client = []
            elif t["speaker"] == "operator" and not t.get("internal"):
                start = pending or last_op
                if start is not None and start.get("offset") is not None and t.get("offset") is not None and \
                        not (pending is None and ack_seen):
                    gap = t["offset"] - start["offset"]
                    if best is None or gap > best["seconds"]:
                        recent = since_client or ([last_op] if last_op else [])
                        warned = any(w in (o.get("text") or "").lower() for o in recent for w in WAIT_PHRASES)
                        best = {"seconds": round(gap), "from_turn": start["index"], "to_turn": t["index"],
                                "operator_warned_before": bool(warned)}
                pending, last_op, ack_seen = None, t, False
                since_client.append(t)
    facts["longest_wait_after_first_reply"] = best
    holds = []
    for t in tr:
        if t.get("hold_event") == "start":
            prev_op = next((p for p in reversed(tr[:t["index"]]) if p["speaker"] == "operator"), None)
            holds.append({"start_turn": t["index"], "operator_warned_before": bool(
                prev_op and any(w in (prev_op.get("text") or "").lower() for w in WAIT_PHRASES))})
        elif t.get("hold_event") == "end" and holds and "end_turn" not in holds[-1]:
            holds[-1]["end_turn"] = t["index"]
            s0, s1 = tr[holds[-1]["start_turn"]].get("offset"), t.get("offset")
            if s0 is not None and s1 is not None:
                holds[-1]["seconds"] = round(s1 - s0)
    facts["holds"] = holds
    non_sys = [t for t in tr if t["speaker"] in ("operator", "client", "bot")]
    end_sys = [t["index"] for t in tr if t["speaker"] == "system" and any(m in (t.get("text") or "").lower() for m in END_MARKERS)]
    facts["ending"] = {"last_speaker": non_sys[-1]["speaker"] if non_sys else None,
                       "last_operator_turn": ops[-1]["index"] if ops else None,
                       "system_end_event_turn": end_sys[-1] if end_sys else None}
    return facts


def format_facts(f):
    lines = []
    if f["first_operator_turn"] is not None:
        tb = f.get("first_operator_turns") or [f["first_operator_turn"]]
        lines.append(f"- Первая реплика человека-оператора: №{f['first_operator_turn']}"
                     + (f" (вместе с подряд идущими №{', №'.join(map(str, tb[1:]))} до ответа клиента)" if len(tb) > 1 else ""))
    else:
        lines.append("- Человек-оператор в диалоге не отвечал.")
    if f.get("system_says_client_authenticated"):
        lines.append("- Система/бот сообщает, что клиент авторизован/верифицирован.")
    lines.append(f"- Язык клиента (эвристика по алфавиту): {f['client_language_guess'] or 'не определён'}; "
                 f"язык в метаданных: {f['language_metadata']}"
                 + (f"; язык первой реплики оператора: {f.get('first_operator_reply_language_guess') or 'не определён'}"
                    if f["first_operator_turn"] is not None else ""))
    w = f["longest_wait_after_first_reply"]
    if w:
        lines.append(f"- Самое долгое ожидание ответа оператора после его первой реплики: {w['seconds']} с "
                     f"(реплики №{w['from_turn']} → №{w['to_turn']}); оператор предупреждал об ожидании перед этим: "
                     f"{'да' if w['operator_warned_before'] else 'нет'}.")
    for h in f["holds"]:
        lines.append(f"- Удержание (системное событие) с реплики №{h['start_turn']}"
                     + (f" до №{h['end_turn']}" if "end_turn" in h else "")
                     + (f", {h['seconds']} с" if "seconds" in h else "")
                     + f"; оператор предупредил перед удержанием: {'да' if h['operator_warned_before'] else 'нет'}.")
    e = f["ending"]
    lines.append(f"- Последний говорящий (без системы): {e['last_speaker']}; последняя реплика оператора: №{e['last_operator_turn']}; "
                 "системное событие завершения: "
                 f"{('№' + str(e['system_end_event_turn'])) if e['system_end_event_turn'] is not None else 'нет'}.")
    return "\n".join(lines)

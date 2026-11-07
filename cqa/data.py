"""Loading, fetching and normalising dialog records."""
import hashlib
import json
import re
from datetime import datetime

import httpx

from . import config

HOLD_START = ("инициирована консультация", "клиент находится на удержании", "consultation initiated", "on hold")
HOLD_END = ("клиент больше не находится на удержании", "no longer on hold", "removed from hold")


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
        if sp == "system" and any(k in low for k in HOLD_END):
            internal = False
        t["internal"] = bool(internal and sp == "operator")
        if sp == "system" and any(k in low for k in HOLD_START):
            internal = True
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

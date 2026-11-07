"""Loading, fetching and normalising dialog records."""
import hashlib
import json
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
        sp = sp.strip().lower()
        if sp not in ("client", "bot", "operator", "system"):
            sp, tx = "client", line
        turns.append({"timestamp": ts, "speaker": sp, "text": tx.strip()})
    return normalize({"conversation_id": "pasted-" + hashlib.md5(text.encode()).hexdigest()[:8],
                      "channel": channel, "language": language, "transcript": turns}, 0)


def _parse_iso(ts):
    try:
        s = ts.replace("Z", "+00:00")
        if "." in s:  # trim 7-digit fractions
            head, rest = s.split(".", 1)
            frac = "".join(ch for ch in rest if ch.isdigit())
            tz = rest[len(frac):]
            s = f"{head}.{frac[:6]}{tz}"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _mmss(ts):
    try:
        parts = [int(x) for x in ts.split(":")]
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
    rec["kind"] = "call" if ("call" in ch or "voice" in ch) else "chat"
    turns = []
    first = None
    internal = False
    for idx, t in enumerate(rec.get("transcript") or []):
        t = dict(t)
        t["index"] = idx
        ts = str(t.get("timestamp") or "")
        off = None
        if rec["kind"] == "call":
            dt = _parse_iso(ts)
            if dt is not None:
                first = first or dt
                off = (dt - first).total_seconds()
        else:
            off = _mmss(ts)
        t["offset"] = off
        low = (t.get("text") or "").lower()
        if t.get("speaker") == "system" and any(k in low for k in HOLD_END):
            internal = False
        t["internal"] = bool(internal and t.get("speaker") == "operator")
        if t.get("speaker") == "system" and any(k in low for k in HOLD_START):
            internal = True
        turns.append(t)
    rec["transcript"] = turns
    rec["hash"] = dialog_hash(rec)
    return rec


def dialog_hash(rec):
    core = [(t.get("speaker"), t.get("text")) for t in rec.get("transcript") or []]
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

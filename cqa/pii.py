"""Deterministic, reversible PII masking for KZ/RU/EN support transcripts.

Every distinct entity value maps to a stable placeholder ``⟨KIND_xxxxxx⟩`` (xxxxxx = sha256 of the
normalised value, salted by ``CQA_PII_SALT``), so the same phone in two calls/turns gets the same
placeholder and the model can still reason "client gave the same number twice". The reverse map
lives only in process memory (``REGISTRY``) — it never leaves the perimeter.

Hook: ``llm.LLM.call`` masks system/cached/user text when ``CQA_PII_MASK=1`` and unmasks every
string in the returned tool arguments, so evidence quotes are validated against (and shown with)
the original transcript text.

Entities: IIN/ЖСН (12 digits + checksum), IBAN KZ, card numbers (Luhn), emails, phones (+7/8),
account/contract numbers after a keyword, dates of birth, person names (after «меня зовут»,
«менің атым», «my name is», «зовут», plus ``CQA_PII_NAMES`` list).
"""
from __future__ import annotations

import hashlib
import os
import re
import threading

REGISTRY: dict[str, str] = {}
_LOCK = threading.Lock()
PH_RE = re.compile(r"⟨(?:IIN|IBAN|CARD|EMAIL|PHONE|ACCOUNT|DOB|NAME)_[0-9a-f]{6}⟩")


def enabled() -> bool:
    return os.environ.get("CQA_PII_MASK", "0").strip().lower() in ("1", "true", "yes", "on")


def iin_valid(s: str) -> bool:
    """Kazakhstan IIN/BIN checksum (weights 1..11, fallback 3..11,1,2; 10 twice → invalid)."""
    d = [int(c) for c in s]
    if len(d) != 12:
        return False
    for w in (range(1, 12), [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]):
        k = sum(a * b for a, b in zip(d[:11], w, strict=True)) % 11
        if k != 10:
            return k == d[11]
    return False


def luhn_valid(s: str) -> bool:
    d = [int(c) for c in s][::-1]
    tot = sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(d))
    return tot % 10 == 0


def _digits(s):
    return re.sub(r"\D", "", s)


def placeholder(kind: str, value: str) -> str:
    norm = value   # exact surface form → lossless unmask (same number in another format = another placeholder)
    salt = os.environ.get("CQA_PII_SALT", "")
    ph = f"⟨{kind}_{hashlib.sha256((salt + kind + norm).encode()).hexdigest()[:6]}⟩"
    with _LOCK:
        REGISTRY.setdefault(ph, value)
    return ph


_NAME = r"[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][a-zа-яёәғқңөұүһі]+(?:-[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][a-zа-яёәғқңөұүһі]+)?"
PATTERNS = [
    ("IBAN", re.compile(r"\bKZ\d{2}(?:\s?[A-Z0-9]){16}\b", re.I), None),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), lambda m: luhn_valid(_digits(m)) and len(_digits(m)) != 12),
    ("IIN", re.compile(r"(?<!\d)\d{12}(?!\d)"), iin_valid),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+7|8|7)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"), None),
    ("ACCOUNT", re.compile(r"(?i)(?:сч[её]т[а-я]*|шот|account|acc|договор[а-я]*|contract|келісімшарт)"
                           r"[\s:№#.]{0,6}(?:номер\s*|number\s*|no\.?\s*|№\s*)?([A-Z]{0,4}\d[\d-]{4,24}\d)"), None),
    ("DOB", re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19[2-9]\d|20[01]\d)(?!\d)"), None),
    ("NAME", re.compile(rf"(?i:меня\s+зовут|зовут|менің\s+атым|атым|my\s+name\s+is)\s+({_NAME}(?:\s+{_NAME})?)"), None),
]
STOP_NAMES = {"Freedom", "Broker", "Finance", "Банк", "Bank", "Фридом"}


def extra_names():
    return [n.strip() for n in os.environ.get("CQA_PII_NAMES", "").split(",") if n.strip()]


def mask(text: str) -> str:
    """Replace PII with stable placeholders. Idempotent (placeholders are never re-masked)."""
    if not text:
        return text
    for kind, rx, check in PATTERNS:
        def rep(m, kind=kind, check=check):
            g = 1 if m.re.groups else 0
            val = m.group(g)
            if PH_RE.search(val) or (check and not check(val)):
                return m.group(0)
            if kind == "NAME":
                if val.split()[0] in STOP_NAMES:
                    return m.group(0)
            s, e = m.span(g)
            o = m.start(0)
            whole = m.group(0)
            return whole[: s - o] + placeholder(kind, val) + whole[s - o + len(val):] if g else placeholder(kind, val)
        text = rx.sub(rep, text)
    for n in extra_names():
        text = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", lambda m: placeholder("NAME", m.group(0)), text)
    return text


def unmask(obj):
    """Recursively restore placeholders in strings/lists/dicts (unknown placeholders are left as is)."""
    if isinstance(obj, str):
        return PH_RE.sub(lambda m: REGISTRY.get(m.group(0), m.group(0)), obj) if "⟨" in obj else obj
    if isinstance(obj, list):
        return [unmask(x) for x in obj]
    if isinstance(obj, dict):
        return {k: unmask(v) for k, v in obj.items()}
    return obj


def find(text: str) -> list[tuple[str, str]]:
    """(kind, value) pairs that ``mask`` would replace — for audit logs / tests."""
    out = []
    for ph in PH_RE.findall(mask(text)):
        out.append((ph[1:].split("_")[0], REGISTRY.get(ph)))
    return out

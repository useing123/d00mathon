"""Loaders for EXTERNAL, public, human-labelled customer-support datasets (external validation).

MAIA-DQE (Mendonça et al., GEM 2023) — real bilingual customer-support chats (agent EN <-> customer DE/PT,
machine-translated by Unbabel), human annotations at sentence / turn / dialogue level.
Source: https://github.com/johndmendonca/MAIA-DQE, licence CC BY-ND 4.0. The raw text is downloaded at run time
into data/external/ (git-ignored) and is NOT redistributed in this repo; we commit only dialogue ids + labels.

Every agent turn is used in its ORIGINAL language (English, ``text_src``); customer turns use the dataset's English
MT (``text_mt``), so the whole transcript is English for both the human annotators' reference and our system.
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from . import config

MAIA_URL = "https://raw.githubusercontent.com/johndmendonca/MAIA-DQE/main/data/{}"
MAIA_SUBSETS = ["de_client_01.json", "de_client_02.json", "pt_br_client_02.json", "pt_br_client_04.json",
                "pt_pt_client_03.json"]
MAIA_LICENSE = "CC BY-ND 4.0 (https://github.com/johndmendonca/MAIA-DQE/blob/main/LICENSE.md)"


def _ext_dir() -> Path:
    d = Path(config.DATA_DIR) / "external" / "maia"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_maia(subsets=None) -> list[Path]:
    out = []
    for name in subsets or MAIA_SUBSETS:
        p = _ext_dir() / name
        if not p.exists():
            urllib.request.urlretrieve(MAIA_URL.format(name), p)   # noqa: S310 (fixed https URL)
        out.append(p)
    return out


def _min_label(turns, key, floor="outbound"):
    vals = [t.get(key) for t in turns if t.get("floor") == floor and t.get(key) is not None]
    return int(min(vals)) if vals else None


def maia_labels(d) -> dict:
    """Dialogue-level human labels derived ONLY from MAIA annotations (no inference)."""
    dl = d.get("dialog") or {}
    ts, dc = dl.get("Task Sucess"), dl.get("Dropped conversation")   # sic: "Sucess" in the source
    turns = d.get("turns") or []
    return {
        "task_success": int(ts) if ts is not None and ts >= 1 else None,          # 1..5
        "dropped": int(dc) if dc is not None and dc >= 0 else None,               # 0/1
        "agent_polite": _min_label(turns, "Politeness"),                         # 1 = every agent turn civil
        "agent_sensible": _min_label(turns, "Sensibleness"),                     # 1 = every agent turn sensible
        "agent_understanding": _min_label(turns, "Understanding"),
    }


def maia_to_record(d, subset="") -> dict:
    """One MAIA dialogue -> our transcript format (cqa.data.normalize-compatible)."""
    tr = []
    for t in d.get("turns") or []:
        agent = t.get("floor") == "outbound"
        sents = t.get("text_src") if agent else t.get("text_mt")
        text = " ".join(s for s in (sents or []) if s).strip()
        if text:
            tr.append({"timestamp": None, "speaker": "operator" if agent else "client", "text": text})
    return {"conversation_id": d["id"], "channel": "agent chat", "language": "EN", "transcript": tr,
            "source": f"MAIA-DQE/{subset}", "human_labels": maia_labels(d)}


def load_maia(paths=None) -> list[dict]:
    recs = []
    for p in paths or fetch_maia():
        p = Path(p)
        for d in json.loads(p.read_text(encoding="utf-8")):
            recs.append(maia_to_record(d, p.stem))
    return recs


def maia_stratum(lab) -> str:
    if lab["agent_polite"] == 0:
        return "impolite"
    ts = lab["task_success"]
    return {1: "ts12", 2: "ts12", 3: "ts3", 4: "ts45", 5: "ts45"}[ts] + f"_dc{lab['dropped']}"


def stratified_sample(recs, n=80, seed=42, max_turns=60):
    """Fixed-seed stratified sample: ALL impolite-agent dialogues (rare negative class), then the rest split as
    evenly as possible across task-success buckets x dropped. Dialogues without TS/DC labels are excluded;
    very long dialogues (> max_turns turns) are excluded to bound cost."""
    ok = [r for r in recs if r["human_labels"]["task_success"] and r["human_labels"]["dropped"] is not None
          and r["human_labels"]["agent_polite"] is not None and len(r["transcript"]) <= max_turns]
    rng = random.Random(seed)
    by = {}
    for r in sorted(ok, key=lambda r: r["conversation_id"]):
        by.setdefault(maia_stratum(r["human_labels"]), []).append(r)
    for v in by.values():
        rng.shuffle(v)
    pick = list(by.pop("impolite", []))[: n // 4]
    keys = sorted(by)
    while len(pick) < n and any(by[k] for k in keys):
        for k in keys:
            if by[k] and len(pick) < n:
                pick.append(by[k].pop())
    uniq = {}   # MAIA ships 43 dialogues twice (pt_br_client_02 contains pt_br_client_04, identical labels)
    for r in pick:
        uniq.setdefault(r["conversation_id"], r)
    return sorted(uniq.values(), key=lambda r: r["conversation_id"])

"""Headline numbers in README / DEMO / deck must equal docs/numbers.json (scripts/pitch_numbers.py)."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("numbers_mod", ROOT / "scripts" / "pitch_numbers.py")
numbers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(numbers)

NUM = json.loads((ROOT / "docs" / "numbers.json").read_text(encoding="utf-8"))
DISPLAY = NUM["display"]


def test_formatters():
    assert numbers.fmt_acc(0.95) == "0.95"
    assert numbers.fmt_acc(0.963) == "0.963"
    assert numbers.fmt_usd(10.2789 / 65) == "$0.158"
    assert numbers.fmt_usd(0.014580) == "$0.0146"


def test_display_matches_raw():
    raw = NUM["raw"]
    for k in ("jury_heldout", "cascade_heldout", "dev_agreement", "local_cascade_heldout"):
        assert DISPLAY[k] == numbers.fmt_acc(raw[k])
    for k in ("jury_usd", "cascade_usd"):
        assert DISPLAY[k] == numbers.fmt_usd(raw[k])
    assert DISPLAY["dev_mean"] == f"{raw['dev_mean']:.2f}"


@pytest.mark.parametrize("doc", sorted(numbers.REQUIRED))
def test_doc_numbers_match_numbers_json(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    found = numbers.SPAN_RE.findall(text)
    assert found, f"{doc}: no <span data-n=...> markers"
    bad = [(k, v, DISPLAY.get(k)) for k, v in found if DISPLAY.get(k) != v]
    assert not bad, f"{doc}: marked numbers differ from docs/numbers.json: {bad}"
    missing = set(numbers.REQUIRED[doc]) - {k for k, _ in found}
    assert not missing, f"{doc}: key metrics not shown: {sorted(missing)}"


@pytest.mark.parametrize("doc", sorted(numbers.REQUIRED))
def test_no_competing_headline_numbers(doc):
    """One number per metric: superseded jury/cascade cost readings must not reappear in the pitch."""
    text = re.sub(numbers.SPAN_RE, "", (ROOT / doc).read_text(encoding="utf-8"))
    for stale in ("$0.16 ", "$0.16<", "$0.16)", "$0.004–0.015", "$0.0745", "11–18"):
        assert stale not in text, f"{doc}: competing number {stale!r}"


def test_deck_pdf_exists():
    assert (ROOT / "docs" / "CQA_Pitch_Deck.pdf").stat().st_size > 10_000

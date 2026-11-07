"""Submission format v1.0: mapping, schema file in sync, validator, committed results/dev/results.json."""
import json
from pathlib import Path

import pytest

from cqa import data, submission
from cqa.evaluator import Evaluator

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_dialogs.json"


@pytest.fixture(scope="module")
def doc():
    recs = data.load_file(SAMPLE)
    dialogs, meta = Evaluator(use_cache=True).evaluate_many(recs)
    return {"group": "DEMO", "meta": meta, "dialogs": dialogs}


def test_v1_shape_and_mapping(doc):
    obj = submission.to_v1(doc, "demo")
    assert submission.validate(obj) == []
    assert obj["format_version"] == "1.0" and obj["turn_numbering"] == "1-based"
    d0, r0 = obj["dialogs"][0], doc["dialogs"][0]
    assert set(d0["criteria"]) == set(submission.SPEC["criteria"].values())
    for r in r0["criteria"]:
        got = d0["criteria"][submission.SPEC["criteria"][r["criterion_id"]]]
        v = r["value"]
        exp = None if v == "NA" else (int(v) if r.get("type") == "scale" else {"YES": True, "NO": False}[v])
        assert got["value"] == exp
        assert got["evidence_turns"] == sorted({e["turn"] for e in r["evidence"]})
        assert all(t >= 1 for t in got["evidence_turns"])
    assert d0["final_score"] == (None if r0["total"] is None else round(r0["total"], 2))
    assert d0["critical_flag"] is bool(r0["gross_error"])


def test_human_override_wins(doc):
    d = json.loads(json.dumps(doc["dialogs"][0]))
    d["human"] = {"reviewed": True, "values": {**{r["criterion_id"]: r["value"] for r in d["criteria"]}, "08": "YES"},
                  "total": 12.5, "gross_error": True}
    o = submission.to_v1({"group": "X", "dialogs": [d]})["dialogs"][0]
    assert o["critical_flag"] is True and o["criteria"]["critical"]["value"] is True and o["final_score"] == 12.5


def test_validator_rejects_bad_docs(doc):
    obj = submission.to_v1(doc, "demo")
    bad = json.loads(json.dumps(obj))
    bad["format_version"] = "0.9"
    bad["dialogs"][0]["criteria"]["contact"]["value"] = "YES"
    bad["dialogs"][0]["criteria"]["closing"]["evidence_turns"] = [0]
    bad["dialogs"].append(bad["dialogs"][0])
    errs = " | ".join(submission.validate(bad))
    assert "format_version" in errs and "contact" in errs and "duplicate" in errs and "closing" in errs


def test_schema_file_in_sync():
    assert json.loads((ROOT / "docs" / "results.schema.json").read_text()) == submission.schema()


@pytest.mark.parametrize("group", ["dev", "prod"])
def test_committed_results_are_valid(group):
    p = ROOT / "results" / group / "results.json"
    if not p.exists():
        pytest.skip(f"{p} not produced yet (prod: `make submit GROUP=<id>` in the control window)")
    assert submission.validate(json.loads(p.read_text())) == []

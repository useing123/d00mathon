"""Submission JSON 1.0 (organizer appendix): mapping, error records, validator rules, schema file, committed files."""
import json
from pathlib import Path

import pytest
import yaml

from cqa import checklist as ckl, config, data, submission
from cqa.evaluator import Evaluator

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_dialogs.json"


@pytest.fixture(scope="module")
def doc():
    recs = data.load_file(SAMPLE)
    dialogs, meta = Evaluator(use_cache=True).evaluate_many(recs)
    return {"group": "DEMO", "meta": meta, "dialogs": dialogs}


def test_shape_and_mapping(doc):
    obj = submission.to_v1(doc)
    assert submission.validate(obj) == ([], [])
    assert obj["schema_version"] == "1.0" and obj["team_name"] == "d00mathon"
    assert obj["model"]["mode"] in ("local", "external_api", "hybrid") and obj["model"]["name"]
    r0, d0 = obj["results"][0], doc["dialogs"][0]
    assert r0["status"] == "ok" and [c["criterion_id"] for c in r0["criteria"]] == submission.CODES
    for c, r in zip(r0["criteria"], d0["criteria"], strict=True):
        v = r["value"]
        exp = None if v == "NA" else (int(v) if r.get("type") == "scale" else {"YES": True, "NO": False}[v])
        assert c["value"] == exp and type(c["value"]) is type(exp)
        assert c["evidence_turns"] == sorted({e["turn"] for e in r["evidence"]}) and all(t >= 1 for t in c["evidence_turns"])
        assert set(c) == {"criterion_id", "value", "rationale", "evidence_turns"}
    assert r0["score"] == {"final_score": round(d0["total"], 2), "critical_flag": d0["gross_error"]}
    crit = next(c for c in r0["criteria"] if c["criterion_id"] == "critical")
    assert r0["score"]["critical_flag"] is crit["value"]


def test_evidence_turns_index_original_transcript(doc):
    recs = {r["conversation_id"]: r for r in data.load_file(SAMPLE)}
    raw = {r["conversation_id"]: r["transcript"] for r in json.loads(SAMPLE.read_text())["data"]["records"]}
    for cid, rec in recs.items():
        assert len(rec["transcript"]) == len(raw[cid])          # normalisation never drops or merges turns
        for t, orig in zip(rec["transcript"], raw[cid], strict=True):
            assert t["text"] == orig["text"] and raw[cid][t["index"] - 1] is not None


def test_error_records_never_zero(doc):
    d = json.loads(json.dumps(doc["dialogs"][0]))
    d["status"], d["total"], d["failed_criteria"] = "incomplete", None, ["04"]
    d["criteria"][3].update(value=None, error="timeout")
    obj = submission.to_v1({"dialogs": [d], "meta": doc["meta"]}, ids=[d["conversation_id"], "missing-1"])
    e1, e2 = obj["results"]
    assert e1 == {"conversation_id": d["conversation_id"], "status": "error", "criteria": [], "score": None, "error": e1["error"]}
    assert "04" in e1["error"] and e2["status"] == "error" and e2["score"] is None
    assert submission.validate(obj)[0] == []


def test_human_override_wins(doc):
    d = json.loads(json.dumps(doc["dialogs"][0]))
    d["human"] = {"reviewed": True, "status": "complete", "values": {**{r["criterion_id"]: r["value"] for r in d["criteria"]}, "08": "YES"},
                  "total": 12.5, "gross_error": True, "overrides": {"08": {"value": "YES", "justification": "апелляция"}}}
    r = submission.to_v1({"dialogs": [d], "meta": {}})["results"][0]
    crit = r["criteria"][-1]
    assert r["score"] == {"final_score": 12.5, "critical_flag": True} and crit["value"] is True and crit["rationale"] == "апелляция"


def test_validator_rules(doc):
    obj = submission.to_v1(doc)
    bad = json.loads(json.dumps(obj))
    r = bad["results"][0]
    r["criteria"][0]["value"] = 1                       # contact must be bool
    r["criteria"][5]["value"] = 5                       # dialogue not in 0/4/7/10
    r["criteria"][4]["value"] = None                    # empathy has no N/A
    r["score"]["critical_flag"] = not r["criteria"][7]["value"]
    r["score"]["final_score"] = 50.123
    bad["results"].append(json.loads(json.dumps(bad["results"][1])))
    bad["results"][1]["criteria"].pop()                 # 7 criteria
    errs = " | ".join(submission.validate(bad)[0])
    for needle in ("contact=1", "dialogue=5", "empathy=None", "critical_flag", "2 decimals", "duplicate", "official 8"):
        assert needle in errs, needle
    assert any("exactly 10" in e for e in submission.validate(obj, prod=True)[0])
    assert any("missing" in e for e in submission.validate(obj, expected_ids=["x"])[0])
    bad2 = json.loads(json.dumps(obj))
    bad2["schema_version"] = "1"
    bad2["model"]["mode"] = "cloud"
    assert len(submission.validate(bad2)[0]) >= 2


def test_replaced_criterion_exports_with_its_id(doc, tmp_path, monkeypatch):
    """Control step: a criterion is replaced in the UI; export must carry the new code."""
    cdir = tmp_path / "checklists"
    cdir.mkdir()
    for p in config.CHECKLIST_DIR.glob("*.yaml"):
        (cdir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config, "CHECKLIST_DIR", cdir)
    cl = ckl.load("freedom_chat_v1")
    cl["criteria"][4] = {"id": "politeness", "name": "Вежливость", "type": "binary", "weight": 15,
                         "rubric": "YES — оператор вежлив.", "rules": ""}
    ckl.save(cl)
    assert yaml.safe_load((cdir / "freedom_chat_v1.yaml").read_text())["criteria"][4]["id"] == "politeness"
    recs = [r for r in data.load_file(SAMPLE) if r["kind"] == "chat"][:1]
    dialogs, meta = Evaluator(use_cache=False).evaluate_many(recs)
    obj = submission.to_v1({"dialogs": dialogs, "meta": meta})
    codes = [c["criterion_id"] for c in obj["results"][0]["criteria"]]
    assert "politeness" in codes and "empathy" not in codes
    errs, warns = submission.validate(obj, custom_ok=True)
    assert errs == [] and any("official 8" in w for w in warns)
    assert any("official 8" in e for e in submission.validate(obj)[0])


def test_schema_file_in_sync():
    assert json.loads((ROOT / "docs" / "results.schema.json").read_text()) == submission.schema()


@pytest.mark.parametrize("group", ["dev", "prod"])
def test_committed_results_are_valid(group):
    p = ROOT / "results" / group / "results.json"
    if not p.exists():
        pytest.skip(f"{p} not produced yet (prod: `make submit GROUP=<id>` in the control window)")
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert submission.validate(obj, prod=group == "prod")[0] == []
    if group == "dev":
        assert len(obj["results"]) == 65

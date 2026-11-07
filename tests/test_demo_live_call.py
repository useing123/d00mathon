"""Control-step demo (docs/DEMO_LIVE.md) on the CALL checklist, over the same HTTP API the UI uses:
add criterion 09 -> save -> re-evaluate one call (only 09 goes to the model) -> v1 export with a custom code
-> replace 05 (only 05 re-evaluated) -> restore the original checklist byte-for-byte."""
import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cqa import checklist as ckl, config, submission
from cqa.llm import USAGE

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_dialogs.json"
CALL = "demo-ru-call"
NEW = {"id": "09", "name": "Предупреждение о записи разговора", "type": "binary", "weight": 5, "allow_na": True,
       "gross": False, "code": "recording_notice",
       "rubric": "YES — клиенту сообщили, что разговор записывается.\nNO — о записи не сообщили.\nNA — разговора с оператором не было.",
       "rules": ""}


@pytest.fixture
def client(tmp_path, monkeypatch):
    cdir = tmp_path / "checklists"
    shutil.copytree(ROOT / "checklists", cdir)
    monkeypatch.setattr(config, "CHECKLIST_DIR", cdir)
    from cqa.server import app
    return TestClient(app), cdir


def _run(cl, body):
    s = {}
    before = USAGE["calls"]
    job = cl.post("/api/evaluate", json=body).json()
    for _ in range(200):
        s = cl.get(f"/api/jobs/{job['id']}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s
    return USAGE["calls"] - before


def _dialog(cl):
    return next(d for d in cl.get("/api/results/DEMO").json()["dialogs"] if d["conversation_id"] == CALL)


def _put(cl, c):
    r = cl.put("/api/checklists/freedom_call_v1", json={"checklist": c})
    assert r.status_code == 200, r.text


def test_add_replace_reeval_export_restore(client):
    cl, cdir = client
    orig = (cdir / "freedom_call_v1.yaml").read_bytes()
    chat_orig = (cdir / "freedom_chat_v1.yaml").read_bytes()
    cl.post("/api/upload?group=DEMO", files={"file": ("s.json", SAMPLE.read_bytes(), "application/json")})
    assert _run(cl, {"group": "DEMO"}) > 0
    base = _dialog(cl)
    assert base["checklist_id"] == "freedom_call_v1" and len(base["criteria"]) == 8

    # 1) add 09 exactly like the UI form does (GET resolved checklist, append, PUT {checklist})
    c = cl.get("/api/checklists/freedom_call_v1").json()["checklist"]
    c["criteria"].append(dict(NEW))
    _put(cl, c)
    assert [x["id"] for x in ckl.load("freedom_call_v1")["criteria"]][-1] == "09"
    assert (cdir / "freedom_chat_v1.yaml").read_bytes() == chat_orig      # the chat base is untouched

    # 2) re-evaluate ONE call with the cache ("Переоценить" on the dialog card) -> only 09 goes to the model
    calls = _run(cl, {"group": "DEMO", "ids": [CALL]})
    d = _dialog(cl)
    by = {x["criterion_id"]: x for x in d["criteria"]}
    assert list(by) == ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    assert [k for k, x in by.items() if not x["cached"]] == ["09"]
    assert calls >= 1 and by["09"]["value"] in ("YES", "NO", "NA")
    for k in by:
        if k != "09":
            assert by[k]["value"] == {x["criterion_id"]: x for x in base["criteria"]}[k]["value"]
    # total recomputed with the new weight set (09 weight 5 counts unless NA)
    from cqa.evaluator import rescore
    doc = rescore({"dialogs": [json.loads(json.dumps(d))], "meta": {}})
    assert doc["dialogs"][0]["total"] == d["total"]
    if by["09"]["value"] != "NA":
        assert d["total"] != base["total"] or by["09"]["value"] == "YES" and base["total"] == 100

    # 3) changing only the export code / weight does not re-run the model
    c = cl.get("/api/checklists/freedom_call_v1").json()["checklist"]
    c["criteria"][-1]["code"] = "recording_notice"
    _put(cl, c)
    assert _run(cl, {"group": "DEMO", "ids": [CALL]}) == 0

    # 4) v1 export carries the custom code and validates with --allow-custom semantics
    v1 = cl.get("/api/export/DEMO/results.json")
    assert v1.status_code == 200, v1.text
    obj = v1.json()
    rec = next(r for r in obj["results"] if r["conversation_id"] == CALL)
    assert "recording_notice" in [x["criterion_id"] for x in rec["criteria"]]
    errs, warns = submission.validate(obj, custom_ok=True)
    assert errs == [] and any("recording_notice" in w for w in warns)
    assert submission.validate(obj)[0]      # strict validation flags the non-official criterion

    # 5) replace 05 (new description) -> only 05 is re-evaluated
    c = cl.get("/api/checklists/freedom_call_v1").json()["checklist"]
    c5 = next(x for x in c["criteria"] if x["id"] == "05")
    c5["rubric"] = c5["rubric"].rstrip() + "\nДополнительно: признать неудобство своими словами."
    _put(cl, c)
    assert _run(cl, {"group": "DEMO", "ids": [CALL]}) >= 1
    assert [x["criterion_id"] for x in _dialog(cl)["criteria"] if not x["cached"]] == ["05"]

    # 6) restore: write the original YAML back (as the YAML editor does) -> byte-identical, 8 criteria, all cached
    r = cl.put("/api/checklists/freedom_call_v1", json={"yaml": orig.decode("utf-8")})
    assert r.status_code == 200, r.text
    assert (cdir / "freedom_call_v1.yaml").read_bytes() == orig     # stored verbatim (extends + comments kept)
    assert len(ckl.load("freedom_call_v1")["criteria"]) == 8
    assert _run(cl, {"group": "DEMO", "ids": [CALL]}) == 0
    d = _dialog(cl)
    assert len(d["criteria"]) == 8 and d["total"] == base["total"]


def test_code_not_in_criterion_hash():
    c = ckl.load("freedom_call_v1")
    cr = dict(c["criteria"][0])
    h = ckl.criterion_hash(c, cr)
    assert ckl.criterion_hash(c, {**cr, "code": "x"}) == h
    assert ckl.criterion_hash(c, {**cr, "rubric": cr["rubric"] + "!"}) != h


def test_yaml_editor_rejects_bad_extends(client):
    cl, _ = client
    bad = "id: freedom_call_v1\nextends: freedom_chat_v1\ncriteria_overrides:\n  - id: '99'\n    weight: 1\n"
    assert cl.put("/api/checklists/freedom_call_v1", json={"yaml": bad}).status_code == 400
    assert cl.put("/api/checklists/freedom_call_v1", json={"yaml": "- 1"}).status_code == 400
    assert len(ckl.load("freedom_call_v1")["criteria"]) == 8

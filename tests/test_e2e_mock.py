"""End-to-end without an API key: CLI-equivalent pipeline + HTTP API on the mock provider."""
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from cqa import data, store
from cqa.evaluator import Evaluator, rescore

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_dialogs.json"


def test_sample_file_schema():
    recs = data.load_file(SAMPLE)
    assert len(recs) == 4
    assert {r["kind"] for r in recs} == {"chat", "call"}
    for r in recs:
        assert r["transcript"] and all({"timestamp", "speaker", "text"} <= set(t) for t in r["transcript"])


def test_pipeline_end_to_end():
    recs = data.load_file(SAMPLE)
    dialogs, meta = Evaluator(use_cache=True).evaluate_many(recs)
    assert meta["errors"] == [] and meta["provider"] == "mock"
    by = {d["conversation_id"]: d for d in dialogs}
    assert by["demo-ru-call"]["checklist_id"] == "freedom_call_v1"
    assert by["demo-ru-chat-good"]["checklist_id"] == "freedom_chat_v1"
    for d in dialogs:
        assert len(d["criteria"]) == 8
        assert d["total"] is None or 0 <= d["total"] <= 100
        for c in d["criteria"]:
            assert c["value"] is not None
            for e in c["evidence"]:  # every surviving quote is verbatim in the cited turn
                rec = next(r for r in recs if r["conversation_id"] == d["conversation_id"])
                assert e["quote"][:20].lower() in rec["transcript"][e["turn"]]["text"].lower()
    # weak dialog (no greeting/name, no farewell) scores below the good one
    assert by["demo-en-chat-weak"]["total"] < by["demo-ru-chat-good"]["total"]
    # second run is served from the per-criterion cache
    again, _ = Evaluator(use_cache=True).evaluate_many(recs)
    assert all(c["cached"] for d in again for c in d["criteria"])
    # persist, override, rescore
    store.save_results("T", dialogs, meta)
    store.save_override("T", "demo-ru-chat-good", "01", "NO", "test")
    doc = store.load_results("T")
    d = next(x for x in doc["dialogs"] if x["conversation_id"] == "demo-ru-chat-good")
    assert d["human"]["diff"] == [{"criterion_id": "01", "ai": "YES", "human": "NO"}]
    assert d["human"]["total"] == 90.0 and d["total"] == 100.0
    assert rescore(doc)["dialogs"][0]["total"] is not None
    csv = store.to_csv(doc, "both")
    assert csv.splitlines()[0].startswith("conversation_id,channel,language,checklist_id,c01")


def test_http_api_end_to_end():
    from cqa.server import app
    cl = TestClient(app)
    assert cl.get("/").status_code == 200
    assert cl.get("/api/health").json()["provider"] == "mock"
    assert len(cl.get("/api/checklists").json()) >= 2
    assert cl.get("/api/checklists/nope").status_code == 404
    r = cl.post("/api/upload?group=DEMO", files={"file": ("s.json", SAMPLE.read_bytes(), "application/json")})
    assert r.json() == {"group": "DEMO", "dialogs": 4}
    assert cl.post("/api/upload?group=BAD", files={"file": ("b.json", b"{not json", "application/json")}).status_code == 400
    job = cl.post("/api/evaluate", json={"group": "DEMO"}).json()
    for _ in range(100):
        s = cl.get(f"/api/jobs/{job['id']}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s
    res = cl.get("/api/results/DEMO").json()
    assert len(res["dialogs"]) == 4
    d = cl.put("/api/overrides/DEMO/demo-kz-chat/07", json={"value": "NO", "justification": "t"}).json()
    assert d["human"]["reviewed"] is True
    assert cl.get("/api/export/DEMO.csv?which=both").text.startswith("conversation_id")
    assert json.loads(cl.get("/api/export/DEMO.json").content)["group"] == "DEMO"
    assert {"group": "DEMO", "dialogs": 4, "evaluated": True} in cl.get("/api/groups").json()
    assert len(cl.get("/api/dialogs/DEMO").json()) == 4
    assert cl.post("/api/rescore/DEMO").status_code == 200
    c = cl.get("/api/checklists/freedom_chat_v1").json()
    assert "criteria" in c["checklist"] and c["yaml"].startswith("id:")
    bad = {"id": "tmp_cl", "criteria": [{"id": "1", "type": "scale"}]}
    assert cl.put("/api/checklists/tmp_cl", json=bad).status_code == 400
    p = cl.post("/api/paste", json={"text": "client: привет\noperator: Здравствуйте, меня зовут Ая"}).json()
    assert p["conversation_id"].startswith("pasted-")

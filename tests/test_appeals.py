"""Two-step appeals: open (mandatory comment) -> resolve confirm/change (mandatory reason), totals recomputed."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from cqa import config, data, store
from cqa.evaluator import Evaluator
from cqa.server import app

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_dialogs.json"


def _setup():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "AP.json").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    recs = data.load_file(config.DATA_DIR / "AP.json")
    dialogs, meta = Evaluator(use_cache=True).evaluate_many(recs[:2])
    store.save_results("AP", dialogs, meta)
    d = next(d for d in dialogs if d["total"] is not None)
    r = next(c for c in d["criteria"] if c["type"] == "binary" and c["value"] in ("YES", "NO") and c["criterion_id"] != "08")
    return d, r


def test_appeal_open_and_change_recomputes_total():
    d, r = _setup()
    c = TestClient(app)
    base = {"group": "AP", "conversation_id": d["conversation_id"], "criterion_id": r["criterion_id"]}
    assert c.post("/api/appeals", json={**base, "comment": "  "}).status_code == 400     # comment mandatory
    a = c.post("/api/appeals", json={**base, "comment": "оператор представился", "author": "user1"}).json()
    assert a["status"] == "open" and a["ai_value"] == r["value"] and a["audit"][0]["action"] == "opened"
    assert c.post("/api/appeals", json={**base, "comment": "ещё раз"}).status_code == 400  # one open per criterion
    lst = c.get("/api/appeals", params={"group": "AP", "status": "open", "fragment": True}).json()
    assert lst["open"] == 1 and lst["items"][0]["id"] == a["id"] and lst["items"][0]["fragment"]
    assert all(t["turn"] >= 1 for t in lst["items"][0]["fragment"])
    url = f"/api/appeals/{a['id']}/resolve"
    new = "NO" if r["value"] == "YES" else "YES"
    assert c.post(url, json={"group": "AP", "decision": "change", "value": new, "reason": ""}).status_code == 400
    res = c.post(url, json={"group": "AP", "decision": "change", "value": new, "reason": "не по рубрике", "reviewer": "qa"}).json()
    ap = res["appeal"]
    assert ap["status"] == "resolved" and ap["resolution"]["decision"] == "changed" and ap["resolution"]["value"] == new
    assert [x["action"] for x in ap["audit"]] == ["opened", "changed"] and ap["audit"][1]["who"] == "qa"
    assert res["dialog"]["human"]["values"][r["criterion_id"]] == new and res["total"] != d["total"]
    ov = store.load_overrides("AP")[d["conversation_id"]][r["criterion_id"]]
    assert ov["value"] == new and a["id"] in ov["justification"]
    assert c.post(url, json={"group": "AP", "decision": "confirm", "reason": "x"}).status_code == 400   # already resolved
    assert c.get("/api/appeals", params={"group": "AP", "status": "open"}).json()["items"] == []
    saved = json.loads((config.RESULTS_DIR / "appeals" / "ap.json").read_text())
    assert saved[0]["status"] == "resolved"


def test_appeal_confirm_keeps_value():
    d, r = _setup()
    c = TestClient(app)
    a = c.post("/api/appeals", json={"group": "AP", "conversation_id": d["conversation_id"],
                                     "criterion_id": r["criterion_id"], "comment": "не согласен"}).json()
    res = c.post(f"/api/appeals/{a['id']}/resolve", json={"group": "AP", "decision": "confirm", "reason": "цитата верна"}).json()
    assert res["appeal"]["resolution"]["decision"] == "confirmed" and res["total"] == d["total"]
    assert store.load_overrides("AP") == {}
    assert c.delete(f"/api/appeals/{a['id']}", params={"group": "AP"}).json() == {"ok": True}
    assert c.get("/api/appeals", params={"group": "AP"}).json()["items"] == []


def test_appeal_bad_inputs():
    d, _ = _setup()
    c = TestClient(app)
    assert c.post("/api/appeals", json={"group": "AP", "conversation_id": "nope", "criterion_id": "01", "comment": "x"}).status_code == 404
    assert c.post("/api/appeals", json={"group": "AP", "conversation_id": d["conversation_id"], "criterion_id": "99",
                                        "comment": "x"}).status_code == 400
    assert c.post("/api/appeals/abc/resolve", json={"group": "AP", "decision": "confirm", "reason": "x"}).status_code == 404
    assert c.get("/api/appeals", params={"group": "AP", "status": "weird"}).status_code == 400

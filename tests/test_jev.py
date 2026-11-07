"""Offline tests for the Jev adapter (no network)."""
import json
from pathlib import Path

from cqa import checklist as ckl, data, gold, jev

PROBE = Path(__file__).resolve().parent.parent / "results" / "jev" / "probe_response.json"


def _cl():
    return ckl.load("freedom_chat_v1")


def test_questions_shape():
    cl = _cl()
    qs = jev.build_questions(cl)
    for cr in cl["criteria"]:
        q = qs[f"c{cr['id']}"]
        if cr.get("type") == "scale":
            assert q["type"] == "score"
            assert len(q["criteria"]) == len(cr["scale"])
            assert q["criteria"][0].startswith(f"{min(cr['scale'])} ")   # ascending
            assert (f"c{cr['id']}_na" in qs) == bool(cr.get("allow_na"))
        else:
            assert q["type"] == "choice"
            assert ("na" in q["criteria"]) == bool(cr.get("allow_na"))


def test_state_has_facts_and_role():
    g = gold.load_gold()[0]
    s = jev.build_state(data.normalize(g["dialog"], 0), _cl())
    assert "ФАКТЫ" in s and "OPERATOR" in s and "[1]" in s


def test_parse_answers_maps_values():
    cl = _cl()
    ans = {"c01": {"choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}},
           "c02": {"choice": "na", "probabilities": {"yes": 0, "no": 0.2, "na": 0.8}},
           "c04": {"score": 2.4, "probabilities": {"0": 0, "1": 0, "2": 0.3, "3": 0.6, "4": 0.1}},
           "c04_na": {"noul": 0.1},
           "c06": {"score": 2.9, "probabilities": {"0": 0, "1": 0, "2": 0.2, "3": 0.8}},
           "c08": {"choice": "no", "probabilities": {"yes": 0.0, "no": 1.0}}}
    crit, vals = jev.parse_answers(cl, ans)
    assert vals["01"] == "YES" and vals["02"] == "NA" and vals["04"] == 15 and vals["06"] == 10 and vals["08"] == "NO"
    c = {x["criterion_id"]: x for x in crit}
    assert c["01"]["evidence"] == [] and c["01"]["confidence"] == 0.9
    assert jev.parse_answers(cl, {"c04": {"score": 3}, "c04_na": {"noul": 0.9}})[1]["04"] == "NA"


def test_evaluate_with_stubbed_api(monkeypatch):
    if not PROBE.exists():
        return
    resp = json.loads(PROBE.read_text())
    monkeypatch.setattr(jev, "_post", lambda *a, **k: resp)
    g = gold.load_gold()[4]
    r = jev.evaluate(data.normalize(g["dialog"], 0), _cl())
    assert r["status"] == "ok" and r["total"] is not None and r["gross_error"] is False

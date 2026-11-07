"""Variance reduction (median of judge samples on scale criteria) and needs_review flags."""
import threading

from cqa import checklist as ckl, config, data, prompts
from cqa.evaluator import Evaluator, build_dialog_result, median_step, review_reasons
from tests.test_review_hardening import REC, Scripted, _cr


class Locked(Scripted):
    """Thread-safe scripted model (judge samples run in parallel)."""
    def __init__(self, outs):
        super().__init__(outs)
        self.lock = threading.Lock()

    def call(self, *a, **k):
        with self.lock:
            return super().call(*a, **k)


def _ev(jurors, judge):
    ev = Evaluator(use_cache=False)
    ev.jurors, ev.judge, ev.ensemble = jurors, judge, True
    return ev, prompts.transcript_block(REC, data.format_transcript(REC))


def test_median_step():
    assert median_step([20, 10, 15]) == 15
    assert median_step([20, 10]) == 10            # lower median on an even count (conservative)
    assert median_step(["NA", "NA", 10]) == "NA" and median_step(["NA", 10, 20]) == 10


def test_scale_disagreement_uses_median_of_three_judge_samples(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_SAMPLES", 3)
    ev_ = [{"turn": 2, "quote": "Тариф"}]
    a, b = Scripted([{"value": "20", "evidence": ev_}]), Scripted([{"value": "10", "evidence": ev_}])
    judge = Locked([{"value": "20", "evidence": ev_}, {"value": "5", "evidence": ev_}, {"value": "15", "evidence": ev_}])
    judge.model = "judge"
    ev, tx = _ev([a, b], judge)
    res = ev._jury(REC, tx, ckl.load("freedom_chat_v1"), _cr("04"), "k" * 16 + "_h")
    assert len(judge.calls) == 3 and sorted(res["judge_samples"]) == [5, 15, 20] and res["value"] == 15
    assert res["needs_review"] and {"jurors_disagreed", "judge_decided", "judge_samples_split"} <= set(res["review_reasons"])


def test_binary_disagreement_single_judge_and_flag(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_SAMPLES", 3)
    a = Scripted([{"value": "YES", "evidence": [{"turn": 2, "quote": "Тариф"}]}])
    b = Scripted([{"value": "NO", "evidence": [], "missing_where": "[2]"}])
    judge = Scripted([{"value": "YES", "evidence": [{"turn": 2, "quote": "Тариф"}]}])
    judge.model = "judge"
    ev, tx = _ev([a, b], judge)
    res = ev._jury(REC, tx, ckl.load("freedom_chat_v1"), _cr("05"), "k" * 16 + "_h")
    assert len(judge.calls) == 1 and res["value"] == "YES" and res["needs_review"]


def test_consensus_not_flagged_unless_low_confidence_scale():
    ev_ = [{"turn": 2, "quote": "Тариф"}]
    a = Scripted([{"value": "YES", "evidence": ev_, "confidence": 0.9}])
    ev, tx = _ev([a, a], Scripted([{}]))
    res = ev._jury(REC, tx, ckl.load("freedom_chat_v1"), _cr("05"), "k" * 16 + "_h")
    assert res["decided_by"] == "consensus" and res["needs_review"] is False
    low = {"value": 7, "confidence": 0.5, "decided_by": "consensus"}
    assert review_reasons(_cr("06"), low) == ["low_confidence_scale"]
    assert review_reasons(_cr("06"), {**low, "confidence": 0.9}) == []


def test_dialog_stability_range():
    cl = ckl.load("freedom_chat_v1")
    crs = []
    for c in cl["criteria"]:
        v = c["scale"][-1] if c["type"] == "scale" else ("NO" if c.get("gross") else "YES")
        crs.append({"criterion_id": c["id"], "name": c["name"], "type": c["type"], "value": v, "votes": [v]})
    st = build_dialog_result(REC, cl, [dict(x) for x in crs])["stability"]
    assert st["agreement"] == 1.0 and st["score_range"] == [100.0, 100.0] and st["score_spread"] == 0
    crs[3] = {**crs[3], "votes": [20, 10, 20], "judge_samples": [20, 10, 15], "needs_review": True}
    st = build_dialog_result(REC, cl, crs)["stability"]
    assert st["criteria_unanimous"] == 7 and st["score_range"][0] == 90.0 and st["score_range"][1] == 100.0
    assert st["flagged_criteria"] == 1


def test_v1_export_has_no_needs_review():
    import inspect

    from cqa import submission
    assert "needs_review" not in inspect.getsource(submission)

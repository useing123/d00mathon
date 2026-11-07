from cqa import checklist as ckl, data, submission
from cqa.evaluator import Evaluator
from cqa.llm import LLM


def test_generic_loads_and_scores_with_mock(monkeypatch):
    c = ckl.load("generic_sales_v1")
    assert [cr["id"] for cr in c["criteria"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert ckl.version_hash(c).startswith("1.0.0+")
    recs = data.load_json([{"conversation_id": f"call-{i}", "channel": "voice call", "language": "RU", "transcript": [
        {"timestamp": "00:01", "speaker": "operator", "text": "Здравствуйте, меня зовут Алина, звоню по поводу тарифа."},
        {"timestamp": "00:08", "speaker": "client", "text": "Да, слушаю."},
        {"timestamp": "00:15", "speaker": "operator", "text": "Давайте договоримся о встрече в пятницу. До свидания!"}]}
        for i in range(3)])   # synthetic: no network / DEV data needed
    ev = Evaluator(llm=LLM(provider="mock"), use_cache=False, ensemble=False)
    out, _ = ev.evaluate_many(recs, checklist_override="generic_sales_v1")
    assert len(out) == 3
    for d in out:
        assert d["checklist_id"] == "generic_sales_v1" and d["checklist_version"] == ckl.version_hash(c)
        assert d["total"] is not None
    v1 = submission.to_v1({"dialogs": out})
    assert {x["criterion_id"] for x in v1["results"][0]["criteria"]} == {
        "GREETING_AGENDA", "NEED_DISCOVERY", "PITCH_FIT", "OBJECTION_HANDLING", "NEXT_STEP"}
    errs, _ = submission.validate(v1, custom_checklist=True)
    assert errs == []

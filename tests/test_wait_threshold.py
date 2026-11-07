"""The wait threshold lives only in the checklist YAML (assumptions.long_wait_seconds)."""
import copy

from cqa import checklist as ckl, data, prompts


def test_yaml_is_single_source():
    c = ckl.load("freedom_chat_v1")
    assert c["assumptions"]["long_wait_seconds"] == 180
    assert not hasattr(data, "LONG_WAIT_S")
    assert ckl.load("freedom_call_v1")["assumptions"]["long_wait_seconds"] == 180


def test_rubric_text_is_templated_from_yaml():
    c = ckl.load("freedom_chat_v1")
    cr = next(x for x in c["criteria"] if x["id"] == "06")
    assert "{long_wait_minutes}" in cr["rules"]
    txt = prompts.criterion_block(c, cr)
    assert "3 мин." in txt and "180 с" in txt and "{long_wait" not in txt and "2 минут" not in txt
    c2 = copy.deepcopy(c)
    c2["assumptions"]["long_wait_seconds"] = 300
    txt2 = prompts.criterion_block(c2, cr)
    assert "5 мин." in txt2 and "300 с" in txt2
    callc = ckl.load("freedom_call_v1")
    assert "{long_wait" not in prompts.criterion_block(callc, next(x for x in callc["criteria"] if x["id"] == "06"))


def test_threshold_changes_facts(monkeypatch):
    from tests.test_review_hardening import _rec as mk
    r = mk([{"timestamp": "00:00", "speaker": "operator", "text": "Здравствуйте, меня зовут Ан"},
            {"timestamp": "00:10", "speaker": "client", "text": "Вопрос"},
            {"timestamp": "02:40", "speaker": "operator", "text": "Ответ"}])
    assert data.compute_facts(r)["long_waits"] == []            # 150 s < 180 s (YAML default)
    f = data.compute_facts(r, 120)
    assert len(f["long_waits"]) == 1 and "дольше 120 с" in data.format_facts(f)
    # changing the YAML value changes the default used by compute_facts
    real = ckl.load
    def patched(cid):
        c = real(cid)
        c["assumptions"]["long_wait_seconds"] = 100
        return c
    monkeypatch.setattr(ckl, "load", patched)
    f2 = data.compute_facts(r)
    assert f2["long_wait_seconds"] == 100 and len(f2["long_waits"]) == 1


def test_bad_threshold_rejected():
    import pytest
    c = copy.deepcopy(ckl.load("freedom_chat_v1"))
    c["assumptions"]["long_wait_seconds"] = -1
    with pytest.raises(ValueError):
        ckl.validate(c)
    c["assumptions"].pop("long_wait_seconds")
    with pytest.raises(ValueError):
        ckl.validate(c)   # rubric uses {long_wait_*} placeholders

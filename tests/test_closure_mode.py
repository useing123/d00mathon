import pytest

from cqa import checklist as C


def _c07(c):
    return next(cr for cr in c["criteria"] if cr["id"] == "07")


def test_default_strict_and_rendered():
    c = C.load("freedom_chat_v1")
    assert c["assumptions"]["closure_mode"] == "strict"
    txt = C.render(_c07(c)["rubric"], c)
    assert "{closure_rule}" not in txt and "ИЛИ предложил" in txt and "попрощался И" in txt
    assert "критерий 04" in _c07(c)["rules"]


def test_lenient_changes_only_07_hash():
    c = C.load("freedom_chat_v1")
    lc = C.load("freedom_chat_v1")
    lc["assumptions"]["closure_mode"] = "lenient"
    for a, b in zip(c["criteria"], lc["criteria"], strict=True):
        same = C.criterion_hash(c, a) == C.criterion_hash(lc, b)
        assert same == (a["id"] != "07"), a["id"]
    assert "ИЛИ предложил" in C.render(_c07(lc)["rubric"], lc)


def test_bad_mode_rejected():
    c = C.load("freedom_chat_v1")
    c["assumptions"]["closure_mode"] = "loose"
    with pytest.raises(ValueError):
        C.validate(c)

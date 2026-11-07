import copy

import pytest

from cqa import checklist as ckl, config


def test_bundled_checklists_valid_and_weights_sum_to_100():
    for cid in ("freedom_chat_v1", "freedom_call_v1"):
        c = ckl.load(cid)
        assert sum(cr["weight"] for cr in c["criteria"] if not cr["gross"]) == 100
        assert [cr["id"] for cr in c["criteria"]] == [f"0{i}" for i in range(1, 9)]
        assert any(cr["gross"] for cr in c["criteria"])


def test_list_checklists():
    ids = {c["id"]: c for c in ckl.list_checklists()}
    assert ids["freedom_chat_v1"]["channel"] == "chat" and ids["freedom_call_v1"]["channel"] == "call"


def test_defaults_applied():
    c = ckl.validate({"id": "x", "criteria": [{"id": 1, "name": "a"}]})
    cr = c["criteria"][0]
    assert cr["id"] == "1" and cr["type"] == "binary" and cr["allow_na"] is False
    assert c["assumptions"]["rounding"] == "half_up"


@pytest.mark.parametrize("bad", [
    {"criteria": [{"id": "1"}]},                                       # no id
    {"id": "x", "criteria": []},                                       # no criteria
    {"id": "x", "criteria": [{"id": "1"}, {"id": "1"}]},               # duplicate ids
    {"id": "x", "criteria": [{"id": "1", "type": "scale"}]},           # scale without steps
    {"id": "x", "criteria": [{"id": "1", "type": "likert"}]},          # unknown type
])
def test_invalid_checklists_rejected(bad):
    with pytest.raises(ValueError):
        ckl.validate(bad)


def test_scale_sorted():
    c = ckl.validate({"id": "x", "criteria": [{"id": "1", "type": "scale", "scale": [10, "0", 5]}]})
    assert c["criteria"][0]["scale"] == [0, 5, 10]


def test_save_rejects_bad_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKLIST_DIR", tmp_path)
    with pytest.raises(ValueError):
        ckl.save({"id": "../evil", "criteria": [{"id": "1", "name": "a"}]})
    ckl.save({"id": "ok_1", "criteria": [{"id": "1", "name": "a", "weight": 5}]})
    assert ckl.load("ok_1")["criteria"][0]["weight"] == 5


def test_channel_autoselect():
    assert ckl.pick_for_channel("call") == "freedom_call_v1"
    assert ckl.pick_for_channel("chat") == "freedom_chat_v1"


def test_kind_from_channel():
    from cqa.data import normalize
    assert normalize({"channel": "voice call", "transcript": []}, 0)["kind"] == "call"
    assert normalize({"channel": "agent chat", "transcript": []}, 0)["kind"] == "chat"
    assert normalize({"transcript": []}, 0)["kind"] == "chat"


def test_criterion_hash_tracks_text_not_weight():
    c = ckl.load("freedom_chat_v1")
    cr = c["criteria"][0]
    h = ckl.criterion_hash(c, cr)
    assert ckl.criterion_hash(c, dict(cr, weight=99)) == h           # reweight -> no LLM re-run
    assert ckl.criterion_hash(c, dict(cr, rubric=cr["rubric"] + " x")) != h
    assert ckl.criterion_hash(c, dict(cr, rules="new rule")) != h
    c2 = copy.deepcopy(c)
    c2["assumptions"]["first_reply"] = "any"
    assert ckl.criterion_hash(c2, cr) != h

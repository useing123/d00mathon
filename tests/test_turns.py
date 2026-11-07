"""1-based turn numbering: one conversion point (cqa.data.TURN_BASE / turn_pos / turn_at)."""
from cqa import data, store
from cqa.evaluator import validate_evidence

REC = data.load_json([{"conversation_id": "t", "channel": "agent chat", "transcript": [
    {"timestamp": "00:00", "speaker": "client", "text": "Где мой тариф?"},
    {"timestamp": "00:10", "speaker": "operator", "text": "Здравствуйте, меня зовут Алина."},
    {"timestamp": "00:20", "speaker": "operator", "text": "Тариф 0.1% за сделку."}]}])[0]


def test_turns_are_one_based_everywhere():
    assert data.TURN_BASE == 1
    assert [t["index"] for t in REC["transcript"]] == [1, 2, 3]
    tx = data.format_transcript(REC)
    assert tx.splitlines()[0].startswith("[1] ") and "[0]" not in tx
    assert data.compute_facts(REC)["first_operator_turn"] == 2


def test_round_trip_number_position():
    for t in REC["transcript"]:
        assert data.turn_at(REC, t["index"]) is t
        assert REC["transcript"][data.turn_pos(t["index"])] is t
    assert data.turn_at(REC, 0) is None and data.turn_at(REC, 4) is None and data.turn_pos("x") == -1


def test_quote_validation_uses_one_based_numbers():
    v, bad = validate_evidence(REC, [{"turn": 3, "quote": "Тариф 0.1%"}])
    assert v == [{"turn": 3, "quote": "Тариф 0.1%", "speaker": "operator", "internal": False}] and not bad
    v, _ = validate_evidence(REC, [{"turn": 0, "quote": "меня зовут Алина"}])   # 0 is out of range -> corrected
    assert v[0]["turn"] == 2 and v[0]["turn_corrected"]


def test_old_zero_based_runs_are_upgraded_once():
    doc = {"meta": {}, "dialogs": [{"criteria": [{"evidence": [{"turn": 0, "quote": "x"}],
                                                   "client_requests": [{"turn": 2}, {"turn": None}]}]}]}
    store.upgrade_turns(doc)
    c = doc["dialogs"][0]["criteria"][0]
    assert c["evidence"][0]["turn"] == 1 and c["client_requests"][0]["turn"] == 3 and doc["meta"]["turn_base"] == 1
    store.upgrade_turns(doc)
    assert c["evidence"][0]["turn"] == 1   # idempotent

from cqa.data import normalize
from cqa.evaluator import validate_evidence

REC = normalize({"conversation_id": "x", "channel": "agent chat", "transcript": [
    {"timestamp": "00:00", "speaker": "client", "text": "Привет, где мой «тариф»?"},
    {"timestamp": "00:10", "speaker": "operator", "text": "Здравствуйте! Меня зовут Алина. Откройте вкладку «Кабинет»."}]}, 0)


def test_valid_and_corrected_and_invalid():
    v, bad = validate_evidence(REC, [{"turn": 1, "quote": "Меня зовут Алина"},
                                     {"turn": 0, "quote": "откройте вкладку Кабинет"},
                                     {"turn": 1, "quote": "Я вам перезвоню"}])
    assert len(v) == 2 and v[1]["turn"] == 1 and v[1].get("turn_corrected")
    assert len(bad) == 1


def test_ellipsis_quote():
    v, bad = validate_evidence(REC, [{"turn": 1, "quote": "Здравствуйте! ... вкладку «Кабинет»"}])
    assert v and not bad


def test_hold_marking():
    r = normalize({"channel": "voice call", "transcript": [
        {"timestamp": "2026-11-03T15:49:13.1900670+00:00", "speaker": "operator", "text": "a"},
        {"timestamp": "2026-11-03T15:49:23.1900670+00:00", "speaker": "system", "text": "Инициирована консультация с агентом."},
        {"timestamp": "2026-11-03T15:49:33.19+00:00", "speaker": "operator", "text": "b"},
        {"timestamp": "2026-11-03T15:50:13.19+00:00", "speaker": "system", "text": "Клиент больше не находится на удержании."},
        {"timestamp": "2026-11-03T15:50:14.19+00:00", "speaker": "operator", "text": "c"}]}, 0)
    assert [t["internal"] for t in r["transcript"]] == [False, False, True, False, False]
    assert abs(r["transcript"][2]["offset"] - 20.0) < 0.01

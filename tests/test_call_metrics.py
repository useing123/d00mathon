from cqa import data
from tests.test_review_hardening import _rec


def test_call_metrics_from_iso():
    t0 = "2026-01-01T10:00:%02d+05:00"
    r = _rec([{"timestamp": t0 % 0, "speaker": "operator", "text": "Здравствуйте меня зовут Ан чем помочь"},
              {"timestamp": t0 % 6, "speaker": "client", "text": "Хочу вывести деньги со счёта сегодня пожалуйста"},
              {"timestamp": t0 % 8, "speaker": "operator", "text": "Понял"},
              {"timestamp": t0 % 20, "speaker": "client", "text": "Ок"}], channel="voice call")
    m = data.call_metrics(r)
    assert m["speaker_changes"] == 3 and m["operator_response_s"]["n"] == 1 and m["operator_response_s"]["median"] == 2.0
    assert m["overlap_rate"] == round(1 / 3, 3)        # operator started 2 s into a 7-word client turn
    assert m["operator_wpm"] == round((6 + 1) / (6 + 12) * 60, 1)


def test_no_call_metrics_for_chats():
    r = _rec([{"timestamp": "00:00", "speaker": "operator", "text": "a"}, {"timestamp": "00:05", "speaker": "client", "text": "b"}])
    assert data.call_metrics(r) is None

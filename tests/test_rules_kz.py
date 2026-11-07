import json
from pathlib import Path

from cqa import data, jev, rules_kz

ROOT = Path(__file__).resolve().parents[1]


def _rec(name="gold-02-kz-wrong-language.json"):
    return data.normalize(json.loads((ROOT / "examples/gold" / name).read_text())["dialog"], 0)


def test_detector_kz_ru_mixed_latin():
    assert rules_kz.detect_lang("Сәлеметсіз бе, шотты қалай толтыруға болады?") == "KZ"
    assert rules_kz.detect_lang("Здравствуйте, как пополнить счёт?") == "RU"
    assert rules_kz.detect_lang("Сәлеметсіз бе, мне нужно пополнить счёт, қалай жасаймын?") == "MIXED"
    assert rules_kz.detect_lang("Salemetsiz be, qalai tolyktyram, rakhmet") == "KZ"
    assert rules_kz.detect_lang("Hello, how can I top up my account?") == "EN"
    assert rules_kz.lang_of(["Сәлем, шот керек", "а комиссия есть?"]) == "MIXED"


def test_lexicons_and_gloss():
    assert rules_kz.FAREWELL.search("Сау болыңыз!") and rules_kz.gloss("Сау бол") == "goodbye"
    assert rules_kz.HELP_OFFER.search("Тағы сұрақтарыңыз бар ма?")
    assert rules_kz.INTRO.search("Менің атым Айгерім")
    assert rules_kz.RED_FLAGS.search("жеке нөмірге жазыңыз") and rules_kz.RED_FLAGS.search("переведите мне на мою карту")


def test_signals_turn_numbers_and_render():
    rec = _rec()
    sig = rules_kz.signals(rec)
    by = {s["type"]: s for s in sig["signals"]}
    assert by["greeting"]["turn"] == 5 and by["introduction"]["turn"] == 5
    assert by["farewell"]["turn"] == 9 and by["farewell"]["lang"] == "RU"
    assert sig["client_lang"] == "KZ" and sig["operator_first_lang"] == "RU"
    txt = rules_kz.format_signals(sig)
    assert "СИГНАЛЫ (парсинг, RU/KZ/EN)" in txt and "«Всего доброго» (№9, RU)" in txt


def test_flag_off_is_byte_identical(monkeypatch):
    rec = _rec()
    monkeypatch.delenv("CQA_KZ_SIGNALS", raising=False)
    off = data.format_facts(data.compute_facts(rec))
    off_state = jev.build_state(rec)
    monkeypatch.setenv("CQA_KZ_SIGNALS", "0")
    assert data.format_facts(data.compute_facts(rec)) == off and jev.build_state(rec) == off_state
    assert "СИГНАЛЫ" not in off
    monkeypatch.setenv("CQA_KZ_SIGNALS", "1")
    on = data.format_facts(data.compute_facts(rec))
    assert on.startswith(off) and "СИГНАЛЫ" in on and "СИГНАЛЫ" in jev.build_state(rec)


def test_red_flags_only_escalate():
    rec = _rec("gold-11-ru-paid-private-consult.json")
    assert rules_kz.red_flags(rec) == [5]
    assert not hasattr(rules_kz, "decide")   # rules never return verdicts

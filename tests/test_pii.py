from cqa import pii

RU = "Здравствуйте, меня зовут Алия Нурланова. Мой ИИН 900101300017, тел +7 (701) 123-45-67 или 87011234567."
KZ = "Сәлеметсіз бе, менің атым Ерлан. Карта 4400 4301 2345 6789, IBAN KZ86125KZT5004100100, пошта a.b@mail.kz"
EN = "Hi, my name is John Smith, account 12345678, date of birth 12.03.1985, contract No. 0012345"


def _kinds(t):
    return sorted({k for k, _ in pii.find(t)})


def test_roundtrip_and_kinds():
    for t in (RU, KZ, EN):
        m = pii.mask(t)
        assert pii.unmask(m) == t
        assert pii.mask(m) == m   # idempotent
    m = pii.mask(RU)
    for v in ("900101300017", "Алия Нурланова", "123-45-67", "87011234567"):
        assert v not in m
    assert _kinds(RU) == ["IIN", "NAME", "PHONE"]
    assert _kinds(KZ) == ["CARD", "EMAIL", "IBAN", "NAME"]
    assert _kinds(EN) == ["ACCOUNT", "DOB", "NAME"]


def test_deterministic_placeholders():
    assert pii.mask("тел 87011234567") == pii.mask("тел 87011234567")
    assert pii.mask(RU).count(pii.mask("87011234567")) == 1


def test_checksums_avoid_false_positives():
    assert pii.mask("сумма 900101300015 тенге") == "сумма 900101300015 тенге"   # bad IIN checksum
    assert pii.mask("код 4400 4301 2345 6781") == "код 4400 4301 2345 6781"     # bad Luhn
    assert pii.mask("Цена 1250.50, акций 300, 10.05.2024") == "Цена 1250.50, акций 300, 10.05.2024"
    assert "Freedom" in pii.mask("Меня зовут Freedom Broker бот")


def test_extra_names(monkeypatch):
    monkeypatch.setenv("CQA_PII_NAMES", "Айгерим")
    m = pii.mask("Айгерим, ваш вопрос решён")
    assert "Айгерим" not in m and pii.unmask(m).startswith("Айгерим")


def test_llm_hook_masks_outbound_and_unmasks_quotes(monkeypatch):
    from cqa import llm
    monkeypatch.setenv("CQA_PII_MASK", "1")
    seen = {}

    class Fake(llm.LLM):
        def __init__(self):
            self.provider, self.model, self.route = "openai", "x", "openai"

        def _openai(self, system, cached_user, user, tool, temperature):
            seen["out"] = system + cached_user + user
            ph = pii.mask("900101300017")
            return {"value": "YES", "evidence": [{"turn": 1, "quote": f"ИИН {ph}"}]}

    r = Fake().call("sys", RU, "q", {})
    assert "900101300017" not in seen["out"] and "Алия" not in seen["out"]
    assert r["evidence"][0]["quote"] == "ИИН 900101300017"

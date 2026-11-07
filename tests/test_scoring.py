import pytest
from cqa.scoring import compute_total
from cqa.checklist import load

CL = load("freedom_chat_v1")["criteria"]
ALL_YES = {"01": "YES", "02": "YES", "03": "YES", "04": 20, "05": "YES", "06": 10, "07": "YES", "08": "NO"}


def t(vals, **kw):
    return compute_total(CL, vals, **kw)


def test_perfect():
    assert t(ALL_YES)["total"] == 100.0


def test_all_no():
    v = {"01": "NO", "02": "NO", "03": "NO", "04": 0, "05": "NO", "06": 0, "07": "NO", "08": "NO"}
    assert t(v)["total"] == 0.0


def test_na_redistribution():
    v = dict(ALL_YES, **{"02": "NA", "03": "NA", "07": "NA"})
    assert t(v)["total"] == 100.0
    v["01"] = "NO"  # 10 of remaining 60 lost
    assert t(v)["total"] == pytest.approx(83.33)


def test_scale_proportional():
    v = dict(ALL_YES, **{"04": 15})  # 20*15/20=15 -> 95
    assert t(v)["total"] == 95.0
    v = dict(ALL_YES, **{"06": 7})  # 15*0.7=10.5 -> 95.5
    assert t(v)["total"] == 95.5


def test_half_up_rounding():
    # weights 40 (scale 0..?) construct 50.625: earned/max = 0.50625
    crit = [{"id": "a", "type": "scale", "scale": [0, 50625], "weight": 100}]
    r = compute_total(crit, {"a": 50625 // 2 * 0 + 50625}, decimals=2)
    assert r["total"] == 100.0
    crit = [{"id": "a", "type": "binary", "weight": 50.625}, {"id": "b", "type": "binary", "weight": 49.375}]
    assert compute_total(crit, {"a": "YES", "b": "NO"})["total"] == 50.63  # float round() gives 50.62
    assert compute_total(crit, {"a": "YES", "b": "NO"}, rounding="bankers")["total"] == 50.62


def test_all_na_is_null():
    crit = [{"id": "a", "type": "binary", "weight": 10, "allow_na": True}, {"id": "g", "type": "binary", "weight": 0, "gross": True}]
    r = compute_total(crit, {"a": "NA", "g": "YES"})
    assert r["total"] is None and r["gross_error"] is True


def test_gross_not_scored():
    v = dict(ALL_YES, **{"08": "YES"})
    r = t(v)
    assert r["total"] == 100.0 and r["gross_error"] is True


def test_missing_value_treated_as_na():
    v = dict(ALL_YES)
    del v["04"]
    assert t(v)["total"] == 100.0


def test_invalid_values():
    with pytest.raises(ValueError):
        t(dict(ALL_YES, **{"04": 12}))
    with pytest.raises(ValueError):
        t(dict(ALL_YES, **{"01": "MAYBE"}))


def test_rounding_two_decimals():
    v = dict(ALL_YES, **{"01": "NO", "07": "NO"})  # 85
    assert t(v)["total"] == 85.0
    v = dict(ALL_YES, **{"02": "NA", "01": "NO"})  # 75/85 = 88.235.. -> 88.24
    assert t(v)["total"] == 88.24

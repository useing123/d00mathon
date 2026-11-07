"""Scoring edge cases (see docs/SCORING.md)."""
import pytest

from cqa.scoring import compute_total, criterion_points

B = {"id": "b", "type": "binary", "weight": 10}
S = {"id": "s", "type": "scale", "scale": [0, 4, 7, 10], "weight": 15}


def test_scale_step_points():
    assert criterion_points(S, 7)[0] == pytest.approx(10.5)
    assert criterion_points(S, "4")[0] == 6


def test_zero_weight_and_gross_skip():
    assert criterion_points(dict(B, weight=0), "YES") is None
    assert criterion_points(dict(B, gross=True), "YES") is None


def test_na_excluded_from_denominator():
    # 01=NO(10) with 02,03,07 NA -> 50/60 = 83.33
    crit = [dict(B, id="1", weight=10), dict(B, id="2", weight=50), dict(B, id="3", weight=40)]
    assert compute_total(crit, {"1": "NO", "2": "YES", "3": "NA"})["total"] == pytest.approx(83.33)


def test_none_value_like_na():
    assert compute_total([B, dict(B, id="c")], {"b": "YES", "c": None})["total"] == 100.0


def test_decimals_configurable():
    crit = [dict(B, id="a", weight=1), dict(B, id="c", weight=2)]
    assert compute_total(crit, {"a": "YES", "c": "NO"}, decimals=0)["total"] == 33.0
    assert compute_total(crit, {"a": "YES", "c": "NO"}, decimals=3)["total"] == 33.333


def test_gross_flag_independent_of_total():
    g = {"id": "g", "type": "binary", "weight": 0, "gross": True}
    r = compute_total([B, g], {"b": "NO", "g": "NO"})
    assert r == {"total": 0.0, "gross_error": False, "max_weight_applicable": 10.0}

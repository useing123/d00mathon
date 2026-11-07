"""Advanced router layer (cqa/routing_policy.py): pure offline logic, no network."""
import json

from cqa import routing_policy as rp


def test_aurc_perfect_and_inverted_ranking():
    good = [(0.9, True), (0.8, True), (0.2, False)]
    bad = [(0.9, False), (0.8, True), (0.2, True)]
    assert rp.aurc(good) == 1 / 3 * (0 + 0 + 1 / 3)
    assert rp.aurc(bad) > rp.aurc(good)
    assert rp.risk_coverage(good)[-1] == (1.0, 1 / 3)


def test_ties_enter_together():
    assert rp.risk_coverage([(1.0, True), (1.0, False)]) == [(1.0, 0.5)]


def test_ece_zero_when_calibrated_and_temperature_reduces_overconfidence():
    assert rp.ece([(1.0, True)] * 5) == 0
    pairs = [(0.99, True)] * 6 + [(0.99, False)] * 4        # 60 % accurate at p = 0.99
    t = rp.fit_temperature(pairs)
    assert t > 1 and rp.ece([(rp.temp_scale(p, t), y) for p, y in pairs]) < rp.ece(pairs)


def test_isotonic_monotone():
    steps = rp.isotonic_fit([(0.1, 1), (0.2, 0), (0.9, 1), (0.95, 1)])
    vals = [v for _, v in steps]
    assert vals == sorted(vals)


def _it(**kw):
    base = {"cid": "d", "k": "01", "exp": "YES", "jev_v": "YES", "jev_p": 0.95, "haiku_v": "NO", "sonnet_v": "YES"}
    return {**base, **kw}


def test_route_item_threshold_s2_and_fail_closed():
    r = {"theta": 0.9, "s2": "none"}
    assert rp.route_item(_it(), r)[1] == 1
    assert rp.route_item(_it(jev_p=0.5), r)[:2] == ("NO", 2)
    assert rp.route_item(_it(jev_p=0.5), {"theta": 0.9, "s2": "disagree"})[:2] == ("YES", 3)
    assert rp.route_item(_it(jev_v=None), {"theta": 0.0, "s2": "none"})[1] == 2           # tier-1 error
    assert "gross_yes" in rp.route_item(_it(k="08"), {"theta": 0.0, "s2": "none"})[2]      # 08 = YES
    assert "tier1_disagree" in rp.route_item(_it(qwen_v="NO"), {"theta": 0, "s2": "none", "agree": True})[2]


def test_pareto_and_budget():
    items = [_it(cid=str(i), jev_p=0.5, jev_v="NO", haiku_v="YES") for i in range(4)]
    pr = {"tier1_dialog": 0.001, "haiku_call": 0.001, "sonnet_call": 0.01}
    front = rp.pareto_policies(items, None, pr)
    costs = [c for c, _, _ in front]
    assert costs == sorted(costs) and front[0][1] == 0 and front[-1][1] == 4
    assert rp.best_under_budget(front, 0.0015)[1] == 0
    assert rp.best_under_budget(front, 0.0025)[1] == 4


def test_live_patch_hook(tmp_path):
    from cqa import router
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"policy": {"01": {"theta": 0.9, "s2": "disagree"}, "04": {"theta": 0.0, "s2": "none"}},
                             "calibration": None}))
    g = {"tier1_reasons": router.tier1_reasons, "tier2_reasons": router.tier2_reasons}
    rp.patch(g, str(p))
    s = router.settings(first="jev")
    assert g["tier1_reasons"]({"id": "04"}, {"value": 10, "confidence": 0.5}, s) == []     # hard rule replaced
    assert g["tier1_reasons"]({"id": "01"}, {"value": "YES", "confidence": 0.5}, s) == ["low_conf_calibrated"]
    assert g["tier2_reasons"]({"value": "YES"}, {"value": "NO", "confidence": 0.99}, s) == ["tier1_haiku_disagree"]
    assert g["tier1_reasons"]({"id": "04"}, {"error": "x"}, s) == ["local_error"]
    assert router.tier1_reasons is not g["tier1_reasons"]       # real module untouched without CQA_ROUTER_POLICY


def test_registry_entries_missing_file(tmp_path):
    assert rp.registry_entries(tmp_path) == []

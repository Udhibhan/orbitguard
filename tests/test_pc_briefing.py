import math

import numpy as np

from orbitguard import briefing, decisions, pc
from orbitguard.data import SIGMA_SENTINEL


def test_pc_formula_matches_closed_form():
    d, s, hbr = 50.0, 100.0, 10.0
    expected = hbr ** 2 / (2 * s ** 2) * math.exp(-d ** 2 / (2 * s ** 2))
    assert math.isclose(10 ** pc.log10_pc(d, s, hbr), expected, rel_tol=1e-9)


def test_pc_sanity_monotonic_and_bounded():
    lp = pc.log10_pc(np.array([10, 100, 1000, 10000.0]), 200.0)
    assert np.all(np.diff(lp) < 0)                  # farther -> less likely
    assert np.all(lp <= 0) and np.all(lp >= -30)    # probability, floored like the dataset
    assert pc.log10_pc(1e6, 1.0) == -30.0           # no underflow to -inf


def test_pc_max_is_upper_bound_over_sigma():
    d = 500.0
    sigmas = np.logspace(0, 5, 2000)
    assert pc.log10_pc(d, sigmas).max() <= pc.log10_pc_max(d) + 1e-9
    assert math.isclose(pc.log10_pc(d, d / math.sqrt(2)), pc.log10_pc_max(d), rel_tol=1e-9)


def test_sentinel_sigma_gives_nan_pc():
    s = pc.combined_sigma([SIGMA_SENTINEL, 30.0], [5.0, 40.0])
    assert np.isnan(s[0]) and math.isclose(s[1], 50.0)
    assert np.isnan(pc.log10_pc(100.0, s[0]))


def _facts():
    return {"event_id": 7, "current_tier": "Elevated", "forecast_tier": "Critical", "latest_pc": "2.50e-05",
            "latest_miss_km": 0.812, "decision_time_to_tca_days": 2.31, "risk_trend_orders_per_day": 0.4,
            "current_rules_fired": ["R3 Pc rule: 1e-6 <= Pc = 2.50e-05 < 1e-4 -> Elevated"],
            "forecast_rules_fired": [], "cdms_before_cutoff": 9, "miss_trend_km_per_day": -0.12, "model": "m",
            "forecast_pc_at_tca": "2.50e-05", "volatile": False, "volatile_reasons": "none",
            "suggested_action": "Act", "suggestion_rule": "A1", "p_critical_pct": 12.5}


def test_verifier_accepts_fact_numbers_and_rejects_invented():
    f = _facts()
    ok, _ = briefing.verify("Miss is 0.812 km (about 0.81 km) at 2.31 days, Pc 2.5e-05; Critical forecast.", "Act", f)
    assert ok
    ok, probs = briefing.verify("Pc will reach 3.1e-4 within 6 hours.", "Act", f)
    assert not ok and any("3.1e-4" in p for p in probs)


def test_verifier_rejects_tier_and_action_changes():
    f = _facts()
    assert not briefing.verify("This event is Safe.", "Act", f)[0]
    assert not briefing.verify("Elevated now, Critical forecast.", "Clear", f)[0]


def test_template_always_verifies_and_no_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = _facts()
    out = briefing.brief(f)
    assert out["source"] == "template" and out["recommendation"] == "Act"
    assert briefing.verify(out["text"], out["recommendation"], f)[0]


def test_suggest_action_rules():
    assert briefing.suggest_action("Critical", "Safe", False, False)[0] == "Act"
    assert briefing.suggest_action("Safe", "Safe", True, False)[0] == "Watch"
    assert briefing.suggest_action("Safe", "Safe", False, True)[0] == "Clear"
    assert briefing.suggest_action("Safe", "Safe", False, False)[0] == "Watch"


def test_decision_log_appends(tmp_path):
    p = tmp_path / "d.jsonl"
    decisions.log_decision({"event_id": 1, "decision": "Watch"}, path=p)
    decisions.log_decision({"event_id": 1, "decision": "Act"}, path=p)
    df = decisions.load_decisions(path=p)
    assert df.decision.tolist() == ["Watch", "Act"] and df.timestamp_utc.notna().all()

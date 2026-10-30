import json

import numpy as np
import pandas as pd

from orbitguard import data, pcnet
from orbitguard.models import GBMTierForecaster
from orbitguard.tiers import CRITICAL, ELEVATED, SAFE


def _cdms():
    return pd.DataFrame({
        "event_id": [1, 1, 1, 2], "time_to_tca": [5.0, 3.0, 2.1, 4.0],
        "miss_distance": [4000.0, 2000.0, 800.0, 30000.0], "relative_speed": [10000.0] * 4,
        "relative_position_r": [10.0, 20.0, 5.0, 100.0], "relative_position_t": [3000.0, 1500.0, 600.0, 20000.0],
        "relative_position_n": [2600.0, 1300.0, 520.0, 22000.0], "c_object_type": ["DEBRIS", "DEBRIS", "DEBRIS", "PAYLOAD"],
        "c_sigma_r": [50.0, 40.0, 6.378136e7, 30.0], "t_sigma_r": [5.0, 4.0, 3.0, 2.0],
        "max_risk_estimate": [-5.0, -4.5, -3.0, -8.0], "F10": [80.0] * 4, "AP": [5.0] * 4,
        "risk": [-9.0, -7.0, -5.0, -30.0], "cdm_row": [0, 1, 2, 3],
    })


def test_pc_engine_never_reads_risk():
    """Changing every `risk` value must not change a single feature (the target cannot leak in)."""
    c = _cdms()
    a = pcnet.row_features(c)
    b = pcnet.row_features(c.assign(risk=np.random.default_rng(0).uniform(-30, 0, len(c))))
    assert not any("risk" == col for col in a.columns)
    pd.testing.assert_frame_equal(a, b)


def test_pc_engine_history_is_causal():
    """Features of a CDM only use that CDM and earlier ones (larger time_to_tca) of the same event."""
    c = _cdms()
    full = pcnet.row_features(c)
    first_two = pcnet.row_features(c.iloc[:2])
    pd.testing.assert_frame_equal(full.iloc[:2], first_two)


def test_sigma_sentinel_is_missing():
    f = pcnet.row_features(_cdms())
    assert np.isnan(f.loc[2, "log_c_sigma_r"])


def test_forecast_tier_rule_thresholds():
    m = GBMTierForecaster()
    m.tau_crit, m.tau_elev = 0.15, 0.20
    pred = pd.DataFrame({"p_safe": [0.9, 0.7, 0.1], "p_elevated": [0.05, 0.25, 0.7], "p_critical": [0.05, 0.05, 0.2]})
    assert list(m.forecast_tiers(pred, None)) == [SAFE, ELEVATED, CRITICAL]


def test_web_export_is_strict_json():
    p = data.ROOT / "web" / "data" / "summary.json"
    if p.exists():
        json.loads(p.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

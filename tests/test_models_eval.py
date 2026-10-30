import numpy as np
import pandas as pd

from orbitguard.evaluate import esa_score, tune_theta
from orbitguard.models import OUTPUT_COLUMNS, ConstantModel, LatestRiskModel, validate_output


def _events():
    return pd.DataFrame({
        "event_id": [1, 2, 3, 4], "latest_risk": [-30.0, -7.0, -5.5, -3.0],
        "latest_miss_distance": [20000.0, 8000.0, 3000.0, 500.0], "volatile": [False, False, True, False],
        "final_tier": ["Safe", "Safe", "Elevated", "Critical"],
    })


def test_lrp_clips_and_abstains():
    ev = _events()
    out = LatestRiskModel(min_count=1).fit(ev).predict(ev)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.pred_risk.tolist() == [-6.001, -6.001, -5.5, -3.0]
    assert out.abstain.tolist() == [False, False, True, False]
    assert np.allclose(out[["p_safe", "p_elevated", "p_critical"]].sum(axis=1), 1)


def test_constant_baseline():
    ev = _events()
    out = ConstantModel().fit(ev).predict(ev)
    assert (out.pred_risk == -5).all() and np.allclose(out.p_safe, 0.5)


def test_esa_score_perfect_and_constant():
    true = np.array([-30, -10, -5, -4.5])
    perfect = esa_score(true, true)
    assert perfect["mse_hr"] == 0 and perfect["f2"] == 1 and perfect["L"] == 0
    const = esa_score(np.full(4, -5.0), true)
    assert const["recall_hr"] == 1 and const["precision_hr"] == 0.5


def test_validate_output_fills_probabilities_from_rules():
    ev = _events()
    out = validate_output(pd.DataFrame({"event_id": [1, 2, 3, 4], "pred_risk": [-8, -5, -7, -8]}), ev, "t")
    assert out.p_safe.tolist() == [1, 0, 0, 0]       # event 3: 3 km miss -> Elevated; event 4: 500 m -> Critical
    assert out.p_critical.tolist() == [0, 0, 0, 1]


def test_theta_keeps_critical_recall():
    pred = pd.DataFrame({"p_safe": [0.99, 0.95, 0.9, 0.5], "abstain": [False] * 4})
    theta = tune_theta(pred, np.array(["Safe", "Safe", "Critical", "Critical"]), target=0.99)
    assert theta == 0.95


def test_teammate_module_plugin_cannot_see_targets(tmp_path, monkeypatch):
    from orbitguard import models
    pkg = tmp_path / "teammate"
    pkg.mkdir()
    (pkg / "model.py").write_text(
        "MODEL_NAME = 'stub'\n"
        "def predict(events):\n"
        "    assert not any(c.startswith('final_') for c in events.columns), 'target leaked'\n"
        "    out = events[['event_id']].copy(); out['pred_risk'] = -7.0; return out\n")
    monkeypatch.setattr(models, "TEAMMATE_PKG", pkg)
    ev = _events().assign(final_risk=-5.0, final_tier="Safe")
    out = models.load_teammate().predict(ev)
    assert out.model_name.iloc[0] == "stub" and list(out.columns) == OUTPUT_COLUMNS

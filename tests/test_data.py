import numpy as np
import pandas as pd
import pytest

from orbitguard import data
from orbitguard.data import ALLOWED_COLUMNS, FEATURE_COLUMNS, build_events, load_cdms


def _synthetic_cdms():
    rows = []
    rng = np.random.default_rng(1)
    for eid in range(40):
        for t in [6.0, 4.5, 3.0, 2.0, 1.5, 0.5]:
            rows.append({"event_id": eid, "time_to_tca": t, "miss_distance": rng.uniform(200, 20000),
                         "relative_speed": 12000.0, "relative_position_r": 10.0, "relative_position_t": 100.0,
                         "relative_position_n": 100.0, "c_object_type": "DEBRIS", "c_sigma_r": 50.0,
                         "t_sigma_r": 5.0, "max_risk_estimate": -5.0, "F10": 70.0, "AP": 5.0,
                         "risk": rng.uniform(-12, -3)})
    df = pd.DataFrame(rows)
    df["cdm_row"] = np.arange(len(df))
    return df


def test_whitelist_drops_forbidden_columns(tmp_path):
    df = _synthetic_cdms().drop(columns="cdm_row")
    df["mission_id"] = 1
    df["mahalanobis_distance"] = 3.0
    df["t_sigma_t"] = 9.0
    df.to_csv(tmp_path / "x.csv", index=False)
    out = load_cdms("x", raw_dir=tmp_path)
    assert set(out.columns) == set(ALLOWED_COLUMNS) | {"cdm_row"}
    assert "mission_id" not in out and "t_sigma_t" not in out


def test_features_use_only_pre_cutoff_cdms():
    cdms = _synthetic_cdms()
    ev1, _ = build_events(cdms, with_target=True)
    tampered = cdms.copy()
    post = tampered["time_to_tca"] < data.CUTOFF_DAYS
    tampered.loc[post, ["risk", "miss_distance", "max_risk_estimate", "c_sigma_r", "F10"]] = [-1.0, 1.0, -1.0, 1.0, 999]
    ev2, _ = build_events(tampered, with_target=True)
    feats = [c for c in FEATURE_COLUMNS if c in ev1]
    pd.testing.assert_frame_equal(ev1[feats], ev2[feats])
    # ...while the target does change, proving the tampered rows were the post-cutoff ones
    assert not np.allclose(ev1["final_risk"], ev2["final_risk"])
    assert (ev1["latest_time_to_tca"] >= data.CUTOFF_DAYS).all()


def test_esa_filter_and_no_pre_cutoff_exclusion():
    cdms = _synthetic_cdms()
    cdms = cdms[~((cdms.event_id == 0) & (cdms.time_to_tca >= 2))]         # no pre-cutoff CDM
    cdms = cdms[~((cdms.event_id == 1) & (cdms.time_to_tca < 2))]          # final CDM too early
    ev, rep = build_events(cdms, with_target=True)
    assert rep["excluded_no_cdm_before_cutoff"] == 1
    assert rep["excluded_final_cdm_not_within_1_day"] == 1
    assert 0 not in ev.event_id.values and 1 not in ev.event_id.values
    assert (ev["final_time_to_tca"] <= data.FINAL_WITHIN_DAYS).all()


def test_sigma_sentinel_is_missing():
    s = pd.Series([5.0, data.SIGMA_SENTINEL, 63781363.00000001])
    assert data.clean_sigma(s).isna().tolist() == [False, True, True]


@pytest.fixture(scope="module")
def real_events():
    if not (data.PROCESSED / "events_all.pkl").exists():
        pytest.skip("run `make data` first")
    return data.load_events()


def test_no_event_in_two_splits(real_events):
    labelled = real_events[real_events.split != "test"]
    assert labelled["event_id"].is_unique
    tr = set(labelled[labelled.split == "train"].event_id)
    va = set(labelled[labelled.split == "val"].event_id)
    assert tr and va and not (tr & va)


def test_real_features_pre_cutoff(real_events):
    assert (real_events["latest_time_to_tca"] >= data.CUTOFF_DAYS).all()
    labelled = real_events[real_events.split != "test"]
    assert (labelled["final_time_to_tca"] <= data.FINAL_WITHIN_DAYS).all()
    assert (labelled["latest_cdm_row"] != labelled["final_cdm_row"]).all()

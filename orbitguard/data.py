"""Data loading with a strict column whitelist, and the event-level dataset builder.

Units (verified from distributions, see scripts/inspect_data.py):
  time_to_tca days; miss_distance, relative_position_*, *_sigma_r metres;
  relative_speed m/s; risk / max_risk_estimate = log10(Pc), floored at -30.

Framing: an analyst decides ~2 days before TCA. Features use ONLY CDMs with
time_to_tca >= CUTOFF_DAYS. The target is the risk of the event's final CDM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tiers import classify_array

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ALLOWED_COLUMNS = [
    "event_id", "time_to_tca", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP",
    "risk",  # target (log10 Pc)
]

CUTOFF_DAYS = 2.0          # decision point: features only from CDMs at/before 2 days out
FINAL_WITHIN_DAYS = 1.0    # ESA filter: last CDM must be <= 1 day from TCA to count as "final"
RISK_FLOOR = -30.0
SIGMA_SENTINEL = 6.378136e7  # ~10 Earth radii in metres = "no covariance" placeholder
HIGH_RISK_LOG = -6.0
VOLATILITY_CLIP = -10.0    # Pc below 1e-10 is "zero" for trend purposes; avoids -30 floor noise

# Volatility rule (pre-cutoff only). Documented, fixed thresholds — routed to Analyst review.
VOL_STD = 1.0        # std of clipped risk over pre-cutoff CDMs (orders of magnitude)
VOL_RISE = 1.0       # clipped risk rose by > 1 order of magnitude from first to latest CDM
VOL_MISS_DROP = 0.5  # latest miss fell below 50% of the first pre-cutoff miss


def load_cdms(name: str, raw_dir: Path = RAW) -> pd.DataFrame:
    """Load one CSV keeping only whitelisted columns. Adds `cdm_row` = 0-based data row in the file."""
    path = raw_dir / f"{name}.csv"
    df = pd.read_csv(path, usecols=lambda c: c in ALLOWED_COLUMNS)
    extra = set(df.columns) - set(ALLOWED_COLUMNS)
    assert not extra, f"non-whitelisted columns leaked: {extra}"
    missing = set(ALLOWED_COLUMNS) - set(df.columns)
    assert not missing, f"expected columns missing: {missing}"
    df = df[ALLOWED_COLUMNS].copy()
    df["cdm_row"] = np.arange(len(df))
    df["c_object_type"] = df["c_object_type"].fillna("UNKNOWN")
    return df


def load_cdms_cached(name: str) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / f"{name}_cdms.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = load_cdms(name)
    df.to_pickle(cache)
    return df


def clean_sigma(s: pd.Series) -> pd.Series:
    """Replace the 6.378e7 m 'no covariance' sentinel with NaN."""
    return s.where(s < SIGMA_SENTINEL * 0.999)


def _slope(g: pd.DataFrame, col: str) -> float:
    """Change of `col` per day moving TOWARD TCA (positive = increasing as TCA approaches)."""
    if len(g) < 2:
        return 0.0
    t = g["time_to_tca"].to_numpy()
    y = g[col].to_numpy()
    if np.ptp(t) == 0:
        return 0.0
    return float(-np.polyfit(t, y, 1)[0])


def build_events(cdms: pd.DataFrame, with_target: bool = True,
                 cutoff: float = CUTOFF_DAYS, final_within: float = FINAL_WITHIN_DAYS):
    """Build one row per event from pre-cutoff CDMs.

    Returns (events_df, report) where report counts exclusions.
    With with_target=True, applies the ESA filter (>=1 CDM at >= cutoff AND final CDM <= final_within days)
    and attaches final_* targets from the last CDM.
    """
    d = cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).copy()
    d["risk_clip"] = d["risk"].clip(lower=VOLATILITY_CLIP)
    pre = d[d["time_to_tca"] >= cutoff]
    all_ids = d["event_id"].unique()
    report = {"events_total": int(len(all_ids))}
    no_pre = np.setdiff1d(all_ids, pre["event_id"].unique())
    report["excluded_no_cdm_before_cutoff"] = int(len(no_pre))

    g = pre.groupby("event_id", sort=True)
    latest = g.tail(1).set_index("event_id")
    first = g.head(1).set_index("event_id")
    ev = pd.DataFrame(index=latest.index)
    ev["n_cdms_pre"] = g.size()
    for c in ["time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
              "relative_position_r", "relative_position_t", "relative_position_n",
              "c_sigma_r", "t_sigma_r", "F10", "AP", "c_object_type", "cdm_row"]:
        ev[f"latest_{c}"] = latest[c]
    ev["latest_c_sigma_r"] = clean_sigma(ev["latest_c_sigma_r"])
    ev["latest_t_sigma_r"] = clean_sigma(ev["latest_t_sigma_r"])
    ev["first_risk"] = first["risk"]
    ev["first_miss_distance"] = first["miss_distance"]
    ev["risk_mean"] = g["risk_clip"].mean()
    ev["risk_std"] = g["risk_clip"].std().fillna(0.0)
    ev["risk_max"] = g["risk"].max()
    ev["max_risk_estimate_max"] = g["max_risk_estimate"].max()
    ev["miss_min"] = g["miss_distance"].min()
    ev["risk_slope"] = g.apply(_slope, col="risk_clip", include_groups=False)
    ev["miss_slope"] = g.apply(_slope, col="miss_distance", include_groups=False)
    ev["risk_rise"] = latest["risk_clip"] - first["risk_clip"]
    ev["miss_ratio"] = latest["miss_distance"] / first["miss_distance"]
    ev["current_tier"] = classify_array(ev["latest_risk"], ev["latest_miss_distance"])

    vol_std = ev["risk_std"] > VOL_STD
    vol_rise = ev["risk_rise"] > VOL_RISE
    vol_miss = ev["miss_ratio"] < VOL_MISS_DROP
    ev["volatile"] = vol_std | vol_rise | vol_miss
    ev["volatile_reasons"] = [
        "; ".join(r for r, f in [(f"risk std {s:.2f} > {VOL_STD}", a),
                                 (f"risk rose {u:.2f} > {VOL_RISE} orders", b),
                                 (f"miss fell to {m:.0%} of first", c)] if f)
        for s, u, m, a, b, c in zip(ev["risk_std"], ev["risk_rise"], ev["miss_ratio"], vol_std, vol_rise, vol_miss)
    ]

    if with_target:
        last = d.groupby("event_id").tail(1).set_index("event_id").loc[ev.index]
        ev["final_risk"] = last["risk"]
        ev["final_miss_distance"] = last["miss_distance"]
        ev["final_time_to_tca"] = last["time_to_tca"]
        ev["final_cdm_row"] = last["cdm_row"]
        ev["final_tier"] = classify_array(ev["final_risk"], ev["final_miss_distance"])
        ev["high_risk"] = ev["final_risk"] >= HIGH_RISK_LOG
        not_final = ev["final_time_to_tca"] > final_within
        report["excluded_final_cdm_not_within_1_day"] = int(not_final.sum())
        ev = ev[~not_final]
    report["events_kept"] = int(len(ev))
    return ev.reset_index(), report


FEATURE_COLUMNS = [
    "n_cdms_pre", "latest_time_to_tca", "latest_risk", "latest_max_risk_estimate",
    "latest_miss_distance", "latest_relative_speed", "latest_relative_position_r",
    "latest_relative_position_t", "latest_relative_position_n", "latest_c_sigma_r",
    "latest_t_sigma_r", "latest_F10", "latest_AP", "latest_c_object_type",
    "first_risk", "first_miss_distance", "risk_mean", "risk_std", "risk_max",
    "max_risk_estimate_max", "miss_min", "risk_slope", "miss_slope", "risk_rise",
    "miss_ratio", "current_tier", "volatile",
]
TARGET_COLUMNS = ["final_risk", "final_miss_distance", "final_time_to_tca",
                  "final_cdm_row", "final_tier", "high_risk"]


def split_events(events: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> pd.Series:
    """Event-level split, stratified on final tier. Returns a Series of 'train'/'val' indexed like events."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=events.index)
    # stratify on tier + high-risk so the 66 high-risk events are spread across both splits
    strata = events["final_tier"].astype(str) + "_" + events["high_risk"].astype(str)
    for _, idx in events.groupby(strata).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        split.loc[idx[:n_val]] = "val"
    return split


def build_all(force: bool = False) -> dict:
    """Build and cache train/val/test event tables. Returns a report dict."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "events_all.pkl"
    if out.exists() and not force:
        return pd.read_pickle(PROCESSED / "build_report.pkl")
    train_cdms = load_cdms_cached("train_data")
    test_cdms = load_cdms_cached("test_data")
    ev, rep_train = build_events(train_cdms, with_target=True)
    ev["split"] = split_events(ev)
    ev_test, rep_test = build_events(test_cdms, with_target=False)
    ev_test["split"] = "test"
    all_ev = pd.concat([ev, ev_test], ignore_index=True)
    all_ev.to_pickle(out)
    for s in ["train", "val", "test"]:
        all_ev[all_ev["split"] == s].to_csv(PROCESSED / f"events_{s}.csv", index=False)
    report = {"train_data": rep_train, "test_data": rep_test,
              "n_train": int((ev["split"] == "train").sum()), "n_val": int((ev["split"] == "val").sum()),
              "n_test": int(len(ev_test))}
    pd.to_pickle(report, PROCESSED / "build_report.pkl")
    return report


def load_events(split: str | None = None) -> pd.DataFrame:
    ev = pd.read_pickle(PROCESSED / "events_all.pkl")
    return ev if split is None else ev[ev["split"] == split].reset_index(drop=True)


def load_event_cdms(dataset: str, event_id: int) -> pd.DataFrame:
    cdms = load_cdms_cached(dataset)
    return cdms[cdms["event_id"] == event_id].sort_values("time_to_tca", ascending=False)


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(force=True), indent=2))

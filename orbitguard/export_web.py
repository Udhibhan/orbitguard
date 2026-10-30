"""Export everything the static website needs into web/data/*.json.

The site has no Python at runtime: every number it shows is computed here (or in the reports this reads).
  summary.json      metrics, Pc Engine scores, 24 h warnings, insights, failure gallery
  events_{split}.json   one row per event (val = hindsight available, test = live view at 2 days)
  cdms_{split}.json     every CDM of those events + the Pc Engine's per-row estimate

Val rows are scored by a Pc Engine fitted on train-split rows only; test rows by the engine fitted on all
train rows (the one used for submissions/). Run after `python run.py all`.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import data, insights, pcnet
from .evaluate import tune_theta
from .models import GBMTierForecaster
from .tiers import CRITICAL, ELEVATED, SAFE, TIERS, classify_array

WEB = data.ROOT / "web" / "data"
REPORTS = data.ROOT / "reports"
TIER_IDX = {t: i for i, t in enumerate(TIERS)}
OBJ = pcnet.OBJ_TYPES


def r(x, n=3):
    """Round, mapping NaN/inf to None so the JSON stays valid."""
    x = np.asarray(x, float)
    out = np.round(x, n).astype(object)
    out[~np.isfinite(x)] = None
    return out


def events_table(ev: pd.DataFrame, pred: pd.DataFrame, theta: float, hindsight: bool, model) -> dict:
    p = pred.set_index("event_id").loc[ev["event_id"]].reset_index()
    fc = model.forecast_tiers(p, ev)
    clear = (p["p_safe"].to_numpy() >= theta) & ~p["abstain"].to_numpy(bool)
    queue = np.where(p["abstain"], 1, np.where(clear, 2, 0))  # 0 queue, 1 analyst review, 2 auto-clear
    cols = {
        "id": ev["event_id"].astype(int), "cur": ev["current_tier"].map(TIER_IDX), "fc": pd.Series(fc).map(TIER_IDX),
        "ps": r(p["p_safe"], 4), "pe": r(p["p_elevated"], 4), "pc": r(p["p_critical"], 4),
        # LRP clips low risk to -6.001 for the ESA metric; the site shows the unclipped latest-risk carry-forward
        "pred": r(np.where(np.isclose(p["pred_risk"], -6.001), ev["latest_risk"], p["pred_risk"])), "risk": r(ev["latest_risk"]), "miss": r(ev["latest_miss_distance"], 1),
        "speed": r(ev["latest_relative_speed"], 1), "obj": ev["latest_c_object_type"].map(lambda o: OBJ.index(o) if o in OBJ else 3),
        "t": r(ev["latest_time_to_tca"]), "n": ev["n_cdms_pre"].astype(int), "q": queue,
        "rr": r(ev["latest_relative_position_r"], 1), "rt": r(ev["latest_relative_position_t"], 1),
        "rn": r(ev["latest_relative_position_n"], 1), "cs": r(ev["latest_c_sigma_r"], 2), "ts": r(ev["latest_t_sigma_r"], 2),
        "mre": r(ev["latest_max_risk_estimate"]), "f10": r(ev["latest_F10"], 1), "ap": r(ev["latest_AP"], 1),
        "rstd": r(ev["risk_std"]), "rrise": r(ev["risk_rise"]), "mratio": r(ev["miss_ratio"]),
        "vol": ev["volatile_reasons"].fillna(""), "row": ev["latest_cdm_row"].astype(int),
    }
    if hindsight:
        cols |= {"ft": ev["final_tier"].map(TIER_IDX), "frisk": r(ev["final_risk"]),
                 "fmiss": r(ev["final_miss_distance"], 1), "ftt": r(ev["final_time_to_tca"]),
                 "frow": ev["final_cdm_row"].astype(int)}
    names = list(cols)
    arrs = [np.asarray(cols[k], dtype=object) for k in names]
    return {"cols": names, "rows": [list(t) for t in zip(*arrs)]}


def cdm_table(cdms: pd.DataFrame, ids, engine: pcnet.PcEngine) -> dict:
    d = cdms[cdms["event_id"].isin(ids)].sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    p = engine.predict(d)
    cols = ["t", "risk", "est", "pcrit", "etier", "miss", "r", "tt", "n", "speed", "cs", "ts", "mre", "row"]
    vals = [r(d["time_to_tca"], 4), r(d["risk"]), r(p["pc_log"]), r(p["p_critical"]),
            p["tier"].map(TIER_IDX).to_numpy(), r(d["miss_distance"], 1), r(d["relative_position_r"], 1),
            r(d["relative_position_t"], 1), r(d["relative_position_n"], 1), r(d["relative_speed"], 1),
            r(data.clean_sigma(d["c_sigma_r"]), 2), r(data.clean_sigma(d["t_sigma_r"]), 2),
            r(d["max_risk_estimate"]), d["cdm_row"].to_numpy()]
    by_event: dict[str, list] = {}
    for i, eid in enumerate(d["event_id"].to_numpy()):
        by_event.setdefault(str(int(eid)), []).append([v[i].item() if hasattr(v[i], "item") else v[i] for v in vals])
    return {"cols": cols, "events": by_event}


def warnings_24h(cdms: pd.DataFrame) -> dict:
    """Retrospective 24 h frame on train CDMs: per event, its latest CDM inside 24 h of TCA."""
    w = insights.latest_within_24h(cdms)
    hours = (24 * w["time_to_tca"]).clip(0, 23.999)
    bins = np.arange(0, 25, 2)
    hist = {t: np.histogram(hours[w["tier"] == t], bins)[0].tolist() for t in TIERS}
    zones = {"under_1km": int((w["miss_distance"] < 1000).sum()),
             "km_1_to_5": int(((w["miss_distance"] >= 1000) & (w["miss_distance"] <= 5000)).sum()),
             "over_5km": int((w["miss_distance"] > 5000).sum())}
    # every CDM (not just the latest) issued inside 24 h, for the "message load" figure
    all24 = cdms[cdms["time_to_tca"] < 1.0]
    miss_hist = np.histogram(np.log10(w["miss_distance"].clip(lower=10)), np.linspace(1, 5.5, 28))
    return {"hour_bins": bins.tolist(), "hist_by_tier": hist, "zones": zones,
            "n_events": int(len(w)), "n_cdms": int(len(all24)),
            "miss_log_hist": {"edges": r(miss_hist[1], 3).tolist(), "counts": miss_hist[0].tolist()}}


def run() -> dict:
    data.build_all()
    WEB.mkdir(parents=True, exist_ok=True)
    ev = data.load_events()
    train, val, test = (ev[ev["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"])
    model = GBMTierForecaster().fit(train)
    theta = tune_theta(model.predict(train).set_index("event_id").loc[train["event_id"]].reset_index(),
                       train["final_tier"].to_numpy())

    tr_cdms = data.load_cdms_cached("train_data")
    te_cdms = data.load_cdms_cached("test_data")
    split = pcnet.row_split(tr_cdms)
    eng_val = pcnet.PcEngine().fit(tr_cdms[split == "train"])
    eng_full_path = data.PROCESSED / "pc_engine.joblib"
    eng_full = joblib.load(eng_full_path) if eng_full_path.exists() else pcnet.PcEngine().fit(tr_cdms)

    out = {"events_val.json": events_table(val, model.predict(val), theta, True, model),
           "events_test.json": events_table(test, model.predict(test), theta, False, model),
           "cdms_val.json": cdm_table(tr_cdms, val["event_id"], eng_val),
           "cdms_test.json": cdm_table(te_cdms, test["event_id"], eng_full)}

    metrics = json.loads((REPORTS / "metrics.json").read_text())
    pce = json.loads((REPORTS / "pc_engine.json").read_text())
    pce["feature_importance"] = dict(list(pce["feature_importance"].items())[:12])
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiers": TIERS, "objects": OBJ, "theta": theta, "model": model.name, "tau_crit": model.tau_crit, "tau_elev": model.tau_elev,
        "cutoff_days": data.CUTOFF_DAYS,
        "forecast_metrics": metrics, "pc_engine": pce,
        "insights_24h": json.loads((REPORTS / "insights_24h.json").read_text()),
        "warnings_24h": warnings_24h(tr_cdms),
        "failure_gallery": json.loads((REPORTS / "failure_gallery.json").read_text()),
        "pc_validation": json.loads((REPORTS / "pc_validation.json").read_text()),
        "counts": {"train_rows": int(len(tr_cdms)), "test_rows": int(len(te_cdms)),
                   "train_events": int(tr_cdms["event_id"].nunique()), "test_events": int(len(test)),
                   "val_events": int(len(val))},
    }
    out["summary.json"] = summary
    shutil.copy(data.ROOT / "submissions" / "test_predictions_rows.csv", WEB / "test_predictions_rows.csv")
    sizes = {}
    for name, obj in out.items():
        p = WEB / name
        p.write_text(json.dumps(_finite(obj), separators=(",", ":"), default=_json_default, allow_nan=False))
        sizes[name] = round(p.stat().st_size / 1e6, 2)
    return sizes


def _finite(o):
    """JSON has no NaN/Infinity: map them to null (e.g. the constant baseline's theta = inf)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

"""Predictions on test_data (no labels): event-level and row-level files in submissions/.

Uses the model with the lowest validation ESA L (from reports/metrics.json), fitted on the train split.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import data, pcnet
from .models import available_models
from .tiers import classify_array, classify_log

SUB = data.ROOT / "submissions"


def best_model_name() -> str | None:
    p = data.ROOT / "reports" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["models"]
    # ESA L first; ties (models sharing LRP's pred_risk) go to the one that clears more events safely
    return min(m, key=lambda k: (round(m[k]["esa"]["L"], 6), -m[k]["workload"]["pct_cleared"]))


def run() -> dict:
    data.build_all()
    SUB.mkdir(exist_ok=True)
    train, test = data.load_events("train"), data.load_events("test")
    models = {m.name: m for m in available_models()}
    name = best_model_name() or "Baseline B: Latest Risk (LRP)"
    if name not in models:
        name = "Baseline B: Latest Risk (LRP)"
    model = models[name].fit(train)
    p = model.predict(test).set_index("event_id").loc[test["event_id"]].reset_index()
    p["pred_tier"] = model.forecast_tiers(p, test)
    p["current_tier"] = test["current_tier"].to_numpy()
    p["latest_miss_distance"] = test["latest_miss_distance"].to_numpy()
    p["latest_cdm_row"] = test["latest_cdm_row"].to_numpy()
    p["volatile"] = test["volatile"].to_numpy()
    ev_cols = ["event_id", "pred_risk", "pred_tier", "p_safe", "p_elevated", "p_critical", "abstain",
               "current_tier", "volatile", "latest_miss_distance", "latest_cdm_row", "model_name"]
    p[ev_cols].to_csv(SUB / "test_predictions_event.csv", index=False)
    # ESA-style file: event_id, predicted_risk
    p[["event_id", "pred_risk"]].rename(columns={"pred_risk": "predicted_risk"}).to_csv(
        SUB / "test_submission_esa.csv", index=False)

    # Row level: the Pc Engine (fit on every train row) estimates Pc and tier for each test CDM.
    # The test `risk` column is never an input; it is only used afterwards for scoring (reports/pc_engine.json).
    cdms = data.load_cdms_cached("test_data")
    eng = pcnet.PcEngine().fit(data.load_cdms_cached("train_data"))
    rp = eng.predict(cdms)
    rows = cdms[["cdm_row", "event_id", "time_to_tca", "miss_distance"]].copy()
    rows["pred_log10_pc"] = rp["pc_log"].round(4)
    rows["pred_pc"] = 10 ** rp["pc_log"]
    rows["pred_tier"] = rp["tier"]
    for c in ["p_safe", "p_elevated", "p_critical"]:
        rows[c] = rp[c].round(4)
    rows["tier_rules"] = ["|".join(r.split()[0] for r in classify_log(a, b)[1])
                          for a, b in zip(rp["pc_log"], rows["miss_distance"])]
    ev_f = p[["event_id", "pred_tier", "p_critical", "abstain"]].rename(
        columns={"pred_tier": "event_forecast_tier_at_tca", "p_critical": "event_p_critical_at_tca",
                 "abstain": "event_analyst_review"})
    rows = rows.merge(ev_f, on="event_id", how="left")
    rows.to_csv(SUB / "test_predictions_rows.csv", index=False)
    joblib.dump(eng, data.PROCESSED / "pc_engine.joblib")
    summary = {"model": name, "n_events": int(len(p)), "n_rows": int(len(rows)),
               "pred_tier_counts": p["pred_tier"].value_counts().to_dict(),
               "analyst_review": int(p["abstain"].sum()),
               "row_model": eng.name, "row_pred_tier_counts": rows["pred_tier"].value_counts().to_dict()}
    (SUB / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

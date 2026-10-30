"""Predictions on test_data (no labels): event-level and row-level files in submissions/.

Uses the model with the lowest validation ESA L (from reports/metrics.json), fitted on the train split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data
from .models import available_models
from .tiers import classify_array, classify_log

SUB = data.ROOT / "submissions"


def best_model_name() -> str | None:
    p = data.ROOT / "reports" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["models"]
    return min(m, key=lambda k: m[k]["esa"]["L"])


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
    p["pred_tier"] = classify_array(p["pred_risk"], test["latest_miss_distance"])
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

    cdms = data.load_cdms_cached("test_data")
    rows = cdms[["cdm_row", "event_id", "time_to_tca", "risk", "miss_distance"]].copy()
    rows["row_tier"] = classify_array(rows["risk"], rows["miss_distance"])
    rows["row_rules"] = ["|".join(r.split()[0] for r in classify_log(a, b)[1])
                         for a, b in zip(rows["risk"], rows["miss_distance"])]
    rows = rows.merge(p[["event_id", "pred_risk", "pred_tier", "p_critical", "abstain"]], on="event_id", how="left")
    rows.to_csv(SUB / "test_predictions_rows.csv", index=False)
    summary = {"model": name, "n_events": int(len(p)), "n_rows": int(len(rows)),
               "pred_tier_counts": p["pred_tier"].value_counts().to_dict(),
               "analyst_review": int(p["abstain"].sum())}
    (SUB / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

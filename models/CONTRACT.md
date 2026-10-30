# Model contract: plugging your model into OrbitGuard

OrbitGuard runs end-to-end on the baselines. Your model replaces nothing: it shows up
as an extra row in every evaluation table and becomes selectable in the dashboard.

## The task

At the **decision cutoff** (2 days before TCA), predict the event's **final risk**
(`risk` = log10 Pc of the last CDM, which is ≤ 1 day from TCA).

- Inputs may only use CDMs with `time_to_tca >= 2.0`. The event tables below already
  enforce this. Don't go back to the raw CSV for post-cutoff rows.
- Only these raw columns are allowed: event_id, time_to_tca, miss_distance, relative_speed,
  relative_position_r/t/n, c_object_type, c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP
  (plus `risk` as the target and as a past-CDM feature).
- Train on `split == "train"` only. `val` is the honest hold-out, and the test set has no labels.

## Input files (run `make data` first)

| file | rows | contents |
|---|---|---|
| `data/processed/events_train.csv` | 6,634 | features + targets |
| `data/processed/events_val.csv` | 1,659 | features + targets (do not train on these) |
| `data/processed/events_test.csv` | 2,167 | features only |

Feature columns (all computed from pre-cutoff CDMs; `orbitguard/data.py: FEATURE_COLUMNS`):
`n_cdms_pre, latest_time_to_tca, latest_risk, latest_max_risk_estimate, latest_miss_distance,
latest_relative_speed, latest_relative_position_{r,t,n}, latest_c_sigma_r, latest_t_sigma_r,
latest_F10, latest_AP, latest_c_object_type, first_risk, first_miss_distance, risk_mean, risk_std,
risk_max, max_risk_estimate_max, miss_min, risk_slope, miss_slope, risk_rise, miss_ratio,
current_tier, volatile`

Notes: `risk` is floored at -30 (Pc ≤ 1e-30). Sigmas equal to the 6.378e7 m "no covariance"
sentinel are already NaN. `risk_*` aggregates use risk clipped at -10. F10/AP have NaNs.

Targets (train/val only): `final_risk` (regression target), `final_tier`, `high_risk` (final_risk ≥ -6),
`final_miss_distance`.

## Output schema

| column | type | meaning |
|---|---|---|
| `event_id` | int | one row per event |
| `pred_risk` | float | predicted final log10 Pc (no NaN) |
| `p_safe`, `p_elevated`, `p_critical` | float | optional; must sum to 1 |
| `abstain` | bool | optional; True = "send to analyst, I'm not confident" |
| `model_name` | str | shown in the UI |

If the probabilities are missing, OrbitGuard fills them as a one-hot of
`classify(pred_risk, latest_miss_distance)`. Your model does **not** decide tiers. The forecast tier is
always `tiers.classify(pred_risk, latest_miss_distance)`. `p_safe` drives the auto-clear policy
(clear if `p_safe >= θ` and not `abstain`, with θ tuned on train for ≥ 99% Critical recall), so
calibrated probabilities directly increase the workload saving.

## Two ways to plug in (pick one)

1. **CSV:** write `models/teammate_predictions.csv` with the schema above, covering
   **every event in train, val and test** (train is needed to tune θ). Remember to fit on train only.
2. **Python module (preferred):** create `models/teammate/model.py` (or `__init__.py`) exposing

   ```python
   MODEL_NAME = "Teammate: GBM v1"
   def fit(train_events: pd.DataFrame) -> None: ...      # optional; called with the train split
   def predict(events: pd.DataFrame) -> pd.DataFrame: ... # returns the output schema
   ```

   `fit` receives the train split with targets. `predict` receives feature columns only:
   OrbitGuard strips `final_*`, `high_risk` and `split` first.

Then run `make eval` (metrics into `reports/`), `make predict` (test submissions) and `make app`.
OrbitGuard never writes to `models/teammate/`.

## Scores to beat (validation, 1,659 events)

Baseline B, Latest Risk Prediction: ESA L = 0.56 (MSE_HR 0.23 / F2 0.41), auto-clears 45.3% at 100% Critical recall.
Current values are in `reports/metrics.json`.

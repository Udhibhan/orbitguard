"""Pc Engine: a learned collision-probability calculator for single CDMs.

Reads one CDM's whitelisted telemetry (plus the telemetry of earlier CDMs in the same event,
never any `risk` value) and outputs:
  * pc_log      : estimated log10 Pc (regression, trained on the dataset's `risk`)
  * p_safe/p_elevated/p_critical : calibrated tier probabilities (classifier)
  * tier        : rules(pc_log, miss)  -- the handbook rules always make the decision

Physics features (isotropic Pc, worst-case Pc_max, Mahalanobis-like d/sigma) are computed
in pc.py and fed in; the model learns the correction for the missing along-track covariance.
Train/validation split is by event_id and matches data.split_events for the events it shares.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import data, pc
from .tiers import CRITICAL, TIERS, classify_array

REPORTS = data.ROOT / "reports"
OBJ_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA"]
SEED = 7


def row_features(cdms: pd.DataFrame) -> pd.DataFrame:
    """Per-row features. History features use only earlier CDMs (larger time_to_tca) of the same event."""
    d = pc.add_pc_columns(cdms).sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    miss = d["miss_distance"].clip(lower=1.0)
    c_sig = data.clean_sigma(d["c_sigma_r"])
    t_sig = data.clean_sigma(d["t_sigma_r"])
    f = pd.DataFrame(index=d.index)
    f["time_to_tca"] = d["time_to_tca"]
    f["log_miss"] = np.log10(miss)
    f["relative_speed"] = d["relative_speed"]
    for ax in "rtn":
        p = d[f"relative_position_{ax}"]
        f[f"pos_{ax}"] = p
        f[f"dir_{ax}"] = p / miss
    f["log_c_sigma_r"] = np.log10(c_sig)
    f["log_t_sigma_r"] = np.log10(t_sig)
    f["log_sigma_comb"] = np.log10(d["sigma_comb"])
    f["d_over_sigma"] = miss / d["sigma_comb"]
    f["abs_r_over_sigma"] = d["relative_position_r"].abs() / d["sigma_comb"]
    f["pc_iso_log"] = d["pc_approx_log"]
    f["pc_max_log"] = d["pc_max_log"]
    f["max_risk_estimate"] = d["max_risk_estimate"]
    f["F10"], f["AP"] = d["F10"], d["AP"]
    f["obj"] = pd.Categorical(d["c_object_type"], categories=OBJ_TYPES).codes
    g = d.groupby("event_id", sort=False)
    f["cdm_index"] = g.cumcount()
    f["prev_log_miss"] = f.groupby(d["event_id"])["log_miss"].shift(1)
    f["d_log_miss"] = f["log_miss"] - f["prev_log_miss"]
    f["first_log_miss"] = f.groupby(d["event_id"])["log_miss"].transform("first")
    f["min_log_miss_sofar"] = f.groupby(d["event_id"])["log_miss"].cummin()
    f["prev_mre"] = g["max_risk_estimate"].shift(1)
    f["d_mre"] = d["max_risk_estimate"] - f["prev_mre"]
    f["prev_log_c_sigma_r"] = f.groupby(d["event_id"])["log_c_sigma_r"].shift(1)
    f["d_log_c_sigma_r"] = f["log_c_sigma_r"] - f["prev_log_c_sigma_r"]
    f["n_unknown_sigma"] = c_sig.isna().astype(int)
    return f.loc[cdms.index]




def row_split(train_cdms: pd.DataFrame) -> pd.Series:
    """'train'/'val' per row, by event. Reuses the event-level split; unfiltered events get a seeded 20%."""
    ev = data.load_events()
    known = ev[ev["split"].isin(["train", "val"])].set_index("event_id")["split"]
    ids = np.sort(train_cdms["event_id"].unique())
    rng = np.random.default_rng(SEED)
    extra = pd.Series(np.where(rng.random(len(ids)) < 0.2, "val", "train"), index=ids)
    lookup = extra.copy()
    lookup.loc[known.index] = known
    return train_cdms["event_id"].map(lookup)


class PcEngine:
    name = "Pc Engine (GBM, physics features)"

    """Hurdle model for Pc + a tier classifier.

    floor : P(Pc above the 1e-30 floor)            -> below 0.5 we report the floor
    reg   : log10 Pc given not floored, rows with Pc >= 1e-8 weighted 5x (that is where tiers change)
    clf   : P(Safe/Elevated/Critical); tier = max(miss-distance rule tier, argmax probability)
    The miss rule is deterministic (miss is observed), so the model only has to judge the Pc side.
    """

    def __init__(self):
        kw = dict(learning_rate=0.05, l2_regularization=1.0, categorical_features=["obj"], random_state=SEED)
        self.floor = HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=63, min_samples_leaf=30, **kw)
        self.reg = HistGradientBoostingRegressor(max_iter=800, max_leaf_nodes=63, min_samples_leaf=30, **kw)
        self.clf = HistGradientBoostingClassifier(max_iter=500, max_leaf_nodes=63, min_samples_leaf=30, **kw)

    def fit(self, cdms: pd.DataFrame) -> "PcEngine":
        X = row_features(cdms)
        y = cdms["risk"].clip(lower=data.RISK_FLOOR).to_numpy()
        nf = y > data.RISK_FLOOR + 0.01
        tier = classify_array(cdms["risk"], cdms["miss_distance"])
        self.floor.fit(X, nf.astype(int))
        self.reg.fit(X[nf], y[nf], sample_weight=np.where(y[nf] >= -8, 5.0, 1.0))
        self.clf.fit(X, pd.Categorical(tier, categories=TIERS).codes)
        return self

    def predict(self, cdms: pd.DataFrame) -> pd.DataFrame:
        X = row_features(cdms)
        p_nf = self.floor.predict_proba(X)[:, 1]
        pc_log = np.where(p_nf >= 0.5, np.clip(self.reg.predict(X), data.RISK_FLOOR, 0.0), data.RISK_FLOOR)
        proba = np.zeros((len(X), 3))
        proba[:, self.clf.classes_] = self.clf.predict_proba(X)
        out = pd.DataFrame({"pc_log": pc_log, "p_safe": proba[:, 0], "p_elevated": proba[:, 1],
                            "p_critical": proba[:, 2]}, index=cdms.index)
        miss_tier = classify_array(np.full(len(X), data.RISK_FLOOR), cdms["miss_distance"])
        rank = np.maximum(proba.argmax(1), pd.Categorical(miss_tier, categories=TIERS).codes)
        out["tier"] = np.array(TIERS)[rank]
        out["tier_by_pc_rule"] = classify_array(pc_log, cdms["miss_distance"])
        return out


def score_rows(cdms: pd.DataFrame, pred: pd.DataFrame) -> dict:
    true_risk = cdms["risk"].to_numpy()
    true_tier = classify_array(true_risk, cdms["miss_distance"])
    pt = pred["tier"].to_numpy()
    nonfloor = true_risk > -29.99
    hr = true_risk >= data.HIGH_RISK_LOG
    phys = pc.add_pc_columns(cdms)["pc_approx_log"].to_numpy()
    ok = np.isfinite(phys) & nonfloor

    def per_tier(t):
        tp = int(((true_tier == t) & (pt == t)).sum())
        return {"support": int((true_tier == t).sum()), "predicted": int((pt == t).sum()),
                "recall": tp / max((true_tier == t).sum(), 1), "precision": tp / max((pt == t).sum(), 1)}

    cm = pd.crosstab(pd.Categorical(true_tier, TIERS), pd.Categorical(pt, TIERS), dropna=False).values.tolist()
    hr_pred = pred["pc_log"].to_numpy() >= data.HIGH_RISK_LOG
    tp = int((hr & hr_pred).sum())
    p_, r_ = tp / max(hr_pred.sum(), 1), tp / max(hr.sum(), 1)
    return {
        "n_rows": int(len(cdms)), "n_events": int(cdms["event_id"].nunique()),
        "tier_accuracy": float((true_tier == pt).mean()),
        "per_tier": {t: per_tier(t) for t in TIERS},
        "confusion": cm, "labels": TIERS,
        "mae_log_pc_nonfloor": float(np.mean(np.abs(pred["pc_log"].to_numpy()[nonfloor] - true_risk[nonfloor]))),
        "mae_log_pc_high_risk": float(np.mean(np.abs(pred["pc_log"].to_numpy()[hr] - true_risk[hr]))) if hr.any() else None,
        "physics_mae_log_pc_nonfloor": float(np.mean(np.abs(phys[ok] - true_risk[ok]))),
        "high_risk_recall": r_, "high_risk_precision": p_,
        "high_risk_f2": 5 * p_ * r_ / (4 * p_ + r_) if (4 * p_ + r_) else 0.0,
        "spearman": float(pd.Series(pred["pc_log"].to_numpy()[nonfloor]).corr(pd.Series(true_risk[nonfloor]), method="spearman")),
    }


def run() -> dict:
    """Fit on train-split rows, score on held-out val rows and on all test rows, then refit on all train rows."""
    data.build_all()
    tr = data.load_cdms_cached("train_data")
    te = data.load_cdms_cached("test_data")
    split = row_split(tr)
    fit_rows = tr[split == "train"]
    val_rows = tr[split == "val"]
    eng = PcEngine().fit(fit_rows)
    report = {"model": PcEngine.name, "val": score_rows(val_rows, eng.predict(val_rows)),
              "test": score_rows(te, eng.predict(te))}
    imp = permutation_like_importance(eng, val_rows)
    report["feature_importance"] = imp
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "pc_engine.json").write_text(json.dumps(report, indent=2))
    return report


def permutation_like_importance(eng: PcEngine, rows: pd.DataFrame, n: int = 20000) -> dict:
    """Increase in MAE (log10 Pc, non-floor rows) when one feature is shuffled. Small sample for speed."""
    rows = rows[rows["risk"] > -29.99]
    rows = rows.sample(min(n, len(rows)), random_state=SEED)
    X = row_features(rows)
    y = rows["risk"].to_numpy()
    base = np.mean(np.abs(eng.reg.predict(X) - y))
    rng = np.random.default_rng(SEED)
    out = {}
    for c in X.columns:
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        out[c] = float(np.mean(np.abs(eng.reg.predict(Xp) - y)) - base)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    r = run()
    for k in ["val", "test"]:
        s = r[k]
        print(k, {x: s[x] for x in ["tier_accuracy", "mae_log_pc_nonfloor", "physics_mae_log_pc_nonfloor",
                                    "mae_log_pc_high_risk", "high_risk_recall", "high_risk_precision", "spearman"]})
        print("  per tier", s["per_tier"])
    print(list(r["feature_importance"].items())[:10])

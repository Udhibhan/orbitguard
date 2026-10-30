"""Model interface, baselines, and the teammate plug-in slot.

Every model maps an event table (one row per event, pre-cutoff features only; see
data.FEATURE_COLUMNS) to a DataFrame with OUTPUT_COLUMNS. Tiers are NOT decided here:
evaluate/app derive forecast_tier = tiers.classify(pred_risk, latest_miss_distance).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, ROOT, TARGET_COLUMNS
from .tiers import CRITICAL, ELEVATED, SAFE, TIERS, classify_array

OUTPUT_COLUMNS = ["event_id", "pred_risk", "p_safe", "p_elevated", "p_critical", "abstain", "model_name"]
MODELS_DIR = ROOT / "models"
TEAMMATE_CSV = MODELS_DIR / "teammate_predictions.csv"
TEAMMATE_PKG = MODELS_DIR / "teammate"

LRP_CLIP = -6.001
MISS_BINS = [0, 1000, 2000, 3500, 5000, 7500, 10000, 15000, 20000, 30000, np.inf]
RISK_BINS = [-np.inf, -29.99, -10, -6, -4, np.inf]   # floor | (-30,-10) | [-10,-6) | [-6,-4) | >=-4


class Model:
    name = "base"

    def fit(self, train_events: pd.DataFrame) -> "Model":
        return self

    def predict(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def forecast_tiers(self, pred: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
        """Default: the handbook rules applied to (predicted final log10 Pc, latest miss distance)."""
        return classify_array(pred["pred_risk"].to_numpy(), events["latest_miss_distance"].to_numpy())


def _frame(events, pred_risk, probs, abstain, name) -> pd.DataFrame:
    out = pd.DataFrame({"event_id": events["event_id"].to_numpy(), "pred_risk": np.asarray(pred_risk, float)})
    out["p_safe"], out["p_elevated"], out["p_critical"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["abstain"] = np.asarray(abstain, bool)
    out["model_name"] = name
    return out[OUTPUT_COLUMNS]


def _tier_onehot(tiers: np.ndarray) -> np.ndarray:
    return np.stack([(tiers == t).astype(float) for t in TIERS], axis=1)


class ConstantModel(Model):
    """Baseline A: predict log10 Pc = -5 for every event. Probabilities = training class priors."""

    def __init__(self, value: float = -5.0):
        self.value = value
        self.name = f"Baseline A: constant ({value:g})"
        self.priors = np.array([1 / 3] * 3)

    def fit(self, train_events):
        vc = train_events["final_tier"].value_counts(normalize=True)
        self.priors = np.array([vc.get(t, 0.0) for t in TIERS])
        return self

    def predict(self, events):
        n = len(events)
        return _frame(events, np.full(n, self.value), np.tile(self.priors, (n, 1)), np.zeros(n, bool), self.name)


class LatestRiskModel(Model):
    """Baseline B: Latest Risk Prediction (LRP), ESA 2019's headline benchmark.

    pred_risk = latest pre-cutoff risk, clipped to -6.001 if below -6.
    Tier probabilities are NOT learned by ML: they are empirical transition frequencies
    P(final tier | latest-risk bin, latest-miss bin) counted on the training split
    (Laplace-smoothed, backing off to the miss-bin marginal for sparse cells).
    abstain = the deterministic volatility flag from data.build_events.
    """
    name = "Baseline B: Latest Risk (LRP)"

    def __init__(self, min_count: int = 20):
        self.min_count = min_count
        self.table: dict = {}
        self.backoff: dict = {}
        self.prior = np.array([1 / 3] * 3)

    @staticmethod
    def _bins(events):
        rb = pd.cut(events["latest_risk"], RISK_BINS, right=False, labels=False)
        mb = pd.cut(events["latest_miss_distance"], MISS_BINS, right=False, labels=False)
        return rb.to_numpy(), mb.to_numpy()

    def fit(self, train_events):
        rb, mb = self._bins(train_events)
        y = train_events["final_tier"].to_numpy()
        onehot = _tier_onehot(y)
        self.prior = onehot.mean(0)
        df = pd.DataFrame(onehot, columns=TIERS)
        df["rb"], df["mb"] = rb, mb
        for m, g in df.groupby("mb"):
            self.backoff[m] = (g[TIERS].sum().to_numpy() + self.prior) / (len(g) + 1)
        for (r, m), g in df.groupby(["rb", "mb"]):
            if len(g) >= self.min_count:
                self.table[(r, m)] = (g[TIERS].sum().to_numpy() + 1) / (len(g) + 3)
        return self

    def probs(self, events) -> np.ndarray:
        rb, mb = self._bins(events)
        return np.array([self.table.get((r, m), self.backoff.get(m, self.prior)) for r, m in zip(rb, mb)])

    def predict(self, events):
        risk = events["latest_risk"].to_numpy(float)
        pred = np.where(risk < -6, LRP_CLIP, risk)
        return _frame(events, pred, self.probs(events), events["volatile"].to_numpy(bool), self.name)


OBJ_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA"]


def event_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """Numeric event features for the GBM (pre-cutoff only, see data.FEATURE_COLUMNS)."""
    x = events[FEATURE_COLUMNS].copy()
    x["latest_c_object_type"] = pd.Categorical(x["latest_c_object_type"], categories=OBJ_TYPES).codes
    x["current_tier"] = pd.Categorical(x["current_tier"], categories=TIERS).codes
    x["volatile"] = x["volatile"].astype(int)
    x["latest_risk_clip"] = events["latest_risk"].clip(lower=-10)
    x["log_miss"] = np.log10(events["latest_miss_distance"].clip(lower=1))
    return x


class GBMTierForecaster(Model):
    """OrbitGuard forecaster: LRP for the Pc value + gradient-boosted tier probabilities.

    No regressor we tried beats LRP on ESA L (consistent with the 2019 competition), so pred_risk stays LRP.
    The GBM learns P(final tier) from the full pre-cutoff history, which sharpens p_safe and lets the
    auto-clear policy clear more events at the same Critical recall. Events seen in fit() get
    out-of-fold probabilities, so θ tuned on train is not tuned on memorised predictions.
    """
    name = "OrbitGuard: LRP + GBM tier probabilities"

    def __init__(self, folds: int = 5, seed: int = 7):
        self.folds, self.seed = folds, seed
        self.kw = dict(random_state=seed, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
                       l2_regularization=2.0, max_iter=400)
        self.clf = None
        self.oof: pd.DataFrame | None = None

    def fit(self, train_events):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        X = event_matrix(train_events)
        y = pd.Categorical(train_events["final_tier"], categories=TIERS).codes
        self.clf = HistGradientBoostingClassifier(**self.kw).fit(X, y)
        cv = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(HistGradientBoostingClassifier(**self.kw), X, y, cv=cv, method="predict_proba")
        self.oof = pd.DataFrame(oof, index=train_events["event_id"].to_numpy(), columns=TIERS)
        self.tau_crit, self.tau_elev = self._tune_thresholds(oof, train_events["final_tier"].to_numpy())
        return self

    @staticmethod
    def _prob_tiers(p: np.ndarray, tau_crit: float, tau_elev: float) -> np.ndarray:
        return np.where(p[:, 2] >= tau_crit, CRITICAL, np.where(p[:, 1] + p[:, 2] >= tau_elev, ELEVATED, SAFE))

    def _tune_thresholds(self, oof: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Forecast-tier rule F1/F2, tuned on out-of-fold train probabilities only.
        Objective favours catching Critical (F2 weights recall 4x), then Elevated, then overall accuracy."""
        def f2(t, c):
            tp = ((t == c) & (y == c)).sum()
            pr, rc = tp / max((t == c).sum(), 1), tp / max((y == c).sum(), 1)
            return 5 * pr * rc / (4 * pr + rc) if pr + rc else 0.0
        best = (-1.0, 0.5, 0.5)
        for tc in np.arange(0.10, 0.70, 0.025):
            for te in np.arange(0.20, 0.80, 0.05):
                t = self._prob_tiers(oof, tc, te)
                score = f2(t, CRITICAL) + 0.5 * f2(t, ELEVATED) + 0.25 * (t == y).mean()
                if score > best[0]:
                    best = (score, float(round(tc, 3)), float(round(te, 3)))
        return best[1], best[2]

    def forecast_tiers(self, pred, events):
        """F1: P(Critical) >= tau_crit -> Critical; F2: P(Elevated or worse) >= tau_elev -> Elevated; else Safe."""
        p = pred[["p_safe", "p_elevated", "p_critical"]].to_numpy()
        return self._prob_tiers(p, self.tau_crit, self.tau_elev)

    def predict(self, events):
        probs = self.clf.predict_proba(event_matrix(events))
        ids = events["event_id"].to_numpy()
        seen = np.isin(ids, self.oof.index) if self.oof is not None else np.zeros(len(ids), bool)
        if seen.any():
            probs[seen] = self.oof.loc[ids[seen], TIERS].to_numpy()
        risk = events["latest_risk"].to_numpy(float)
        pred = np.where(risk < -6, LRP_CLIP, risk)
        return _frame(events, pred, probs, events["volatile"].to_numpy(bool), self.name)


def load_teammate() -> Model | None:
    """Plug-in slot. Prefers a Python package models/teammate/ (with predict(), optional fit()),
    else models/teammate_predictions.csv. Never modifies anything in models/teammate/."""
    init = TEAMMATE_PKG / "__init__.py"
    mod_file = init if init.exists() else (TEAMMATE_PKG / "model.py")
    if mod_file.exists():
        spec = importlib.util.spec_from_file_location("teammate_model", mod_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _Wrapped(Model):
            name = getattr(mod, "MODEL_NAME", "Teammate model")

            def fit(self, train_events):
                if hasattr(mod, "fit"):
                    mod.fit(train_events)
                return self

            def predict(self, events):
                # strip targets so a plug-in cannot read the answer at predict time
                feats = events.drop(columns=TARGET_COLUMNS + ["split"], errors="ignore")
                return validate_output(mod.predict(feats.copy()), events, self.name)
        return _Wrapped()
    if TEAMMATE_CSV.exists():
        preds = pd.read_csv(TEAMMATE_CSV)

        class _FromCsv(Model):
            name = str(preds["model_name"].iloc[0]) if "model_name" in preds and len(preds) else "Teammate model"

            def predict(self, events):
                sub = preds[preds["event_id"].isin(events["event_id"])]
                sub = events[["event_id"]].merge(sub, on="event_id", how="inner")
                return validate_output(sub, events, self.name, allow_missing=True)
        return _FromCsv()
    return None


def validate_output(out: pd.DataFrame, events: pd.DataFrame, name: str, allow_missing=False) -> pd.DataFrame:
    """Enforce models/CONTRACT.md. Missing probabilities are filled from a one-hot of the tier implied by pred_risk."""
    out = out.copy()
    assert "event_id" in out and "pred_risk" in out, f"{name}: needs event_id and pred_risk"
    assert out["event_id"].is_unique, f"{name}: duplicate event_id rows"
    if not allow_missing:
        assert set(out["event_id"]) == set(events["event_id"]), f"{name}: must predict every event"
    assert out["pred_risk"].notna().all(), f"{name}: NaN pred_risk"
    if not {"p_safe", "p_elevated", "p_critical"} <= set(out.columns):
        miss = events.set_index("event_id").loc[out["event_id"], "latest_miss_distance"].to_numpy()
        oh = _tier_onehot(classify_array(out["pred_risk"], miss))
        out["p_safe"], out["p_elevated"], out["p_critical"] = oh.T
    s = out[["p_safe", "p_elevated", "p_critical"]].sum(axis=1)
    assert np.allclose(s, 1, atol=1e-3), f"{name}: probabilities must sum to 1"
    out["abstain"] = out["abstain"].astype(bool) if "abstain" in out else False
    out["model_name"] = name
    return out[OUTPUT_COLUMNS]


def available_models() -> list[Model]:
    models: list[Model] = [ConstantModel(), LatestRiskModel(), GBMTierForecaster()]
    tm = load_teammate()
    if tm is not None:
        models.append(tm)
    return models

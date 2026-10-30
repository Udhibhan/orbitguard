"""Event-level evaluation of every available model on the held-out validation split.

All numbers here are computed in code; nothing comes from an LLM. Outputs go to reports/
(JSON + PNG + CSV) and the dashboard only reads these cached files.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .models import available_models
from .tiers import CRITICAL, ELEVATED, SAFE, TIERS, classify_array

REPORTS = data.ROOT / "reports"
TARGET_CRIT_RECALL = 0.99
N_BOOT = 1000
LATE_JUMP_ORDERS = 2.0  # final risk exceeds latest pre-cutoff risk by > 2 orders of magnitude

plt.style.use("dark_background")


def esa_score(pred_risk, true_risk) -> dict:
    """ESA Kelvins metric: L = MSE on true high-risk events (true >= -6) / F2 (beta=2, threshold -6)."""
    pred_risk, true_risk = np.asarray(pred_risk, float), np.asarray(true_risk, float)
    hr_true = true_risk >= data.HIGH_RISK_LOG
    hr_pred = pred_risk >= data.HIGH_RISK_LOG
    mse = float(np.mean((pred_risk[hr_true] - true_risk[hr_true]) ** 2)) if hr_true.any() else float("nan")
    tp = int((hr_true & hr_pred).sum()); fp = int((~hr_true & hr_pred).sum()); fn = int((hr_true & ~hr_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return {"L": mse / f2 if f2 else float("inf"), "mse_hr": mse, "f2": f2,
            "precision_hr": p, "recall_hr": r, "tp": tp, "fp": fp, "fn": fn}


def tier_metrics(true_tier, pred_tier) -> dict:
    true_tier, pred_tier = np.asarray(true_tier), np.asarray(pred_tier)
    cm = pd.crosstab(pd.Categorical(true_tier, TIERS), pd.Categorical(pred_tier, TIERS), dropna=False)
    tc, pc = true_tier == CRITICAL, pred_tier == CRITICAL
    tp = int((tc & pc).sum())
    rec = tp / tc.sum() if tc.sum() else float("nan")
    prec = tp / pc.sum() if pc.sum() else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    return {"confusion": cm.values.tolist(), "labels": TIERS, "accuracy": float((true_tier == pred_tier).mean()),
            "critical_recall": rec, "critical_precision": prec, "critical_f2": f2}


def clear_mask(pred: pd.DataFrame, theta: float) -> np.ndarray:
    """Auto-clear policy: p_safe >= theta AND the model did not abstain (volatile -> analyst)."""
    return (pred["p_safe"].to_numpy() >= theta) & ~pred["abstain"].to_numpy(bool)


def tune_theta(pred: pd.DataFrame, true_tier: np.ndarray, target=TARGET_CRIT_RECALL) -> float:
    """Lowest p_safe threshold (= most events cleared) keeping Critical recall >= target on TRAIN."""
    crit = true_tier == CRITICAL
    best = np.inf
    for theta in np.sort(np.unique(pred["p_safe"]))[::-1]:
        cleared = clear_mask(pred, theta)
        recall = 1 - (cleared & crit).sum() / max(crit.sum(), 1)
        if recall >= target:
            best = theta
        else:
            break
    return float(best)


def workload(pred: pd.DataFrame, ev: pd.DataFrame, theta: float) -> dict:
    cleared = clear_mask(pred, theta)
    t = ev["final_tier"].to_numpy()
    crit, elev = t == CRITICAL, t == ELEVATED
    return {"theta": theta, "n_events": int(len(ev)), "n_cleared": int(cleared.sum()),
            "pct_cleared": float(cleared.mean()),
            "critical_recall": float(1 - (cleared & crit).sum() / max(crit.sum(), 1)),
            "critical_auto_cleared": int((cleared & crit).sum()),
            "elevated_auto_cleared": int((cleared & elev).sum()),
            "n_analyst_review": int(pred["abstain"].sum())}


def bootstrap(pred: pd.DataFrame, ev: pd.DataFrame, theta: float, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(ev)
    pr, tr = pred["pred_risk"].to_numpy(), ev["final_risk"].to_numpy()
    cleared = clear_mask(pred, theta)
    crit = (ev["final_tier"] == CRITICAL).to_numpy()
    Ls, pcts, recs = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        Ls.append(esa_score(pr[i], tr[i])["L"])
        pcts.append(cleared[i].mean())
        c = crit[i]
        recs.append(1 - (cleared[i] & c).sum() / max(c.sum(), 1))
    ci = lambda a: [float(np.nanpercentile(np.where(np.isinf(a), np.nan, a), 2.5)),
                    float(np.nanpercentile(np.where(np.isinf(a), np.nan, a), 97.5))]
    return {"L_ci95": ci(np.array(Ls)), "pct_cleared_ci95": ci(np.array(pcts)),
            "critical_recall_ci95": ci(np.array(recs))}


def failure_gallery(ev: pd.DataFrame, preds: dict[str, pd.DataFrame], top=15) -> list[dict]:
    """Late surprises: risk jumped after the cutoff, or the tier escalated Safe/Elevated -> Critical.
    These are events no pre-cutoff model could see; we record whether the volatility flag caught them."""
    jump = ev["final_risk"] - ev["latest_risk"].clip(lower=data.VOLATILITY_CLIP)
    escalated = (ev["final_tier"] == CRITICAL) & (ev["current_tier"] != CRITICAL)
    mask = (jump > LATE_JUMP_ORDERS) | escalated
    sub = ev[mask].assign(jump=jump[mask]).sort_values("jump", ascending=False)
    rows = []
    for _, r in sub.iterrows():
        row = {"event_id": int(r.event_id), "latest_risk": float(r.latest_risk), "final_risk": float(r.final_risk),
               "jump_orders": float(r.jump), "latest_miss_m": float(r.latest_miss_distance),
               "final_miss_m": float(r.final_miss_distance), "current_tier": r.current_tier,
               "final_tier": r.final_tier, "volatile_flag": bool(r.volatile), "volatile_reasons": r.volatile_reasons,
               "latest_cdm_row": int(r.latest_cdm_row), "final_cdm_row": int(r.final_cdm_row)}
        for name, p in preds.items():
            pp = p.set_index("event_id").loc[r.event_id]
            row[f"pred[{name}]"] = float(pp.pred_risk)
        rows.append(row)
    return rows


def _plot_confusions(results: dict, path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["tiers"]["confusion"])
        ax.imshow(cm, cmap="magma")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] < cm.max() / 2 else "k")
        ax.set_xticks(range(3), TIERS); ax.set_yticks(range(3), TIERS)
        ax.set_xlabel("forecast tier"); ax.set_ylabel("true final tier"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_calibration(ev, preds, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (col, tier) in zip(axes, [("p_critical", CRITICAL), ("p_safe", SAFE)]):
        y = (ev["final_tier"] == tier).to_numpy()
        for name, p in preds.items():
            prob = p[col].to_numpy()
            if np.unique(prob).size < 3:
                continue
            bins = np.quantile(prob, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(prob, np.unique(bins)[1:-1]), 0, None)
            xs = [prob[idx == k].mean() for k in np.unique(idx)]
            ys = [y[idx == k].mean() for k in np.unique(idx)]
            ax.plot(xs, ys, "o-", label=name)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel(f"predicted {col}"); ax.set_ylabel(f"observed rate of {tier}")
        ax.set_title(f"Calibration: {tier}", fontsize=10); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run() -> dict:
    data.build_all()
    REPORTS.mkdir(exist_ok=True)
    train, val = data.load_events("train"), data.load_events("val")
    results, val_preds = {}, {}
    for model in available_models():
        model.fit(train)
        p_tr = model.predict(train)
        p_val = model.predict(val).set_index("event_id").loc[val["event_id"]].reset_index()
        theta = tune_theta(p_tr.set_index("event_id").loc[train["event_id"]].reset_index(),
                           train["final_tier"].to_numpy())
        ftier = classify_array(p_val["pred_risk"], val["latest_miss_distance"])
        abst = p_val["abstain"].to_numpy(bool)
        crit = (val["final_tier"] == CRITICAL).to_numpy()
        silent_miss = int((crit & (ftier != CRITICAL) & ~abst).sum())
        results[model.name] = {
            "esa": esa_score(p_val["pred_risk"], val["final_risk"]),
            "tiers": tier_metrics(val["final_tier"], ftier),
            "critical_recall_with_review": float(1 - silent_miss / max(crit.sum(), 1)),
            "critical_silent_misses": silent_miss,
            "workload": workload(p_val, val, theta),
            "workload_train": workload(p_tr.set_index("event_id").loc[train["event_id"]].reset_index(), train, theta),
            "bootstrap": bootstrap(p_val, val, theta),
        }
        p_val["forecast_tier"] = ftier
        p_val["auto_cleared"] = clear_mask(p_val, theta)
        val_preds[model.name] = p_val

    persistence = tier_metrics(val["final_tier"], val["current_tier"])
    gallery = failure_gallery(val, val_preds)
    metrics = {
        "split": {"n_train": len(train), "n_val": len(val),
                  "val_tier_counts": val["final_tier"].value_counts().to_dict(),
                  "val_high_risk": int(val["high_risk"].sum())},
        "build_report": data.build_all(),
        "models": results,
        "current_tier_persistence": persistence,
        "volatility": {"rate_val": float(val["volatile"].mean()),
                       "late_surprises": len(gallery),
                       "late_surprises_flagged": int(sum(g["volatile_flag"] for g in gallery))},
        "config": {"cutoff_days": data.CUTOFF_DAYS, "final_within_days": data.FINAL_WITHIN_DAYS,
                   "target_critical_recall": TARGET_CRIT_RECALL, "late_jump_orders": LATE_JUMP_ORDERS,
                   "vol_std": data.VOL_STD, "vol_rise": data.VOL_RISE, "vol_miss_drop": data.VOL_MISS_DROP},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (REPORTS / "failure_gallery.json").write_text(json.dumps(gallery, indent=2, default=float))
    pd.concat(val_preds.values()).to_csv(REPORTS / "val_predictions.csv", index=False)
    _plot_confusions(results, REPORTS / "confusion.png")
    _plot_calibration(val, val_preds, REPORTS / "calibration.png")
    return metrics


def summary_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, r in metrics["models"].items():
        rows.append({"model": name, "ESA L (lower=better)": r["esa"]["L"], "MSE_HR": r["esa"]["mse_hr"],
                     "F2_HR": r["esa"]["f2"], "Crit recall": r["tiers"]["critical_recall"],
                     "Crit precision": r["tiers"]["critical_precision"], "Crit F2": r["tiers"]["critical_f2"],
                     "Crit recall incl. review": r["critical_recall_with_review"],
                     "Auto-cleared %": 100 * r["workload"]["pct_cleared"],
                     "Crit recall @clear": r["workload"]["critical_recall"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    m = run()
    pd.set_option("display.width", 250)
    print(summary_table(m).round(3).to_string(index=False))
    print("volatility:", m["volatility"])
    for n, r in m["models"].items():
        print(n, "workload:", r["workload"], "boot:", r["bootstrap"])

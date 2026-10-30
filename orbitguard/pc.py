"""Independent collision-probability estimate. This is an APPROXIMATION, not a replacement for `risk`.

Only the radial position sigmas (c_sigma_r, t_sigma_r) are allowed, so we assume an
ISOTROPIC combined covariance  sigma^2 = c_sigma_r^2 + t_sigma_r^2  and the 2D
small-hard-body-radius formula:

    Pc     ~= HBR^2 / (2 sigma^2) * exp(-d^2 / (2 sigma^2))
    Pc_max ~= HBR^2 / (e d^2)            (worst case over sigma, at sigma = d / sqrt 2)

HBR (combined hard-body radius) defaults to 10 m. That is an ASSUMPTION; the dataset gives no object sizes.
Known reasons this diverges from the dataset's `risk`: the real covariance is anisotropic
(along-track sigma is usually far larger than radial), it is projected onto the encounter
plane, and there are cross-correlation terms. None of these are available to us.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .tiers import classify_array

DEFAULT_HBR_M = 10.0
LOG_FLOOR = data.RISK_FLOOR
LN10 = np.log(10.0)


def combined_sigma(c_sigma_r, t_sigma_r):
    c = data.clean_sigma(pd.Series(np.asarray(c_sigma_r, float))).to_numpy()
    t = data.clean_sigma(pd.Series(np.asarray(t_sigma_r, float))).to_numpy()
    return np.sqrt(c ** 2 + t ** 2)


def log10_pc(miss_m, sigma_m, hbr_m: float = DEFAULT_HBR_M):
    """log10 of the isotropic small-HBR Pc, computed in log space (no underflow), floored at -30, capped at 0."""
    d, s = np.asarray(miss_m, float), np.asarray(sigma_m, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log10(hbr_m ** 2 / (2 * s ** 2)) - d ** 2 / (2 * s ** 2) / LN10
    return np.clip(lp, LOG_FLOOR, 0.0)  # NaN stays NaN when sigma is unavailable


def log10_pc_max(miss_m, hbr_m: float = DEFAULT_HBR_M):
    d = np.asarray(miss_m, float)
    return np.clip(np.log10(hbr_m ** 2 / (np.e * d ** 2)), LOG_FLOOR, 0.0)


def implied_hbr(max_risk_estimate, miss_m):
    """HBR that would make our Pc_max equal the dataset's max_risk_estimate: d * sqrt(e * 10^mre)."""
    return np.asarray(miss_m, float) * np.sqrt(np.e * 10 ** np.asarray(max_risk_estimate, float))


def add_pc_columns(cdms: pd.DataFrame, hbr_m: float = DEFAULT_HBR_M) -> pd.DataFrame:
    out = cdms.copy()
    out["sigma_comb"] = combined_sigma(out["c_sigma_r"], out["t_sigma_r"])
    out["pc_approx_log"] = log10_pc(out["miss_distance"], out["sigma_comb"], hbr_m)
    out["pc_max_log"] = log10_pc_max(out["miss_distance"], hbr_m)
    return out


def validate(hbr_m: float = DEFAULT_HBR_M) -> dict:
    """Compare the approximation with the dataset's `risk` on all train CDMs. Writes reports/pc_validation.json + PNG."""
    cdms = add_pc_columns(data.load_cdms_cached("train_data"), hbr_m)
    n_all = len(cdms)
    ok = cdms["sigma_comb"].notna()
    d = cdms[ok]
    floor = d["risk"] <= LOG_FLOOR
    nf = d[~floor]
    band = lambda r: np.where(r >= -4, "Pc>=1e-4", np.where(r >= -6, "1e-6..1e-4", "<1e-6"))
    tier_ds = classify_array(d["risk"], d["miss_distance"])
    tier_ap = classify_array(d["pc_approx_log"], d["miss_distance"])
    band_ds, band_ap = band(d["risk"]), band(d["pc_approx_log"])
    diff = nf["pc_approx_log"] - nf["risk"]
    sig_ratio = (nf["c_sigma_r"] / nf["t_sigma_r"])
    by_sigma = (pd.DataFrame({"sigma_comb_bin": pd.qcut(nf["sigma_comb"], 5), "diff": diff})
                .groupby("sigma_comb_bin", observed=True)["diff"].median())
    by_miss = (pd.DataFrame({"miss_bin": pd.cut(nf["miss_distance"], [0, 1000, 5000, 10000, 20000, 1e6]), "diff": diff})
               .groupby("miss_bin", observed=True)["diff"].median())
    hbr_impl = implied_hbr(cdms["max_risk_estimate"], cdms["miss_distance"])
    # Divergent high-risk cases: dataset says Pc >= 1e-6, we say < 1e-6
    missed_hr = (d["risk"] >= -6) & (d["pc_approx_log"] < -6)
    report = {
        "hbr_m": hbr_m,
        "n_cdms": int(n_all),
        "n_sigma_unavailable": int((~ok).sum()),
        "n_dataset_floor_-30": int(floor.sum()),
        "spearman_non_floor": float(nf["pc_approx_log"].corr(nf["risk"], method="spearman")),
        "pearson_non_floor": float(nf["pc_approx_log"].corr(nf["risk"])),
        "median_log_diff_non_floor": float(diff.median()),
        "mae_log_non_floor": float(diff.abs().median()),
        "pc_band_agreement": float((band_ds == band_ap).mean()),
        "tier_agreement": float((tier_ds == tier_ap).mean()),
        "tier_agreement_note": "inflated by the miss-distance rules, which are identical on both sides",
        "dataset_high_risk_cdms": int((d["risk"] >= -6).sum()),
        "dataset_high_risk_missed_by_approx": int(missed_hr.sum()),
        "approx_floor_when_dataset_not_floor": float((nf["pc_approx_log"] <= LOG_FLOOR).mean()),
        "median_diff_by_sigma_quintile": {str(k): float(v) for k, v in by_sigma.items()},
        "median_diff_by_miss_bin": {str(k): float(v) for k, v in by_miss.items()},
        "pc_max_vs_max_risk_estimate": {
            "spearman": float(pd.Series(cdms["pc_max_log"]).corr(cdms["max_risk_estimate"], method="spearman")),
            "median_implied_hbr_m": float(np.nanmedian(hbr_impl)),
            "iqr_implied_hbr_m": [float(np.nanpercentile(hbr_impl, 25)), float(np.nanpercentile(hbr_impl, 75))],
        },
        "c_over_t_sigma_ratio_median": float(sig_ratio.median()),
    }
    _plot(d, nf, report)
    data.ROOT.joinpath("reports").mkdir(exist_ok=True)
    (data.ROOT / "reports" / "pc_validation.json").write_text(json.dumps(report, indent=2))
    return report


def _plot(d, nf, report):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    s = nf.sample(min(20000, len(nf)), random_state=0)
    axes[0].scatter(s["risk"], s["pc_approx_log"], s=2, alpha=0.25, color="#4cc9f0")
    axes[0].plot([-30, 0], [-30, 0], "--", color="grey")
    for v in (-4, -6):
        axes[0].axvline(v, color="#555", lw=0.8); axes[0].axhline(v, color="#555", lw=0.8)
    axes[0].set_xlabel("dataset risk (log10 Pc)"); axes[0].set_ylabel(f"approx log10 Pc (HBR={report['hbr_m']:g} m)")
    axes[0].set_title(f"Isotropic approximation vs dataset (Spearman {report['spearman_non_floor']:.2f})", fontsize=9)
    s2 = d.sample(min(20000, len(d)), random_state=0)
    axes[1].scatter(s2["max_risk_estimate"], s2["pc_max_log"], s=2, alpha=0.25, color="#f72585")
    axes[1].plot([-12, 0], [-12, 0], "--", color="grey")
    axes[1].set_xlabel("dataset max_risk_estimate"); axes[1].set_ylabel("approx log10 Pc_max")
    axes[1].set_title(f"Worst case: implied HBR median {report['pc_max_vs_max_risk_estimate']['median_implied_hbr_m']:.1f} m",
                      fontsize=9)
    fig.tight_layout(); fig.savefig(data.ROOT / "reports" / "pc_validation.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))

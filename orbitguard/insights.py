"""24-hour warnings and auto-generated insights. All numbers and sentences are computed in code.

Uses the raw train CDMs (descriptive, no model): for each event, its latest CDM with time_to_tca < 1 day.
The test set has no CDMs under 2 days, so this view is necessarily retrospective.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data
from .tiers import CRITICAL, ELEVATED, SAFE, TIERS, classify_array


def latest_within_24h(cdms: pd.DataFrame) -> pd.DataFrame:
    w = cdms[cdms["time_to_tca"] < 1.0].sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    last = w.groupby("event_id").tail(1).copy()
    last["tier"] = classify_array(last["risk"], last["miss_distance"])
    last["pc_rule"] = last["risk"] >= -4
    last["miss_rule"] = last["miss_distance"] < 1000
    return last


def compute(cdms: pd.DataFrame | None = None) -> dict:
    cdms = data.load_cdms_cached("train_data") if cdms is None else cdms
    w = latest_within_24h(cdms)
    n = len(w)
    crit = w[w["tier"] == CRITICAL]
    by_tier = w["tier"].value_counts().reindex(TIERS, fill_value=0).to_dict()
    driver = {"miss_rule_only": int((crit.miss_rule & ~crit.pc_rule).sum()),
              "pc_rule_only": int((crit.pc_rule & ~crit.miss_rule).sum()),
              "both": int((crit.pc_rule & crit.miss_rule).sum())}
    obj = (w.groupby("c_object_type")
           .agg(events=("event_id", "size"), critical=("tier", lambda s: int((s == CRITICAL).sum())),
                under_1km=("miss_rule", "sum"), median_miss_km=("miss_distance", lambda s: float(s.median() / 1000)))
           .assign(critical_rate=lambda d: d.critical / d.events).sort_values("events", ascending=False))
    sw = w.dropna(subset=["F10", "AP"])
    f10_bins = pd.qcut(sw["F10"], 3, duplicates="drop")
    solar = (sw.groupby(f10_bins, observed=True)
             .agg(events=("event_id", "size"), critical_rate=("tier", lambda s: float((s == CRITICAL).mean())),
                  high_risk_rate=("risk", lambda s: float((s >= -6).mean()))))
    ap_storm = sw["AP"] >= 27  # Kp >= 4 roughly: active geomagnetic conditions
    rho_f10 = float(sw["risk"].clip(lower=-10).corr(sw["F10"], method="spearman"))
    rho_ap = float(sw["risk"].clip(lower=-10).corr(sw["AP"], method="spearman"))

    top_obj = obj["critical_rate"].idxmax()
    insights = [
        f"{n:,} events had a CDM within 24 h of TCA: {by_tier[CRITICAL]:,} Critical, {by_tier[ELEVATED]:,} Elevated, "
        f"{by_tier[SAFE]:,} Safe; {int(w.miss_rule.sum()):,} with miss distance under 1 km.",
        f"{100 * driver['miss_rule_only'] / max(len(crit), 1):.1f}% of Critical flags are driven by the miss-distance rule "
        f"alone, {100 * driver['pc_rule_only'] / max(len(crit), 1):.1f}% by Pc alone and "
        f"{100 * driver['both'] / max(len(crit), 1):.1f}% by both. The 1 km rule dominates the alerting load.",
        f"'{top_obj}' secondaries have the highest Critical rate ({100 * obj.loc[top_obj, 'critical_rate']:.1f}% of "
        f"{int(obj.loc[top_obj, 'events']):,} events) versus {100 * len(crit) / max(n, 1):.1f}% overall.",
        f"Space weather shows a weak link with risk (Spearman vs F10 {rho_f10:+.2f}, vs AP {rho_ap:+.2f}); Critical rate "
        f"is {100 * (sw[ap_storm].tier == CRITICAL).mean():.1f}% when AP >= 27 vs "
        f"{100 * (sw[~ap_storm].tier == CRITICAL).mean():.1f}% otherwise ({int(ap_storm.sum())} active-geomagnetic events). "
        f"Treat this as descriptive, not causal.",
    ]
    out = {"n_events": n, "by_tier": by_tier, "under_1km": int(w.miss_rule.sum()), "critical_drivers": driver,
           "by_object_type": obj.reset_index().to_dict(orient="records"),
           "by_f10_tercile": [{"f10_range": str(k), **v} for k, v in solar.to_dict(orient="index").items()],
           "spearman_risk_f10": rho_f10, "spearman_risk_ap": rho_ap, "insights": insights}
    (data.ROOT / "reports").mkdir(exist_ok=True)
    (data.ROOT / "reports" / "insights_24h.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    for s in compute()["insights"]:
        print("-", s)

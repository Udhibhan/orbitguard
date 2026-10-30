"""Deterministic risk tiers. This is the ONLY place tier decisions are made.

Rules (precedence Critical > Elevated > Safe):
  Critical : Pc >= 1e-4            OR miss < 1.0 km
  Elevated : 1e-6 <= Pc < 1e-4     OR 1.0 km <= miss <= 5.0 km
  Safe     : Pc < 1e-6             AND miss > 5.0 km

Comparisons are done in log10 space so that the dataset's `risk` (log10 Pc)
hits the boundaries exactly (risk == -4.0 is Critical, risk == -6.0 is Elevated).
If Pc is unknown (NaN) we cannot confirm Safe, so the event is at least Elevated.
"""
from __future__ import annotations

import math

import numpy as np

CRITICAL, ELEVATED, SAFE = "Critical", "Elevated", "Safe"
TIERS = [SAFE, ELEVATED, CRITICAL]  # ordered low -> high
TIER_RANK = {t: i for i, t in enumerate(TIERS)}

LOG_PC_CRITICAL = -4.0   # Pc >= 1e-4
LOG_PC_ELEVATED = -6.0   # Pc >= 1e-6
MISS_CRITICAL_M = 1000.0  # miss < 1 km
MISS_ELEVATED_M = 5000.0  # miss <= 5 km


def _fmt_pc(log_pc: float) -> str:
    return f"{10 ** log_pc:.2e}" if log_pc > -30 else "<=1e-30 (floor)"


def classify_log(log_pc: float, miss_distance_m: float) -> tuple[str, list[str]]:
    """Classify from log10(Pc) and miss distance in metres. Returns (tier, reasons)."""
    reasons: list[str] = []
    pc_known = log_pc is not None and not math.isnan(log_pc)
    pc_crit = pc_known and log_pc >= LOG_PC_CRITICAL
    pc_elev = pc_known and LOG_PC_ELEVATED <= log_pc < LOG_PC_CRITICAL
    miss_crit = miss_distance_m < MISS_CRITICAL_M
    miss_elev = MISS_CRITICAL_M <= miss_distance_m <= MISS_ELEVATED_M
    km = miss_distance_m / 1000.0

    if pc_crit:
        reasons.append(f"R1 Pc rule: Pc = {_fmt_pc(log_pc)} >= 1e-4 -> Critical")
    if miss_crit:
        reasons.append(f"R2 miss rule: miss = {km:.3f} km < 1.0 km -> Critical")
    if pc_crit or miss_crit:
        return CRITICAL, reasons

    if pc_elev:
        reasons.append(f"R3 Pc rule: 1e-6 <= Pc = {_fmt_pc(log_pc)} < 1e-4 -> Elevated")
    if miss_elev:
        reasons.append(f"R4 miss rule: 1.0 km <= miss = {km:.3f} km <= 5.0 km -> Elevated")
    if not pc_known:
        reasons.append("R5 Pc unavailable: cannot confirm Pc < 1e-6 -> at least Elevated")
    if pc_elev or miss_elev or not pc_known:
        return ELEVATED, reasons

    reasons.append(f"R6 safe rule: Pc = {_fmt_pc(log_pc)} < 1e-6 AND miss = {km:.3f} km > 5.0 km -> Safe")
    return SAFE, reasons


def classify(pc: float, miss_distance_m: float) -> tuple[str, list[str]]:
    """Classify from a raw probability Pc (0..1) and miss distance in metres."""
    if pc is None or (isinstance(pc, float) and math.isnan(pc)):
        log_pc = float("nan")
    elif pc <= 0:
        log_pc = -30.0
    else:
        log_pc = math.log10(pc)
    return classify_log(log_pc, miss_distance_m)


def classify_array(log_pc, miss_distance_m) -> np.ndarray:
    """Vectorised classify_log returning tier names (same rules, no reasons)."""
    r = np.asarray(log_pc, dtype=float)
    m = np.asarray(miss_distance_m, dtype=float)
    unknown = np.isnan(r)
    crit = (r >= LOG_PC_CRITICAL) | (m < MISS_CRITICAL_M)
    elev = (r >= LOG_PC_ELEVATED) | (m <= MISS_ELEVATED_M) | unknown
    return np.where(crit, CRITICAL, np.where(elev, ELEVATED, SAFE))


def fired_rule_codes(log_pc, miss_distance_m) -> list[str]:
    """Short rule codes, e.g. ['R2'] — used for the insights breakdown."""
    _, reasons = classify_log(log_pc, miss_distance_m)
    return [r.split()[0] for r in reasons]

import math

import numpy as np
import pytest

from orbitguard.tiers import CRITICAL, ELEVATED, SAFE, classify, classify_array, classify_log

FAR = 50_000.0  # miss distance that never fires a miss rule
TINY = 1e-12    # Pc that never fires a Pc rule


@pytest.mark.parametrize("pc,miss,tier", [
    # Pc boundaries (miss far away)
    (1e-4, FAR, CRITICAL), (1.0001e-4, FAR, CRITICAL), (0.9999e-4, FAR, ELEVATED),
    (1e-6, FAR, ELEVATED), (1.0001e-6, FAR, ELEVATED), (0.9999e-6, FAR, SAFE),
    (0.0, FAR, SAFE),
    # miss boundaries (Pc tiny)
    (TINY, 999.999, CRITICAL), (TINY, 1000.0, ELEVATED), (TINY, 1000.001, ELEVATED),
    (TINY, 5000.0, ELEVATED), (TINY, 5000.001, SAFE), (TINY, 4999.999, ELEVATED),
    # precedence
    (1e-3, 3000.0, CRITICAL), (1e-5, 500.0, CRITICAL), (1e-5, 10_000.0, ELEVATED),
    (1e-8, 2000.0, ELEVATED),
])
def test_boundaries(pc, miss, tier):
    assert classify(pc, miss)[0] == tier


def test_log_space_exact_boundaries():
    assert classify_log(-4.0, FAR)[0] == CRITICAL
    assert classify_log(-4.0000001, FAR)[0] == ELEVATED
    assert classify_log(-6.0, FAR)[0] == ELEVATED
    assert classify_log(-6.001, FAR)[0] == SAFE   # LRP clip value lands in Safe
    assert classify_log(-30.0, FAR)[0] == SAFE


def test_reasons_list_every_fired_rule():
    tier, reasons = classify(2e-4, 800.0)
    assert tier == CRITICAL
    assert any(r.startswith("R1") for r in reasons) and any(r.startswith("R2") for r in reasons)
    tier, reasons = classify(1e-9, 20_000.0)
    assert tier == SAFE and reasons[0].startswith("R6") and "20.000 km" in reasons[0]


def test_unknown_pc_never_safe():
    tier, reasons = classify(float("nan"), FAR)
    assert tier == ELEVATED and any("R5" in r for r in reasons)
    assert classify(float("nan"), 500.0)[0] == CRITICAL


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.uniform(-12, -2, 500), [-4, -6, -6.001, -30, np.nan]])
    m = np.concatenate([rng.uniform(0, 10_000, 500), [1000, 5000, 5000.0001, 999.99, 7000]])
    vec = classify_array(r, m)
    for i in range(len(r)):
        assert vec[i] == classify_log(r[i], m[i])[0]

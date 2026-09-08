"""Validity properties of the conformal layer."""

import numpy as np
import pytest

from sentinel_e.conformal import (
    ConformalCalibrator,
    MondrianConformalCalibrator,
    QuantileTaxonomy,
    ResidualConformalCalibrator,
    beta_calibration_levels,
    dkw_inflation,
    effective_sample_size,
    roc_auc,
    weighted_dkw_inflation,
)


def test_split_conformal_is_marginally_super_uniform():
    """Marginal validity averages over the calibration draw, so the draw is resampled.

    Conditioning on a single fixed calibration set would leave an O(n^{-1/2})
    deviation -- which is not a bug but precisely the gap that the Beta levels
    of ``mode='beta'`` are there to close.
    """
    rng = np.random.default_rng(0)
    n_cal, n_test, reps = 500, 400, 400
    hits = {u: 0 for u in (0.01, 0.05, 0.1, 0.5)}
    for _ in range(reps):
        cal = ConformalCalibrator(delta=0.0, mode="none").fit(rng.normal(size=n_cal))
        p = cal.p_values(rng.normal(size=n_test))
        for u in hits:
            hits[u] += int((p <= u).sum())
    total = reps * n_test
    for u, k in hits.items():
        assert k / total <= u + 4 * np.sqrt(u * (1 - u) / total)


def test_calibration_conditional_gap_is_real_without_correction():
    """A single fixed calibration set is *not* conditionally uniform."""
    rng = np.random.default_rng(10)
    worst = 0.0
    for _ in range(40):
        cal = ConformalCalibrator(delta=0.0, mode="none").fit(rng.normal(size=400))
        p = cal.p_values(rng.normal(size=20000))
        worst = max(worst, abs((p <= 0.5).mean() - 0.5))
    # The deviation is of order 1/sqrt(n_cal) = 0.05, far above test noise.
    assert worst > 0.01


def test_p_values_decrease_with_score():
    rng = np.random.default_rng(1)
    cal = ConformalCalibrator(delta=0.0, mode="none").fit(rng.normal(size=1000))
    p = cal.p_values(np.array([-3.0, 0.0, 3.0]))
    assert p[0] > p[1] > p[2]


def test_beta_levels_are_monotone_and_tighter_than_dkw():
    n, delta = 2000, 1e-3
    levels = beta_calibration_levels(n, delta)
    assert levels.size == n + 1
    assert np.all(np.diff(levels) >= -1e-12)
    assert levels[-1] == pytest.approx(1.0)
    assert levels[0] < 1.0 / (n + 1) + dkw_inflation(n, delta)


def test_beta_levels_give_calibration_conditional_validity():
    """On at least 1 - delta of calibration draws the levels must hold conditionally."""
    rng = np.random.default_rng(2)
    n, delta, reps = 800, 0.10, 200
    levels = beta_calibration_levels(n, delta)
    probe = [0, 3, 9, 49]
    failures = 0
    for _ in range(reps):
        cal = np.sort(rng.normal(size=n))
        test = rng.normal(size=6000)
        n_ge = n - np.searchsorted(cal, test, side="left")
        p = levels[n_ge]
        for j in probe:
            u = levels[j]
            if (p <= u).mean() > u + 4 * np.sqrt(u * (1 - u) / 6000):
                failures += 1
                break
    assert failures <= 0.5 * delta * reps  # comfortably inside the budget


def test_dkw_and_effective_size():
    assert dkw_inflation(100, 0.0) == 0.0
    assert dkw_inflation(100, 0.05) > dkw_inflation(1000, 0.05)
    w = np.ones(50) / 50
    assert effective_sample_size(w) == pytest.approx(50.0)
    skewed = np.zeros(50)
    skewed[0] = 1.0
    assert effective_sample_size(skewed) == pytest.approx(1.0)
    assert weighted_dkw_inflation(skewed, 0.05) > weighted_dkw_inflation(w, 0.05)


def test_roc_auc_matches_brute_force():
    rng = np.random.default_rng(3)
    s = rng.normal(size=200)
    y = rng.random(200) < 0.4
    pos, neg = s[y], s[~y]
    brute = np.mean(
        (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    )
    assert roc_auc(s, y) == pytest.approx(brute, abs=1e-12)


def test_mondrian_is_group_conditionally_valid_under_drift():
    """Pooled calibration fails per-group; Mondrian does not."""
    rng = np.random.default_rng(4)
    n = 24000
    ctx_cal = rng.uniform(-1, 1, size=(n, 1))
    cal = rng.normal(loc=2.0 * ctx_cal[:, 0], scale=1.0)

    tax = QuantileTaxonomy(continuous_cols=(0,), categorical_cols=(), n_bins=6)
    mon = MondrianConformalCalibrator(taxonomy=tax, delta=1e-3, min_bin=50).fit(
        cal, ctx_cal
    )
    pooled = ConformalCalibrator(delta=1e-3).fit(cal)

    ctx_hi = np.full((20000, 1), 0.8)
    test_hi = rng.normal(loc=2.0 * 0.8, scale=1.0, size=20000)
    p_mon, _ = mon.p_values(test_hi, ctx_hi)
    p_pool = pooled.p_values(test_hi)
    assert (p_mon <= 0.1).mean() <= 0.1 + 0.02
    assert (p_pool <= 0.1).mean() > 0.2  # pooled is badly anti-conservative here


def test_residual_conformal_handles_multivariate_context():
    rng = np.random.default_rng(5)
    n = 20000
    x = rng.normal(size=(n, 3))
    mu = 0.7 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 0] * x[:, 2]
    sd = np.exp(0.3 * x[:, 2])
    cal = mu + sd * rng.normal(size=n)

    rc = ResidualConformalCalibrator(delta=1e-3).fit(cal, x)
    xt = rng.normal(size=(30000, 3))
    mt = 0.7 * xt[:, 0] - 0.4 * xt[:, 1] + 0.3 * xt[:, 0] * xt[:, 2]
    st = mt + np.exp(0.3 * xt[:, 2]) * rng.normal(size=30000)
    p, active = rc.p_values(st, xt)
    p = p[active]
    for u in (0.05, 0.1, 0.25):
        assert (p <= u).mean() <= u + 0.02
    assert active.mean() > 0.9


def test_residual_conformal_abstains_out_of_support():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(4000, 2))
    s = x[:, 0] + rng.normal(size=4000)
    rc = ResidualConformalCalibrator(delta=1e-3).fit(s, x)
    far = np.full((100, 2), 12.0)
    p, active = rc.p_values(np.zeros(100), far)
    assert not active.any()
    assert np.allclose(p, 1.0)


def test_calibrator_rejects_empty_and_unfitted():
    with pytest.raises(ValueError):
        ConformalCalibrator().fit([])
    with pytest.raises(RuntimeError):
        ConformalCalibrator().p_values([0.1])

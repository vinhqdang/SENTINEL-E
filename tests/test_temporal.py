"""Temporal decorrelation utilities."""

import numpy as np
import pytest

from sentinel_e.temporal import (
    acf,
    detrend,
    estimate_decorrelation_lag,
    ljung_box,
    thin_indices,
)


def _ar1(n, rho, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal() * np.sqrt(1 - rho**2)
    return x


def test_acf_of_white_noise_is_near_zero():
    rng = np.random.default_rng(0)
    r = acf(rng.normal(size=20000), n_lags=10)
    assert r[0] == pytest.approx(1.0)
    assert np.abs(r[1:]).max() < 0.05


@pytest.mark.parametrize("rho", [0.5, 0.85, 0.95])
def test_acf_recovers_the_ar1_coefficient(rho):
    rng = np.random.default_rng(1)
    r = acf(_ar1(60000, rho, rng), n_lags=3)
    assert r[1] == pytest.approx(rho, abs=0.03)
    assert r[2] == pytest.approx(rho**2, abs=0.05)


def test_decorrelation_lag_grows_with_persistence():
    rng = np.random.default_rng(2)
    lo = estimate_decorrelation_lag(_ar1(40000, 0.5, rng))
    hi = estimate_decorrelation_lag(_ar1(40000, 0.95, rng))
    assert lo < hi
    # rho^k < 0.05  =>  k > log(0.05)/log(rho)
    assert hi == pytest.approx(np.log(0.05) / np.log(0.95), rel=0.4)


def test_detrend_removes_a_context_driven_drift():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 8 * np.pi, 20000)
    ctx = np.stack([np.sin(t), np.cos(t)], axis=1)
    y = 3.0 * ctx[:, 0] - 2.0 * ctx[:, 1] ** 2 + rng.normal(size=20000) * 0.1
    assert abs(acf(y, 50)[50]) > 0.2                 # trend dominates the raw ACF
    assert abs(acf(detrend(y, ctx), 50)[50]) < 0.05  # residual is white


def test_lag_estimation_ignores_predictable_drift():
    rng = np.random.default_rng(4)
    t = np.arange(40000)
    ctx = np.sin(2 * np.pi * t / 12000)[:, None]
    y = 2.0 * ctx[:, 0] + _ar1(40000, 0.8, rng)
    assert estimate_decorrelation_lag(y) > 60                       # confounded
    assert estimate_decorrelation_lag(y, context=ctx) < 30          # corrected


def test_thin_indices_spacing():
    idx = thin_indices(100, 7)
    assert idx[0] == 0 and np.all(np.diff(idx) == 7) and idx[-1] < 100
    assert np.array_equal(thin_indices(10, 1), np.arange(10))


def test_ljung_box_separates_dependent_from_independent():
    rng = np.random.default_rng(5)
    x = _ar1(20000, 0.85, rng)
    assert ljung_box(x, 20)[1] < 1e-6
    assert ljung_box(x[::25], 20)[1] > 0.01
    assert ljung_box(rng.normal(size=20000), 20)[1] > 0.01

"""Block betting: combiner math, block super-uniformity, and the unchanged default path."""

import numpy as np
import pytest
from scipy import stats

from sentinel_e.block_betting import block_pvalues, block_starts, combine_block
from sentinel_e.pipeline import SentinelE
from sentinel_e.temporal import thin_indices


def _ar1_uniform(rng, n_rows, length, rho):
    """Rows of a stationary Gaussian AR(1) pushed through Phi: uniform marginals, strong serial dependence."""
    z = np.empty((n_rows, length))
    z[:, 0] = rng.standard_normal(n_rows)
    innov = rng.standard_normal((n_rows, length))
    s = np.sqrt(1.0 - rho * rho)
    for t in range(1, length):
        z[:, t] = rho * z[:, t - 1] + s * innov[:, t]
    return stats.norm.cdf(z)


# --------------------------------------------------------------------------- #
# Combiner math
# --------------------------------------------------------------------------- #
def test_bonferroni_formula_and_clipping():
    P = np.array([[0.02, 0.5, 0.9], [0.4, 0.45, 0.6]])
    np.testing.assert_allclose(combine_block(P, "bonferroni"), [min(1, 3 * 0.02), min(1, 3 * 0.4)])
    # the minimum is clipped at 1
    np.testing.assert_allclose(combine_block(np.array([[0.5, 0.6]]), "bonferroni"), [1.0])


def test_mean2_formula_and_clipping():
    P = np.array([[0.02, 0.5, 0.9], [0.4, 0.45, 0.6]])
    np.testing.assert_allclose(combine_block(P, "mean2"), [2 * P[0].mean(), 2 * P[1].mean()])
    np.testing.assert_allclose(combine_block(np.array([[0.7, 0.9]]), "mean2"), [1.0])


def test_single_frame_block():
    """B = 1: Bonferroni is the identity; twice-the-mean doubles the frame p-value (clipped at 1)."""
    p = np.array([[0.03], [0.5], [0.99]])
    np.testing.assert_allclose(combine_block(p, "bonferroni"), [0.03, 0.5, 0.99])
    np.testing.assert_allclose(combine_block(p, "mean2"), [0.06, 1.0, 1.0])


def test_unknown_combiner_rejected():
    with pytest.raises(ValueError):
        combine_block(np.array([[0.1, 0.2]]), "harmonic")


def test_block_starts_and_gap():
    np.testing.assert_array_equal(block_starts(10, 5, 0), [0, 5])
    np.testing.assert_array_equal(block_starts(12, 5, 2), [0, 7])   # blocks [0,5) and [7,12)
    np.testing.assert_array_equal(block_starts(4, 5, 0), [])         # no complete block


def test_block_pvalues_indices_active_and_gap():
    p = np.arange(1, 13, dtype=float) / 13.0
    act = np.ones(12, dtype=bool)
    act[7] = False
    pb, ends, ab = block_pvalues(p, 3, gap=1, combiner="mean2", active=act)
    # blocks start at 0, 4, 8 (length 3, gap 1); the last block [8,11) is complete
    np.testing.assert_array_equal(ends, [2, 6, 10])
    np.testing.assert_allclose(pb, [min(1, 2 * p[0:3].mean()), min(1, 2 * p[4:7].mean()),
                                    min(1, 2 * p[8:11].mean())])
    np.testing.assert_array_equal(ab, [True, True, True])
    ab2 = block_pvalues(p, 3, gap=1, active=act)[2]
    np.testing.assert_array_equal(ab2, [True, True, True])
    act[5] = False                       # frame 5 lies in block [4,7)
    np.testing.assert_array_equal(block_pvalues(p, 3, gap=1, active=act)[2], [True, False, True])


# --------------------------------------------------------------------------- #
# Super-uniformity of a single block p-value under arbitrary within-block dependence
# --------------------------------------------------------------------------- #
GRID = np.array([0.01, 0.05, 0.1, 0.2, 0.5])


@pytest.mark.parametrize("combiner", ["bonferroni", "mean2"])
@pytest.mark.parametrize("B", [1, 5, 10])
def test_block_p_super_uniform_iid(combiner, B):
    rng = np.random.default_rng(1234 + B)
    n = 40000
    P = rng.uniform(size=(n, B))
    pB = combine_block(P, combiner)
    emp = np.array([np.mean(pB <= u) for u in GRID])
    tol = 4.0 * np.sqrt(GRID * (1 - GRID) / n)
    assert np.all(emp <= GRID + tol), (combiner, B, emp)


@pytest.mark.parametrize("combiner", ["bonferroni", "mean2"])
@pytest.mark.parametrize("B", [5, 10, 20])
def test_block_p_super_uniform_strong_ar1_copula(combiner, B):
    """Every frame's p-value is uniform, but consecutive frames are strongly dependent (rho = 0.95)."""
    rng = np.random.default_rng(99 + B)
    n = 40000
    U = _ar1_uniform(rng, n, B, rho=0.95)
    # each row is one block: its frames are uniform marginally but serially dependent
    pB = combine_block(U, combiner)
    emp = np.array([np.mean(pB <= u) for u in GRID])
    tol = 4.0 * np.sqrt(GRID * (1 - GRID) / n)
    assert np.all(emp <= GRID + tol), (combiner, B, emp)


def test_block_p_super_uniform_under_anti_dependence():
    """Worst-case style: pairs with p2 = 1 - p1 (negatively dependent) and comonotone pairs."""
    rng = np.random.default_rng(5)
    n = 40000
    u = rng.uniform(size=n)
    for P in (np.column_stack([u, 1 - u]), np.column_stack([u, u])):
        for c in ("bonferroni", "mean2"):
            pB = combine_block(P, c)
            emp = np.array([np.mean(pB <= g) for g in GRID])
            assert np.all(emp <= GRID + 4.0 * np.sqrt(GRID * (1 - GRID) / n)), (c, emp)


# --------------------------------------------------------------------------- #
# Library: default path unchanged, block path behaves as specified
# --------------------------------------------------------------------------- #
def _seeded_stream(seed=7, n_cal=3000, n_test=2000, shift_from=1500):
    rng = np.random.default_rng(seed)
    cal = rng.gamma(2.0, 1.0, size=n_cal)
    test = rng.gamma(2.0, 1.0, size=n_test)
    test[shift_from:] += 3.0
    return cal, test


def test_default_path_bit_identical_to_thinned_reference():
    cal, test = _seeded_stream()
    model = SentinelE(alpha=0.05, calibration="pooled", lag=3, restart=True).fit(cal)
    res = model.run(test)
    # reference: the pre-existing thinned path, written out by hand
    idx = thin_indices(test.size, 3)
    ref_model = SentinelE(alpha=0.05, calibration="pooled", lag=3, restart=True).fit(cal)
    p, active = ref_model.p_values(test[idx])
    ref_model.detector.reset()
    lw, al = ref_model.detector.run(p, active=active)
    np.testing.assert_array_equal(res.bet_index, idx)
    np.testing.assert_array_equal(np.flatnonzero(res.alarms), idx[np.flatnonzero(al)])
    np.testing.assert_array_equal(res.p_values[idx], p)
    np.testing.assert_array_equal(res.log_wealth[idx], lw)


def test_block_size_none_equals_default():
    cal, test = _seeded_stream(seed=3)
    a = SentinelE(alpha=0.05, calibration="pooled", lag=4, restart=True).fit(cal).run(test)
    b = SentinelE(alpha=0.05, calibration="pooled", lag=4, restart=True, block_size=None,
                  block_combiner="mean2", block_gap=9).fit(cal).run(test)
    np.testing.assert_array_equal(a.alarms, b.alarms)
    np.testing.assert_array_equal(a.p_values, b.p_values)
    np.testing.assert_array_equal(a.log_wealth, b.log_wealth)


def test_block_size_one_equals_lag_one_thinning():
    """With B = 1 and no gap the block p-value is the frame p-value, so block betting equals lag-1 thinning."""
    cal, test = _seeded_stream(seed=11)
    blk = SentinelE(alpha=0.05, calibration="pooled", lag=1, restart=True,
                    block_size=1, block_combiner="bonferroni").fit(cal).run(test)
    thin = SentinelE(alpha=0.05, calibration="pooled", lag=1, restart=True).fit(cal).run(test)
    np.testing.assert_array_equal(blk.bet_index, np.arange(test.size))
    np.testing.assert_array_equal(blk.alarms, thin.alarms)


def test_block_path_places_bets_at_block_ends_and_detects_shift():
    cal, test = _seeded_stream(seed=21, shift_from=1000)
    test[1000:] += 5.0   # a strong shift: post-shift p-values are near zero
    m = SentinelE(alpha=0.05, calibration="pooled", lag=2, restart=True,
                  block_size=10, block_combiner="mean2", block_gap=10)
    res = m.fit(cal).run(test)
    np.testing.assert_array_equal(res.bet_index, np.arange(9, test.size, 20))
    alarms = np.flatnonzero(res.alarms)
    assert alarms.size > 0
    assert set(alarms.tolist()) <= set(res.bet_index.tolist())
    assert alarms[0] >= 1000   # no alarm before the shift on this seeded stream


def test_block_mode_false_alarm_rate_on_null_is_controlled_in_practice():
    """A single seeded null check, not a proof: the block detector does not alarm on a null stream."""
    cal, test = _seeded_stream(seed=31, shift_from=10**9)
    res = SentinelE(alpha=0.05, calibration="pooled", restart=False, block_size=20,
                    block_combiner="bonferroni").fit(cal).run(test)
    assert not res.alarms.any()


@pytest.mark.parametrize("kw", [
    {"block_size": 0}, {"block_combiner": "harmonic"}, {"block_gap": -1},
])
def test_invalid_block_options_rejected(kw):
    with pytest.raises(ValueError):
        SentinelE(alpha=0.05, calibration="pooled", **kw)

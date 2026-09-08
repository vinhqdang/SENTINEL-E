"""e-BH and e-value merging."""

import numpy as np
import pytest

from sentinel_e.ebh import ebh, ebh_rejections, ebh_threshold, global_e_merge
from sentinel_e.metrics import fdr_power


def test_threshold_matches_the_definition():
    e = np.array([100.0, 40.0, 3.0, 0.5])
    alpha, K = 0.1, 4
    k, cutoff = ebh_threshold(e, alpha)
    order = np.sort(e)[::-1]
    expected = max(
        [j for j in range(1, K + 1) if order[j - 1] >= K / (alpha * j)], default=0
    )
    assert k == expected
    if k:
        assert cutoff == pytest.approx(order[k - 1])


def test_no_rejections_when_evidence_is_weak():
    assert not ebh(np.ones(20), 0.1).any()
    assert ebh_rejections(np.ones(20), 0.1).size == 0


def test_rejects_the_strongest_hypotheses():
    e = np.array([1.0, 1.0, 5000.0, 1.0, 4000.0])
    rej = ebh(e, 0.1)
    assert rej[2] and rej[4]
    assert not rej[[0, 1, 3]].any()


def test_fdr_controlled_under_arbitrary_dependence():
    """Perfectly correlated null e-values: BH-on-p needs a penalty, e-BH does not."""
    rng = np.random.default_rng(0)
    alpha, K, reps = 0.2, 20, 3000
    fdps = np.empty(reps)
    for r in range(reps):
        # One shared shock drives every camera: maximal positive dependence.
        shared = rng.exponential()
        e = np.exp(shared - 1.0) * np.ones(K)   # mean 1 => valid e-values
        fdps[r] = fdr_power(ebh(e, alpha), np.zeros(K, dtype=bool))["fdp"]
    assert fdps.mean() <= alpha


def test_fdr_controlled_with_a_mix_of_true_and_false_nulls():
    rng = np.random.default_rng(1)
    alpha, K, n_alt, reps = 0.1, 40, 10, 2000
    truth = np.zeros(K, dtype=bool)
    truth[:n_alt] = True
    fdps, powers = np.empty(reps), np.empty(reps)
    for r in range(reps):
        e = np.empty(K)
        e[truth] = rng.exponential(scale=200.0, size=n_alt)
        # Null e-values with mean exactly one.
        e[~truth] = rng.exponential(scale=1.0, size=K - n_alt)
        m = fdr_power(ebh(e, alpha), truth)
        fdps[r], powers[r] = m["fdp"], m["power"]
    assert fdps.mean() <= alpha
    assert powers.mean() > 0.5


def test_average_merge_is_an_e_value_under_dependence():
    rng = np.random.default_rng(2)
    reps = 20000
    merged = np.array(
        [global_e_merge(rng.exponential(size=8) * np.ones(8)) for _ in range(reps)]
    )
    se = merged.std(ddof=1) / np.sqrt(reps)
    assert merged.mean() <= 1.0 + 4 * se


def test_merge_validation():
    assert global_e_merge([]) == 1.0
    assert global_e_merge([2.0, 8.0], "product") == pytest.approx(16.0)
    with pytest.raises(ValueError):
        global_e_merge([1.0], "median")
    with pytest.raises(ValueError):
        ebh([-1.0], 0.1)
    with pytest.raises(ValueError):
        ebh([1.0], 1.5)

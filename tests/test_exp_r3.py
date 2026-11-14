"""Leave-one-video-out calibration and the lag/knob selection rules of protocol v3."""

import numpy as np
import pytest

from experiments.real_data.exp_r3_per_video import (
    _runs, build_calibration, cp_interval, lovo_folds, poisson_rate_interval,
    select_knob, select_lag,
)


def _synthetic_train(n_videos=7, length=40):
    """Video j takes values in [1000 j, 1000 j + length), so membership is unambiguous."""
    return [1000.0 * j + np.arange(length, dtype=float) for j in range(n_videos)]


@pytest.mark.parametrize("held", [[0], [3], [6]])
def test_lovo_calibration_never_contains_the_monitored_video(held):
    train = _synthetic_train()
    cal = build_calibration(train, held)
    for j in held:
        assert not np.isin(train[j], cal).any()
    others = [v for j, v in enumerate(train) if j not in held]
    assert cal.size == sum(v.size for v in others)
    for v in others:
        assert np.isin(v, cal).all()


def test_lovo_folds_are_single_videos_and_mod10_folds_partition_the_videos():
    assert lovo_folds(5, "lovo") == [[0], [1], [2], [3], [4]]
    folds = lovo_folds(23, "mod10")
    assert len(folds) == 10
    assert sorted(j for f in folds for j in f) == list(range(23))
    train = _synthetic_train(n_videos=23)
    for fold in folds:
        cal = build_calibration(train, fold)
        for j in fold:
            assert not np.isin(train[j], cal).any()


def test_build_calibration_refuses_an_empty_calibration_set():
    with pytest.raises(ValueError):
        build_calibration(_synthetic_train(n_videos=1), [0])


def test_select_lag_picks_the_smallest_admissible_grid_lag():
    grid = [1, 2, 3, 5, 10]
    curve = {1: (3, 16), 2: (1, 16), 3: (0, 16), 5: (0, 16), 10: (0, 16)}
    # alpha 0.1: 3/16 is not admissible at lag 1, 1/16 is at lag 2, so L* = 2 even though 3 and 5 also qualify.
    assert select_lag(curve, 0.1, grid) == (2, False)
    # Grid order given unsorted must not change the rule.
    assert select_lag(curve, 0.1, [10, 3, 1, 5, 2]) == (2, False)


def test_select_lag_flags_failure_and_returns_the_largest_lag():
    grid = [1, 2, 3]
    curve = {1: (16, 16), 2: (9, 16), 3: (5, 16)}
    assert select_lag(curve, 0.1, grid) == (3, True)


def test_select_knob_takes_the_first_admissible_knob_in_sweep_order():
    props = [(3, 16), (1, 16), (0, 16)]          # most aggressive first
    assert select_knob(props, 0.1) == (1, False)  # index 1 is the first admissible one


def test_select_knob_flags_when_nothing_is_admissible():
    props = [(9, 16), (5, 16), (4, 16)]
    assert select_knob(props, 0.1) == (2, True)   # most conservative knob, flagged


def test_clopper_pearson_zero_successes_upper_bound():
    lo, hi = cp_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** 0.1, abs=1e-9)


def test_poisson_interval_zero_count_upper_bound_per_hour():
    lo, hi = poisson_rate_interval(0, 2.0)
    assert lo == 0.0
    # Exact upper limit for a zero count is chi2_{0.975}(2)/2 = 3.6889 events, divided by 2 hours.
    assert hi == pytest.approx(3.6889 / 2.0, rel=1e-3)


def test_runs_split_labels_into_maximal_true_segments():
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 1])
    assert _runs(labels) == [(1, 3), (5, 8)]
    assert _runs(np.array([1, 1])) == [(0, 2)]
    assert _runs(np.zeros(4, bool)) == []

"""Warm-up normalisation helpers and the pre-registered warm-up selection rule of exp_r4."""

import numpy as np
import pytest

from experiments.real_data.exp_r4_warmup import (
    EPS, choose_w, onset_eligible, post_warmup, usable_null, warmup_normalise,
)


def _video(n=200, n_warm=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 0.5, size=n)


def test_output_is_post_warmup_part_only():
    s = _video()
    z = warmup_normalise(s, 50)
    assert z.size == s.size - 50


def test_standardisation_uses_median_and_mad_of_the_warmup():
    s = np.concatenate([np.array([1.0, 2.0, 3.0, 4.0, 100.0]), np.array([10.0, 20.0])])
    # warm-up = first 5 values: median 3, MAD = median(|x-3|) = median(2,1,0,1,97) = 1
    z = warmup_normalise(s, 5)
    np.testing.assert_allclose(z, (np.array([10.0, 20.0]) - 3.0) / (1.0 + EPS))


def test_invariant_to_affine_rescaling_of_the_whole_video():
    s = _video()
    a, b = 7.5, -3.0
    # exact without the eps term; with the fixed eps the difference is only the eps effect
    np.testing.assert_allclose(warmup_normalise(s, 50, eps=0.0),
                               warmup_normalise(a * s + b, 50, eps=0.0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(warmup_normalise(s, 50), warmup_normalise(a * s + b, 50), rtol=1e-3, atol=1e-4)


def test_warmup_statistics_do_not_depend_on_the_monitored_part():
    s = _video()
    s2 = s.copy()
    s2[50:] = s2[50:] * 10 + 100
    med = np.median(s[:50])
    mad = np.median(np.abs(s[:50] - med))
    np.testing.assert_allclose(warmup_normalise(s2, 50), (s2[50:] - med) / (mad + EPS))


def test_constant_warmup_gives_finite_values():
    s = np.concatenate([np.full(10, 2.0), np.array([3.0, 2.0])])
    z = warmup_normalise(s, 10)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z, np.array([1.0, 0.0]) / EPS)


@pytest.mark.parametrize("n_warm", [0, 200, 250])
def test_rejects_warmup_without_post_warmup_part(n_warm):
    with pytest.raises(ValueError):
        warmup_normalise(_video(200), n_warm)


def test_raw_arm_is_the_unchanged_post_warmup_segment():
    s = _video()
    np.testing.assert_array_equal(post_warmup(s, 50, "raw"), s[50:])
    with pytest.raises(ValueError):
        post_warmup(s, 50, "bogus")


def test_usable_null_and_onset_boundaries():
    assert usable_null(20, 20) and not usable_null(19, 20)
    # onset must be at or after W + 2 s: with 10 fps and W = 5, n_warm = 50, margin = 20 frames
    assert onset_eligible(70, 50, 20)
    assert not onset_eligible(69, 50, 20)
    assert not onset_eligible(None, 50, 20)
    assert not onset_eligible(-1, 50, 20)


def test_choose_w_picks_smallest_feasible_when_detection_is_not_evaluable():
    feas = {5: True, 10: True, 20: True}
    assert choose_w(feas, {5: None, 10: None, 20: None}) == (5, "tie-break: training-positive detection not evaluable; smallest feasible W")


def test_choose_w_maximises_detection_among_feasible_only():
    feas = {5: True, 10: False, 20: True}
    W, _ = choose_w(feas, {5: 0.4, 10: 0.9, 20: 0.7})
    assert W == 20
    feas = {5: True, 10: True, 20: True}
    W, _ = choose_w(feas, {5: 0.5, 10: 0.5, 20: 0.4})
    assert W == 5


def test_choose_w_no_feasible_window():
    W, reason = choose_w({5: False, 10: False, 20: False}, {5: 0.5, 10: 0.5, 20: 0.5})
    assert W is None and "no admissible W" in reason

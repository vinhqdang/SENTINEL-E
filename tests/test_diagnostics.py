"""Drift monitor: quiet under valid (uniform) p-values, flags a small-p excess."""

import numpy as np
import pytest

from sentinel_e.diagnostics import drift_monitor


def test_uniform_p_values_raise_no_flags():
    rng = np.random.default_rng(12345)
    p = rng.uniform(size=2000 * 200)              # 200 windows of W = 2000
    out = drift_monitor(p, window=2000, q=0.1, z=4.0)
    assert out["fraction"].size == 200
    assert not out["flags"].any()
    assert out["first_flag"] == -1 and out["first_flag_window"] == -1
    assert out["threshold"] == pytest.approx(0.1 + 4 * np.sqrt(0.09 / 2000))


def test_beta_small_p_excess_is_flagged():
    rng = np.random.default_rng(7)
    p = rng.beta(0.7, 1.0, size=2000 * 20)        # P(p <= 0.1) = 0.1**0.7 ~ 0.20
    out = drift_monitor(p, window=2000, q=0.1, z=4.0)
    assert out["flags"].all()
    assert out["first_flag_window"] == 0
    assert out["first_flag"] == 1999              # last betting instant of window 0


def test_active_mask_and_index_mapping():
    rng = np.random.default_rng(3)
    n = 6000
    p = rng.uniform(size=n)
    active = np.ones(n, dtype=bool)
    active[::2] = False                           # half the instants are inactive
    p[~active] = 0.0                              # inactive values must be ignored
    out = drift_monitor(p, active=active, window=1000, q=0.1, z=4.0)
    assert out["fraction"].size == 3              # 3000 active values -> 3 windows
    assert np.all(active[out["ends"]])            # window ends are active instants
    assert not out["flags"].any()


def test_window_longer_than_data_gives_no_windows():
    out = drift_monitor(np.full(10, 0.01), window=50)
    assert out["fraction"].size == 0 and out["first_flag"] == -1


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        drift_monitor(np.ones(10), window=0)
    with pytest.raises(ValueError):
        drift_monitor(np.ones(10), active=np.ones(5, dtype=bool), window=2)

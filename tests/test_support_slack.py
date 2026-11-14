"""Opt-in stricter support region: default unchanged, smaller slack abstains more."""

import numpy as np
import pytest

from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator


def _stream(activity_shift=1.0, seed=3, T=30_000):
    return StreamSimulator(StreamConfig(T=T, activity_shift=activity_shift),
                           seed=seed).simulate_camera(change_point=None)


def test_default_is_unchanged_bit_identical():
    st = _stream()
    a = SentinelE(alpha=0.01, calibration="residual", restart=True)
    b = SentinelE(alpha=0.01, calibration="residual", restart=True, support_slack=1.5)
    ra, rb = a.run_stream(st), b.run_stream(st)
    assert a.calibrator.mahalanobis_slack == 1.5
    assert np.array_equal(ra.p_values, rb.p_values)
    assert np.array_equal(ra.active, rb.active)
    assert np.array_equal(ra.alarms, rb.alarms)


def test_smaller_slack_is_a_subset_and_abstains_more():
    st = _stream(activity_shift=3.0)
    base = SentinelE(alpha=0.01, calibration="residual", restart=True)
    strict = SentinelE(alpha=0.01, calibration="residual", restart=True, support_slack=0.5)
    rb, rs = base.run_stream(st), strict.run_stream(st)
    assert np.all(rs.active <= rb.active)          # stricter support only removes bets
    assert rs.abstention_rate > rb.abstention_rate
    assert rs.active.sum() < rb.active.sum()


def test_support_slack_rejects_non_positive():
    with pytest.raises(ValueError):
        SentinelE(calibration="residual", support_slack=0.0)
    with pytest.raises(ValueError):
        SentinelE(calibration="residual", support_slack=-1.0)

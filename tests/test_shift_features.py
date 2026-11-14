"""Phase-invariant shift gate: ``shift_features`` restricts the gate and the weights.

``shift_features=None`` must reproduce the original full-context weights exactly.
A column subset must be used consistently by the classifier gate and by the
density ratio that produces the weights.
"""

import numpy as np
import pytest

from sentinel_e.conformal import (
    LogisticDensityRatio,
    MondrianConformalCalibrator,
    QuantileTaxonomy,
    ResidualConformalCalibrator,
)
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices

THRESHOLD = 0.60


def _synthetic(seed=0, n_cal=800, n_dep=400, dep_shift=1.0):
    rng = np.random.default_rng(seed)
    ctx = rng.normal(size=(n_cal, 4))
    scores = rng.normal(size=n_cal) + 0.1 * ctx[:, 0]
    dep = rng.normal(size=(n_dep, 4))
    dep[:, 0] += dep_shift
    return scores, ctx, dep


# --------------------------------------------------------------------------- #
# None reproduces the original weights exactly
# --------------------------------------------------------------------------- #
def test_residual_none_reproduces_full_context_weights():
    scores, ctx, dep = _synthetic()
    cal = ResidualConformalCalibrator(weighted=True).fit(scores, ctx)
    cal.update_shift(dep)
    assert cal._weights is not None, "the synthetic shift must be detected"
    assert cal.shift_auc >= THRESHOLD

    # Original code path: density ratio fitted and evaluated on all columns.
    dr = LogisticDensityRatio(clip=cal.weight_clip).fit(cal._sorted_ctx, dep)
    w = np.clip(dr.ratio(cal._sorted_ctx),
                np.exp(-cal.weight_clip), np.exp(cal.weight_clip))
    assert np.array_equal(cal._weights, w)


def test_residual_explicit_all_columns_equals_none():
    scores, ctx, dep = _synthetic(seed=1)
    a = ResidualConformalCalibrator(weighted=True).fit(scores, ctx)
    b = ResidualConformalCalibrator(weighted=True, shift_features=[0, 1, 2, 3]).fit(scores, ctx)
    a.update_shift(dep)
    b.update_shift(dep)
    assert a.shift_auc == pytest.approx(b.shift_auc, abs=1e-12)
    # Column selection copies the array, so BLAS sums may differ in the last bit.
    assert np.allclose(a._weights, b._weights, rtol=1e-12, atol=1e-12)
    assert a.n_eff == pytest.approx(b.n_eff, rel=1e-12)


def test_mondrian_none_reproduces_full_context_weights():
    scores, ctx, dep = _synthetic(seed=2, n_cal=1200, n_dep=800)

    def make(features):
        tax = QuantileTaxonomy(continuous_cols=(0,), categorical_cols=(), n_bins=3)
        return MondrianConformalCalibrator(
            taxonomy=tax, weighted=True, min_bin=50, shift_features=features
        ).fit(scores, ctx)

    cal = make(None)
    cal.update_shift(dep)
    detected = [b for b, rec in cal._bins.items() if "dr" in rec]
    assert detected, "the synthetic shift must be detected in at least one bin"

    # Original code path, bin by bin.
    dep_bins = cal.taxonomy(dep)
    for b in detected:
        rec = cal._bins[b]
        dep_b = dep[dep_bins == b]
        dr = LogisticDensityRatio(clip=cal.weight_clip).fit(rec["context"], dep_b)
        w = np.clip(dr.ratio(rec["context"]),
                    np.exp(-cal.weight_clip), np.exp(cal.weight_clip))
        order = np.argsort(cal._cal_scores[cal._cal_bin == b])
        assert np.array_equal(rec["weights"], w[order])

    explicit = make([0, 1, 2, 3])
    explicit.update_shift(dep)
    for b in cal._bins:
        assert np.allclose(cal._bins[b]["weights"], explicit._bins[b]["weights"],
                           rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Plumbing and validation
# --------------------------------------------------------------------------- #
def test_pipeline_passes_shift_features_through():
    scores, ctx, _ = _synthetic(seed=3)
    model = SentinelE(calibration="residual", weighted=True, shift_features=[1, 3])
    model.fit(scores, ctx)
    assert model.calibrator.shift_features == (1, 3)
    assert SentinelE(calibration="residual").shift_features is None


def test_out_of_range_or_unsupported_features_rejected():
    scores, ctx, _ = _synthetic(seed=4)
    with pytest.raises(ValueError):
        ResidualConformalCalibrator(shift_features=[7]).fit(scores, ctx)
    with pytest.raises(ValueError):
        ResidualConformalCalibrator(shift_features=[]).fit(scores, ctx)
    with pytest.raises(ValueError):
        SentinelE(calibration="pooled", shift_features=[1, 3])


# --------------------------------------------------------------------------- #
# Stream-level gate behaviour
# --------------------------------------------------------------------------- #
def _block_aucs(seed, activity_shift, features, T=90_000, gate="fixed"):
    """Held-out gate AUC for each deployment block, as in exp3c's gate loop."""
    st = StreamSimulator(
        StreamConfig(T=T, activity_shift=activity_shift), seed=seed
    ).simulate_camera(change_point=None)
    model = SentinelE(alpha=0.01, calibration="residual", weighted=True,
                      restart=True, shift_features=features, shift_gate=gate)
    model.fit(st.calibration_scores, st.calibration_context)
    idx = thin_indices(st.T, model.decorrelation_lag)
    ctx = np.atleast_2d(np.asarray(st.context, dtype=float))[idx]
    blk = model.shift_block
    aucs = []
    for start in range(blk, ctx.shape[0], blk):
        model.calibrator.update_shift(ctx[start - blk:start])
        aucs.append(model.calibrator.shift_auc)
    return np.asarray(aucs), model.calibrator.gate_threshold


# Measured on this simulator (see the report): with the fixed threshold 0.60 and
# shift_features=[1,3], a stationary null reached AUC 0.60 on 96% of blocks
# (median 0.67), because the binary weather column is nearly constant within a
# block.  The calibrated gate with all columns is tested below.
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Calibrated gate (all columns, gate_level 0.05) is not quiet on a stationary "
        "null in this simulator: about 0.49 of blocks fire over 8 seeds (90 blocks), "
        "because deployment null blocks have median AUC 0.86 against reference-window "
        "median 0.74. The cause is not isolated."
    ),
)
def test_calibrated_gate_quiet_on_stationary_null():
    rates = []
    for seed in (31, 32, 33, 34):
        aucs, thr = _block_aucs(seed=seed, activity_shift=1.0, features=None, gate="calibrated")
        rates.append(np.mean(aucs >= thr))
    assert np.mean(rates) <= 0.15


def test_calibrated_gate_reference_and_fallback():
    scores, ctx, _ = _synthetic(seed=6, n_cal=2000)
    cal = ResidualConformalCalibrator(weighted=True, shift_gate="calibrated",
                                      gate_window=400, gate_reference_windows=50).fit(scores, ctx)
    assert cal.gate_status == "calibrated"
    assert cal.gate_window_used == 400
    assert cal.gate_reference.shape == (50,)
    assert cal.gate_threshold == pytest.approx(np.quantile(cal.gate_reference, 0.95))
    # The reference uses the calibration half (n_cal // 2 rows).  If that holds
    # fewer than two windows, the window shrinks to half of it and is still used.
    short = ResidualConformalCalibrator(weighted=True, shift_gate="calibrated",
                                        gate_window=400).fit(*_synthetic(seed=7, n_cal=200)[:2])
    assert short.gate_status == "calibrated" and short.gate_window_used == 50
    # Too short for any usable window: falls back to the fixed threshold.
    tiny = ResidualConformalCalibrator(weighted=True, shift_gate="calibrated",
                                       gate_window=400).fit(*_synthetic(seed=8, n_cal=60)[:2])
    assert tiny.gate_status == "fallback_fixed"
    assert tiny.gate_threshold == tiny.shift_auc_threshold


def test_fixed_gate_is_default_and_uses_fixed_threshold():
    scores, ctx, _ = _synthetic(seed=9)
    cal = ResidualConformalCalibrator(weighted=True).fit(scores, ctx)
    assert cal.shift_gate == "fixed" and cal.gate_status == "fixed"
    assert cal.gate_threshold == cal.shift_auc_threshold
    assert cal.gate_reference is None


def test_activity_shift_fires_calibrated_gate():
    aucs, thr = _block_aucs(seed=22, activity_shift=3.0, features=None, gate="calibrated")
    assert aucs.size >= 5
    assert np.mean(aucs >= thr) >= 0.8


def test_p_values_use_the_same_column_subset_as_the_weights():
    """Once a shift is detected, the stored density ratio must be evaluated on the
    same columns it was fitted on, in both calibrators' p-value paths."""
    scores, ctx, dep = _synthetic(seed=5, n_cal=1200, n_dep=800)
    dep[:, 0] -= 1.0    # undo the column-0 shift: only column 3 moves
    dep[:, 3] += 1.0
    test_ctx = dep[:50]
    test_scores = np.linspace(-1.0, 2.0, 50)

    res = ResidualConformalCalibrator(weighted=True, shift_features=[1, 3]).fit(scores, ctx)
    res.update_shift(dep)
    assert res._dr is not None
    p, _ = res.p_values(test_scores, test_ctx)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))

    tax = QuantileTaxonomy(continuous_cols=(0,), categorical_cols=(), n_bins=3)
    mon = MondrianConformalCalibrator(
        taxonomy=tax, weighted=True, min_bin=50, shift_features=[1, 3]
    ).fit(scores, ctx)
    mon.update_shift(dep)
    assert any("dr" in rec for rec in mon._bins.values())
    p_m, _ = mon.p_values(test_scores, test_ctx)
    assert np.all(np.isfinite(p_m)) and np.all((p_m >= 0) & (p_m <= 1))

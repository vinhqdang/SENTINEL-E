"""End-to-end pipeline behaviour, including the guarantees under composition."""

import numpy as np
import pytest

from sentinel_e.gnn import FeatureTracker, HeuristicSpatialPrior, SpatialPriorConfig
from sentinel_e.metrics import detection_delay, fdr_power, summarize_runs
from sentinel_e.pipeline import FleetSentinelE, SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices


def _cfg(**kw):
    base = dict(T=40_000, calibration_regime="representative")
    base.update(kw)
    return StreamConfig(**base)


# --------------------------------------------------------------------------- #
# Single camera
# --------------------------------------------------------------------------- #
def test_run_reports_on_the_original_frame_grid():
    st = StreamSimulator(_cfg(T=20_000), seed=0).simulate_camera()
    res = SentinelE(alpha=0.01).run_stream(st)
    assert res.p_values.shape == (st.T,)
    assert res.alarms.shape == (st.T,)
    assert res.log_wealth.shape == (st.T,)
    assert res.lag > 1
    # Bets happen only on the thinned grid; every other frame is a neutral 1.
    assert np.array_equal(res.bet_index, thin_indices(st.T, res.lag))
    off_grid = np.setdiff1d(np.arange(st.T), res.bet_index)
    assert np.allclose(res.p_values[off_grid], 1.0)
    assert not res.alarms[off_grid].any()


def test_wealth_is_constant_between_bets():
    st = StreamSimulator(_cfg(T=10_000), seed=1).simulate_camera()
    res = SentinelE(alpha=0.01).run_stream(st)
    for a, b in zip(res.bet_index[:-1], res.bet_index[1:]):
        assert np.allclose(res.log_wealth[a:b], res.log_wealth[a])


def test_detects_an_event_and_stays_quiet_without_one():
    sim = StreamSimulator(_cfg(T=40_000), seed=2)
    quiet = SentinelE(alpha=0.01).run_stream(sim.simulate_camera())
    assert not quiet.alarms.any()
    loud = SentinelE(alpha=0.01).run_stream(sim.simulate_camera(change_point=20_000))
    assert loud.alarms.any()
    assert detection_delay(loud.alarms, 20_000) is not None


@pytest.mark.parametrize("calibration", ["residual", "mondrian", "pooled"])
def test_all_calibration_modes_run(calibration):
    st = StreamSimulator(_cfg(T=15_000), seed=3).simulate_camera(change_point=7_000)
    res = SentinelE(alpha=0.01, calibration=calibration).run_stream(st)
    assert np.all((res.p_values > 0) & (res.p_values <= 1.0))


def test_pooled_calibration_needs_no_context_but_others_do():
    st = StreamSimulator(_cfg(T=5_000), seed=4).simulate_camera()
    SentinelE(calibration="pooled").fit(st.calibration_scores)
    with pytest.raises(ValueError):
        SentinelE(calibration="residual").fit(st.calibration_scores)
    with pytest.raises(ValueError):
        SentinelE(calibration="mondrian").fit(st.calibration_scores)
    with pytest.raises(ValueError):
        SentinelE(calibration="nonsense")


def test_calibration_is_thinned_to_near_independence():
    st = StreamSimulator(_cfg(T=5_000, n_calibration=60_000), seed=5).simulate_camera()
    m = SentinelE(alpha=0.01, thin_calibration=True)
    m.fit(st.calibration_scores, st.calibration_context)
    assert m.n_calibration_effective == pytest.approx(60_000 / m.decorrelation_lag, rel=0.05)
    m2 = SentinelE(alpha=0.01, thin_calibration=False)
    m2.fit(st.calibration_scores, st.calibration_context)
    assert m2.n_calibration_effective == 60_000


def test_false_alarm_rate_is_controlled_on_null_streams():
    """The end-to-end guarantee, measured rather than asserted."""
    sim = StreamSimulator(_cfg(T=40_000), seed=6)
    alarms = [SentinelE(alpha=0.05).run_stream(sim.simulate_camera()).alarms
              for _ in range(40)]
    assert summarize_runs(alarms).pfa <= 0.05


def test_a_tighter_level_never_alarms_earlier():
    st = StreamSimulator(_cfg(T=40_000), seed=7).simulate_camera(change_point=20_000)
    loose = SentinelE(alpha=0.1).run_stream(st)
    tight = SentinelE(alpha=0.001).run_stream(st)
    d_loose = detection_delay(loose.alarms, 20_000)
    d_tight = detection_delay(tight.alarms, 20_000)
    if d_loose is not None and d_tight is not None:
        assert d_tight >= d_loose


# --------------------------------------------------------------------------- #
# Fleet
# --------------------------------------------------------------------------- #
def test_fleet_shapes_and_ebh_output():
    fleet = StreamSimulator(_cfg(T=20_000, n_cameras=8), seed=8).simulate_fleet(n_events=2)
    res = FleetSentinelE(alpha=0.01, fdr_level=0.1).run(fleet)
    K, B = res.log_wealth.shape
    assert K == 8 and B == res.bet_index.size
    assert res.p_values.shape == (K, B)
    assert res.ebh_rejected.shape == (K,)
    assert np.all(res.e_values >= 0)
    assert res.alarms_on_frame_grid(fleet.T).shape == (K, fleet.T)


def test_fleet_recovers_affected_cameras():
    fleet = StreamSimulator(_cfg(T=40_000, n_cameras=10), seed=9).simulate_fleet(n_events=3)
    res = FleetSentinelE(alpha=0.01, fdr_level=0.1).run(fleet)
    m = fdr_power(res.ebh_rejected, fleet.affected)
    assert m["power"] > 0.5
    assert m["fdp"] <= 0.5


def test_fleet_controls_false_rejections_on_a_null_fleet():
    fleet = StreamSimulator(_cfg(T=40_000, n_cameras=10), seed=10).simulate_fleet(n_events=0)
    res = FleetSentinelE(alpha=0.01, fdr_level=0.1).run(fleet)
    assert not res.ebh_rejected.any()
    assert res.global_e < 10.0


def test_heuristic_controller_runs_and_modulates():
    fleet = StreamSimulator(_cfg(T=20_000, n_cameras=8), seed=11).simulate_fleet(n_events=2)
    prior = HeuristicSpatialPrior(fleet.graph.normalized_adjacency())
    res = FleetSentinelE(alpha=0.01, controller=prior).run(fleet)
    assert res.hazards.min() >= SpatialPriorConfig().rho_min - 1e-12
    assert res.hazards.max() <= SpatialPriorConfig().rho_max + 1e-12
    assert res.hazards.std() > 0     # the controller is actually doing something


def test_an_adversarial_predictable_controller_cannot_break_validity():
    """Theorem 2: any predictable modulation preserves the guarantee."""
    rng = np.random.default_rng(12)

    def adversary(features):
        K = features.shape[0]
        # Maximally aggressive whenever any neighbour has positive wealth.
        hot = features[:, 0] > 0
        return np.where(hot, 5e-2, 1e-3), np.where(hot, 1.0, 1.0)

    fired = 0
    reps = 25
    for r in range(reps):
        fleet = StreamSimulator(
            _cfg(T=40_000, n_cameras=6), seed=200 + r
        ).simulate_fleet(n_events=0)
        res = FleetSentinelE(alpha=0.05, controller=adversary).run(fleet)
        fired += int(res.alarms.any())
    # Any alarm on a null fleet is a false alarm; with six cameras at alpha=0.05
    # the union bound allows 0.3, and the per-camera guarantee is what is tested.
    assert fired / reps <= 0.3


def test_feature_tracker_is_predictable_and_bounded():
    from sentinel_e.gnn import N_FEATURES

    tracker = FeatureTracker(n_cameras=4, log_threshold=np.log(100.0))
    f0 = tracker.features()
    assert f0.shape == (4, N_FEATURES)
    tracker.update(np.array([0.5, 0.01, 0.9, 0.2]), np.array([0.0, 3.0, -1.0, 0.5]),
                   np.array([0.1, 0.9, 0.0, 0.4]))
    f1 = tracker.features()
    assert np.allclose(f1[:, 1], [0.1, 0.9, 0.0, 0.4])   # the episode posterior
    assert not np.allclose(f0, f1)
    assert np.all(np.abs(f1[:, 0]) <= 2.0)
    tracker.reset()
    assert np.allclose(tracker.features(), f0)

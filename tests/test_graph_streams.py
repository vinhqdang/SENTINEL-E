"""Camera graph, stream simulator and clip-splicing protocol."""

import numpy as np
import pytest

from sentinel_e.graph import CameraGraph
from sentinel_e.streams import StreamConfig, StreamSimulator, build_stream_from_clips
from sentinel_e.temporal import acf


# --------------------------------------------------------------------------- #
def test_graph_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(0)
    g = CameraGraph.from_positions(rng.uniform(0, 1000, size=(15, 2)), k_neighbors=3)
    assert np.allclose(g.weights, g.weights.T)
    assert np.allclose(np.diag(g.weights), 0.0)
    assert g.n_cameras == 15


def test_nearby_cameras_are_connected_and_distant_ones_are_not():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [5000.0, 5000.0]])
    g = CameraGraph.from_positions(pos, length_scale=100.0, k_neighbors=1)
    assert g.weights[0, 1] > 0
    assert g.weights[0, 2] == 0.0


def test_normalized_adjacency_is_symmetric_and_bounded():
    rng = np.random.default_rng(1)
    g = CameraGraph.from_positions(rng.uniform(0, 800, size=(12, 2)), k_neighbors=4)
    a = g.normalized_adjacency()
    assert np.allclose(a, a.T)
    assert np.all(np.linalg.eigvalsh(a) <= 1.0 + 1e-8)


def test_edge_list_matches_the_weight_matrix():
    rng = np.random.default_rng(2)
    g = CameraGraph.from_positions(rng.uniform(0, 900, size=(10, 2)), k_neighbors=3)
    edges = g.edge_list()
    assert np.all(g.weights[edges[:, 0], edges[:, 1]] > 0)
    assert edges.shape[0] == int((g.weights > 0).sum() // 2)


def test_graph_rejects_mismatched_context():
    with pytest.raises(ValueError):
        CameraGraph.from_positions(np.zeros((5, 2)), context=np.zeros((4, 2)))


# --------------------------------------------------------------------------- #
def test_simulated_stream_has_the_requested_shape():
    cfg = StreamConfig(T=4000, n_calibration=3000)
    st = StreamSimulator(cfg, seed=0).simulate_camera(change_point=2000)
    assert st.scores.shape == (4000,)
    assert st.context.shape == (4000, 4)
    assert st.calibration_scores.size == 3000
    assert st.change_point == 2000
    assert st.labels[:2000].sum() == 0 and st.labels[2000:].sum() > 0
    assert np.all((st.scores > 0) & (st.scores < 1))


def test_event_frames_score_higher_than_background():
    cfg = StreamConfig(T=8000, n_calibration=3000, event_length=3000)
    st = StreamSimulator(cfg, seed=1).simulate_camera(change_point=3000)
    assert st.scores[st.labels].mean() > st.scores[~st.labels].mean() + 0.1


def test_null_stream_is_autocorrelated_as_advertised():
    cfg = StreamConfig(T=30000, n_calibration=1000, ar_rho=0.9)
    st = StreamSimulator(cfg, seed=2).simulate_camera()
    assert acf(st.scores, 1)[1] > 0.5


def test_calibration_regimes_differ_in_coverage():
    cfg_full = StreamConfig(T=1000, n_calibration=6000, calibration_regime="representative")
    cfg_day = StreamConfig(T=1000, n_calibration=6000, calibration_regime="daytime_only")
    full = StreamSimulator(cfg_full, seed=3).simulate_camera()
    day = StreamSimulator(cfg_day, seed=3).simulate_camera()
    # The daytime-only window covers a much narrower slice of the clock context.
    assert np.ptp(day.calibration_context[:, 0]) < np.ptp(full.calibration_context[:, 0])
    assert full.calibration_context[:, 1].max() == 1.0     # some adverse weather
    assert day.calibration_context[:, 1].max() == 0.0      # none


def test_unknown_calibration_regime_rejected():
    cfg = StreamConfig(T=100, n_calibration=100, calibration_regime="bogus")
    with pytest.raises(ValueError):
        StreamSimulator(cfg, seed=0).simulate_camera()


def test_fleet_has_a_graph_and_propagated_events():
    cfg = StreamConfig(T=3000, n_calibration=2000, n_cameras=10, propagation_prob=1.0)
    fleet = StreamSimulator(cfg, seed=4).simulate_fleet(n_events=2)
    assert fleet.n_cameras == 10 and fleet.T == 3000
    assert fleet.affected.sum() >= 2          # seeds plus propagated neighbours
    assert len(fleet.change_points) == 10


# --------------------------------------------------------------------------- #
def test_clip_splicing_builds_a_stream_with_a_change_point():
    rng = np.random.default_rng(5)
    clips = [rng.beta(2, 12, size=200) for _ in range(60)]
    clips += [rng.beta(6, 4, size=200) for _ in range(10)]
    labels = [0] * 60 + [1] * 10
    st = build_stream_from_clips(clips, labels, n_frames=5000,
                                 n_calibration=1000, n_events=2, rng=rng)
    assert st.scores.size == 5000
    assert st.calibration_scores.size == 1000
    assert st.change_point is not None
    assert st.labels.sum() > 0
    assert st.scores[st.labels].mean() > st.scores[~st.labels].mean()


def test_clip_splicing_keeps_calibration_clips_out_of_the_stream():
    rng = np.random.default_rng(6)
    clips = [np.full(100, float(i)) for i in range(40)] + [np.full(100, 99.0)]
    labels = [0] * 40 + [1]
    st = build_stream_from_clips(clips, labels, n_frames=2000,
                                 n_calibration=500, n_events=1, rng=rng)
    cal_values = set(np.unique(st.calibration_scores).tolist())
    bg = st.scores[~st.labels]
    assert not (set(np.unique(bg).tolist()) & cal_values)


def test_clip_splicing_validates_inputs():
    with pytest.raises(ValueError):
        build_stream_from_clips([np.zeros(10)], [1], n_frames=100)
    with pytest.raises(ValueError):
        build_stream_from_clips([np.zeros(10)], [0], n_frames=100, n_events=1)

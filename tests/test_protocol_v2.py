"""Behaviour of the v2 real-video protocol helpers (whole videos, episode scoring)."""

import numpy as np

from experiments.real_data.protocol_v2 import make_stream, run_alarms, score_labelled_stream
from experiments.runner import Spec, alarm_indices_on_stream, run_spec_on_stream
import sentinel_e.datasets.real_vad as real_vad
from sentinel_e.datasets.real_vad import (
    RealVADCorpus, concat_whole_videos, train_video_groups,
)
from sentinel_e.streams import Stream


def _three_videos():
    # Values encode (video, frame) so every frame is identifiable after shuffling.
    scores = [1000.0 * v + np.arange(n, dtype=float) for v, n in ((0, 10), (1, 7), (2, 5))]
    labels = [
        np.isin(np.arange(10), [7, 8, 9]),           # episode runs to the video's last frame
        np.ones(7, dtype=bool),                      # starts and ends inside an anomaly
        np.isin(np.arange(5), [0, 1]),               # starts with an episode
    ]
    return scores, labels


def test_concat_uses_every_frame_once_and_never_merges_across_videos():
    scores, labels = _three_videos()
    for seed in range(20):
        s, lab, eps = concat_whole_videos(scores, np.random.default_rng(seed), labels)
        # Each frame appears exactly once: the multiset of values matches.
        assert s.size == 22
        assert sorted(s.tolist()) == sorted(np.concatenate(scores).tolist())
        # Labels travel with their frames.
        expected = {(v, t): labels[v][t] for v in range(3) for t in range(labels[v].size)}
        for value, flag in zip(s, lab):
            v, t = divmod(int(round(value)), 1000)
            assert flag == expected[(v, t)]
        # Three per-video episodes, not two: the adjacent anomalous ends of
        # consecutive videos are not joined into one run.
        assert len(eps) == 3
        for a, b in eps:
            vids = {int(round(x)) // 1000 for x in s[a:b]}
            assert len(vids) == 1, "an episode crossed a video boundary"
        assert sum(b - a for a, b in eps) == int(lab.sum())


def test_concat_without_labels_has_no_anomalies_or_episodes():
    scores, _ = _three_videos()
    s, lab, eps = concat_whole_videos(scores, np.random.default_rng(0))
    assert s.size == 22 and not lab.any() and eps == []


def test_score_labelled_stream_hand_built_example():
    labels = np.zeros(100, dtype=bool)
    labels[10:20] = True
    # 12 is inside the episode (delay 2); 23 is in the grace window [20, 25);
    # 3 and 50 are false alarms.
    out = score_labelled_stream([3, 12, 23, 50], labels, [(10, 20)], grace=5, fps=25.0)
    assert out["n_episodes"] == 1
    assert out["n_detected"] == 1
    assert out["delays"] == [2]
    assert out["n_false_alarms"] == 2
    # 90 normal frames, minus the 5 normal frames [20, 25) in the grace window.
    assert out["normal_frames"] == 85
    assert np.isclose(out["normal_hours"], 85 / 25.0 / 3600.0)


def test_score_labelled_stream_missed_episode_and_grace_only_alarm():
    labels = np.zeros(100, dtype=bool)
    labels[10:20] = True
    labels[60:70] = True
    # Only a grace-window alarm for the first episode, nothing for the second.
    out = score_labelled_stream([22], labels, [(10, 20), (60, 70)], grace=5, fps=25.0)
    assert out["n_detected"] == 1 and out["delays"] == [12]
    assert out["n_false_alarms"] == 0


def _stream(seed=0):
    rng = np.random.default_rng(seed)
    cal = rng.normal(size=600)
    scores = np.concatenate([rng.normal(size=300), rng.normal(loc=2.0, size=200),
                             rng.normal(size=300)])
    labels = np.zeros(scores.size, dtype=bool)
    labels[300:500] = True
    ctx = np.zeros((scores.size, 1))
    return Stream(
        scores=scores, context=ctx, labels=labels, change_point=300,
        calibration_scores=cal, calibration_context=np.zeros((cal.size, 1)),
        episodes=[(300, 500)],
    )


def test_alarm_indices_first_element_matches_trace_first_alarm():
    st = _stream()
    for method, knob in (("SENTINEL-E", 0.1), ("CUSUM", 5.0), ("Fixed threshold", 1e-2)):
        spec = Spec(method, knob, 0, 300, {},
                    options=tuple(sorted({"calibration": "pooled", "lag": 3,
                                          "restart": True}.items())))
        idx, abstention, lag = alarm_indices_on_stream(st, spec)
        tr = run_spec_on_stream(st, spec)
        assert tr.first_alarm == (int(idx[0]) if idx.size else -1), method
        assert tr.n_alarms == idx.size, method
        assert tr.lag == lag and tr.abstention == abstention, method
        assert np.all(np.diff(idx) > 0), method
    # The helper returns the same indices as the dispatch in run_alarms.
    spec_idx = run_alarms("CUSUM", 5.0, st, 3)
    assert spec_idx.size > 0 and spec_idx[0] >= 300


def test_make_stream_sets_change_point_from_first_episode():
    st = make_stream(np.zeros(50), np.zeros(50, bool), [(20, 30), (40, 45)], np.zeros(80))
    assert st.change_point == 20 and st.episodes == [(20, 30), (40, 45)]
    assert st.context.shape == (50, 1) and st.calibration_context.shape == (80, 1)
    assert make_stream(np.zeros(5), np.zeros(5, bool), [], np.zeros(9)).change_point is None


def _corpus(n_train=7, n_test=3):
    rng = np.random.default_rng(1)
    tr_lens = [10 + i for i in range(n_train)]
    cal = rng.normal(size=sum(tr_lens))
    tr_b, lo = [], 0
    for i, n in enumerate(tr_lens):
        tr_b.append((f"Train{i:03d}", lo, lo + n))
        lo += n
    te_lens = [6, 4, 8][:n_test]
    te_scores = rng.normal(size=sum(te_lens))
    te_labels = np.zeros(sum(te_lens), dtype=bool)
    te_labels[2:4] = True
    te_labels[6:9] = True   # video 1 is only 4 frames, so this lies in video 2
    te_b, lo = [], 0
    for i, n in enumerate(te_lens):
        te_b.append((f"Test{i:03d}", lo, lo + n))
        lo += n
    return RealVADCorpus(
        name="toy", calibration_scores=cal, test_scores=te_scores, test_labels=te_labels,
        video_boundaries=te_b, episodes=[], backbone_auc=float("nan"), train_boundaries=tr_b,
    )


def test_train_groups_are_disjoint_and_follow_index_residue():
    c = _corpus(n_train=7)
    g = train_video_groups(c)
    lengths = [hi - lo for _n, lo, hi in c.train_boundaries]
    for key, residue in (("cal", 0), ("sel", 1), ("null", 2)):
        expected = [lengths[i] for i in range(len(lengths)) if i % 3 == residue]
        assert [v.size for v in g[key]] == expected
    assert sum(v.size for k in g for v in g[k]) == c.calibration_scores.size


def test_videos_returns_each_video_in_official_order():
    c = _corpus()
    tv = real_vad.test_videos(c)
    assert [s.size for s, _ in tv] == [6, 4, 8]
    assert np.array_equal(np.concatenate([s for s, _ in tv]), c.test_scores)
    assert np.array_equal(np.concatenate([l for _, l in tv]), c.test_labels)

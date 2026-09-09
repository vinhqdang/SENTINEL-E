"""Baseline detectors, metrics and dataset adapters."""

import numpy as np
import pytest

from sentinel_e.baselines import (
    CUSUM,
    FixedThreshold,
    ParametricEDetector,
    PValueThreshold,
    ShiryaevRoberts,
)
from sentinel_e.datasets import (
    ADAPTERS,
    ScoredCorpus,
    corpus_from_frame_table,
    describe_datasets,
    load_dataset,
    load_scored_corpus,
    write_scored_corpus,
)
from sentinel_e.metrics import (
    delay_far_curve,
    detection_delay,
    fdr_power,
    first_alarm_index,
    per_episode_outcomes,
    run_length,
    summarize_per_episode,
    summarize_runs,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_fixed_threshold_and_persistence_filter():
    s = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1])
    assert FixedThreshold(0.5).run(s).tolist() == [False, True, False, True, True, False]
    strict = FixedThreshold(0.5, k_of_m=(2, 3)).run(s)
    assert not strict[1]          # a lone spike is suppressed
    assert strict[4]              # two hits within three frames are not


def test_fixed_threshold_calibration_hits_the_target_rate():
    rng = np.random.default_rng(0)
    cal = rng.uniform(size=200000)
    det = FixedThreshold()
    det.calibrate_threshold(cal, per_frame_rate=1e-3)
    assert det.run(rng.uniform(size=400000)).mean() == pytest.approx(1e-3, rel=0.25)


def test_pvalue_threshold_consumes_p_values():
    det = PValueThreshold(level=0.05)
    assert det.run_pvalues([0.5, 0.01, 0.2]).tolist() == [False, True, False]
    with pytest.raises(NotImplementedError):
        det.run([0.5])


@pytest.mark.parametrize("cls", [CUSUM, ShiryaevRoberts, ParametricEDetector])
def test_baselines_are_quiet_on_matched_null_data(cls):
    rng = np.random.default_rng(1)
    det = cls().fit(rng.normal(size=20000))
    knob = {"CUSUM": ("h", 12.0), "ShiryaevRoberts": ("A", 1e6),
            "ParametricEDetector": ("alpha", 1e-4)}[cls.__name__]
    setattr(det, knob[0], knob[1])
    assert det.run(rng.normal(size=20000)).mean() < 0.01


@pytest.mark.parametrize("cls", [CUSUM, ShiryaevRoberts, ParametricEDetector])
def test_baselines_react_to_a_large_shift(cls):
    """With a threshold tight enough to stay quiet on the null half, the shift fires."""
    rng = np.random.default_rng(2)
    det = cls().fit(rng.normal(size=20000))
    knob, value = {"CUSUM": ("h", 12.0), "ShiryaevRoberts": ("A", 1e5),
                   "ParametricEDetector": ("alpha", 1e-4)}[cls.__name__]
    setattr(det, knob, value)
    s = np.concatenate([rng.normal(size=500), rng.normal(loc=3.0, size=500)])
    idx = np.flatnonzero(det.run(s))
    assert idx.size and idx[0] >= 500


def test_cusum_is_monotone_in_its_threshold():
    rng = np.random.default_rng(3)
    s = rng.normal(size=40000)
    det = CUSUM(delta=1.0).fit(rng.normal(size=20000))
    counts = []
    for h in (2.0, 5.0, 10.0):
        det.h = h
        counts.append(int(det.run(s).sum()))
    assert counts[0] >= counts[1] >= counts[2]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_first_alarm_and_run_length():
    assert first_alarm_index([False, False, True, False]) == 2
    assert first_alarm_index([False, False]) is None
    assert run_length([False, False, True, False]) == 3
    assert run_length([False, False]) == 2
    assert run_length([False, False], horizon=99) == 99


def test_detection_delay_ignores_pre_change_alarms():
    a = [True, False, False, False, True, False]
    assert detection_delay(a, 3) == 1
    assert detection_delay([False] * 6, 3) is None
    with pytest.raises(ValueError):
        detection_delay(a, 10)


def test_summarize_runs_aggregates_correctly():
    null = [[False] * 10, [False] * 4 + [True] + [False] * 5]
    sig = [[False] * 5 + [True] + [False] * 4, [False] * 10]
    res = summarize_runs(null, sig, [5, 5])
    assert res.pfa == pytest.approx(0.5)
    assert res.arl0 == pytest.approx((10 + 5) / 2)
    assert res.add == pytest.approx(0.0)
    assert res.miss_rate == pytest.approx(0.5)
    assert res.pre_change_fa == pytest.approx(0.0)
    assert res.pfa_ci[0] <= 0.5 <= res.pfa_ci[1]


def test_wilson_interval_edges():
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == pytest.approx(1.0)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_delay_far_curve_is_sorted_by_false_alarm_rate():
    outs = [summarize_runs([[True]] * 4, [[True]] * 2, [0] * 2),
            summarize_runs([[False]] * 4, [[True]] * 2, [0] * 2)]
    curve = delay_far_curve(outs)
    assert np.all(np.diff(curve["pfa"]) >= 0)


def test_per_episode_outcomes_scores_each_episode_independently():
    # Stream-level scoring (summarize_runs' miss_rate) would call this a clean
    # detection: an alarm follows the first onset at 100. Per-episode scoring
    # reveals two of the three episodes were slept through.
    episodes = [(100, 200), (300, 400), (500, 600)]
    alarms = [False] * 700
    alarms[650] = True
    outs = per_episode_outcomes(alarms, episodes, horizon=700)
    assert [o.missed for o in outs] == [True, True, False]
    assert outs[2].delay == 150
    agg = summarize_per_episode([outs])
    assert agg["n_episodes"] == 3
    assert agg["episode_miss_rate"] == pytest.approx(2 / 3)
    assert agg["episode_add"] == pytest.approx(150.0)
    # censored: the two misses are charged their own window's remainder
    # (window 0: 300-100=200; window 1: 500-300=200; window 2: delay 150)
    assert agg["episode_add_censored"] == pytest.approx((200 + 200 + 150) / 3)


def test_per_episode_outcomes_detects_within_its_own_window():
    episodes = [(0, 50), (100, 150)]
    alarms = [False] * 200
    alarms[20] = True    # inside episode 0's window
    alarms[120] = True   # inside episode 1's window
    outs = per_episode_outcomes(alarms, episodes, horizon=200)
    assert [o.missed for o in outs] == [False, False]
    assert outs[0].delay == 20
    assert outs[1].delay == 20
    agg = summarize_per_episode([outs])
    assert agg["episode_miss_rate"] == 0.0


def test_summarize_per_episode_handles_no_episodes():
    assert summarize_per_episode([[]])["n_episodes"] == 0


def test_fdr_power_counts():
    m = fdr_power([True, True, False], [True, False, False])
    assert m["fdp"] == pytest.approx(0.5)
    assert m["power"] == pytest.approx(1.0)
    assert fdr_power([False, False], [True, False])["fdp"] == 0.0
    with pytest.raises(ValueError):
        fdr_power([True], [True, False])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def test_corpus_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    corpus = ScoredCorpus(
        clip_scores=[rng.random(7), rng.random(3)],
        clip_labels=[0, 1],
        clip_ids=["a", "b"],
        context=[rng.random((7, 2)), rng.random((3, 2))],
        name="toy",
        fps=12.0,
    )
    path = write_scored_corpus(corpus, tmp_path / "toy.npz")
    back = load_scored_corpus(path)
    assert back.n_clips == 2 and back.n_frames == 10 and back.n_positive == 1
    assert back.fps == pytest.approx(12.0)
    assert np.allclose(back.clip_scores[0], corpus.clip_scores[0])
    assert np.allclose(back.context[1], corpus.context[1])
    assert back.summary()["clips"] == 2


def test_corpus_split_is_disjoint():
    rng = np.random.default_rng(5)
    c = ScoredCorpus([rng.random(4) for _ in range(10)], [0] * 10,
                     [str(i) for i in range(10)])
    a, b = c.split(0.3, seed=1)
    assert a.n_clips == 3 and b.n_clips == 7
    assert not (set(a.clip_ids) & set(b.clip_ids))


def test_corpus_from_frame_table_groups_by_clip():
    c = corpus_from_frame_table(
        frame_ids=["f0", "f1", "f2", "f3"],
        clip_ids=["v1", "v1", "v2", "v1"],
        scores=[0.1, 0.2, 0.9, 0.3],
        clip_labels={"v1": 0, "v2": 1},
    )
    assert c.clip_ids == ["v1", "v2"]
    assert np.allclose(c.clip_scores[0], [0.1, 0.2, 0.3])
    assert c.clip_labels.tolist() == [0, 1]
    with pytest.raises(KeyError):
        corpus_from_frame_table(["f"], ["vX"], [0.1], {"v1": 0})


def test_corpus_validates_alignment():
    with pytest.raises(ValueError):
        ScoredCorpus([np.zeros(3)], [0, 1], ["a"])


def test_adapters_are_documented_and_fail_loudly_when_absent(tmp_path):
    assert set(ADAPTERS) >= {"mivia_iwdd", "taco", "zerowaste", "uavvaste",
                             "aerialwaste", "marida", "mados"}
    for spec in ADAPTERS.values():
        assert spec.url.startswith("http") and spec.role and spec.backbone_output
    with pytest.raises(FileNotFoundError) as e:
        load_dataset("mivia_iwdd", root=tmp_path)
    assert "write_scored_corpus" in str(e.value)
    with pytest.raises(KeyError):
        load_dataset("not_a_dataset", root=tmp_path)
    rows = describe_datasets(root=tmp_path)
    assert len(rows) == len(ADAPTERS) and not any(r["available"] for r in rows)


def test_adapter_reads_a_produced_intermediate(tmp_path):
    rng = np.random.default_rng(6)
    corpus = ScoredCorpus([rng.random(5) for _ in range(4)], [0, 0, 1, 0],
                          [f"c{i}" for i in range(4)])
    write_scored_corpus(corpus, ADAPTERS["taco"].path(tmp_path))
    loaded = load_dataset("taco", root=tmp_path)
    assert loaded.n_clips == 4
    assert loaded.name == ADAPTERS["taco"].name


def test_censored_delay_charges_misses_the_remaining_horizon():
    """Conditional ADD drops misses; the censored figure must not."""
    detected = [False] * 5 + [True] + [False] * 4      # delay 1 from nu = 4
    missed = [False] * 10
    res = summarize_runs([[False] * 10], [detected, missed], [4, 4])
    assert res.add == pytest.approx(1.0)               # conditional: only the hit
    assert res.miss_rate == pytest.approx(0.5)
    # censored: (1 + 6) / 2, the miss charged the 6 remaining frames.
    assert res.add_censored == pytest.approx(3.5)


def test_censored_delay_equals_conditional_when_nothing_is_missed():
    a = [False, False, True, False]
    res = summarize_runs([[False] * 4], [a, a], [1, 1])
    assert res.add == pytest.approx(res.add_censored)
    assert res.miss_rate == 0.0

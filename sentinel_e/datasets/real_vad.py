"""Real video-anomaly-detection corpora, adapted to SENTINEL-E's clip interface.

Illegal-dumping video (Mivia-IWDD) is access-gated; no other public corpus of
continuous illegal-dumping surveillance video exists (see the manuscript's
data section). What *is* freely downloadable, real, and genuinely continuous
surveillance video with frame-level anomaly onsets is the classical video
anomaly detection (VAD) benchmark pair used here:

CUHK Avenue
    :cite:`lu2013avenue`. 16 training / 21 testing videos, campus walkway,
    pixel-level per-frame ground truth. Most test videos contain *several*
    separate anomalous runs (loitering, running, throwing objects, wrong
    direction), which is the real analogue of this paper's "recurring
    episode" structure.
UCSD Ped2
    :cite:`mahadevan2010ucsd`. 16 training / 12 testing videos, pedestrian
    walkway, anomalies are non-pedestrian traffic (bikes, carts, skaters).

Neither dataset ships per-frame *detector scores* -- only raw video. The
scores used throughout this paper come from a real, if deliberately simple,
per-frame backbone: Farneback optical-flow magnitude, averaged per frame
(``experiments/real_data/extract_scores.py``; see the manuscript's data
section for why this backbone, not a heavier one, was used). Everything
downstream of that scalar is identical to the simulated-stream pipeline this
paper used until its previous revision: the same conformal calibration, the
same episodic e-process, the same baselines.

This module does *not* re-run the extraction (that step touches pixels and
is documented separately); it loads the cached per-frame scores and frame
labels each dataset's extraction pass already produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from sentinel_e.streams import Stream

__all__ = [
    "RealVADCorpus",
    "load_real_vad",
    "clips_from_runs",
    "build_real_stream",
    "causal_local_context",
]

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_vad"

#: Detector-score range varies by backbone; SentinelE's power-family betting
#: functions are defined on p-values in (0, 1], which the conformal layer
#: produces regardless, so no rescaling of the raw optical-flow scores is
#: needed before they reach :class:`sentinel_e.pipeline.SentinelE`.


@dataclass
class RealVADCorpus:
    """One real dataset's calibration pool plus its labelled test stream.

    Attributes
    ----------
    name : str
    calibration_scores : (n,) float array
        Real per-frame backbone scores from the official *training* split,
        which is confirmed anomaly-free by construction for both datasets
        used here.
    test_scores, test_labels : (m,) arrays
        The real, continuous concatenation of every official test video, in
        official order, with real frame-level ground truth (1 = inside an
        annotated anomalous run).
    video_boundaries : list of (name, start, end)
        Frame index ranges of each source test video inside the concatenated
        stream, so per-video structure (and cross-video splicing) is
        recoverable.
    episodes : list of (start, end)
        Maximal runs of ``test_labels == 1`` in the concatenated stream --
        the real analogue of the simulator's episode windows.
    backbone_auc : float
        Frame-level ROC-AUC of the backbone on this dataset's real test set,
        reported for transparency about how strong (or weak) the real signal
        is before any sequential machinery touches it.
    """

    name: str
    calibration_scores: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    video_boundaries: List[Tuple[str, int, int]]
    episodes: List[Tuple[int, int]]
    backbone_auc: float
    train_boundaries: List[Tuple[str, int, int]]
    fps: float = 25.0


def causal_local_context(scores: np.ndarray, window: int = 50) -> np.ndarray:
    """A real, data-derived context feature: the trailing local mean score.

    This is the real-data stand-in for the synthetic context features
    (time-of-day, weather flag) the simulated experiments used to drive
    :class:`sentinel_e.pipeline.SentinelE`'s ``calibration='residual'``
    layer. Neither dataset used here carries side-channel metadata, but a
    genuinely real, legitimate covariate is available directly from the
    score sequence itself: the local baseline "how busy is this scene right
    now" level, computed causally (only frames strictly before ``i``
    contribute to ``ctx[i]``, so nothing about the future or about frame
    ``i``'s own score leaks into its own context) so it can be computed
    identically at calibration time and at deployment time.
    """
    n = scores.size
    ctx = np.empty(n, dtype=float)
    if n == 0:
        return ctx
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    for i in range(n):
        lo = max(0, i - window)
        if i == 0:
            ctx[i] = scores[0]
        else:
            ctx[i] = (csum[i] - csum[lo]) / (i - lo)
    return ctx


def _episodes_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(labels):
        if v and not in_run:
            start, in_run = i, True
        if not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(labels)))
    return runs


def load_real_vad(name: str) -> RealVADCorpus:
    """Load the cached real per-frame scores and labels for ``name``.

    ``name`` is ``'ped2'`` or ``'avenue'``. Raises ``FileNotFoundError`` with
    the extraction recipe if the cached arrays are absent, matching the
    fail-loudly convention of :mod:`sentinel_e.datasets.adapters`.
    """
    root = DATA_ROOT / name
    required = ["calibration_scores.npy", "test_scores.npy", "test_labels.npy",
                "test_meta.json"]
    missing = [f for f in required if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"real VAD corpus {name!r} is missing {missing} under {root}. "
            "Run experiments/real_data/download_and_extract.py first; it "
            "downloads the official Avenue/UCSD Ped2 releases and computes "
            "real per-frame optical-flow backbone scores. No synthetic data "
            "is substituted."
        )
    cal = np.load(root / "calibration_scores.npy").astype(float)
    scores = np.load(root / "test_scores.npy").astype(float)
    labels = np.load(root / "test_labels.npy").astype(bool)
    meta = json.loads((root / "test_meta.json").read_text())

    boundaries: List[Tuple[str, int, int]] = []
    idx = 0
    for m in meta:
        n = int(m["n_frames"])
        boundaries.append((m["name"], idx, idx + n))
        idx += n
    if idx != scores.size:
        raise ValueError(
            f"{name}: test_meta frame count {idx} != test_scores length {scores.size}"
        )

    # Episodes must respect video boundaries: two real videos where one ends
    # mid-anomaly and the next begins mid-anomaly are not one continuous
    # event, and naively running a run-length scan over the raw concatenation
    # would silently merge them into a fictitious longer episode.
    episodes: List[Tuple[int, int]] = []
    for _vid_name, lo, hi in boundaries:
        for s, e in _episodes_from_labels(labels[lo:hi]):
            episodes.append((lo + s, lo + e))

    auc_path = root / "backbone_auc.json"
    auc = json.loads(auc_path.read_text())["auc"] if auc_path.exists() else float("nan")

    train_meta_path = root / "train_meta.json"
    train_boundaries: List[Tuple[str, int, int]] = []
    if train_meta_path.exists():
        tidx = 0
        for m in json.loads(train_meta_path.read_text()):
            n = int(m["n_frames"])
            train_boundaries.append((m["name"], tidx, tidx + n))
            tidx += n
        if tidx != cal.size:
            raise ValueError(
                f"{name}: train_meta frame count {tidx} != calibration_scores length {cal.size}"
            )

    return RealVADCorpus(
        name=name,
        calibration_scores=cal,
        test_scores=scores,
        test_labels=labels,
        video_boundaries=boundaries,
        episodes=episodes,
        backbone_auc=float(auc),
        train_boundaries=train_boundaries,
    )


def clips_from_runs(
    corpus: RealVADCorpus, include_calibration_as_negative: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Decompose the real test stream into maximal normal/anomalous runs.

    Every run is a genuine, unmodified slice of the real backbone-score
    sequence -- no synthetic value is ever introduced. This is the same
    "clip" representation :func:`sentinel_e.streams.build_stream_from_clips`
    already consumes for the (formerly simulated) splicing protocol, so the
    downstream pipeline needs no changes: only where the clips come from.
    """
    scores: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []

    for name, lo, hi in corpus.video_boundaries:
        seg_labels = corpus.test_labels[lo:hi]
        seg_scores = corpus.test_scores[lo:hi]
        run_start = 0
        cur = bool(seg_labels[0]) if seg_labels.size else False
        for i in range(1, seg_labels.size + 1):
            at_end = i == seg_labels.size
            if at_end or bool(seg_labels[i]) != cur:
                scores.append(seg_scores[run_start:i].copy())
                labels.append(1 if cur else 0)
                ids.append(f"{corpus.name}/{name}/{run_start}-{i}")
                if not at_end:
                    run_start, cur = i, bool(seg_labels[i])

    if include_calibration_as_negative:
        # The official training split: confirmed anomaly-free by construction
        # for both datasets used here, and disjoint from every test frame.
        scores.append(corpus.calibration_scores.copy())
        labels.append(0)
        ids.append(f"{corpus.name}/train")

    return scores, np.asarray(labels, dtype=int), ids


def _split_negative_runs(corpus: RealVADCorpus) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Partition the test split's real normal runs into two disjoint halves.

    One half backs "representative" calibration, the other backs the
    background/spliced-event pool a deployment stream is built from -- so a
    frame used for calibration is *never* also eligible to appear in the
    stream being monitored, exactly as the official-train-split regime
    already guarantees by construction (there, the two pools are different
    dataset splits; here, they are alternating runs of the same split, split
    by run index so every video contributes to both halves rather than one
    half being drawn from only a subset of scenes).
    """
    neg_runs, labels_arr, _ids = clips_from_runs(corpus, include_calibration_as_negative=False)
    neg = [neg_runs[i] for i in np.flatnonzero(labels_arr == 0)]
    return neg[0::2], neg[1::2]


def _representative_calibration_pool(corpus: RealVADCorpus) -> np.ndarray:
    """Held-out real *test*-split normal frames, not the official train split.

    The official train/test split of both datasets used here turns out not
    to be exchangeable with itself: the marginal score distribution on
    confirmed-normal *test* frames differs from the official *train* split
    enough to badly inflate the false-alarm rate under plain split-conformal
    calibration (see the manuscript's data section, which reports this as a
    real, disclosed finding rather than working around it silently). This
    pool is the "representative calibration" fix, the real-data analogue of
    the ``calibration_regime='representative'`` condition already used
    throughout this paper's simulated experiments.
    """
    cal_runs, _ = _split_negative_runs(corpus)
    return np.concatenate(cal_runs) if cal_runs else np.zeros(0)


def _train_video_halves(corpus: RealVADCorpus) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split whole official-training videos (not sub-run fragments) in half.

    Alternates by video, so each half gets roughly equal real footage from
    across the whole training split rather than, say, the first half of the
    recording session. Used for the "train-holdout" calibration regime: the
    lowest-seam-count real test available, since each unit is a genuinely
    continuous few-hundred-to-thousand-frame real recording rather than one
    of the much shorter (median well under 100 frames) normal/anomalous runs
    :func:`clips_from_runs` extracts from the labelled test split.
    """
    videos = [corpus.calibration_scores[lo:hi] for _n, lo, hi in corpus.train_boundaries]
    if not videos:
        raise ValueError(f"{corpus.name}: no train_meta.json boundaries available")
    return videos[0::2], videos[1::2]


def build_real_stream(
    corpus: RealVADCorpus,
    n_frames: int,
    n_calibration: int = 2_000,
    n_events: int = 1,
    rng: Optional[np.random.Generator] = None,
    require_normal: bool = False,
    change_point: Optional[int] = None,
    calibration_regime: str = "official",
    min_event_length: int = 0,
) -> Stream:
    """Build one monitoring stream entirely out of real per-frame scores.

    Unlike :func:`clips_from_runs` + :func:`sentinel_e.streams.build_stream_from_clips`,
    this keeps calibration and deployment ironclad-disjoint by construction
    rather than by shuffling: the background/spliced-event pool is always
    drawn from the officially labelled runs inside the *test* split, and no
    frame used for calibration is ever eligible to appear in the stream being
    monitored (the representative regime below draws calibration and
    deployment from disjoint halves of that same pool, drawn independently
    per stream so different replicates see different calibration draws too).

    Parameters mirror :func:`sentinel_e.streams.build_stream_from_clips`.
    ``require_normal=True`` builds a null stream (``n_events=0``, background
    only), used for the false-alarm-rate sweeps. ``calibration_regime``:
    ``'official'`` draws calibration from the dataset's official training
    split (what a practitioner would do with no other guidance);
    ``'representative'`` draws it instead from held-out normal runs inside
    the test split, fixing the shift ``'official'`` exposes.
    """
    rng = np.random.default_rng() if rng is None else rng
    background_runs = None
    if calibration_regime == "official":
        cal = corpus.calibration_scores
    elif calibration_regime == "representative":
        cal_runs, deploy_runs = _split_negative_runs(corpus)
        cal_pool = np.concatenate(cal_runs) if cal_runs else np.zeros(0)
        background_runs = deploy_runs  # deployment draws only from the OTHER half
        if cal_pool.size > n_calibration:
            start = int(rng.integers(0, cal_pool.size - n_calibration))
            cal = cal_pool[start:start + n_calibration]
        else:
            cal = cal_pool
    elif calibration_regime == "train_holdout":
        # Minimal-seam variant: whole official-training *videos* (typically
        # hundreds to low thousands of genuinely continuous real frames each)
        # split by video, one half for calibration and the other for the
        # background a null or signal stream is built from -- as opposed to
        # 'representative', which uses much shorter labelled-run fragments
        # from the test split and stitches many more of them together.
        cal_videos, bg_videos = _train_video_halves(corpus)
        cal_pool = np.concatenate(cal_videos) if cal_videos else np.zeros(0)
        background_runs = bg_videos
        if cal_pool.size > n_calibration:
            start = int(rng.integers(0, cal_pool.size - n_calibration))
            cal = cal_pool[start:start + n_calibration]
        else:
            cal = cal_pool
    else:
        raise ValueError(f"unknown calibration_regime {calibration_regime!r}")
    if n_calibration > cal.size:
        raise ValueError(
            f"{corpus.name}: requested n_calibration={n_calibration} exceeds "
            f"the available {calibration_regime} pool ({cal.size} frames)"
        )
    calibration = cal[:n_calibration].copy()
    return _splice(corpus, calibration, n_frames, n_events, rng, require_normal,
                   change_point, background_runs=background_runs,
                   min_event_length=min_event_length)


def _splice(
    corpus: RealVADCorpus,
    calibration: np.ndarray,
    n_frames: int,
    n_events: int,
    rng: np.random.Generator,
    require_normal: bool,
    fixed_change_point: Optional[int] = None,
    background_runs: Optional[List[np.ndarray]] = None,
    min_event_length: int = 0,
) -> Stream:
    scores_list, labels_arr, _ids = clips_from_runs(corpus, include_calibration_as_negative=False)
    pos_idx = np.flatnonzero(labels_arr == 1)
    if min_event_length > 0:
        # Restricts splicing to genuinely sustained real anomalous runs,
        # excluding few-frame labelling flickers this paper's episodic model
        # was never designed to catch (a single-digit-frame "event" is below
        # any sequential test's identifiability floor regardless of backbone
        # quality). Disclosed and reported separately from the unrestricted
        # pool, never silently substituted for it.
        long_enough = np.array([scores_list[k].size >= min_event_length for k in pos_idx])
        pos_idx = pos_idx[long_enough]
    if n_events > 0 and pos_idx.size == 0:
        raise ValueError(f"{corpus.name}: no real anomalous runs in the test split "
                          f"(after min_event_length={min_event_length} filter)")

    if background_runs is None:
        neg_idx = np.flatnonzero(labels_arr == 0)
        if neg_idx.size == 0:
            raise ValueError(f"{corpus.name}: no real negative runs in the test split")
        background_pool = [scores_list[k] for k in neg_idx]
    else:
        if len(background_runs) == 0:
            raise ValueError(f"{corpus.name}: empty background run pool")
        background_pool = background_runs

    parts, total = [], 0
    while total < n_frames:
        s = background_pool[int(rng.integers(0, len(background_pool)))]
        parts.append(s)
        total += s.size
    stream_scores = np.concatenate(parts)[:n_frames]
    stream_labels = np.zeros(n_frames, dtype=bool)
    episodes: List[Tuple[int, int]] = []

    change_point: Optional[int] = None
    n_events = 0 if require_normal else n_events
    if n_events > 0:
        if fixed_change_point is not None:
            # Matches the simulator's convention (a fixed CHANGE_POINT frame
            # index shared across the whole replicate ensemble, only the
            # spliced content varying by seed), which is what
            # experiments/real_data/runner.py relies on to pass a single
            # known onset to summarize_runs across all signal replicates.
            first = int(fixed_change_point)
            rest = n_events - 1
            onsets = np.array([first], dtype=int)
            if rest > 0:
                extra = np.sort(rng.choice(
                    np.arange(first + 1, int(0.95 * n_frames)),
                    size=min(rest, max(1, (n_frames - first) // 20)), replace=False,
                ))
                onsets = np.concatenate([onsets, extra])
        else:
            onsets = np.sort(rng.choice(
                np.arange(n_frames // 10, int(0.85 * n_frames)),
                size=min(n_events, max(1, n_frames // 20)), replace=False,
            ))
        for onset in onsets:
            k = int(rng.choice(pos_idx))
            ev = scores_list[k]
            end = min(n_frames, onset + ev.size)
            stream_scores[onset:end] = ev[: end - onset]
            stream_labels[onset:end] = True
            episodes.append((int(onset), int(end)))
        change_point = int(onsets[0])

    return Stream(
        scores=stream_scores,
        context=causal_local_context(stream_scores).reshape(-1, 1),
        labels=stream_labels,
        change_point=change_point,
        calibration_scores=calibration,
        calibration_context=causal_local_context(calibration).reshape(-1, 1),
        episodes=episodes,
    )

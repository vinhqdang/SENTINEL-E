"""Shared pieces of the v2 real-video protocol (whole videos, train-only lag selection).

Two changes from the v1 protocol motivated the v2 code paths here:

1. Streams are concatenations of *whole* videos drawn in a random order with
   no frame reused inside a stream (:func:`sentinel_e.datasets.real_vad.concat_whole_videos`),
   replacing six-thousand-frame splices drawn with replacement from short runs.
2. The betting-grid lag is selected on official *training* videos only, and
   detection metrics are computed on the real labelled test split at the lag
   fixed that way (see ``exp_r1_validity.py``).

This module holds the stream constructor, the single-detector alarm
dispatch, and the episode-level scoring rule. It contains no experiment
choices; those live in the experiment scripts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.runner import Spec, alarm_indices_on_stream
from sentinel_e.datasets.real_vad import (
    causal_local_context, load_real_vad, test_videos, train_video_groups,
)
from sentinel_e.metrics import wilson_interval
from sentinel_e.streams import Stream

#: Alarms this many frames after an episode's end still count as detections
#: of that episode (one second at 25 fps). Frames in the grace window are
#: neither false alarms nor normal frames.
GRACE_FRAMES = 25
FPS = 25.0

#: Master seed. Each stream's permutation is seeded by (SEED, dataset, stage, rep),
#: so every method and every knob sees the same streams (common random numbers).
SEED = 20_260_101
_DATASET_ID = {"ped2": 1, "avenue": 2}
_STAGE = {"sel": 1, "null": 2, "test": 3}


def rng_for(dataset: str, stage: str, rep: int) -> np.random.Generator:
    """Generator for one stream's video permutation."""
    return np.random.default_rng([SEED, _DATASET_ID.get(dataset, 9), _STAGE[stage], int(rep)])


def iid_rng(dataset: str) -> np.random.Generator:
    """Generator for the i.i.d. reference streams (distinct from every stream stage)."""
    return np.random.default_rng([SEED, _DATASET_ID.get(dataset, 9), 4, 0])


_STATE: Dict[str, dict] = {}


def state(dataset: str) -> dict:
    """Per-process cache: the corpus, its training groups, test videos, calibration set."""
    if dataset not in _STATE:
        corpus = load_real_vad(dataset)
        groups = train_video_groups(corpus)
        _STATE[dataset] = {
            "corpus": corpus,
            "groups": groups,
            "cal": np.concatenate(groups["cal"]),
            "tv": test_videos(corpus),
        }
    return _STATE[dataset]


def rate(k: int, n: int) -> Dict:
    """Proportion with its Wilson 95% interval."""
    lo, hi = wilson_interval(int(k), int(n))
    return {"k": int(k), "n": int(n), "pfa": (k / n) if n else float("nan"), "ci": [lo, hi]}


def test_summary(rows: List[Dict], fps: float = FPS) -> Dict:
    """Pool per-stream :func:`score_labelled_stream` outputs into one summary.

    Detection rate is detected over total episodes; delays are pooled over all
    detected episodes; false alarms per hour are false alarms over total
    normal hours.
    """
    n_ep = sum(r["n_episodes"] for r in rows)
    n_det = sum(r["n_detected"] for r in rows)
    delays = [d for r in rows for d in r["delays"]]
    n_fa = sum(r["n_false_alarms"] for r in rows)
    hours = sum(r["normal_hours"] for r in rows)
    return {
        "n_streams": len(rows),
        "n_episodes": int(n_ep),
        "n_detected": int(n_det),
        "episode_detection_rate": (n_det / n_ep) if n_ep else float("nan"),
        "mean_delay_s": (float(np.mean(delays)) / fps) if delays else None,
        "median_delay_s": (float(np.median(delays)) / fps) if delays else None,
        "n_false_alarms": int(n_fa),
        "normal_hours": float(hours),
        "false_alarms_per_hour": (n_fa / hours) if hours > 0 else float("nan"),
    }


def make_stream(
    scores: np.ndarray,
    labels: np.ndarray,
    episodes: Sequence[Tuple[int, int]],
    cal_scores: np.ndarray,
) -> Stream:
    """Wrap one real stream and its calibration set in a :class:`Stream`.

    ``context`` is the causal trailing local mean (see
    :func:`causal_local_context`); it is computed from the stream's own scores
    only, so it never looks ahead. The change point is the first episode onset,
    or None when the stream holds no labelled episode.
    """
    scores = np.asarray(scores, dtype=float)
    cal_scores = np.asarray(cal_scores, dtype=float)
    episodes = [(int(s), int(e)) for s, e in episodes]
    return Stream(
        scores=scores,
        context=causal_local_context(scores).reshape(-1, 1),
        labels=np.asarray(labels, dtype=bool),
        change_point=episodes[0][0] if episodes else None,
        calibration_scores=cal_scores,
        calibration_context=causal_local_context(cal_scores).reshape(-1, 1),
        episodes=episodes,
    )


def run_alarms(
    method: str,
    knob: float,
    st: Stream,
    lag: int,
    extra_options: Optional[Dict] = None,
) -> np.ndarray:
    """All alarm frame indices (sorted) raised by ``method`` at ``knob`` on ``st``.

    Every method uses the pooled conformal calibration, the same betting-grid
    ``lag`` and restart-after-alarm, so no method is thinned differently from
    another. ``extra_options`` overrides these defaults.
    """
    opts = {"calibration": "pooled", "lag": int(lag), "restart": True}
    opts.update(extra_options or {})
    spec = Spec(
        method=method, knob=knob, seed=0, change_point=st.change_point,
        cfg={}, options=tuple(sorted(opts.items())),
    )
    idx, _abstention, _lag = alarm_indices_on_stream(st, spec)
    return np.asarray(idx, dtype=np.int64)


def score_labelled_stream(
    alarm_idx: Sequence[int],
    labels: np.ndarray,
    episodes: Sequence[Tuple[int, int]],
    grace: int,
    fps: float = FPS,
) -> dict:
    """Episode-level detection, delay and false-alarm accounting for one stream.

    An episode ``(s, e)`` is detected iff some alarm ``a`` satisfies
    ``s <= a < e + grace``; its delay is the first such ``a`` minus ``s``.
    An alarm is a false alarm iff it lies in no window ``[s, e + grace)``.
    ``normal_frames`` counts frames with ``labels == False`` that lie in no
    window; ``normal_hours`` converts that to hours at ``fps``.

    Returns a dict with ``n_episodes``, ``n_detected``, ``delays`` (frames,
    one per detected episode), ``n_false_alarms``, ``normal_frames`` and
    ``normal_hours``.
    """
    labels = np.asarray(labels, dtype=bool)
    T = labels.size
    alarms = np.sort(np.asarray(alarm_idx, dtype=np.int64))
    episodes = [(int(s), int(e)) for s, e in episodes]

    covered = np.zeros(T, dtype=bool)
    delays: List[int] = []
    n_detected = 0
    in_any_window = np.zeros(alarms.size, dtype=bool)
    for s, e in episodes:
        hi = min(e + grace, T)
        covered[s:hi] = True
        in_win = (alarms >= s) & (alarms < e + grace)
        in_any_window |= in_win
        hit = alarms[in_win]
        if hit.size:
            n_detected += 1
            delays.append(int(hit[0]) - s)

    n_false = int(np.count_nonzero(~in_any_window))
    normal_frames = int(np.count_nonzero(~labels & ~covered))
    return {
        "n_episodes": len(episodes),
        "n_detected": int(n_detected),
        "delays": delays,
        "n_false_alarms": n_false,
        "normal_frames": normal_frames,
        "normal_hours": normal_frames / float(fps) / 3600.0,
    }

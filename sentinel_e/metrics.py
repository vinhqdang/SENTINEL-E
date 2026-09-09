"""Evaluation metrics for streaming detection.

The quantities that matter operationally are not clip-level precision and
recall but

``PFA``
    probability that a *null* stream of a given length produces at least one
    alarm --- the quantity a fixed threshold silently inflates;
``ARL0``
    average run length to a false alarm, in frames, on a null stream;
``ADD``
    average detection delay after the true onset of dumping, counted only on
    runs that alarm after the onset;
``censored ADD``
    the same, but a run that never alarms is charged the whole remaining
    horizon rather than dropped;
``missed-detection rate``
    fraction of post-change segments that end before any alarm.

The distinction between the two delay metrics is not pedantry. Conditional
``ADD`` is averaged over a *different subset of runs* for each method, so a
method that misses the hard events looks fast --- it is being scored only on the
easy ones. Whenever two methods differ in miss rate, and throughout this work
they do, the censored figure is the comparable one and is what head-to-head
claims should rest on.

The headline plot of the paper is delay as a function of realised PFA (or of
ARL0), obtained by sweeping each method's own tuning knob.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "first_alarm_index",
    "run_length",
    "detection_delay",
    "RunOutcome",
    "summarize_runs",
    "delay_far_curve",
    "fdr_power",
    "wilson_interval",
    "PerEpisodeOutcome",
    "per_episode_outcomes",
    "summarize_per_episode",
]


def first_alarm_index(alarms: Sequence[bool]) -> Optional[int]:
    """Index of the first ``True`` in ``alarms``, or ``None``."""
    idx = np.flatnonzero(np.asarray(alarms, dtype=bool))
    return int(idx[0]) if idx.size else None


def run_length(alarms: Sequence[bool], horizon: Optional[int] = None) -> int:
    """Frames until the first alarm; the (censored) horizon if none occurs."""
    a = np.asarray(alarms, dtype=bool)
    idx = first_alarm_index(a)
    T = a.size if horizon is None else int(horizon)
    return T if idx is None else idx + 1


def detection_delay(alarms: Sequence[bool], change_point: int) -> Optional[int]:
    """Frames between the onset ``change_point`` and the first alarm at or after it.

    Alarms strictly *before* the onset are false alarms and are ignored here;
    they are accounted for separately by ``PFA``.  Returns ``None`` when the
    post-change segment ends without an alarm (a missed detection).
    """
    a = np.asarray(alarms, dtype=bool)
    if change_point < 0 or change_point >= a.size:
        raise ValueError("change_point outside the stream")
    post = np.flatnonzero(a[change_point:])
    return int(post[0]) if post.size else None


@dataclass
class RunOutcome:
    """Aggregate outcome over a set of Monte-Carlo streams."""

    n_null: int
    n_signal: int
    pfa: float                    # P(at least one alarm on a null stream)
    pfa_ci: tuple                 # Wilson 95% interval for pfa
    arl0: float                   # mean run length on null streams (frames)
    add: float                    # mean detection delay (frames), detected only
    add_median: float
    add_se: float
    add_censored: float           # misses charged the remaining horizon
    add_censored_se: float
    miss_rate: float              # fraction of post-change segments never flagged
    pre_change_fa: float          # P(alarm before onset) on signal streams

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion (stable at k = 0 or n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def summarize_runs(
    null_alarms: Sequence[Sequence[bool]],
    signal_alarms: Sequence[Sequence[bool]] = (),
    change_points: Sequence[int] = (),
) -> RunOutcome:
    """Summarise null and post-change streams into the operational metrics."""
    null_alarms = list(null_alarms)
    n_null = len(null_alarms)
    fa_flags = [first_alarm_index(a) is not None for a in null_alarms]
    n_fa = int(np.sum(fa_flags))
    pfa = n_fa / n_null if n_null else float("nan")
    rls = [run_length(a) for a in null_alarms]
    arl0 = float(np.mean(rls)) if rls else float("nan")

    signal_alarms = list(signal_alarms)
    change_points = list(change_points)
    if len(signal_alarms) != len(change_points):
        raise ValueError("signal_alarms and change_points must align")
    delays: List[int] = []
    censored: List[int] = []
    misses = 0
    pre_fa = 0
    for a, nu in zip(signal_alarms, change_points):
        arr = np.asarray(a, dtype=bool)
        if np.any(arr[:nu]):
            pre_fa += 1
        d = detection_delay(arr, nu)
        if d is None:
            misses += 1
            censored.append(int(arr.size - nu))   # charged the full remainder
        else:
            delays.append(d)
            censored.append(d)
    n_signal = len(signal_alarms)
    add = float(np.mean(delays)) if delays else float("nan")
    add_median = float(np.median(delays)) if delays else float("nan")
    add_se = float(np.std(delays, ddof=1) / np.sqrt(len(delays))) if len(delays) > 1 else float("nan")
    add_c = float(np.mean(censored)) if censored else float("nan")
    add_c_se = (
        float(np.std(censored, ddof=1) / np.sqrt(len(censored)))
        if len(censored) > 1 else float("nan")
    )

    return RunOutcome(
        n_null=n_null,
        n_signal=n_signal,
        pfa=pfa,
        pfa_ci=wilson_interval(n_fa, n_null) if n_null else (float("nan"),) * 2,
        arl0=arl0,
        add=add,
        add_median=add_median,
        add_se=add_se,
        add_censored=add_c,
        add_censored_se=add_c_se,
        miss_rate=misses / n_signal if n_signal else float("nan"),
        pre_change_fa=pre_fa / n_signal if n_signal else float("nan"),
    )


def delay_far_curve(outcomes: Iterable[RunOutcome]) -> Dict[str, np.ndarray]:
    """Extract the ``(PFA, ADD)`` operating curve from a knob sweep."""
    out = list(outcomes)
    pfa = np.array([o.pfa for o in out], dtype=float)
    add = np.array([o.add for o in out], dtype=float)
    addc = np.array([o.add_censored for o in out], dtype=float)
    arl0 = np.array([o.arl0 for o in out], dtype=float)
    order = np.argsort(pfa)
    return {"pfa": pfa[order], "add": add[order],
            "add_censored": addc[order], "arl0": arl0[order]}


def fdr_power(rejected: Sequence[bool], truth: Sequence[bool]) -> Dict[str, float]:
    """False-discovery proportion and power of a fleet-level decision."""
    r = np.asarray(rejected, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if r.shape != t.shape:
        raise ValueError("rejected and truth must have the same shape")
    n_rej = int(r.sum())
    n_false = int((r & ~t).sum())
    n_true_alt = int(t.sum())
    return {
        "fdp": n_false / n_rej if n_rej else 0.0,
        "power": int((r & t).sum()) / n_true_alt if n_true_alt else float("nan"),
        "n_rejected": float(n_rej),
    }


@dataclass(frozen=True)
class PerEpisodeOutcome:
    """Whether and how fast *this one episode* was flagged.

    ``summarize_runs``'s ``miss_rate`` is stream-level: with several episodes
    per camera, a stream counts as detected the moment *any* alarm follows the
    *first* onset, so a detector that sleeps through episodes 1 through k-1 and
    only fires on episode k is scored a clean detection. That hides exactly the
    mechanism the episodic e-process claims credit for --- recovering after a
    missed episode rather than carrying a burned wealth process into the next
    one. This scores each episode's own detection window separately.
    """

    episode_index: int
    start: int
    window_end: int          # exclusive: the next episode's start, or the horizon
    delay: Optional[int]     # frames from ``start`` to the first alarm in-window
    missed: bool


def per_episode_outcomes(
    alarms: Sequence[bool],
    episodes: Sequence[tuple],
    horizon: Optional[int] = None,
) -> List[PerEpisodeOutcome]:
    """Score each episode's own detection window.

    ``episodes`` is the stream's ``[(start, end), ...]`` list, in order. Each
    episode's detection window runs from its own ``start`` to the *next*
    episode's start (or the horizon, for the last one) --- so an alarm counts
    toward the episode whose window it falls in, whether it fires during the
    episode itself or during the quiet gap that follows, and a restart between
    alarms (the convention used throughout this paper) means each window's
    result is independent of the others.
    """
    a = np.asarray(alarms, dtype=bool)
    T = a.size if horizon is None else int(horizon)
    out: List[PerEpisodeOutcome] = []
    for i, (start, _end) in enumerate(episodes):
        window_end = int(episodes[i + 1][0]) if i + 1 < len(episodes) else T
        window_end = min(window_end, T)
        start = int(start)
        if start >= window_end:
            continue
        d = detection_delay(a[:window_end], start)
        out.append(PerEpisodeOutcome(
            episode_index=i, start=start, window_end=window_end,
            delay=d, missed=d is None,
        ))
    return out


def summarize_per_episode(
    per_stream: Sequence[Sequence[PerEpisodeOutcome]],
) -> Dict[str, float]:
    """Pool per-episode outcomes across streams into aggregate rates.

    Unlike ``summarize_runs``'s stream-level ``miss_rate``, this is the
    fraction of individual episodes --- not streams --- that end unflagged,
    which is the definition \\S3 gives and the one the per-episode mechanism
    claim needs.
    """
    all_outcomes = [o for stream in per_stream for o in stream]
    n = len(all_outcomes)
    if n == 0:
        return {"n_episodes": 0, "episode_miss_rate": float("nan"),
                "episode_add": float("nan"), "episode_add_se": float("nan"),
                "episode_add_censored": float("nan"),
                "episode_add_censored_se": float("nan")}
    misses = sum(o.missed for o in all_outcomes)
    delays = [o.delay for o in all_outcomes if o.delay is not None]
    censored = [
        o.delay if o.delay is not None else (o.window_end - o.start)
        for o in all_outcomes
    ]
    return {
        "n_episodes": n,
        "episode_miss_rate": misses / n,
        "episode_add": float(np.mean(delays)) if delays else float("nan"),
        "episode_add_se": (
            float(np.std(delays, ddof=1) / np.sqrt(len(delays)))
            if len(delays) > 1 else float("nan")
        ),
        "episode_add_censored": float(np.mean(censored)),
        "episode_add_censored_se": (
            float(np.std(censored, ddof=1) / np.sqrt(len(censored)))
            if len(censored) > 1 else float("nan")
        ),
    }

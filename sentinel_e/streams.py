"""Streaming benchmarks: long monitoring streams for a camera fleet.

Two constructors are provided.

:class:`StreamSimulator`
    A generative model of *per-frame detector scores* rather than of pixels.
    Because SENTINEL-E sits on top of a frozen backbone and only ever sees the
    backbone's scalar output, simulating that output is the faithful way to
    stress the guarantee layer at the timescales that matter (weeks of video,
    :math:`10^6` frames), which no released clip dataset covers.  The model is
    deliberately unfriendly to the baselines in the ways real footage is:

    * a diurnal illumination cycle and a two-state weather chain shift the
      null score distribution *after* calibration;
    * scores are temporally autocorrelated (AR(1)) --- consecutive frames are
      not independent evidence;
    * the null tail is heavy (Student-:math:`t`), so Gaussian CUSUM/SR
      mis-specify it;
    * dumping events are *intermittent*: within an event window only a fraction
      of frames are actually anomalous (the offender is occluded, walks out of
      frame, returns);
    * events propagate along the camera graph with a lag, which is the
      structure the spatial prior is meant to exploit.

:func:`build_stream_from_clips`
    The protocol used with a real clip-labelled corpus such as Mivia-IWDD:
    concatenate confirmed no-dumping clips into a long background stream and
    splice labelled dumping clips in at random onsets.  This turns any
    fixed-sample-size video benchmark into a streaming one without collecting
    new data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

from sentinel_e.graph import CameraGraph

__all__ = [
    "StreamConfig",
    "Stream",
    "FleetStream",
    "StreamSimulator",
    "build_stream_from_clips",
]


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Stream:
    """One camera's monitoring stream."""

    scores: np.ndarray                 # per-frame detector scores in (0, 1)
    context: np.ndarray                # per-frame context features, (T, d)
    labels: np.ndarray                 # per-frame dumping indicator (bool)
    change_point: Optional[int]        # onset of the first dumping episode
    calibration_scores: np.ndarray     # confirmed no-dumping scores
    calibration_context: np.ndarray
    episodes: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def T(self) -> int:
        return int(self.scores.size)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)


@dataclass
class FleetStream:
    """Synchronous streams for a whole camera fleet plus the graph."""

    streams: List[Stream]
    graph: CameraGraph
    change_points: List[Optional[int]]

    @property
    def n_cameras(self) -> int:
        return len(self.streams)

    @property
    def T(self) -> int:
        return self.streams[0].T

    @property
    def affected(self) -> np.ndarray:
        """Boolean mask of cameras that actually experience a dumping event."""
        return np.array([cp is not None for cp in self.change_points], dtype=bool)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class StreamConfig:
    """Parameters of the score-level generative model.

    Defaults correspond to a competent but imperfect lightweight backbone: on
    calibration-matched frames the null score sits near 0.12 and a fully
    anomalous frame near 0.45, i.e. the per-frame signal is weak and only
    accumulates over time --- exactly the regime where sequential methods earn
    their keep and single-frame thresholds do not.
    """

    T: int = 100_000                   # frames per stream (~67 min at 25 fps)
    n_calibration: int = 150_000       # raw no-dumping frames (~100 min at 25 fps)
    fps: float = 25.0                  # native frame rate of the backbone

    # Null score model (on the logit scale).
    mu_null: float = -2.0
    sigma_null: float = 0.8
    ar_rho: float = 0.85               # AR(1) coefficient of the latent noise
    t_df: float = 4.0                  # Student-t degrees of freedom (heavy tail)

    # Non-stationarity.
    diurnal_amp: float = 0.6           # logit-scale swing over a day
    diurnal_period: int = 40_000       # frames per simulated day
    weather_shift: float = 0.5         # logit-scale offset in the "bad weather" state
    weather_p_enter: float = 3.0e-5    # per-frame transition probabilities
    weather_p_exit: float = 8.0e-5
    luminance_noise: float = 0.15      # measurement noise on the ambient-light proxy

    # Scene activity: pedestrians, traffic, street cleaning.  It drives the
    # backbone's false-positive rate but is *not* part of the operator's
    # obvious (clock, weather) taxonomy, so it is the covariate that the
    # likelihood-ratio weighting layer exists to handle.  ``activity_shift``
    # multiplies the deployment activity level relative to the labelling window
    # (a quiet week used for labelling, a busy month in deployment).
    activity_mean: float = 1.0
    activity_coef: float = 0.55        # logit-scale effect of unit activity
    activity_shift: float = 1.0        # deployment / calibration activity ratio
    activity_timescale: float = 250.0  # OU correlation time in frames (~10 s)
    activity_log_sd: float = 0.45      # stationary spread on the log scale

    # How the per-camera calibration set of confirmed no-dumping frames was
    # collected.  ``full_cycle_benign`` models a labelling window that spans a
    # full day-night cycle but happens to contain no adverse weather, leaving
    # that regime uncalibrated.
    # ``daytime_only`` is the stress test in which even the diurnal cycle is
    # only partially covered.  ``representative`` is the realistic default: a
    # labelling window long enough to span both the diurnal cycle and the
    # weather regimes that occur during deployment.
    calibration_regime: str = "representative"

    # Dumping events.
    event_shift: float = 2.6           # logit-scale mean shift on anomalous frames
    event_length: int = 1_500          # frames (= 60 s at 25 fps)
    event_intermittency: float = 0.5   # fraction of event frames that are anomalous
    event_ramp: int = 150              # frames over which the shift ramps in/out

    # Illegal dumping recurs: a site that has been used once is used again.
    # ``n_episodes`` consecutive episodes are placed at the same camera,
    # separated by gaps drawn around ``episode_gap``.  This is the structure the
    # episodic e-process is built for, and the regime in which a
    # changepoint-mixture detector degrades.
    n_episodes: int = 1
    episode_gap: int = 6_000           # mean quiet gap between episodes (frames)

    # Fleet.
    n_cameras: int = 12
    area_size: float = 1_200.0         # metres
    graph_length_scale: float = 300.0
    k_neighbors: int = 4
    weather_correlation: float = 0.9   # fraction of the weather state shared fleet-wide
    propagation_prob: float = 0.45     # chance an event recurs at a graph neighbour
    propagation_lag: int = 2_000       # mean lag in frames (~80 s at 25 fps)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class StreamSimulator:
    """Generates null and post-change streams for single cameras or fleets."""

    def __init__(self, config: Optional[StreamConfig] = None, seed: int = 0) -> None:
        self.cfg = config if config is not None else StreamConfig()
        self.rng = np.random.default_rng(seed)

    # -- latent processes -------------------------------------------------- #
    @staticmethod
    def _ar1(rho: float, innovations: np.ndarray, x0: float) -> np.ndarray:
        """Vectorised AR(1) recursion ``x_t = rho x_{t-1} + e_t`` with ``x_0 = x0``.

        A Python loop over a million frames dominates the cost of every
        experiment, so the recursion is delegated to an IIR filter; the result is
        bit-comparable to the loop up to floating-point associativity.
        """
        if rho == 0.0:
            out = innovations.copy()
            out[0] = x0
            return out
        e = innovations.copy()
        e[0] = 0.0
        return lfilter([1.0], [1.0, -rho], e, zi=[rho * x0])[0]

    def _ar_noise(self, T: int) -> np.ndarray:
        """Stationary AR(1) sequence with Student-t innovations, unit marginal sd."""
        cfg = self.cfg
        df = cfg.t_df
        # Scale innovations so the stationary marginal has unit variance.
        t_var = df / (df - 2.0) if df > 2 else 1.0
        innov_sd = np.sqrt((1.0 - cfg.ar_rho**2) / t_var)
        e = self.rng.standard_t(df, size=T) * innov_sd
        x0 = float(self.rng.standard_t(df) / np.sqrt(t_var))
        return self._ar1(cfg.ar_rho, e, x0)

    def _weather(self, T: int, shared: Optional[np.ndarray] = None) -> np.ndarray:
        """Two-state (benign / adverse) Markov chain, optionally fleet-coupled."""
        cfg = self.cfg
        # Sample the two-state chain by alternating geometric sojourn times,
        # which is exact and costs O(number of switches) rather than O(T).
        own = np.zeros(T, dtype=bool)
        state = False
        t = 0
        while t < T:
            p_leave = cfg.weather_p_exit if state else cfg.weather_p_enter
            run = int(self.rng.geometric(max(p_leave, 1e-12)))
            end = min(T, t + run)
            if state:
                own[t:end] = True
            t = end
            state = not state
        if shared is None:
            return own
        mix = self.rng.random(T) < cfg.weather_correlation
        return np.where(mix, shared, own)

    def _diurnal(self, T: int, phase: float) -> np.ndarray:
        cfg = self.cfg
        t = np.arange(T)
        return cfg.diurnal_amp * np.sin(2 * np.pi * (t / cfg.diurnal_period) + phase)

    # -- one camera --------------------------------------------------------- #
    def simulate_camera(
        self,
        change_point: Optional[int] = None,
        T: Optional[int] = None,
        phase: float = 0.0,
        shared_weather: Optional[np.ndarray] = None,
        event_length: Optional[int] = None,
    ) -> Stream:
        """Simulate a single camera stream, optionally with a dumping onset."""
        cfg = self.cfg
        T = cfg.T if T is None else int(T)
        L = cfg.event_length if event_length is None else int(event_length)

        diurnal = self._diurnal(T, phase)
        weather = self._weather(T, shared_weather)
        noise = self._ar_noise(T)
        activity = self._activity(T, cfg.activity_mean * cfg.activity_shift)

        logit = (
            cfg.mu_null
            + diurnal
            + cfg.weather_shift * weather
            + cfg.activity_coef * activity
            + cfg.sigma_null * noise
        )

        labels = np.zeros(T, dtype=bool)
        episodes: List[Tuple[int, int]] = []
        if change_point is not None:
            cp = int(np.clip(change_point, 0, T - 1))
            start = cp
            for _ in range(max(cfg.n_episodes, 1)):
                if start >= T - 1:
                    break
                end = min(T, start + L)
                # Intermittent anomalous frames inside the episode, with a
                # smooth ramp so the onset is not an implausible step.
                idx = np.arange(start, end)
                ramp = np.clip((idx - start) / max(cfg.event_ramp, 1), 0.0, 1.0)
                ramp *= np.clip((end - idx) / max(cfg.event_ramp, 1), 0.0, 1.0)
                active = self.rng.random(idx.size) < cfg.event_intermittency
                logit[start:end] += cfg.event_shift * ramp * active
                labels[start:end] |= active
                episodes.append((start, int(end)))
                start = int(end + self.rng.exponential(cfg.episode_gap))
        else:
            cp = None

        scores = 1.0 / (1.0 + np.exp(-logit))
        context = self._context(diurnal, weather.astype(float), activity)

        cal_scores, cal_context = self._calibration(phase)
        return Stream(
            scores=scores,
            context=context,
            labels=labels,
            change_point=cp,
            calibration_scores=cal_scores,
            calibration_context=cal_context,
            episodes=episodes,
        )

    def _activity(self, T: int, mean: float) -> np.ndarray:
        """Slowly varying, strictly positive scene-activity level.

        A log-scale Ornstein--Uhlenbeck process: activity persists over minutes
        (a passing delivery round, a market closing) rather than fluctuating
        frame to frame, while keeping a genuine stationary spread.  A moving
        average of i.i.d. draws would look smooth but have almost no variance
        left, which would make any deployment shift trivially out-of-support.
        """
        cfg = self.cfg
        phi = float(np.exp(-1.0 / max(cfg.activity_timescale, 1.0)))
        sd = cfg.activity_log_sd
        innov = self.rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), size=T)
        z = self._ar1(phi, innov, float(self.rng.normal(0.0, sd)))
        return mean * np.exp(z - 0.5 * sd**2)

    def _context(
        self, diurnal: np.ndarray, weather: np.ndarray, activity: np.ndarray
    ) -> np.ndarray:
        """Observable per-frame context: clock phase, weather, luminance, activity.

        All four are available to a deployed camera without privileged
        knowledge: the clock drives the first column, a weather feed or rain
        sensor the second, a scene-brightness statistic the third and a
        background-subtraction motion measure the fourth.  The latent detector
        noise is deliberately *not* exposed.
        """
        cfg = self.cfg
        lum = (
            cfg.mu_null
            + diurnal
            - cfg.weather_shift * weather
            + cfg.luminance_noise * self.rng.normal(size=diurnal.size)
        )
        return np.stack([diurnal, weather, lum, activity], axis=1)

    def _calibration(self, phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """Confirmed no-dumping frames as an operator would actually collect them."""
        cfg = self.cfg
        n = cfg.n_calibration
        regime = cfg.calibration_regime
        if regime == "daytime_only":
            # A single clean daytime labelling window: most of the diurnal
            # cycle is never observed, so most deployment bins are uncalibrated.
            t = np.linspace(0.15, 0.35, n) * cfg.diurnal_period
            diurnal = cfg.diurnal_amp * np.sin(2 * np.pi * (t / cfg.diurnal_period) + phase)
            weather = np.zeros(n)
        elif regime == "full_cycle_benign":
            # A labelling window spanning whole days, but with no adverse
            # weather: the diurnal cycle is covered, the weather regime is not.
            t = np.linspace(0.0, 1.0, n) * cfg.diurnal_period
            diurnal = cfg.diurnal_amp * np.sin(2 * np.pi * (t / cfg.diurnal_period) + phase)
            weather = np.zeros(n)
        elif regime == "representative":
            diurnal = self._diurnal(n, phase)
            weather = (self.rng.random(n) < 0.25).astype(float)
        else:
            raise ValueError(f"unknown calibration_regime {regime!r}")
        noise = self._ar_noise(n)
        activity = self._activity(n, cfg.activity_mean)
        logit = (
            cfg.mu_null
            + diurnal
            + cfg.weather_shift * weather
            + cfg.activity_coef * activity
            + cfg.sigma_null * noise
        )
        scores = 1.0 / (1.0 + np.exp(-logit))
        return scores, self._context(diurnal, weather, activity)

    # -- fleet -------------------------------------------------------------- #
    def simulate_fleet(
        self,
        n_events: int = 3,
        T: Optional[int] = None,
        propagate: bool = True,
    ) -> FleetStream:
        """Simulate a synchronous fleet with graph-correlated events."""
        cfg = self.cfg
        T = cfg.T if T is None else int(T)
        K = cfg.n_cameras

        positions = self.rng.uniform(0.0, cfg.area_size, size=(K, 2))
        static_ctx = self.rng.normal(size=(K, 2))
        graph = CameraGraph.from_positions(
            positions,
            context=static_ctx,
            length_scale=cfg.graph_length_scale,
            k_neighbors=cfg.k_neighbors,
            context_scale=2.0,
        )

        change_points: List[Optional[int]] = [None] * K
        seeds = self.rng.choice(K, size=min(n_events, K), replace=False)
        for i in seeds:
            change_points[int(i)] = int(self.rng.integers(T // 5, int(0.8 * T)))
        if propagate:
            for i in list(seeds):
                for j in graph.neighbors(int(i)):
                    if change_points[int(j)] is not None:
                        continue
                    if self.rng.random() < cfg.propagation_prob:
                        lag = int(self.rng.exponential(cfg.propagation_lag))
                        cp = change_points[int(i)] + lag
                        if cp < T - cfg.event_length:
                            change_points[int(j)] = cp

        shared_weather = self._weather(T)
        phases = self.rng.uniform(0.0, 2 * np.pi, size=K)
        streams = [
            self.simulate_camera(
                change_point=change_points[i],
                T=T,
                phase=float(phases[i]),
                shared_weather=shared_weather,
            )
            for i in range(K)
        ]
        return FleetStream(streams=streams, graph=graph, change_points=change_points)


# --------------------------------------------------------------------------- #
# Clip-based stream construction (real corpora)
# --------------------------------------------------------------------------- #
def build_stream_from_clips(
    clip_scores: Sequence[np.ndarray],
    clip_labels: Sequence[int],
    n_frames: int,
    n_calibration: int = 2_000,
    n_events: int = 1,
    rng: Optional[np.random.Generator] = None,
    clip_context: Optional[Sequence[np.ndarray]] = None,
) -> Stream:
    """Splice clip-level detector outputs into a long monitoring stream.

    Parameters
    ----------
    clip_scores : sequence of 1-D arrays
        Per-frame detector scores for each clip of the source corpus.
    clip_labels : sequence of int
        ``0`` for a confirmed no-dumping clip, ``1`` for a dumping clip.
    n_frames : int
        Length of the constructed monitoring stream.
    n_calibration : int
        Frames reserved, from *held-out* negative clips, as the calibration set.
        Calibration clips are never reused in the stream, which keeps the
        exchangeability assumption of the conformal layer intact.
    n_events : int
        Number of dumping clips to splice in at random onsets.

    Returns
    -------
    Stream
        With ``change_point`` set to the onset of the first spliced event.
    """
    rng = np.random.default_rng() if rng is None else rng
    labels_arr = np.asarray(clip_labels, dtype=int)
    neg_idx = np.flatnonzero(labels_arr == 0)
    pos_idx = np.flatnonzero(labels_arr == 1)
    if neg_idx.size == 0:
        raise ValueError("at least one no-dumping clip is required")
    if n_events > 0 and pos_idx.size == 0:
        raise ValueError("no dumping clips available to splice")

    rng.shuffle(neg_idx)
    # Reserve negative clips for calibration until the budget is met.
    cal_parts, cal_ctx_parts, used = [], [], 0
    for k in neg_idx:
        cal_parts.append(np.asarray(clip_scores[k], dtype=float).ravel())
        if clip_context is not None:
            cal_ctx_parts.append(np.atleast_2d(clip_context[k]))
        used += 1
        if sum(len(c) for c in cal_parts) >= n_calibration:
            break
    calibration = np.concatenate(cal_parts)[:n_calibration]
    cal_context = (
        np.vstack(cal_ctx_parts)[:n_calibration]
        if cal_ctx_parts
        else np.zeros((calibration.size, 1))
    )
    background_pool = neg_idx[used:]
    if background_pool.size == 0:
        raise ValueError(
            "all negative clips were consumed by calibration; "
            "reduce n_calibration or supply more clips"
        )

    # Background: sample negative clips with replacement until long enough.
    parts, ctx_parts, total = [], [], 0
    while total < n_frames:
        k = int(rng.choice(background_pool))
        s = np.asarray(clip_scores[k], dtype=float).ravel()
        parts.append(s)
        ctx_parts.append(
            np.atleast_2d(clip_context[k]) if clip_context is not None
            else np.zeros((s.size, 1))
        )
        total += s.size
    scores = np.concatenate(parts)[:n_frames]
    context = np.vstack(ctx_parts)[:n_frames]
    labels = np.zeros(n_frames, dtype=bool)

    change_point: Optional[int] = None
    if n_events > 0:
        onsets = np.sort(rng.choice(
            np.arange(n_frames // 10, int(0.85 * n_frames)),
            size=n_events, replace=False,
        ))
        for onset in onsets:
            k = int(rng.choice(pos_idx))
            ev = np.asarray(clip_scores[k], dtype=float).ravel()
            end = min(n_frames, onset + ev.size)
            scores[onset:end] = ev[: end - onset]
            labels[onset:end] = True
        change_point = int(onsets[0])

    return Stream(
        scores=scores,
        context=context,
        labels=labels,
        change_point=change_point,
        calibration_scores=calibration,
        calibration_context=cal_context,
    )

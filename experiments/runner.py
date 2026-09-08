"""Monte-Carlo workers shared by the single-camera experiments.

A worker returns only the two frame indices the metrics actually depend on ---
the first alarm anywhere, and the first alarm at or after the onset --- rather
than a 60,000-element boolean array, which keeps inter-process traffic
negligible while reproducing exactly what
:func:`sentinel_e.metrics.summarize_runs` would compute from the full arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.common import parallel_map
from sentinel_e.baselines import (
    CUSUM,
    FixedThreshold,
    ParametricEDetector,
    PValueThreshold,
    ShiryaevRoberts,
)
from sentinel_e.metrics import RunOutcome, summarize_runs
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices

#: Methods that consume conformal p-values rather than raw scores.
PVALUE_METHODS = {"SENTINEL-E", "p-value threshold", "Oracle p-values"}


@dataclass(frozen=True)
class Spec:
    """One Monte-Carlo run."""

    method: str
    knob: float
    seed: int
    change_point: Optional[int]
    cfg: Dict
    options: Tuple = ()

    @property
    def opt(self) -> Dict:
        return dict(self.options)


@dataclass(frozen=True)
class Trace:
    """Compact outcome of one run."""

    T: int
    first_alarm: int          # -1 when the stream is never flagged
    first_after_nu: int       # -1 when the post-change segment is never flagged
    abstention: float
    lag: int
    n_alarms: int


def _to_alarm_array(tr: Trace) -> np.ndarray:
    a = np.zeros(tr.T, dtype=bool)
    if tr.first_alarm >= 0:
        a[tr.first_alarm] = True
    if tr.first_after_nu >= 0:
        a[tr.first_after_nu] = True
    return a


def _first(idx: np.ndarray, start: int = 0) -> int:
    hit = idx[idx >= start]
    return int(hit[0]) if hit.size else -1


def run_spec(spec: Spec) -> Trace:
    """Simulate one stream and run one detector over it."""
    cfg = StreamConfig(**spec.cfg)
    sim = StreamSimulator(cfg, seed=spec.seed)
    st = sim.simulate_camera(change_point=spec.change_point)
    opt = spec.opt
    nu = spec.change_point if spec.change_point is not None else 0

    if spec.method == "Oracle p-values":
        # Diagnostic: feed the same e-detector exactly-uniform p-values on the
        # same betting grid.  Any remaining gap to the nominal level is the
        # slack in Ville's inequality itself, not a cost of the conformal layer.
        model = SentinelE(calibration="residual", weighted=False, alpha=spec.knob,
                          restart=False)
        model.fit(st.calibration_scores, st.calibration_context)
        bet = thin_indices(st.T, model.decorrelation_lag)
        rng = np.random.default_rng(spec.seed + 7_000_000)
        p = rng.uniform(size=bet.size)
        if spec.change_point is not None:
            post = bet >= spec.change_point
            p[post] = p[post] ** opt.get("oracle_power", 8.0)
        model.detector.reset()
        _, al = model.detector.run(p)
        idx = bet[np.flatnonzero(al)]
        return Trace(st.T, _first(idx), _first(idx, nu), 0.0,
                     model.decorrelation_lag, int(idx.size))

    if spec.method in PVALUE_METHODS:
        cal_kw = dict(
            calibration=opt.get("calibration", "residual"),
            weighted=opt.get("weighted", False),
            family=opt.get("family", "power"),
            n_grid=opt.get("n_grid", 32),
            rho=opt.get("rho", 1e-3),
            prior_kind=opt.get("prior_kind", "geometric"),
            delta=opt.get("delta", 1e-3),
            restart=False,
            lag=opt.get("lag", None),
            thin_calibration=opt.get("thin_calibration", True),
        )
        if spec.method == "SENTINEL-E":
            model = SentinelE(alpha=spec.knob, **cal_kw)
            res = model.run_stream(st)
            idx = np.flatnonzero(res.alarms)
            return Trace(st.T, _first(idx), _first(idx, nu),
                         res.abstention_rate, res.lag, int(idx.size))
        # p-value threshold: identical conformal front end, naive stopping rule.
        model = SentinelE(alpha=0.01, **cal_kw)
        model.fit(st.calibration_scores, st.calibration_context)
        bet = thin_indices(st.T, model.decorrelation_lag)
        p, active = model.p_values(st.scores[bet], st.context[bet])
        fired = PValueThreshold(level=spec.knob).run_pvalues(p) & active
        idx = bet[np.flatnonzero(fired)]
        return Trace(st.T, _first(idx), _first(idx, nu), float(1 - active.mean()),
                     model.decorrelation_lag, int(idx.size))

    # Score-based baselines: calibrated on the same confirmed no-dumping frames.
    if spec.method == "Fixed threshold":
        det = FixedThreshold(k_of_m=opt.get("k_of_m", (1, 1)))
        det.calibrate_threshold(st.calibration_scores, per_frame_rate=spec.knob)
    elif spec.method == "CUSUM":
        det = CUSUM(delta=opt.get("delta_shift", 1.0), h=spec.knob, restart=False)
        det.fit(st.calibration_scores)
    elif spec.method == "Shiryaev--Roberts":
        det = ShiryaevRoberts(delta=opt.get("delta_shift", 1.0), A=spec.knob, restart=False)
        det.fit(st.calibration_scores)
    elif spec.method == "Parametric e-detector":
        det = ParametricEDetector(alpha=spec.knob, delta=opt.get("delta_shift", 1.0),
                                  rho=opt.get("rho", 1e-3), restart=False)
        det.fit(st.calibration_scores)
    else:
        raise ValueError(f"unknown method {spec.method!r}")

    # Baselines are given the same decorrelated betting grid, so no method is
    # penalised or helped by the frame-rate choice.
    if opt.get("thin_baselines", True):
        model = SentinelE(calibration="residual", weighted=False)
        model.fit(st.calibration_scores, st.calibration_context)
        bet = thin_indices(st.T, model.decorrelation_lag)
        lag = model.decorrelation_lag
    else:
        bet, lag = np.arange(st.T), 1
    fired = det.run(st.scores[bet])
    idx = bet[np.flatnonzero(fired)]
    return Trace(st.T, _first(idx), _first(idx, nu), 0.0, lag, int(idx.size))


def evaluate(
    method: str,
    knob: float,
    cfg: Dict,
    reps: int,
    change_point: Optional[int],
    options: Optional[Dict] = None,
    seed0: int = 10_000,
    workers: Optional[int] = None,
) -> Tuple[RunOutcome, List[Trace], List[Trace]]:
    """Run ``reps`` null streams and ``reps`` post-change streams for one setting."""
    opts = tuple(sorted((options or {}).items()))
    null_specs = [Spec(method, knob, seed0 + i, None, cfg, opts) for i in range(reps)]
    sig_specs = [
        Spec(method, knob, seed0 + 500_000 + i, change_point, cfg, opts)
        for i in range(reps)
    ]
    null = parallel_map(run_spec, null_specs, workers)
    sig = parallel_map(run_spec, sig_specs, workers) if change_point is not None else []
    outcome = summarize_runs(
        [_to_alarm_array(t) for t in null],
        [_to_alarm_array(t) for t in sig],
        [change_point] * len(sig),
    )
    return outcome, null, sig

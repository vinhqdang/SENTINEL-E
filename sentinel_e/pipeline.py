"""End-to-end SENTINEL-E pipelines.

:class:`SentinelE`
    One camera: temporal decorrelation -> conformal calibration -> betting
    e-detector.
:class:`FleetSentinelE`
    A whole fleet stepped in lockstep, with the graph controller supplying a
    predictable hazard / stake modulation and e-BH combining the per-camera
    e-values into a fleet-level decision with FDR control.

The three preprocessing choices --- how often to bet, which calibration bin to
compare against, and whether the current context is calibrated at all --- are
all made from the calibration set and from strictly past frames, so nothing the
detector does at frame ``t`` depends on frame ``t`` itself.  That is what keeps
the wealth process a supermartingale end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sentinel_e.conformal import (
    ConformalCalibrator,
    MondrianConformalCalibrator,
    QuantileTaxonomy,
    ResidualConformalCalibrator,
    WeightedConformalCalibrator,
)
from sentinel_e.ebh import ebh, global_e_merge
from sentinel_e.edetector import ChangepointPrior, EDetector
from sentinel_e.gnn import FeatureTracker
from sentinel_e.graph import CameraGraph
from sentinel_e.streams import FleetStream, Stream
from sentinel_e.temporal import estimate_decorrelation_lag, thin_indices

__all__ = ["SentinelE", "FleetSentinelE", "FleetResult", "CameraResult"]


@dataclass
class CameraResult:
    """Per-camera output, reported on the original frame grid."""

    p_values: np.ndarray        # (T,) 1.0 on frames where no bet was placed
    active: np.ndarray          # (T,) bool: a bet was placed and it was informative
    log_wealth: np.ndarray      # (T,) held constant between bets
    alarms: np.ndarray          # (T,) bool
    bet_index: np.ndarray       # indices of frames on which a bet was placed
    lag: int                    # decorrelation lag actually used
    abstention_rate: float


class SentinelE:
    """Conformal calibration plus a betting e-detector for one camera.

    Parameters
    ----------
    alpha : float
        Time-uniform false-alarm level; the wealth threshold is ``1/alpha``.
    cc_mode : {'beta', 'dkw', 'none'}
        Calibration-conditional correction: exact order-statistic levels, the
        additive DKW inflation, or none.
    delta : float
        Calibration-conditional failure probability.  The
        end-to-end guarantee is ``alpha + delta``, so ``delta`` an order of
        magnitude below ``alpha`` costs little and buys honesty about the finite
        calibration set.
    calibration : {'residual', 'mondrian', 'pooled', 'weighted'}
        ``residual`` (default) conformalises the context-normalised residual and
        scales to several context variables at once; ``mondrian`` bins the
        context and is exactly group-conditional but limited to one or two
        variables; ``pooled`` and ``weighted`` are the ablations.
    weighted : bool
        Enable likelihood-ratio shift weighting inside bins.
    lag : int or None
        Bets are placed once every ``lag`` frames.  ``None`` estimates it from
        the calibration autocorrelation, which is the recommended setting.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        delta: float = 1e-3,
        calibration: str = "residual",
        weighted: bool = False,
        family: str = "power",
        n_grid: int = 32,
        rho: float = 1e-3,
        prior_kind: str = "geometric",
        restart: bool = False,
        lag: Optional[int] = None,
        thin_calibration: bool = True,
        warmup: bool = True,
        acf_target: float = 0.05,
        max_lag: int = 200,
        shift_block: int = 400,
        n_context_bins: int = 8,
        min_bin: int = 100,
        cc_mode: str = "beta",
        continuous_cols: Sequence[int] = (0,),
        categorical_cols: Sequence[int] = (1,),
    ) -> None:
        if calibration not in {"residual", "mondrian", "pooled", "weighted"}:
            raise ValueError(
                "calibration must be 'residual', 'mondrian', 'pooled' or 'weighted'"
            )
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.calibration = calibration
        self.weighted = bool(weighted)
        self.lag = lag
        self.thin_calibration = bool(thin_calibration)
        self.warmup = bool(warmup)
        self.acf_target = float(acf_target)
        self.max_lag = int(max_lag)
        self.shift_block = int(shift_block)
        self.n_context_bins = int(n_context_bins)
        self.min_bin = int(min_bin)
        self.cc_mode = str(cc_mode)
        self.continuous_cols = tuple(continuous_cols)
        self.categorical_cols = tuple(categorical_cols)

        self.detector = EDetector(
            alpha=alpha,
            family=family,
            n_grid=n_grid,
            prior=ChangepointPrior(kind=prior_kind, rho=rho),
            restart=restart,
        )
        self.calibrator = None
        self._lag_used = 1
        self._n_calibration_effective = 0

    # -- fitting ---------------------------------------------------------- #
    def fit(
        self,
        calibration_scores: Sequence[float],
        calibration_context: Optional[np.ndarray] = None,
    ) -> "SentinelE":
        s_raw = np.asarray(calibration_scores, dtype=float).ravel()
        ctx_raw = (
            None
            if calibration_context is None
            else np.atleast_2d(np.asarray(calibration_context, dtype=float))
        )
        self._lag_used = (
            int(self.lag)
            if self.lag is not None
            else estimate_decorrelation_lag(
                s_raw, self.acf_target, self.max_lag, context=ctx_raw
            )
        )
        # The calibration frames are consecutive video too, so they are no more
        # independent than the deployment stream.  Both the DKW and the exact
        # order-statistic bounds assume i.i.d. calibration draws, so the
        # calibration set is thinned by the same lag.  The honest calibration
        # size is therefore n_raw / lag, and it is that number the guarantee is
        # stated in terms of -- not the raw frame count.
        keep = thin_indices(s_raw.size, self._lag_used) if self.thin_calibration else (
            np.arange(s_raw.size)
        )
        s = s_raw[keep]
        calibration_context = None if ctx_raw is None else ctx_raw[keep]
        self._n_calibration_effective = int(s.size)

        if self.calibration == "residual":
            if calibration_context is None:
                raise ValueError("residual calibration requires context features")
            self.calibrator = ResidualConformalCalibrator(
                delta=self.delta, mode=self.cc_mode
            ).fit(s, calibration_context)
        elif self.calibration == "mondrian":
            if calibration_context is None:
                raise ValueError("mondrian calibration requires context features")
            tax = QuantileTaxonomy(
                continuous_cols=self.continuous_cols,
                categorical_cols=self.categorical_cols,
                n_bins=self.n_context_bins,
            )
            self.calibrator = MondrianConformalCalibrator(
                taxonomy=tax,
                delta=self.delta,
                mode=self.cc_mode,
                min_bin=self.min_bin,
                weighted=self.weighted,
            ).fit(s, calibration_context)
        elif self.calibration == "weighted":
            self.calibrator = WeightedConformalCalibrator(delta=self.delta).fit(
                s, calibration_context
            )
        else:
            self.calibrator = ConformalCalibrator(
                delta=self.delta, mode=self.cc_mode
            ).fit(s)
        return self

    @property
    def decorrelation_lag(self) -> int:
        return self._lag_used

    @property
    def n_calibration_effective(self) -> int:
        """Number of near-independent calibration frames actually used."""
        return self._n_calibration_effective

    # -- p-values --------------------------------------------------------- #
    def _p_values_block(
        self, scores: np.ndarray, context: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.calibration in {"residual", "mondrian"}:
            return self.calibrator.p_values(scores, context)
        if self.calibration == "weighted":
            p = self.calibrator.p_values(scores, context)
        else:
            p = self.calibrator.p_values(scores)
        return p, np.ones(scores.size, dtype=bool)

    def p_values(
        self, scores: Sequence[float], context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Conformal p-values, refreshing shift weights block by block.

        Weights applied to block ``b`` are estimated from block ``b-1``'s
        context, so they never depend on the frames they score.
        """
        if self.calibrator is None:
            raise RuntimeError("SentinelE.fit must be called first")
        s = np.asarray(scores, dtype=float).ravel()
        ctx = None if context is None else np.atleast_2d(np.asarray(context, float))

        supports_shift = self.weighted and ctx is not None and hasattr(
            self.calibrator, "update_shift"
        )
        if not supports_shift:
            return self._p_values_block(s, ctx)

        p = np.empty(s.size)
        active = np.ones(s.size, dtype=bool)
        for start in range(0, s.size, self.shift_block):
            stop = min(start + self.shift_block, s.size)
            if start > 0:
                self.calibrator.update_shift(ctx[max(0, start - self.shift_block):start])
            p[start:stop], active[start:stop] = self._p_values_block(
                s[start:stop], ctx[start:stop]
            )
        if self.warmup:
            # The first block is scored before any deployment context has been
            # seen, so its weights are necessarily uniform and its p-values are
            # uncorrected.  Betting on them would spend the whole false-alarm
            # budget before the shift estimate exists, so the detector abstains
            # through the warm-up and uses it only to measure the deployment
            # context distribution.
            active[: min(self.shift_block, s.size)] = False
        return p, active

    # -- streaming -------------------------------------------------------- #
    def run(
        self, scores: Sequence[float], context: Optional[np.ndarray] = None
    ) -> CameraResult:
        """Run the full pipeline; results are expanded to the original frame grid."""
        s = np.asarray(scores, dtype=float).ravel()
        ctx = None if context is None else np.atleast_2d(np.asarray(context, float))
        T = s.size
        idx = thin_indices(T, self._lag_used)

        p_thin, active_thin = self.p_values(
            s[idx], None if ctx is None else ctx[idx]
        )
        self.detector.reset()
        lw_thin, al_thin = self.detector.run(p_thin, active=active_thin)

        # Expand onto the full frame grid: the wealth is a step function that
        # only changes on betting frames.
        p_full = np.ones(T)
        act_full = np.zeros(T, dtype=bool)
        lw_full = np.zeros(T)
        al_full = np.zeros(T, dtype=bool)
        p_full[idx] = p_thin
        act_full[idx] = active_thin
        al_full[idx] = al_thin
        pos = np.searchsorted(idx, np.arange(T), side="right") - 1
        lw_full[pos >= 0] = lw_thin[pos[pos >= 0]]

        rate = getattr(self.calibrator, "abstention_rate", 0.0)
        return CameraResult(
            p_values=p_full,
            active=act_full,
            log_wealth=lw_full,
            alarms=al_full,
            bet_index=idx,
            lag=self._lag_used,
            abstention_rate=float(rate),
        )

    def run_stream(self, stream: Stream) -> CameraResult:
        """Fit on the stream's own calibration set, then run."""
        self.fit(stream.calibration_scores, stream.calibration_context)
        return self.run(stream.scores, stream.context)


# --------------------------------------------------------------------------- #
# Fleet
# --------------------------------------------------------------------------- #
@dataclass
class FleetResult:
    """Output of a fleet run (all arrays on the thinned betting grid)."""

    log_wealth: np.ndarray        # (K, B)
    alarms: np.ndarray            # (K, B) bool
    p_values: np.ndarray          # (K, B)
    hazards: np.ndarray           # (K, B)
    stakes: np.ndarray            # (K, B)
    bet_index: np.ndarray         # (B,) frame index of each betting step
    e_values: np.ndarray          # (K,) wealth at the per-camera stopping time
    ebh_rejected: np.ndarray      # (K,) bool
    global_e: float               # fleet-level merged e-value

    def alarms_on_frame_grid(self, T: int) -> np.ndarray:
        """Expand the alarm matrix back onto the original ``T``-frame grid."""
        out = np.zeros((self.alarms.shape[0], T), dtype=bool)
        out[:, self.bet_index] = self.alarms
        return out


class FleetSentinelE:
    """Fleet-level SENTINEL-E: graph-steered e-detectors plus e-BH.

    Parameters
    ----------
    controller : callable or None
        Maps ``(K, F)`` predictable node features to ``(hazard, stake)``.  Pass
        a trained :class:`~sentinel_e.gnn.SpatialPriorGNN` via :meth:`from_gnn`,
        a :class:`~sentinel_e.gnn.HeuristicSpatialPrior`, or ``None`` for
        independent per-camera detectors.
    restart : bool
        ``False`` at fleet level: e-BH reads each camera's wealth at the
        stopping time ``min(first alarm, horizon)``, and restarting would throw
        away exactly the evidence that decision needs.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        delta: float = 1e-3,
        calibration: str = "residual",
        weighted: bool = False,
        family: str = "power",
        n_grid: int = 16,
        rho: float = 1e-3,
        restart: bool = False,
        controller: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        fdr_level: float = 0.1,
        lag: Optional[int] = None,
        **camera_kw,
    ) -> None:
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.calibration = calibration
        self.weighted = bool(weighted)
        self.family = family
        self.n_grid = int(n_grid)
        self.rho = float(rho)
        self.restart = bool(restart)
        self.controller = controller
        self.fdr_level = float(fdr_level)
        self.lag = lag
        self.camera_kw = camera_kw

    @classmethod
    def from_gnn(cls, model, graph: CameraGraph, **kw) -> "FleetSentinelE":
        """Build a fleet detector driven by a trained :class:`SpatialPriorGNN`."""
        a_hat = graph.normalized_adjacency()

        def controller(features: np.ndarray):
            return model.predict(features, a_hat)

        return cls(controller=controller, **kw)

    # -- run --------------------------------------------------------------- #
    def _camera(self) -> SentinelE:
        return SentinelE(
            alpha=self.alpha,
            delta=self.delta,
            calibration=self.calibration,
            weighted=self.weighted,
            family=self.family,
            n_grid=self.n_grid,
            rho=self.rho,
            restart=self.restart,
            lag=self.lag,
            **self.camera_kw,
        )

    def compute_p_values(self, fleet: FleetStream) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Conformal p-values for the fleet on a common betting grid."""
        cams = [
            self._camera().fit(s.calibration_scores, s.calibration_context)
            for s in fleet.streams
        ]
        # A single betting grid keeps the cameras synchronous, which the graph
        # layer needs; the lag is the largest required by any camera.
        lag = max(c.decorrelation_lag for c in cams)
        idx = thin_indices(fleet.T, lag)
        p_list, a_list = [], []
        for c, s in zip(cams, fleet.streams):
            c._lag_used = lag
            p, a = c.p_values(s.scores[idx], s.context[idx])
            p_list.append(p)
            a_list.append(a)
        return np.stack(p_list), np.stack(a_list), idx

    def run(self, fleet: FleetStream, inspect_at: Optional[int] = None) -> FleetResult:
        K = fleet.n_cameras
        p, active, idx = self.compute_p_values(fleet)
        B = idx.size

        detectors = [
            EDetector(
                alpha=self.alpha,
                family=self.family,
                n_grid=self.n_grid,
                prior=ChangepointPrior(kind="geometric", rho=self.rho),
                restart=self.restart,
            )
            for _ in range(K)
        ]
        static = (
            fleet.graph.context
            if fleet.graph.context is not None
            else np.zeros((K, 2))
        )
        tracker = FeatureTracker(
            n_cameras=K,
            static_context=static,
            degree=fleet.graph.degree,
            log_threshold=float(np.log(1.0 / self.alpha)),
        )

        log_wealth = np.zeros((K, B))
        alarms = np.zeros((K, B), dtype=bool)
        hazards = np.zeros((K, B))
        stakes = np.ones((K, B))

        for t in range(B):
            if self.controller is None:
                rho_t, stake_t = np.full(K, self.rho), np.ones(K)
            else:
                rho_t, stake_t = self.controller(tracker.features())
                rho_t = np.asarray(rho_t, dtype=float).ravel()
                stake_t = np.asarray(stake_t, dtype=float).ravel()

            lw = np.empty(K)
            for i in range(K):
                res = detectors[i].step(
                    p[i, t],
                    hazard=float(rho_t[i]),
                    stake_scale=float(stake_t[i]),
                    abstain=not bool(active[i, t]),
                )
                lw[i] = res.log_wealth
                alarms[i, t] = res.alarm
            log_wealth[:, t] = lw
            hazards[:, t] = rho_t
            stakes[:, t] = stake_t
            tracker.update(p[:, t], lw)

        # e-values are read at tau_i = min(first alarm, horizon): optional
        # stopping keeps E[W_tau] <= 1, which is what e-BH requires.  A running
        # maximum would *not* be a valid e-value.
        last = B - 1 if inspect_at is None else int(np.clip(inspect_at, 0, B - 1))
        stop_idx = np.full(K, last, dtype=int)
        for i in range(K):
            hit = np.flatnonzero(alarms[i, : last + 1])
            if hit.size:
                stop_idx[i] = int(hit[0])
        e_values = np.exp(
            np.clip(log_wealth[np.arange(K), stop_idx], -50.0, np.log(1e12))
        )
        return FleetResult(
            log_wealth=log_wealth,
            alarms=alarms,
            p_values=p,
            hazards=hazards,
            stakes=stakes,
            bet_index=idx,
            e_values=e_values,
            ebh_rejected=ebh(e_values, self.fdr_level),
            global_e=global_e_merge(e_values, "average"),
        )

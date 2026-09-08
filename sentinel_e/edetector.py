"""Layer 2 --- the betting e-detector with time-uniform false-alarm control.

Problem
-------
A camera streams p-values :math:`p_1, p_2, \\ldots` produced by
:mod:`sentinel_e.conformal`.  Under the no-dumping null they are (conditionally)
i.i.d. super-uniform.  A dumping event may begin at an unknown frame
:math:`\\nu`, after which the p-values become stochastically small.  We want a
stopping rule that alarms quickly after :math:`\\nu` yet satisfies

.. math:: \\mathbb P_{\\infty}\\bigl(\\exists\\, t \\ge 1 : \\text{alarm at } t\\bigr) \\le \\alpha,

where :math:`\\mathbb P_\\infty` is the law under "no dumping ever".  Note the
quantifier: the bound holds *simultaneously over all frames*, so monitoring for
longer never erodes it.  This is what a fixed threshold cannot deliver --- a
per-frame false-positive rate of :math:`\\alpha` accumulates to a near-certain
alarm over a month of video.

Construction
------------
Put a prior :math:`w_1, w_2, \\ldots` (summing to one) on the changepoint
location and let :math:`f` be a betting function.  Define the mixture wealth

.. math::
    W_t \\;=\\; \\underbrace{\\sum_{j \\le t} w_j \\prod_{s=j}^{t} f(p_s)}_{R_t}
        \\;+\\; \\underbrace{\\sum_{j > t} w_j}_{Q_t},

i.e. hypotheses "the change started at ``j``" that have not yet been reached
contribute their prior mass unlevered.  Then :math:`W_0 = 1` and

.. math::
    \\mathbb E[W_t \\mid \\mathcal F_{t-1}]
      = (R_{t-1} + w_t)\\,\\mathbb E[f(p_t)] + Q_t
      \\le R_{t-1} + w_t + Q_t = W_{t-1},

so :math:`(W_t)` is a non-negative supermartingale started at one and Ville's
inequality gives :math:`\\mathbb P_\\infty(\\exists t : W_t \\ge 1/\\alpha) \\le \\alpha`.
Keeping the unreached prior mass :math:`Q_t` inside the statistic is what turns
the familiar Shiryaev--Roberts sum into an exact e-process; dropping it (as the
classical statistic does) breaks the inequality.

The whole thing collapses to the ``O(1)`` recursion

.. math:: R_t = (R_{t-1} + w_t)\\, f(p_t), \\qquad Q_t = Q_{t-1} - w_t,

evaluated in log-space for numerical stability.  A hazard-rate parameterisation
:math:`w_t = \\rho_t Q_{t-1}`, :math:`Q_t = (1-\\rho_t) Q_{t-1}` keeps the
telescoping identity :math:`Q_{t-1} = Q_t + w_t` exact even when the hazard
:math:`\\rho_t` varies over time --- provided it is *predictable*.  That is the
hook through which the spatial graph prior of :mod:`sentinel_e.gnn` acts
without costing any validity (Theorem 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sentinel_e.betting import AdaptiveLinearBet, MixtureBet

__all__ = ["ChangepointPrior", "EDetectorState", "StepResult", "EDetector"]

_NEG_INF = -np.inf
_EPS = 1e-300


def _logaddexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


# --------------------------------------------------------------------------- #
# Changepoint prior
# --------------------------------------------------------------------------- #
@dataclass
class ChangepointPrior:
    """Prior over the (unknown) frame at which dumping begins.

    Parameters
    ----------
    kind : {'geometric', 'scale_free', 'point'}
        ``geometric``
            :math:`w_t = \\rho (1-\\rho)^{t-1}`.  Constant hazard ``rho``;
            the natural choice when events arrive at a roughly known rate.
        ``scale_free``
            :math:`w_t = 1 / (t (t+1))`, so :math:`Q_t = 1/(t+1)`.  Heavy tailed:
            no timescale is privileged, at the price of a slowly decaying hazard.
        ``point``
            All mass on ``t = 1``; the statistic degenerates to the plain
            product martingale, appropriate when the stream is known to start
            at the moment of interest.
    rho : float
        Hazard rate for ``kind='geometric'``.
    """

    kind: str = "geometric"
    rho: float = 1e-3

    def __post_init__(self) -> None:
        if self.kind not in {"geometric", "scale_free", "point"}:
            raise ValueError(f"unknown changepoint prior {self.kind!r}")
        if self.kind == "geometric" and not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0, 1)")

    def hazard(self, t: int) -> float:
        """Hazard :math:`\\rho_t = w_t / Q_{t-1}` at frame ``t`` (1-indexed)."""
        if self.kind == "geometric":
            return self.rho
        if self.kind == "scale_free":
            # w_t = 1/(t(t+1)), Q_{t-1} = 1/t  =>  rho_t = 1/(t+1).
            return 1.0 / (t + 1.0)
        return 1.0 if t == 1 else 0.0


# --------------------------------------------------------------------------- #
# Detector state and per-step result
# --------------------------------------------------------------------------- #
@dataclass
class EDetectorState:
    """Mutable state of a single-camera e-detector (log-space)."""

    log_R: np.ndarray            # per-grid-point log of the reached-mass wealth
    log_Q: float                 # log of the unreached changepoint prior mass
    t: int = 0                   # frames seen since the last restart
    n_restarts: int = 0

    def copy(self) -> "EDetectorState":
        return EDetectorState(self.log_R.copy(), self.log_Q, self.t, self.n_restarts)


@dataclass
class StepResult:
    """Outcome of processing one frame."""

    log_wealth: float
    alarm: bool
    hazard: float
    stake_scale: float


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class EDetector:
    """Anytime-valid betting e-detector for one camera.

    Parameters
    ----------
    alpha : float
        Target time-uniform false-alarm probability.  Alarms are raised the
        first time the wealth reaches ``1/alpha``.
    family : {'power', 'linear', 'adaptive'}
        Betting family.  ``power`` and ``linear`` are prior mixtures over a grid
        (parameter-free, recommended); ``adaptive`` learns a single stake online
        with ONS.  Only ``linear`` and ``adaptive`` accept an external stake
        modulation from the graph layer.
    n_grid : int
        Size of the betting-parameter grid for the mixture families.
    prior : ChangepointPrior
        Prior over the change location.
    restart : bool
        If ``True`` the wealth is reset to one after each alarm, so the detector
        keeps monitoring.  Each *run* then enjoys the ``alpha`` guarantee, which
        yields an average run length to false alarm of at least ``1/alpha``
        frames.  If ``False`` the detector stops at the first alarm and the
        guarantee is the stronger time-uniform probability bound.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        family: str = "power",
        n_grid: int = 32,
        prior: Optional[ChangepointPrior] = None,
        restart: bool = True,
        adaptive_lam0: float = 0.2,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.log_threshold = float(np.log(1.0 / alpha))
        self.family = family.lower()
        if self.family not in {"power", "linear", "adaptive"}:
            raise ValueError("family must be 'power', 'linear' or 'adaptive'")
        self.prior = prior if prior is not None else ChangepointPrior()
        self.restart = bool(restart)

        if self.family == "adaptive":
            self.mixture: Optional[MixtureBet] = None
            self._ons = AdaptiveLinearBet(lam0=adaptive_lam0)
            self.k = 1
            self.log_prior = np.zeros(1)
        else:
            self.mixture = MixtureBet(family=self.family, n_grid=n_grid)
            self._ons = None
            self.k = len(self.mixture)
            self.log_prior = np.log(self.mixture.prior)

        self.state = self._fresh_state()
        self.alarm_times: List[int] = []
        self._global_t = 0

    # -- lifecycle -------------------------------------------------------- #
    def _fresh_state(self) -> EDetectorState:
        return EDetectorState(
            log_R=np.full(self.k, _NEG_INF), log_Q=0.0, t=0, n_restarts=0
        )

    def reset(self, keep_history: bool = False) -> None:
        """Full reset, including the global frame counter and alarm history."""
        n_restarts = self.state.n_restarts if keep_history else 0
        self.state = self._fresh_state()
        self.state.n_restarts = n_restarts
        if self._ons is not None:
            self._ons.reset()
        if not keep_history:
            self.alarm_times = []
            self._global_t = 0

    def _restart_run(self) -> None:
        n = self.state.n_restarts + 1
        self.state = self._fresh_state()
        self.state.n_restarts = n
        if self._ons is not None:
            self._ons.reset()

    # -- betting factors -------------------------------------------------- #
    @staticmethod
    def _modulated_kappa(grid: np.ndarray, stake_scale: float) -> np.ndarray:
        """Interpolate the power exponents towards the neutral bet.

        ``kappa = 1`` gives ``f(p) = 1``, the neutral bet, so

        .. math:: \\kappa_{\\mathrm{eff}} = 1 - s\\,(1 - \\kappa)

        sweeps continuously from the grid's own aggressiveness at ``s = 1`` to
        no bet at all at ``s = 0``.  Every intermediate value is a legitimate
        betting function, so a *predictable* ``s`` supplied by the graph layer
        preserves the supermartingale property exactly (Theorem 2).  This is the
        power-family analogue of scaling a linear stake, and it matters because
        linear stakes are bounded by ``1 + lambda <= 2`` and therefore extract
        far less evidence from a very small p-value than a power bet does.
        """
        return np.clip(1.0 - stake_scale * (1.0 - grid), 1e-4, 1.0)

    def _log_factors(self, p: float, stake_scale: float) -> np.ndarray:
        """Log betting factors for the current grid at p-value ``p``."""
        p = float(min(max(p, 1e-12), 1.0))
        if self.family == "power":
            kappa = self._modulated_kappa(self.mixture.grid, stake_scale)
            return np.log(kappa) + (kappa - 1.0) * np.log(p)
        if self.family == "linear":
            lam = np.clip(self.mixture.grid * stake_scale, 0.0, 1.0)
            return np.log(np.maximum(1.0 + lam * (1.0 - 2.0 * p), _EPS))
        lam = float(np.clip(self._ons.lam * stake_scale, 0.0, 1.0))
        return np.array([np.log(max(1.0 + lam * (1.0 - 2.0 * p), _EPS))])

    # -- core recursion --------------------------------------------------- #
    def step(
        self,
        p: float,
        hazard: Optional[float] = None,
        stake_scale: float = 1.0,
        abstain: bool = False,
    ) -> StepResult:
        """Process one frame.

        Parameters
        ----------
        p : float
            Conformal p-value of the current frame.
        hazard : float, optional
            Predictable override of the changepoint hazard ``rho_t`` for this
            frame, supplied by the spatial prior.  Must be measurable with
            respect to information available *before* ``p`` was observed;
            otherwise the supermartingale property is lost.
        stake_scale : float
            Predictable multiplicative modulation in ``[0, 1]`` of the betting
            stake (``linear``/``adaptive`` families only).
        abstain : bool
            Place the neutral bet ``f = 1`` instead of betting on ``p``.  Used
            when the conformal layer reports that the frame falls outside the
            calibrated context region.  A constant factor of one is trivially an
            e-value, so abstaining can never break the guarantee --- it only
            forgoes evidence.
        """
        st = self.state
        st.t += 1
        self._global_t += 1

        rho = self.prior.hazard(st.t) if hazard is None else float(hazard)
        rho = float(np.clip(rho, 0.0, 1.0))
        stake_scale = float(np.clip(stake_scale, 0.0, 1.0))

        # w_t = rho * Q_{t-1};  Q_t = (1 - rho) * Q_{t-1}.
        log_w = st.log_Q + np.log(rho) if rho > 0 else _NEG_INF
        log_Q_new = st.log_Q + np.log1p(-rho) if rho < 1.0 else _NEG_INF

        log_f = (
            np.zeros(self.k) if abstain else self._log_factors(p, stake_scale)
        )
        # R_t = (R_{t-1} + w_t) * f(p_t), computed stably in log-space.
        log_R_new = np.logaddexp(st.log_R, log_w) + log_f

        st.log_R = log_R_new
        st.log_Q = log_Q_new

        if self._ons is not None and not abstain:
            self._ons.update(p)  # strictly after the bet: keeps predictability

        log_wealth = self.log_wealth()
        alarm = bool(log_wealth >= self.log_threshold)
        if alarm:
            self.alarm_times.append(self._global_t)
            if self.restart:
                self._restart_run()
        return StepResult(log_wealth, alarm, rho, stake_scale)

    # -- readouts --------------------------------------------------------- #
    def log_wealth(self) -> float:
        """Current :math:`\\log W_t`.  ``exp`` of this is a valid e-value."""
        st = self.state
        mixed = float(np.logaddexp.reduce(self.log_prior + st.log_R))
        return _logaddexp(mixed, st.log_Q)

    def e_value(self, cap: float = 1e12) -> float:
        """Current wealth as an e-value, capped for numerical sanity.

        By optional stopping, :math:`\\mathbb E_\\infty[W_\\tau] \\le 1` at every
        stopping time :math:`\\tau`, which is exactly the input e-BH requires.
        """
        return float(min(np.exp(min(self.log_wealth(), np.log(cap))), cap))

    # -- vectorised batch driver ------------------------------------------- #
    #: Frames per vectorised block.  Bounded so that the running log-product
    #: cannot drift far enough for the final subtraction to lose precision.
    BLOCK = 4096

    def _hazard_vector(self, t0: int, n: int) -> np.ndarray:
        """Hazards for frames ``t0+1 .. t0+n`` (1-indexed within the run)."""
        t = np.arange(t0 + 1, t0 + n + 1, dtype=float)
        kind = self.prior.kind
        if kind == "geometric":
            return np.full(n, self.prior.rho)
        if kind == "scale_free":
            return 1.0 / (t + 1.0)
        return (t == 1.0).astype(float)

    def _log_factor_matrix(self, p: np.ndarray, active: Optional[np.ndarray]) -> np.ndarray:
        """``(n, K)`` log betting factors, with zero rows wherever we abstain."""
        pc = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)[:, None]
        if self.family == "power":
            kappa = self.mixture.grid[None, :]   # unmodulated: the batch path is
                                                 # only used when no controller
                                                 # supplies a stake scale
            log_f = np.log(kappa) + (kappa - 1.0) * np.log(pc)
        else:
            lam = np.clip(self.mixture.grid, 0.0, 1.0)[None, :]
            log_f = np.log(np.maximum(1.0 + lam * (1.0 - 2.0 * pc), _EPS))
        if active is not None:
            log_f = np.where(np.asarray(active, dtype=bool)[:, None], log_f, 0.0)
        return log_f

    def _run_block(
        self, p: np.ndarray, active: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised wealth trajectory over one block, without touching state.

        Uses the closed form of the recursion.  Writing
        :math:`P_t = \\prod_{s \\le t} f(p_s)`, the identity
        :math:`R_t = P_t\bigl(R_0 + \\sum_{j \\le t} w_j / P_{j-1}\bigr)`
        turns the sequential update into one cumulative log-sum-exp, which numpy
        evaluates with ``logaddexp.accumulate`` in a single pass.
        """
        st = self.state
        n = p.size
        rho = np.clip(self._hazard_vector(st.t, n), 0.0, 1.0)

        # rho == 0 and rho == 1 are both legitimate (the point prior uses both)
        # and give -inf logs, which the log-space recursion handles correctly.
        with np.errstate(divide="ignore"):
            log_rho = np.log(rho)
            log_one_minus = np.log1p(-rho)
        cum = np.cumsum(log_one_minus)
        log_Q = st.log_Q + cum                       # Q_t
        log_Q_prev = st.log_Q + np.concatenate([[0.0], cum[:-1]])
        log_w = log_Q_prev + log_rho                 # w_t = rho_t Q_{t-1}

        log_f = self._log_factor_matrix(p, active)
        log_P = np.cumsum(log_f, axis=0)
        log_P_prev = np.vstack([np.zeros((1, self.k)), log_P[:-1]])

        terms = np.vstack([st.log_R[None, :], log_w[:, None] - log_P_prev])
        acc = np.logaddexp.accumulate(terms, axis=0)[1:]
        log_R = log_P + acc

        log_W = np.logaddexp(
            np.logaddexp.reduce(log_R + self.log_prior[None, :], axis=1), log_Q
        )
        return log_W, log_R, log_Q

    def run(
        self,
        p_values: Sequence[float],
        hazards: Optional[Sequence[float]] = None,
        stake_scales: Optional[Sequence[float]] = None,
        active: Optional[Sequence[bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a whole stream.

        Returns
        -------
        log_wealth : ndarray of shape (T,)
        alarms : boolean ndarray of shape (T,)
        """
        p_values = np.asarray(p_values, dtype=float).ravel()
        T = p_values.size
        act = None if active is None else np.asarray(active, dtype=bool)
        if hazards is None and stake_scales is None and self.family != "adaptive":
            return self._run_vectorized(p_values, act)

        lw = np.empty(T)
        al = np.zeros(T, dtype=bool)
        for i in range(T):
            h = None if hazards is None else float(hazards[i])
            s = 1.0 if stake_scales is None else float(stake_scales[i])
            ab = False if active is None else (not bool(active[i]))
            res = self.step(p_values[i], hazard=h, stake_scale=s, abstain=ab)
            lw[i] = res.log_wealth
            al[i] = res.alarm
            if res.alarm and not self.restart:
                lw[i + 1:] = res.log_wealth
                break
        return lw, al

    def _run_vectorized(
        self, p: np.ndarray, active: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Block-vectorised equivalent of the scalar :meth:`step` loop."""
        T = p.size
        lw = np.empty(T)
        al = np.zeros(T, dtype=bool)
        i = 0
        while i < T:
            j = min(i + self.BLOCK, T)
            chunk_active = None if active is None else active[i:j]
            log_W, log_R, log_Q = self._run_block(p[i:j], chunk_active)
            lw[i:j] = log_W

            hit = np.flatnonzero(log_W >= self.log_threshold)
            if hit.size == 0:
                st = self.state
                st.log_R, st.log_Q = log_R[-1].copy(), float(log_Q[-1])
                st.t += j - i
                self._global_t += j - i
                i = j
                continue

            k = int(hit[0])
            al[i + k] = True
            self._global_t += k + 1
            self.alarm_times.append(self._global_t)
            if not self.restart:
                lw[i + k + 1:] = log_W[k]
                st = self.state
                st.log_R, st.log_Q = log_R[k].copy(), float(log_Q[k])
                st.t += k + 1
                return lw, al
            self._restart_run()
            i = i + k + 1
        return lw, al

    def first_alarm(self, p_values: Sequence[float], **kw) -> Optional[int]:
        """Index (0-based) of the first alarm, or ``None`` if the stream is quiet."""
        _, alarms = self.run(p_values, **kw)
        idx = np.flatnonzero(alarms)
        return int(idx[0]) if idx.size else None

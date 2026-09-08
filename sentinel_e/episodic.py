"""The episodic e-process: the algorithmic core of SENTINEL-E.

Why the standard construction is the wrong shape
------------------------------------------------
Every sequential change detector in use --- CUSUM, Shiryaev--Roberts, the
e-detectors of Shin et al., E-SHIFT --- is built for a *persistent* change: at an
unknown frame :math:`\\nu` the law switches from :math:`P_0` to :math:`P_1` and
stays there forever.  Their optimality theory, their thresholds and their
mixtures over :math:`\\nu` all presume it.

An illegal dumping event is not that.  It is an *episode*: a van pulls up, the
offender unloads for forty seconds and drives away.  Inside the episode the
signal is *intermittent* --- the offender is occluded by the vehicle, steps out
of frame, comes back --- so only a fraction of the frames are anomalous.  Then
the episode ends and the stream returns to its null behaviour, and weeks later
another one begins.

Modelling that as a persistent change is not a small mismatch.  A
changepoint-mixture wealth process keeps betting at full aggression after the
episode is over, so it hands back the wealth the episode earned; and because it
assumes every anomalous-looking frame is evidence of the same permanent change,
it is punished by the quiet frames *inside* the episode rather than treating
them as expected.  Both effects lengthen detection delay, and the second gets
worse the more intermittent the event is --- which is precisely the regime this
problem lives in.

The construction
----------------
We replace the mixture over changepoints with a mixture over *episodes*.  Let
:math:`z_t \\in \\{\\text{quiet}, \\text{active}\\}` be a two-state Markov chain
with :math:`\\mathbb P(\\text{quiet} \\to \\text{active}) = \\rho` and
:math:`\\mathbb P(\\text{active} \\to \\text{quiet}) = \\eta`, and let the wealth
mix over every trajectory of that chain:

.. math::
   W_t \;=\; \\sum_{z_{1:t}} \\mathbb P(z_{1:t})
              \\prod_{s \\le t} f\\bigl(p_s\\bigr)^{\\mathbf 1\\{z_s = \\text{active}\\}} .

The sum has :math:`2^t` terms and collapses to two numbers.  Writing
:math:`A_t` and :math:`Q_t` for the parts of the sum ending in the active and
quiet states,

.. math::
   A_t = \\bigl[(1-\\eta) A_{t-1} + \\rho\\, Q_{t-1}\\bigr] f(p_t),
   \\qquad
   Q_t = \\eta\\, A_{t-1} + (1-\\rho)\\, Q_{t-1},

with :math:`A_0 = 0`, :math:`Q_0 = 1`, and :math:`W_t = A_t + Q_t`.  This is the
HMM forward recursion, and it is :math:`O(1)` per frame in both time and memory.

Two properties make it the right object rather than merely a plausible one.

*It is an exact e-process.*  Under the null,

.. math::
   \\mathbb E[W_t \\mid \\mathcal F_{t-1}]
     = \\bigl[(1-\\eta)A_{t-1} + \\rho Q_{t-1}\\bigr]\\mathbb E[f(p_t)]
       + \\eta A_{t-1} + (1-\\rho) Q_{t-1}
     \\le A_{t-1} + Q_{t-1} = W_{t-1},

so :math:`(W_t)` is a non-negative supermartingale started at one and Ville's
inequality controls it *simultaneously at every frame*.  No new machinery is
needed for the guarantee: the episodic structure is free.

*It contains the classical construction as a boundary case.*  Setting
:math:`\\eta = 0` forbids the episode from ever ending and recovers the
changepoint mixture; the classical Shiryaev--Roberts statistic is that in turn
with the unreached prior mass discarded.  The episodic process is therefore a
strict generalisation along an axis --- episode duration --- that the change
detection literature has left fixed at infinity.

On cost: a mixture over candidate starts evaluated term by term is ``O(t)`` per
observation, and under a geometric prior it telescopes to ``O(1)``, which is
exactly the ``eta = 0`` case of the recursion above.  The claim here is not that
this is cheaper than that special case but that it *matches* it while spanning
episodes of finite duration --- the extra generality costs nothing
arithmetically, which is what keeps month-long monitoring viable on a camera.

Intermittency, by dilation
--------------------------
Within an episode only a fraction :math:`\\pi` of frames actually show the act.
Betting the full :math:`f` on all of them is wrong in both directions: too
aggressive on the occluded frames, and the losses there cancel the gains.  We
therefore *dilate* the emission,

.. math:: f_\\pi(p) = (1-\\pi) + \\pi\\, f(p),

which is a convex combination of the neutral bet and :math:`f`.  Convexity is
what keeps it legal: :math:`\\int_0^1 f_\\pi \\le 1` and :math:`f_\\pi` is
non-increasing, so it is a betting function and the supermartingale property is
untouched.  It says exactly the right thing --- *this frame belongs to an episode,
but it may or may not show the act* --- and :math:`\\pi = 1` recovers the
undilated bet.

Everything above is a prior specification, so we do not have to know
:math:`(\\rho, \\eta, \\pi, \\kappa)`.  The detector runs one forward recursion per
grid point and mixes them, which keeps it parameter-free and still :math:`O(1)`
per frame with a small constant.

A quantity worth passing between cameras
----------------------------------------
The recursion yields, for free, the posterior probability that the camera is
inside an episode right now,

.. math:: \\Pi_t = A_t / (A_t + Q_t) \\in [0, 1],

which is far more informative than raw wealth: it is bounded, interpretable, and
comparable across cameras with different histories.  :mod:`sentinel_e.gnn` uses
it as the message the camera graph exchanges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["EpisodePrior", "EpisodicState", "EpisodicResult", "EpisodicEDetector"]

_TINY = 1e-300


# --------------------------------------------------------------------------- #
# Prior grid
# --------------------------------------------------------------------------- #
@dataclass
class EpisodePrior:
    """Prior grid over episode dynamics and betting aggressiveness.

    Parameters
    ----------
    kappa : sequence of float
        Power-betting exponents in ``(0, 1]``; smaller bets harder on small
        p-values.
    eta : sequence of float
        Per-frame probability that an episode ends, i.e. the reciprocal of its
        expected length *in betting steps*.  Include a very small value so the
        grid spans persistent changes as well as short bursts.
    pi : sequence of float
        Fraction of frames inside an episode that actually show the act.
    rho : float
        Per-frame probability that an episode begins.  Matched to the monitoring
        horizon, or supplied per frame by the graph controller.
    """

    kappa: Sequence[float] = (0.02, 0.05, 0.12, 0.28, 0.55, 0.85)
    eta: Sequence[float] = (1e-4, 3e-3, 1e-2, 5e-2)
    pi: Sequence[float] = (0.25, 0.55, 1.0)
    rho: float = 1e-3

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flattened ``(kappa, eta, pi, log_prior)`` arrays over the product grid."""
        k, e, p = np.meshgrid(
            np.asarray(self.kappa, dtype=float),
            np.asarray(self.eta, dtype=float),
            np.asarray(self.pi, dtype=float),
            indexing="ij",
        )
        k, e, p = k.ravel(), e.ravel(), p.ravel()
        if np.any((k <= 0) | (k > 1)):
            raise ValueError("kappa must lie in (0, 1]")
        if np.any((e < 0) | (e >= 1)):
            raise ValueError("eta must lie in [0, 1)")
        if np.any((p <= 0) | (p > 1)):
            raise ValueError("pi must lie in (0, 1]")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0, 1)")
        log_prior = np.full(k.size, -np.log(k.size))
        return k, e, p, log_prior

    def __len__(self) -> int:
        return len(self.kappa) * len(self.eta) * len(self.pi)


@dataclass
class EpisodicState:
    """Forward state, held in scaled form with a running log normaliser.

    ``a`` and ``q`` are renormalised to sum to one at every step and the scale is
    accumulated in ``log_scale``.  This is the standard HMM scaling trick and it
    is both faster and better conditioned than carrying the recursion in log
    space, because the two-state simplex never underflows.
    """

    a: np.ndarray
    q: np.ndarray
    log_scale: np.ndarray
    t: int = 0
    n_restarts: int = 0

    def copy(self) -> "EpisodicState":
        return EpisodicState(self.a.copy(), self.q.copy(), self.log_scale.copy(),
                             self.t, self.n_restarts)


@dataclass
class EpisodicResult:
    """Outcome of processing one frame."""

    log_wealth: float
    alarm: bool
    episode_posterior: float
    hazard: float
    stake_scale: float


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class EpisodicEDetector:
    """Anytime-valid episodic e-detector for one camera.

    Parameters
    ----------
    alpha : float
        Target time-uniform false-alarm probability; the wealth threshold is
        ``1/alpha``.
    prior : EpisodePrior
        Grid over episode dynamics and betting aggressiveness.
    restart : bool
        Reset the wealth after an alarm and keep monitoring.  Note that the
        episodic process does *not* need a restart to survive a finished episode
        --- that is what ``eta`` is for --- so ``restart`` only governs behaviour
        after an alarm is actually raised.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        prior: Optional[EpisodePrior] = None,
        restart: bool = False,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.log_threshold = float(np.log(1.0 / alpha))
        self.prior = prior if prior is not None else EpisodePrior()
        self.restart = bool(restart)

        self.kappa, self.eta, self.pi, self.log_prior = self.prior.grid()
        self.k = int(self.kappa.size)
        self.state = self._fresh_state()
        self.alarm_times: List[int] = []
        self._global_t = 0

    # -- lifecycle -------------------------------------------------------- #
    def _fresh_state(self) -> EpisodicState:
        return EpisodicState(
            a=np.zeros(self.k), q=np.ones(self.k), log_scale=np.zeros(self.k)
        )

    def reset(self, keep_history: bool = False) -> None:
        n = self.state.n_restarts if keep_history else 0
        self.state = self._fresh_state()
        self.state.n_restarts = n
        if not keep_history:
            self.alarm_times = []
            self._global_t = 0

    def _restart_run(self) -> None:
        n = self.state.n_restarts + 1
        self.state = self._fresh_state()
        self.state.n_restarts = n

    # -- betting ----------------------------------------------------------- #
    def _emission(self, p: float, stake_scale: float) -> np.ndarray:
        """Dilated power bet ``(1-pi) + pi * kappa_eff * p^(kappa_eff-1)``.

        ``stake_scale`` interpolates the exponents towards the neutral bet
        ``kappa = 1``, so a predictable scale supplied by the graph controller
        keeps every grid point a legal betting function.
        """
        p = float(min(max(p, 1e-12), 1.0))
        kappa = np.clip(1.0 - stake_scale * (1.0 - self.kappa), 1e-4, 1.0)
        f = kappa * p ** (kappa - 1.0)
        return (1.0 - self.pi) + self.pi * f

    # -- core recursion ---------------------------------------------------- #
    def step(
        self,
        p: float,
        hazard: Optional[float] = None,
        stake_scale: float = 1.0,
        abstain: bool = False,
    ) -> EpisodicResult:
        """Process one betting instant.

        Parameters
        ----------
        p : float
            Conformal p-value.
        hazard : float, optional
            Predictable override of the episode-onset probability ``rho`` for
            this frame, supplied by the graph controller.  It must be measurable
            with respect to information available strictly before ``p``.
        stake_scale : float
            Predictable modulation in ``[0, 1]`` of betting aggressiveness.
        abstain : bool
            Place the neutral bet ``f = 1``; used when the conformal layer
            reports the frame outside the calibrated context region.
        """
        st = self.state
        st.t += 1
        self._global_t += 1

        rho = self.prior.rho if hazard is None else float(hazard)
        rho = float(np.clip(rho, 0.0, 1.0))
        stake_scale = float(np.clip(stake_scale, 0.0, 1.0))
        f = np.ones(self.k) if abstain else self._emission(p, stake_scale)

        # Forward step of the two-state chain, then renormalise.
        a_new = ((1.0 - self.eta) * st.a + rho * st.q) * f
        q_new = self.eta * st.a + (1.0 - rho) * st.q
        c = a_new + q_new
        c = np.maximum(c, _TINY)
        st.a, st.q = a_new / c, q_new / c
        st.log_scale = st.log_scale + np.log(c)

        log_wealth = self.log_wealth()
        posterior = self.episode_posterior()
        alarm = bool(log_wealth >= self.log_threshold)
        if alarm:
            self.alarm_times.append(self._global_t)
            if self.restart:
                self._restart_run()
        return EpisodicResult(log_wealth, alarm, posterior, rho, stake_scale)

    # -- readouts ---------------------------------------------------------- #
    def log_wealth(self) -> float:
        """:math:`\\log W_t`; ``exp`` of this is a valid e-value at any stopping time."""
        return _logsumexp(self.log_prior + self.state.log_scale)

    def episode_posterior(self) -> float:
        """Prior-weighted posterior probability of being inside an episode now.

        Bounded in ``[0, 1]`` and comparable across cameras, which is what makes
        it the right message for the graph layer to pass.
        """
        w = self.log_prior + self.state.log_scale
        m = float(np.max(w))
        if not np.isfinite(m):
            return 0.0
        weights = np.exp(w - m)
        return float(np.sum(weights * self.state.a) / max(np.sum(weights), _TINY))

    def e_value(self, cap: float = 1e12) -> float:
        return float(min(np.exp(min(self.log_wealth(), np.log(cap))), cap))

    # -- batch driver ------------------------------------------------------ #
    def run(
        self,
        p_values: Sequence[float],
        hazards: Optional[Sequence[float]] = None,
        stake_scales: Optional[Sequence[float]] = None,
        active: Optional[Sequence[bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a stream, returning ``(log_wealth, alarms)``."""
        p = np.asarray(p_values, dtype=float).ravel()
        if hazards is None and stake_scales is None:
            return self._run_fast(p, None if active is None else np.asarray(active, bool))
        T = p.size
        lw = np.empty(T)
        al = np.zeros(T, dtype=bool)
        for i in range(T):
            res = self.step(
                p[i],
                hazard=None if hazards is None else float(hazards[i]),
                stake_scale=1.0 if stake_scales is None else float(stake_scales[i]),
                abstain=False if active is None else (not bool(active[i])),
            )
            lw[i], al[i] = res.log_wealth, res.alarm
            if res.alarm and not self.restart:
                lw[i + 1:] = res.log_wealth
                break
        return lw, al

    def _run_fast(
        self, p: np.ndarray, active: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tight loop for the common case of a constant hazard and full stake.

        The two-state recursion is genuinely sequential --- unlike the
        changepoint mixture it has no cumulative-sum closed form --- so this
        precomputes the emission matrix and keeps the per-step work to a handful
        of vector operations on the grid.
        """
        T = p.size
        pc = np.clip(p, 1e-12, 1.0)
        # Emission matrix (T, K): dilated power bets on the prior grid.
        logf = np.log(self.kappa)[None, :] + (self.kappa - 1.0)[None, :] * np.log(pc)[:, None]
        F = (1.0 - self.pi)[None, :] + self.pi[None, :] * np.exp(logf)
        if active is not None:
            F = np.where(active[:, None], F, 1.0)

        rho = self.prior.rho
        one_minus_eta, one_minus_rho = 1.0 - self.eta, 1.0 - rho
        a, q, logc = self.state.a.copy(), self.state.q.copy(), self.state.log_scale.copy()
        lw = np.empty(T)
        al = np.zeros(T, dtype=bool)
        lp = self.log_prior

        i = 0
        while i < T:
            a_new = (one_minus_eta * a + rho * q) * F[i]
            q_new = self.eta * a + one_minus_rho * q
            c = np.maximum(a_new + q_new, _TINY)
            a, q = a_new / c, q_new / c
            logc = logc + np.log(c)
            w = lp + logc
            m = w.max()
            lw[i] = m + np.log(np.sum(np.exp(w - m)))
            self._global_t += 1
            if lw[i] >= self.log_threshold:
                al[i] = True
                self.alarm_times.append(self._global_t)
                if not self.restart:
                    lw[i + 1:] = lw[i]
                    self.state.a, self.state.q, self.state.log_scale = a, q, logc
                    self.state.t += i + 1
                    return lw, al
                self._restart_run()
                a, q, logc = self.state.a.copy(), self.state.q.copy(), self.state.log_scale.copy()
            i += 1
        self.state.a, self.state.q, self.state.log_scale = a, q, logc
        self.state.t += T
        return lw, al

    def posteriors(self, p_values: Sequence[float]) -> np.ndarray:
        """Episode posterior at each betting instant (diagnostic; resets state)."""
        self.reset()
        out = np.empty(len(p_values))
        for i, x in enumerate(p_values):
            out[i] = self.step(float(x)).episode_posterior
        return out

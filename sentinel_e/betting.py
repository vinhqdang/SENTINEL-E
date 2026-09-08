"""Betting functions that turn a p-value into an e-value.

A *betting function* is any measurable ``f : (0, 1] -> [0, inf)`` with
:math:`\\int_0^1 f(p)\\,dp \\le 1`.  If ``f`` is additionally non-increasing then
for any super-uniform p-value (:math:`P(p \\le u) \\le u`)

.. math:: \\mathbb E_{H_0}[f(p)] \\le \\int_0^1 f(u)\\,du \\le 1,

so ``f(p)`` is an *e-value*: the gambler who reinvests wealth at odds ``f(p)``
never expects to profit under the null.  Products of such factors are exactly
the non-negative supermartingales that Ville's inequality controls.

The monotonicity requirement is what makes super-uniformity (rather than exact
uniformity) sufficient, which matters here because the DKW-inflated conformal
p-values of :mod:`sentinel_e.conformal` are strictly conservative by
construction.

Implemented families
--------------------
``PowerBet``
    :math:`f_\\kappa(p) = \\kappa p^{\\kappa - 1}`, :math:`\\kappa \\in (0, 1)`.
    The classical power martingale; ``kappa`` small bets hard on tiny p-values.
``LinearBet``
    :math:`f_\\lambda(p) = 1 + \\lambda (1 - 2p)`, :math:`\\lambda \\in [0, 1]`.
    Bounded, numerically forgiving, and differentiable in ``lambda``, which is
    what lets the graph neural network of :mod:`sentinel_e.gnn` steer it.
``MixtureBet``
    A finite prior mixture over a grid of either family.  Mixing is done at the
    level of *wealth*, not of the betting factor, so the mixture is itself a
    supermartingale and stays parameter-free.
``AdaptiveLinearBet``
    Online-Newton-step learning of ``lambda`` from past p-values only, hence
    predictable and validity-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "BettingFunction",
    "PowerBet",
    "LinearBet",
    "MixtureBet",
    "AdaptiveLinearBet",
]

_EPS = 1e-12


class BettingFunction:
    """Interface for a betting function."""

    #: Whether the factor depends on internal state that must be advanced.
    stateful: bool = False

    def factor(self, p: float) -> float:
        """Return the betting factor ``f(p)`` (an e-value under the null)."""
        raise NotImplementedError

    def update(self, p: float) -> None:
        """Advance any internal state *after* the bet on ``p`` was settled."""

    def reset(self) -> None:
        """Return to the initial state (used when a detector restarts)."""


@dataclass
class PowerBet(BettingFunction):
    """Power betting function ``kappa * p ** (kappa - 1)`` with ``0 < kappa < 1``."""

    kappa: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.kappa < 1.0:
            raise ValueError(f"kappa must lie in (0, 1), got {self.kappa}")

    def factor(self, p: float) -> float:
        p = min(max(float(p), _EPS), 1.0)
        return float(self.kappa * p ** (self.kappa - 1.0))

    def factors(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0)
        return self.kappa * p ** (self.kappa - 1.0)


@dataclass
class LinearBet(BettingFunction):
    """Linear betting function ``1 + lambda * (1 - 2p)`` with ``0 <= lambda <= 1``.

    ``lambda`` is the fraction of wealth staked.  It may be re-set between
    frames by an external, *predictable* controller (the GNN spatial prior)
    without disturbing the supermartingale property, because
    :math:`\\mathbb E[f_\\lambda(p)] \\le 1` holds for every fixed
    :math:`\\lambda \\in [0, 1]`.
    """

    lam: float = 0.5

    def __post_init__(self) -> None:
        self.set_lambda(self.lam)

    def set_lambda(self, lam: float) -> None:
        self.lam = float(np.clip(lam, 0.0, 1.0))

    def factor(self, p: float) -> float:
        p = min(max(float(p), 0.0), 1.0)
        return float(max(1.0 + self.lam * (1.0 - 2.0 * p), 0.0))

    def factors(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        return np.maximum(1.0 + self.lam * (1.0 - 2.0 * p), 0.0)


class MixtureBet(BettingFunction):
    """Prior mixture over a grid of betting parameters.

    ``MixtureBet`` is *not* a pointwise betting function: mixing must happen at
    the level of accumulated wealth for the result to remain a supermartingale.
    :class:`~sentinel_e.edetector.EDetector` therefore runs one wealth recursion
    per grid point and averages them with :attr:`prior`.  This class only holds
    the grid and the prior; per-frame cost is ``O(K)`` with ``K = len(grid)``
    (32 by default), i.e. a few dozen floating-point operations per frame.
    """

    def __init__(
        self,
        family: str = "power",
        grid: Sequence[float] | None = None,
        prior: Sequence[float] | None = None,
        n_grid: int = 32,
    ) -> None:
        family = family.lower()
        if family not in {"power", "linear"}:
            raise ValueError("family must be 'power' or 'linear'")
        self.family = family
        if grid is None:
            if family == "power":
                # Log-spaced towards aggressive bets: small kappa detects strong,
                # short bursts; kappa near 1 accumulates weak persistent evidence.
                grid = np.exp(np.linspace(np.log(0.02), np.log(0.95), n_grid))
            else:
                grid = np.linspace(0.02, 0.95, n_grid)
        self.grid = np.asarray(grid, dtype=float)
        if prior is None:
            prior = np.ones_like(self.grid)
        prior = np.asarray(prior, dtype=float)
        if prior.shape != self.grid.shape:
            raise ValueError("prior and grid must have the same shape")
        if np.any(prior < 0):
            raise ValueError("prior must be non-negative")
        self.prior = prior / prior.sum()

    def factors(self, p: float) -> np.ndarray:
        """Vector of betting factors, one per grid point."""
        p = min(max(float(p), _EPS), 1.0)
        if self.family == "power":
            return self.grid * p ** (self.grid - 1.0)
        return np.maximum(1.0 + self.grid * (1.0 - 2.0 * p), 0.0)

    def factor(self, p: float) -> float:
        """Prior-averaged factor (used only for diagnostics, not for wealth)."""
        return float(self.prior @ self.factors(p))

    def __len__(self) -> int:
        return int(self.grid.size)


class AdaptiveLinearBet(BettingFunction):
    """Online-Newton-step (ONS) learning of the linear betting fraction.

    The stake used at frame ``t`` is a function of ``p_1..p_{t-1}`` only, so it
    is predictable with respect to the observation filtration and the wealth
    process remains a supermartingale (Lemma 1 in the paper).  ONS on the
    log-wealth objective is the standard universal-portfolio-style choice; it
    approaches the growth rate of the best fixed stake in hindsight.
    """

    stateful = True

    def __init__(self, lam0: float = 0.2, lam_max: float = 0.9, a: float = 1.0) -> None:
        self.lam0 = float(np.clip(lam0, 0.0, lam_max))
        self.lam_max = float(lam_max)
        self.a = float(a)
        self.reset()

    def reset(self) -> None:
        self.lam = self.lam0
        self._sum_sq_grad = 1.0
        self._t = 0

    def factor(self, p: float) -> float:
        p = min(max(float(p), 0.0), 1.0)
        return float(max(1.0 + self.lam * (1.0 - 2.0 * p), 0.0))

    def update(self, p: float) -> None:
        p = min(max(float(p), 0.0), 1.0)
        x = 1.0 - 2.0 * p                      # payoff direction
        denom = max(1.0 + self.lam * x, 1e-6)
        grad = x / denom                       # d log(1 + lam x) / d lam
        self._sum_sq_grad += grad * grad
        step = self.a * grad / self._sum_sq_grad
        self.lam = float(np.clip(self.lam + step, 0.0, self.lam_max))
        self._t += 1

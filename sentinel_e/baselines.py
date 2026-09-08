"""Baseline stream detectors.

All detectors expose the same interface --- ``fit(calibration_scores)`` then
``run(scores) -> boolean alarm array`` --- and each exposes a single scalar knob
that is swept to trace an operating curve, so that every method is compared at
its *realised* false-alarm rate rather than at its nominal one.

======================  ==========================================  ===============
Detector                Knob                                        Guarantee
======================  ==========================================  ===============
``FixedThreshold``       score threshold                            none over time
``PValueThreshold``      per-frame p-value level                    per frame only
``CUSUM``                decision interval ``h``                    asymptotic, parametric
``ShiryaevRoberts``      threshold ``A``                            ARL0 >= A, parametric
``ParametricEDetector``  ``alpha``                                  anytime-valid *if* the
                                                                    Gaussian model holds
======================  ==========================================  ===============

The last row is the honest statement of what a generic e-detector buys without
the conformal layer: the wealth process is a supermartingale only under the
assumed pre-change law, so a mis-specified tail or an un-modelled illumination
shift destroys the guarantee.  Supplying that missing piece is Layer 1 of
SENTINEL-E.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from sentinel_e.edetector import ChangepointPrior

__all__ = [
    "StreamDetector",
    "FixedThreshold",
    "PValueThreshold",
    "CUSUM",
    "ShiryaevRoberts",
    "ParametricEDetector",
]


class StreamDetector:
    """Common interface for baseline stream detectors."""

    #: Name of the scalar attribute swept to trace the operating curve.
    knob: str = ""

    def fit(self, calibration_scores: Sequence[float]) -> "StreamDetector":
        return self

    def run(self, scores: Sequence[float]) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
@dataclass
class FixedThreshold(StreamDetector):
    """Threshold the raw detector score, with an optional persistence filter.

    This is what deployed systems do.  Its false-alarm probability over a
    horizon of ``T`` frames is :math:`1 - (1-q)^{T}` for a per-frame exceedance
    probability ``q``, which reaches one for any fixed ``q > 0``: the very
    failure mode SENTINEL-E is built to remove.  ``k_of_m`` persistence
    suppresses isolated spikes but only rescales ``q``; it does not change the
    fact that the bound degrades with ``T``.
    """

    threshold: float = 0.5
    k_of_m: tuple = (1, 1)
    knob: str = "threshold"

    def run(self, scores: Sequence[float]) -> np.ndarray:
        s = np.asarray(scores, dtype=float).ravel()
        hit = s >= self.threshold
        k, m = self.k_of_m
        if m <= 1:
            return hit
        c = np.convolve(hit.astype(int), np.ones(m, dtype=int), mode="full")[: s.size]
        return c >= k

    def calibrate_threshold(self, calibration_scores, per_frame_rate: float) -> float:
        """Set the threshold to a target *per-frame* exceedance rate."""
        s = np.asarray(calibration_scores, dtype=float)
        self.threshold = float(np.quantile(s, 1.0 - per_frame_rate))
        return self.threshold


# --------------------------------------------------------------------------- #
@dataclass
class PValueThreshold(StreamDetector):
    """Alarm the first time a conformal p-value falls below ``level``.

    Included to isolate the optional-stopping effect: the p-values are perfectly
    valid *per frame*, and the procedure still fails catastrophically over a
    long stream, because validity at each fixed time says nothing about the
    minimum over a growing window.
    """

    level: float = 0.01
    knob: str = "level"

    def run_pvalues(self, p_values: Sequence[float]) -> np.ndarray:
        p = np.asarray(p_values, dtype=float).ravel()
        return p <= self.level

    def run(self, scores: Sequence[float]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("PValueThreshold consumes p-values, not scores")


# --------------------------------------------------------------------------- #
@dataclass
class CUSUM(StreamDetector):
    """Page's CUSUM on calibration-standardised scores.

    Reference statistic :math:`z_t = (s_t - \\hat\\mu_0)/\\hat\\sigma_0` with the
    Gaussian log-likelihood ratio for a shift of ``delta`` standard deviations:

    .. math:: S_t = \\max(0,\\; S_{t-1} + \\delta z_t - \\delta^2/2),

    alarming when :math:`S_t \\ge h`.  Optimal in Lorden's minimax sense under a
    correctly specified i.i.d. Gaussian pre-change law --- an assumption that
    autocorrelated, heavy-tailed, diurnally drifting detector scores violate.
    """

    delta: float = 1.0
    h: float = 5.0
    restart: bool = True
    knob: str = "h"
    _mu: float = field(default=0.0, repr=False)
    _sd: float = field(default=1.0, repr=False)

    def fit(self, calibration_scores: Sequence[float]) -> "CUSUM":
        s = np.asarray(calibration_scores, dtype=float)
        self._mu = float(s.mean())
        self._sd = float(s.std()) or 1.0
        return self

    def run(self, scores: Sequence[float]) -> np.ndarray:
        z = (np.asarray(scores, dtype=float).ravel() - self._mu) / self._sd
        incr = self.delta * z - 0.5 * self.delta**2
        alarms = np.zeros(z.size, dtype=bool)
        S = 0.0
        for t in range(z.size):
            S = max(0.0, S + incr[t])
            if S >= self.h:
                alarms[t] = True
                if self.restart:
                    S = 0.0
                else:
                    break
        return alarms


# --------------------------------------------------------------------------- #
@dataclass
class ShiryaevRoberts(StreamDetector):
    """Shiryaev--Roberts statistic :math:`R_t = (1 + R_{t-1})\\,\\Lambda_t`.

    Asymptotically optimal for the average detection delay at a given ARL0 under
    a correctly specified model, and the classical parametric ancestor of the
    e-process used here.  The difference is instructive: SR drops the prior mass
    that has not yet been reached, so ``R_t`` is *not* a supermartingale and only
    an ARL0 bound is available; restoring that mass (see
    :class:`~sentinel_e.edetector.EDetector`) upgrades the guarantee to a
    time-uniform probability bound.
    """

    delta: float = 1.0
    A: float = 100.0
    restart: bool = True
    knob: str = "A"
    _mu: float = field(default=0.0, repr=False)
    _sd: float = field(default=1.0, repr=False)

    def fit(self, calibration_scores: Sequence[float]) -> "ShiryaevRoberts":
        s = np.asarray(calibration_scores, dtype=float)
        self._mu = float(s.mean())
        self._sd = float(s.std()) or 1.0
        return self

    def run(self, scores: Sequence[float]) -> np.ndarray:
        z = (np.asarray(scores, dtype=float).ravel() - self._mu) / self._sd
        log_lr = self.delta * z - 0.5 * self.delta**2
        alarms = np.zeros(z.size, dtype=bool)
        log_R = -np.inf
        log_A = np.log(self.A)
        for t in range(z.size):
            log_R = np.logaddexp(0.0, log_R) + log_lr[t]
            if log_R >= log_A:
                alarms[t] = True
                if self.restart:
                    log_R = -np.inf
                else:
                    break
        return alarms


# --------------------------------------------------------------------------- #
@dataclass
class ParametricEDetector(StreamDetector):
    """Generic e-detector on raw scores with a parametric Gaussian bet.

    Same changepoint-mixture e-process as SENTINEL-E, but the betting factor is
    the Gaussian likelihood ratio estimated from the calibration set instead of
    a conformal p-value bet.  It therefore inherits anytime validity *only* when
    the pre-change law really is the fitted Gaussian; the experiments show that
    under diurnal drift and heavy tails its realised false-alarm rate exceeds
    the nominal ``alpha`` by more than an order of magnitude.  This is the
    closest generic prior art to SENTINEL-E and the fairest way to isolate the
    contribution of the conformal layer.
    """

    alpha: float = 0.01
    delta: float = 1.0
    rho: float = 1e-3
    restart: bool = True
    knob: str = "alpha"
    _mu: float = field(default=0.0, repr=False)
    _sd: float = field(default=1.0, repr=False)

    def fit(self, calibration_scores: Sequence[float]) -> "ParametricEDetector":
        s = np.asarray(calibration_scores, dtype=float)
        self._mu = float(s.mean())
        self._sd = float(s.std()) or 1.0
        return self

    def run(self, scores: Sequence[float]) -> np.ndarray:
        z = (np.asarray(scores, dtype=float).ravel() - self._mu) / self._sd
        log_lr = self.delta * z - 0.5 * self.delta**2
        prior = ChangepointPrior(kind="geometric", rho=self.rho)
        log_thr = np.log(1.0 / self.alpha)

        alarms = np.zeros(z.size, dtype=bool)
        log_R, log_Q = -np.inf, 0.0
        for t in range(z.size):
            rho = prior.hazard(t + 1)
            log_w = log_Q + np.log(rho)
            log_Q = log_Q + np.log1p(-rho)
            log_R = np.logaddexp(log_R, log_w) + log_lr[t]
            if np.logaddexp(log_R, log_Q) >= log_thr:
                alarms[t] = True
                if self.restart:
                    log_R, log_Q = -np.inf, 0.0
                else:
                    break
        return alarms

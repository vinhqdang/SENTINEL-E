"""Layer 1 --- conformal calibration of raw detector scores.

A frozen per-frame dumping detector emits a real-valued score ``s_t`` for every
frame of a camera stream.  Absolute score values are not comparable across
cameras, weather regimes or times of day, so they cannot be thresholded with any
statistical meaning.  This module converts them into *p-values* that are
(super-)uniform under the no-dumping null hypothesis, which is exactly the input
required by the betting e-detector of :mod:`sentinel_e.edetector`.

Three ingredients
-----------------

1. **Split conformal p-values.**  Given a per-camera calibration set of
   ``n`` confirmed no-dumping frames with scores ``s_1..s_n``, the p-value of a
   deployment frame with score ``s`` is

   .. math:: p = \\frac{1 + |\\{i : s_i \\ge s\\}|}{n + 1}.

   Under exchangeability of the calibration frames and the test frame this is
   super-uniform, :math:`P(p \\le u) \\le u`.

2. **Likelihood-ratio-weighted conformal.**  Deployment conditions drift away
   from calibration (day to night, clear to rain, camera swap).  Under a
   covariate-shift model with density ratio ``w(x)`` on a low-dimensional
   context vector ``x`` (illumination, motion energy, weather flag, ...) the
   weighted p-value

   .. math::
       p = \\frac{\\sum_i w(x_i)\\,\\mathbf 1\\{s_i \\ge s\\} + w(x)}
                 {\\sum_i w(x_i) + w(x)}

   restores super-uniformity.  ``w`` is estimated by a logistic
   density-ratio classifier that separates calibration contexts from an
   *earlier, disjoint* window of deployment contexts.

3. **Calibration-conditional inflation.**  This is the ingredient that makes the
   downstream anytime-valid claim honest.  A fixed calibration set is reused for
   every frame, so the deployment p-values are *not* independent marginally ---
   they are i.i.d. only *conditionally* on the calibration set, and their
   conditional distribution is the empirical one, which is only approximately
   uniform.  Betting on marginally-super-uniform-but-dependent p-values does not
   yield a supermartingale.  We therefore inflate every p-value by the
   Dvoretzky--Kiefer--Wolfowitz (DKW) deviation

   .. math:: \\varepsilon_n(\\delta) = \\sqrt{\\log(2/\\delta) / (2 n_{\\mathrm{eff}})},

   with :math:`n_{\\mathrm{eff}} = n` in the unweighted case and
   :math:`n_{\\mathrm{eff}} = 1/\\lVert \\tilde w \\rVert_2^2` in the weighted
   case.  On an event of probability at least :math:`1-\\delta` over the draw of
   the calibration set, the inflated p-values are *conditionally* i.i.d. and
   super-uniform, which is precisely what the wealth process needs.
   See :func:`dkw_inflation` and Theorem 1 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "beta_calibration_levels",
    "ResidualConformalCalibrator",
    "roc_auc",
    "MondrianConformalCalibrator",
    "QuantileTaxonomy",
    "effective_sample_size",
    "dkw_inflation",
    "weighted_dkw_inflation",
    "ConformalCalibrator",
    "WeightedConformalCalibrator",
    "LogisticDensityRatio",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Calibration-conditional inflation
# --------------------------------------------------------------------------- #
def dkw_inflation(n: int, delta: float) -> float:
    """DKW uniform deviation of an empirical CDF built from ``n`` i.i.d. draws.

    Returns ``eps`` such that, with probability at least ``1 - delta`` over the
    calibration draw, ``sup_u |F_n(u) - F(u)| <= eps``.

    Parameters
    ----------
    n : int
        Calibration set size (must be positive).
    delta : float
        Calibration-conditional failure probability in ``(0, 1)``.  ``delta = 0``
        disables the correction and returns ``0.0`` (marginal-validity mode).
    """
    if delta <= 0.0:
        return 0.0
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in [0, 1), got {delta}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))


def beta_calibration_levels(
    n: int, delta: float, spending: str = "inverse_square"
) -> np.ndarray:
    """Exact calibration-conditional levels for split-conformal p-values.

    The DKW inflation of :func:`dkw_inflation` is additive and therefore
    crippling in the tail: with ``n = 2000`` and ``delta = 1e-3`` it pushes the
    smallest attainable p-value from ``1/2001`` to ``0.051``, capping the
    evidence a single frame can contribute at well under one nat.  The tail is
    exactly where a betting detector earns its wealth, so this is worth doing
    properly.

    Conditional on the calibration set, the exact conditional level of the
    discrete p-value :math:`(1+j)/(n+1)` is

    .. math:: 1 - F\bigl(s_{(n-j)}\bigr) \\sim \\mathrm{Beta}(j+1,\\, n-j),

    because :math:`F(s_{(k)}) \\sim \\mathrm{Beta}(k, n+1-k)` for the order
    statistics of an i.i.d. sample.  Replacing each attainable level by its
    upper :math:`(1-\\delta_j)` Beta quantile therefore gives *exact*
    calibration-conditional super-uniformity rather than a concentration bound,
    and it is an order of magnitude tighter in the tail: for the same ``n`` and
    ``delta`` the smallest level becomes ``0.0037``.

    The budget is split over levels as :math:`\\delta_j \\propto (j+1)^{-2}`
    (``spending='inverse_square'``), which concentrates confidence where the
    betting functions actually look, or uniformly (``spending='uniform'``, plain
    Bonferroni).  Either way the union bound over levels keeps the total failure
    probability at ``delta``.

    Returns
    -------
    ndarray of shape ``(n + 1,)``
        ``levels[j]`` is the calibration-conditional p-value to report when
        exactly ``j`` calibration scores are greater than or equal to the test
        score.  Monotonically increasing, with ``levels[n] = 1``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if delta <= 0.0:
        return (1.0 + np.arange(n + 1)) / (n + 1.0)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in [0, 1)")
    try:
        from scipy.special import betaincinv
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "beta_calibration_levels needs SciPy; pass mode='dkw' to fall back "
            "to the additive DKW inflation"
        ) from exc

    j = np.arange(n)
    if spending == "inverse_square":
        weights = 1.0 / (j + 1.0) ** 2
    elif spending == "uniform":
        weights = np.ones(n)
    else:
        raise ValueError("spending must be 'inverse_square' or 'uniform'")
    delta_j = delta * weights / weights.sum()

    levels = betaincinv(j + 1.0, (n - j).astype(float), 1.0 - delta_j)
    levels = np.concatenate([levels, [1.0]])
    # Numerical monotonicity: quantiles are increasing in j, enforce exactly.
    return np.clip(np.maximum.accumulate(levels), 0.0, 1.0)


def weighted_dkw_inflation(normalized_weights: np.ndarray, delta: float) -> float:
    """Weighted-DKW deviation for a weighted empirical CDF.

    For normalized weights :math:`\\tilde w` (summing to one) the weighted
    empirical CDF is a weighted average of independent bounded indicators, so a
    Hoeffding/DKW-type bound holds with the sample size replaced by the Kish
    effective sample size :math:`n_{\\mathrm{eff}} = 1/\\lVert\\tilde w\\rVert_2^2`.
    Heavily-shifted deployment conditions concentrate the weights on few
    calibration frames, ``n_eff`` collapses, and the inflation grows --- the
    method automatically becomes more conservative exactly when calibration
    transfer is least trustworthy.
    """
    w = np.asarray(normalized_weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("normalized_weights must be a non-empty 1-D array")
    total = w.sum()
    if total <= 0:
        raise ValueError("normalized_weights must have a positive sum")
    w = w / total
    n_eff = 1.0 / float(np.sum(w**2))
    return dkw_inflation(max(n_eff, _EPS), delta) if delta > 0 else 0.0


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (ties averaged); no SciPy or scikit-learn needed."""
    sc = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(sc, kind="mergesort")
    ranks = np.empty(sc.size, dtype=float)
    ranks[order] = np.arange(1, sc.size + 1, dtype=float)
    # Average ranks within ties so the statistic is exact for discrete scores.
    uniq, inv, counts = np.unique(sc, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def effective_sample_size(normalized_weights: np.ndarray) -> float:
    """Kish effective sample size ``1 / sum(w_i^2)`` for normalized weights."""
    w = np.asarray(normalized_weights, dtype=float)
    w = w / max(w.sum(), _EPS)
    return 1.0 / max(float(np.sum(w**2)), _EPS)


# --------------------------------------------------------------------------- #
# Unweighted split conformal
# --------------------------------------------------------------------------- #
@dataclass
class ConformalCalibrator:
    """Split-conformal p-values against a fixed per-camera calibration set.

    Parameters
    ----------
    delta : float, default 0.01
        Calibration-conditional failure probability.  ``delta=0`` disables the
        correction and gives plain marginal validity.
    mode : {'beta', 'dkw', 'none'}, default 'beta'
        How calibration-conditional validity is enforced.  ``beta`` uses the
        exact order-statistic levels of :func:`beta_calibration_levels` and is
        strongly preferred; ``dkw`` uses the additive inflation and is retained
        for the ablation; ``none`` is marginal validity only.
    randomized : bool, default False
        Use smoothed (randomized) conformal p-values, which are exactly uniform
        under continuity rather than merely super-uniform.  Slightly sharper but
        introduces auxiliary randomness into the alarm decision.
    clip_min : float, default 1e-6
        Lower clip on the returned p-value.  Betting functions take
        ``log p``-like quantities, so an exact zero must be avoided.
    """

    delta: float = 0.01
    mode: str = "beta"
    randomized: bool = False
    clip_min: float = 1e-6
    _sorted_scores: Optional[np.ndarray] = field(default=None, repr=False)
    _levels: Optional[np.ndarray] = field(default=None, repr=False)
    _eps: float = field(default=0.0, repr=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )

    # -- fitting ---------------------------------------------------------- #
    def fit(self, calibration_scores: Sequence[float]) -> "ConformalCalibrator":
        """Store the calibration scores of confirmed no-dumping frames."""
        s = np.asarray(calibration_scores, dtype=float).ravel()
        if s.size == 0:
            raise ValueError("calibration set is empty")
        if not np.all(np.isfinite(s)):
            raise ValueError("calibration scores contain non-finite values")
        if self.mode not in {"beta", "dkw", "none"}:
            raise ValueError("mode must be 'beta', 'dkw' or 'none'")
        self._sorted_scores = np.sort(s)
        if self.mode == "beta" and self.delta > 0:
            self._levels = beta_calibration_levels(s.size, self.delta)
            self._eps = float(self._levels[0] - 1.0 / (s.size + 1.0))
        else:
            self._levels = None
            self._eps = (
                dkw_inflation(s.size, self.delta) if self.mode == "dkw" else 0.0
            )
        return self

    @property
    def n(self) -> int:
        self._check_fitted()
        return int(self._sorted_scores.size)

    @property
    def inflation(self) -> float:
        """Gap between the smallest attainable p-value and its raw value ``1/(n+1)``.

        For ``mode='dkw'`` this is exactly the additive inflation; for
        ``mode='beta'`` it is the (much smaller) tail cost of the exact
        order-statistic bound, reported on the same scale for comparability.
        """
        self._check_fitted()
        return self._eps

    def _check_fitted(self) -> None:
        if self._sorted_scores is None:
            raise RuntimeError("ConformalCalibrator.fit must be called first")

    # -- p-values --------------------------------------------------------- #
    def p_value(self, score: float) -> float:
        """Conformal p-value of a single deployment score."""
        return float(self.p_values(np.asarray([score], dtype=float))[0])

    def p_values(self, scores: Sequence[float]) -> np.ndarray:
        """Vectorised conformal p-values.  Higher score => smaller p-value."""
        self._check_fitted()
        s = np.asarray(scores, dtype=float).ravel()
        cal = self._sorted_scores
        n = cal.size
        # |{i : cal_i >= s}| and |{i : cal_i > s}| via sorted-array searches.
        n_ge = n - np.searchsorted(cal, s, side="left")
        if self._levels is not None:
            # Exact calibration-conditional remapping of each attainable level.
            return np.clip(self._levels[n_ge], self.clip_min, 1.0)
        if self.randomized:
            n_gt = n - np.searchsorted(cal, s, side="right")
            n_eq = n_ge - n_gt
            u = self._rng.uniform(size=s.shape)
            p = (n_gt + u * (n_eq + 1.0)) / (n + 1.0)
        else:
            p = (1.0 + n_ge) / (n + 1.0)
        p = np.minimum(1.0, p + self._eps)
        return np.clip(p, self.clip_min, 1.0)


# --------------------------------------------------------------------------- #
# Density-ratio estimation
# --------------------------------------------------------------------------- #
class LogisticDensityRatio:
    """Likelihood ratio ``w(x) = q(x)/p(x)`` by probabilistic classification.

    A regularised logistic regression is trained to separate calibration
    contexts (label 0, density ``p``) from deployment contexts (label 1, density
    ``q``).  With balanced-class correction the ratio is recovered as

    .. math:: w(x) = \\frac{\\hat\\pi(x)}{1 - \\hat\\pi(x)}\\cdot\\frac{n_p}{n_q}.

    Fitted with plain gradient descent so the package needs no scikit-learn.
    The estimate is deliberately fitted on a *disjoint, earlier* deployment
    window so that ``w`` is a predictable function of the data actually being
    monitored (see :meth:`WeightedConformalCalibrator.update_shift`).
    """

    def __init__(
        self,
        l2: float = 1.0,
        lr: float = 0.5,
        n_iter: int = 400,
        clip: float = 20.0,
        random_state: int = 0,
    ) -> None:
        self.l2 = float(l2)
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.clip = float(clip)
        self.random_state = int(random_state)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x_cal: np.ndarray, x_dep: np.ndarray) -> "LogisticDensityRatio":
        x_cal = np.atleast_2d(np.asarray(x_cal, dtype=float))
        x_dep = np.atleast_2d(np.asarray(x_dep, dtype=float))
        if x_cal.shape[1] != x_dep.shape[1]:
            raise ValueError("calibration and deployment contexts differ in width")
        x = np.vstack([x_cal, x_dep])
        y = np.concatenate([np.zeros(len(x_cal)), np.ones(len(x_dep))])

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.scale_

        n, d = z.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.n_iter):
            logits = np.clip(z @ w + b, -30.0, 30.0)
            pr = 1.0 / (1.0 + np.exp(-logits))
            resid = pr - y
            gw = z.T @ resid / n + self.l2 * w / n
            gb = float(resid.mean())
            w -= self.lr * gw
            b -= self.lr * gb
        self.coef_, self.intercept_ = w, b
        self._log_prior_ratio = float(np.log(max(len(x_cal), 1) / max(len(x_dep), 1)))
        return self

    def ratio(self, x: np.ndarray) -> np.ndarray:
        """Likelihood ratio at contexts ``x``, clipped to ``[e^-clip, e^clip]``."""
        if self.coef_ is None:
            raise RuntimeError("LogisticDensityRatio.fit must be called first")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        z = (x - self.mean_) / self.scale_
        logit = z @ self.coef_ + self.intercept_ + self._log_prior_ratio
        return np.exp(np.clip(logit, -self.clip, self.clip))


# --------------------------------------------------------------------------- #
# Weighted split conformal
# --------------------------------------------------------------------------- #
@dataclass
class WeightedConformalCalibrator:
    """Covariate-shift-robust conformal p-values with an effective-size penalty.

    The calibration weights are held fixed between :meth:`update_shift` calls,
    so within a monitoring block ``w`` is a deterministic function and the
    weighted p-values inherit the weighted-conformal validity guarantee.
    """

    delta: float = 0.01
    clip_min: float = 1e-6
    weight_clip: float = 20.0
    ratio_l2: float = 1.0

    _sorted_scores: Optional[np.ndarray] = field(default=None, repr=False)
    _sorted_weights: Optional[np.ndarray] = field(default=None, repr=False)
    _cal_context: Optional[np.ndarray] = field(default=None, repr=False)
    _order: Optional[np.ndarray] = field(default=None, repr=False)
    _dr: Optional[LogisticDensityRatio] = field(default=None, repr=False)
    _eps: float = field(default=0.0, repr=False)
    _n_eff: float = field(default=0.0, repr=False)

    # -- fitting ---------------------------------------------------------- #
    def fit(
        self,
        calibration_scores: Sequence[float],
        calibration_context: Optional[np.ndarray] = None,
    ) -> "WeightedConformalCalibrator":
        s = np.asarray(calibration_scores, dtype=float).ravel()
        if s.size == 0:
            raise ValueError("calibration set is empty")
        self._order = np.argsort(s)
        self._sorted_scores = s[self._order]
        if calibration_context is None:
            self._cal_context = None
        else:
            ctx = np.atleast_2d(np.asarray(calibration_context, dtype=float))
            if len(ctx) != s.size:
                raise ValueError("context rows must match number of scores")
            self._cal_context = ctx[self._order]
        # Start unweighted: uniform weights until the first update_shift call.
        self._sorted_weights = np.ones(s.size)
        self._refresh_inflation(test_weight=1.0)
        return self

    def update_shift(self, deployment_context: np.ndarray) -> float:
        """Re-estimate the density ratio from a window of deployment contexts.

        Returns the resulting effective calibration size ``n_eff``.  Call this
        at block boundaries with contexts observed strictly *before* the frames
        that will be scored with the refreshed weights.
        """
        self._check_fitted()
        if self._cal_context is None:
            raise RuntimeError("calibrator was fitted without context features")
        dep = np.atleast_2d(np.asarray(deployment_context, dtype=float))
        if dep.shape[0] < 2:
            return self._n_eff
        self._dr = LogisticDensityRatio(
            l2=self.ratio_l2, clip=self.weight_clip
        ).fit(self._cal_context, dep)
        w = self._dr.ratio(self._cal_context)
        w = np.clip(w, np.exp(-self.weight_clip), np.exp(self.weight_clip))
        self._sorted_weights = w
        self._refresh_inflation(test_weight=float(np.median(w)))
        return self._n_eff

    def _refresh_inflation(self, test_weight: float) -> None:
        w = np.append(self._sorted_weights, max(test_weight, _EPS))
        w = w / w.sum()
        self._n_eff = effective_sample_size(w)
        self._eps = weighted_dkw_inflation(w, self.delta)

    def _check_fitted(self) -> None:
        if self._sorted_scores is None:
            raise RuntimeError("WeightedConformalCalibrator.fit must be called first")

    @property
    def n(self) -> int:
        self._check_fitted()
        return int(self._sorted_scores.size)

    @property
    def n_eff(self) -> float:
        """Kish effective calibration size under the current weights."""
        self._check_fitted()
        return self._n_eff

    @property
    def inflation(self) -> float:
        self._check_fitted()
        return self._eps

    # -- p-values --------------------------------------------------------- #
    def p_values(
        self, scores: Sequence[float], context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Weighted conformal p-values for deployment ``scores``."""
        self._check_fitted()
        s = np.asarray(scores, dtype=float).ravel()
        cal = self._sorted_scores
        w = self._sorted_weights

        # Weight of the test point itself.
        if context is not None and self._dr is not None:
            ctx = np.atleast_2d(np.asarray(context, dtype=float))
            if len(ctx) != s.size:
                raise ValueError("context rows must match number of scores")
            w_test = np.clip(
                self._dr.ratio(ctx),
                np.exp(-self.weight_clip),
                np.exp(self.weight_clip),
            )
        else:
            w_test = np.ones(s.size)

        # Suffix sums of calibration weights for {cal_i >= s}.
        suffix = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
        idx = np.searchsorted(cal, s, side="left")
        num = suffix[idx] + w_test
        den = suffix[0] + w_test
        p = num / np.maximum(den, _EPS)
        p = np.minimum(1.0, p + self._eps)
        return np.clip(p, self.clip_min, 1.0)

    def p_value(self, score: float, context: Optional[np.ndarray] = None) -> float:
        ctx = None if context is None else np.atleast_2d(context)
        return float(self.p_values(np.asarray([score], dtype=float), ctx)[0])


# --------------------------------------------------------------------------- #
# Group-conditional (Mondrian) conformal calibration
# --------------------------------------------------------------------------- #
class QuantileTaxonomy:
    """Partition context space into bins by calibration quantiles.

    Continuous columns (time of day, illumination) are cut at calibration
    quantiles; categorical columns (weather state, camera mode) contribute their
    integer level directly.  The bin index is the mixed-radix combination.
    """

    def __init__(
        self,
        continuous_cols: Sequence[int] = (0,),
        categorical_cols: Sequence[int] = (),
        n_bins: int = 8,
        n_levels: int = 2,
    ) -> None:
        self.continuous_cols = tuple(int(c) for c in continuous_cols)
        self.categorical_cols = tuple(int(c) for c in categorical_cols)
        self.n_bins = int(n_bins)
        self.n_levels = int(n_levels)
        self._edges: Optional[list] = None

    def fit(self, context: np.ndarray) -> "QuantileTaxonomy":
        ctx = np.atleast_2d(np.asarray(context, dtype=float))
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self._edges = [np.quantile(ctx[:, c], qs) for c in self.continuous_cols]
        return self

    @property
    def n_groups(self) -> int:
        return (self.n_bins ** len(self.continuous_cols)) * (
            self.n_levels ** len(self.categorical_cols)
        )

    def __call__(self, context: np.ndarray) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError("QuantileTaxonomy.fit must be called first")
        ctx = np.atleast_2d(np.asarray(context, dtype=float))
        idx = np.zeros(len(ctx), dtype=int)
        radix = 1
        for j, c in enumerate(self.continuous_cols):
            b = np.searchsorted(self._edges[j], ctx[:, c], side="right")
            idx += b * radix
            radix *= self.n_bins
        for c in self.categorical_cols:
            lvl = np.clip(ctx[:, c].astype(int), 0, self.n_levels - 1)
            idx += lvl * radix
            radix *= self.n_levels
        return idx


@dataclass
class MondrianConformalCalibrator:
    """Group-conditional conformal p-values with optional shift weighting.

    Pooling all calibration frames and comparing a night-time score against a
    calibration set dominated by daylight is *marginally* valid and
    *conditionally* useless: the p-values are far too small during the hours
    when the backbone runs hot, and a sequential test converts that local
    miscalibration into an early false alarm.  Mondrian calibration restores
    exactness by conformalising within a bin of the context taxonomy, which is
    a strictly stronger, group-conditional guarantee:

    .. math:: \\mathbb P\\bigl(p \\le u \\mid \\text{bin}(x) = b\\bigr) \\le u
              \\quad \\text{for every bin } b.

    Layered on top, the likelihood-ratio weights absorb residual within-bin
    shift.  When deployment moves into a region the calibration set never
    covered --- adverse weather that never occurred during the labelling window,
    say --- the weights concentrate, the effective bin size collapses, the DKW
    inflation grows past one and the p-value saturates at one.  The detector
    then places a neutral bet instead of an unjustified one, which is the
    correct behaviour: a regime that was never calibrated cannot be certified.
    :attr:`abstention_rate` reports how often that happens.

    Parameters
    ----------
    taxonomy : QuantileTaxonomy
        Context partition.  Fitted on the calibration context inside :meth:`fit`.
    delta : float
        Calibration-conditional failure probability, spent *per bin* with a
        Bonferroni correction so the guarantee holds simultaneously over bins.
    min_bin : int
        Bins with fewer calibration frames than this are merged into the pooled
        fallback, whose p-values carry the pooled (larger) inflation.
    weighted : bool
        Enable within-bin likelihood-ratio weighting.
    shift_auc_threshold : float
        Held-out AUC above which a bin is declared shifted and the weighted
        (conservative) path is taken.  ``0.60`` keeps the exact levels whenever
        calibration and deployment contexts are essentially indistinguishable.
    weighted_bound : {'effective_beta', 'dkw'}
        Calibration-conditional correction used once a bin is declared shifted.
    """

    taxonomy: "QuantileTaxonomy"
    delta: float = 1e-3
    mode: str = "beta"
    min_bin: int = 100
    weighted: bool = False
    weight_clip: float = 10.0
    shift_auc_threshold: float = 0.60
    weighted_bound: str = "effective_beta"
    clip_min: float = 1e-6
    _per_bin_delta: float = 1e-3

    _bins: dict = field(default_factory=dict, repr=False)
    _pooled: Optional[ConformalCalibrator] = field(default=None, repr=False)
    _cal_context: Optional[np.ndarray] = field(default=None, repr=False)
    _cal_bin: Optional[np.ndarray] = field(default=None, repr=False)
    _cal_scores: Optional[np.ndarray] = field(default=None, repr=False)
    _dr: Optional[LogisticDensityRatio] = field(default=None, repr=False)
    _n_scored: int = field(default=0, repr=False)
    _n_abstained: int = field(default=0, repr=False)

    # -- fitting ---------------------------------------------------------- #
    def fit(
        self, calibration_scores: Sequence[float], calibration_context: np.ndarray
    ) -> "MondrianConformalCalibrator":
        s = np.asarray(calibration_scores, dtype=float).ravel()
        ctx = np.atleast_2d(np.asarray(calibration_context, dtype=float))
        if len(ctx) != s.size:
            raise ValueError("context rows must match number of calibration scores")
        self.taxonomy.fit(ctx)
        bins = self.taxonomy(ctx)
        self._cal_scores, self._cal_context, self._cal_bin = s, ctx, bins

        # Bonferroni over the populated bins keeps the guarantee simultaneous.
        populated = [b for b in np.unique(bins) if int((bins == b).sum()) >= self.min_bin]
        per_bin_delta = self.delta / max(len(populated), 1)

        self._bins = {}
        self._per_bin_delta = per_bin_delta
        for b in populated:
            m = bins == b
            nb = int(m.sum())
            self._bins[int(b)] = {
                "scores": np.sort(s[m]),
                "context": ctx[m],
                "weights": np.ones(nb),
                "eps": 0.0,
                "levels": (
                    beta_calibration_levels(nb, per_bin_delta)
                    if self.mode == "beta" and per_bin_delta > 0
                    else None
                ),
            }
        self._pooled = ConformalCalibrator(delta=self.delta, mode=self.mode).fit(s)
        self._n_scored = self._n_abstained = 0
        return self

    def update_shift(self, deployment_context: np.ndarray) -> dict:
        """Refresh within-bin likelihood-ratio weights from an earlier window.

        Returns a mapping from bin index to effective calibration size.
        """
        if not self.weighted:
            return {}
        self._check_fitted()
        dep = np.atleast_2d(np.asarray(deployment_context, dtype=float))
        if dep.shape[0] < 8:
            return {}
        dep_bins = self.taxonomy(dep)
        per_bin_delta = self.delta / max(len(self._bins), 1)
        n_eff = {}
        for b, rec in self._bins.items():
            dep_b = dep[dep_bins == b]
            if len(dep_b) < 8:
                rec["weights"] = np.ones(len(rec["scores"]))
                rec["eps"] = 0.0
                rec.pop("dr", None)
                n_eff[b] = float(len(rec["scores"]))
                continue
            # Classifier two-sample test: fit the density ratio on one half and
            # measure, on the held-out half, whether calibration and deployment
            # contexts are separable at all.  Without this gate the logistic
            # model always finds *some* spurious direction, the effective sample
            # size drops, the conservative fallback engages and detection power
            # is thrown away for no reason.  Weighting should cost something
            # only when there is a shift to correct.
            auc, dr = self._shift_test(rec["context"], dep_b)
            if auc < self.shift_auc_threshold:
                rec["weights"] = np.ones(len(rec["scores"]))
                rec["eps"] = 0.0
                rec.pop("dr", None)
                rec["shift_auc"] = auc
                n_eff[b] = float(len(rec["scores"]))
                continue
            rec["shift_auc"] = auc
            w = np.clip(
                dr.ratio(rec["context"]),
                np.exp(-self.weight_clip),
                np.exp(self.weight_clip),
            )
            # Weights must follow the same sort order as the stored scores.
            order = np.argsort(self._cal_scores[self._cal_bin == b])
            rec["weights"] = w[order]
            wn = np.append(rec["weights"], float(np.median(w)))
            wn = wn / wn.sum()
            eff = effective_sample_size(wn)
            n_eff[b] = eff
            rec["dr"] = dr
            if eff >= 0.9 * len(rec["scores"]):
                # Weights are effectively uniform: keep the exact Beta levels,
                # which are far tighter than any concentration bound.
                rec["weights"] = np.ones(len(rec["scores"]))
                rec["eps"] = 0.0
                rec.pop("dr", None)
            else:
                # Genuine shift.  The exact order-statistic argument no longer
                # applies once the calibration points carry unequal weights, so
                # the calibration-conditional correction is taken at the
                # effective sample size instead.
                #
                #   weighted_bound='dkw'
                #       the rigorous but loose additive weighted-DKW inflation;
                #   weighted_bound='effective_beta'
                #       the Beta levels evaluated at n_eff, an approximation
                #       (exact only for uniform weights) that is far tighter in
                #       the tail and is validated empirically in the paper.
                #
                # Either way, as deployment moves away from the calibrated
                # regime n_eff collapses, the correction grows past one, and the
                # p-value saturates: the detector abstains rather than certify a
                # regime it has no calibration for.
                rec["levels"] = None
                rec["n_eff"] = eff
                if self.weighted_bound == "effective_beta":
                    rec["eps"] = 0.0
                    rec["levels_eff"] = beta_calibration_levels(
                        max(int(round(eff)), 1), per_bin_delta
                    )
                else:
                    rec["levels_eff"] = None
                    rec["eps"] = weighted_dkw_inflation(wn, per_bin_delta)
        return n_eff

    def _shift_test(self, cal_ctx: np.ndarray, dep_ctx: np.ndarray):
        """Held-out AUC of a calibration-vs-deployment classifier, plus a
        density ratio refitted on all of the data once a shift is established."""
        rng = np.random.default_rng(0)
        ic = rng.permutation(len(cal_ctx))
        idp = rng.permutation(len(dep_ctx))
        ca, cb = ic[: len(ic) // 2], ic[len(ic) // 2:]
        da, db = idp[: len(idp) // 2], idp[len(idp) // 2:]
        if min(len(ca), len(cb), len(da), len(db)) < 4:
            return 0.5, None
        probe = LogisticDensityRatio(l2=1.0, clip=self.weight_clip).fit(
            cal_ctx[ca], dep_ctx[da]
        )
        held = np.vstack([cal_ctx[cb], dep_ctx[db]])
        y = np.concatenate([np.zeros(len(cb)), np.ones(len(db))])
        auc = roc_auc(np.log(probe.ratio(held) + 1e-12), y)
        auc = max(auc, 1.0 - auc)
        full = LogisticDensityRatio(l2=1.0, clip=self.weight_clip).fit(cal_ctx, dep_ctx)
        return float(auc), full

    def _check_fitted(self) -> None:
        if self._pooled is None:
            raise RuntimeError("MondrianConformalCalibrator.fit must be called first")

    # -- p-values --------------------------------------------------------- #
    def p_values(
        self, scores: Sequence[float], context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_values, active)``.

        ``active[t]`` is ``False`` when the frame falls in an uncalibrated
        region and the p-value saturated at one; the detector should place a
        neutral bet on those frames.
        """
        self._check_fitted()
        s = np.asarray(scores, dtype=float).ravel()
        ctx = np.atleast_2d(np.asarray(context, dtype=float))
        if len(ctx) != s.size:
            raise ValueError("context rows must match number of scores")
        bins = self.taxonomy(ctx)
        p = np.ones(s.size)
        active = np.ones(s.size, dtype=bool)

        for b in np.unique(bins):
            m = bins == b
            rec = self._bins.get(int(b))
            if rec is None:
                # Never-calibrated bin: fall back to the pooled calibrator, whose
                # validity is only marginal, and mark the frames inactive.
                p[m] = self._pooled.p_values(s[m])
                active[m] = False
                continue
            cal, w, eps = rec["scores"], rec["weights"], rec["eps"]
            if self.weighted and "dr" in rec:
                w_test = np.clip(
                    rec["dr"].ratio(ctx[m]),
                    np.exp(-self.weight_clip),
                    np.exp(self.weight_clip),
                )
            else:
                w_test = np.ones(int(m.sum()))
            idx = np.searchsorted(cal, s[m], side="left")
            levels = rec.get("levels")
            use_exact = levels is not None and "dr" not in rec
            if use_exact:
                pb = levels[len(cal) - idx]
            else:
                suffix = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
                pb = (suffix[idx] + w_test) / np.maximum(suffix[0] + w_test, _EPS)
                lv = rec.get("levels_eff")
                if lv is not None:
                    # Read the weighted p-value off the effective-size level
                    # ladder: rank j corresponds to a raw p-value of
                    # (1 + j) / (n_eff + 1).
                    n_eff = lv.size - 1
                    j = np.clip(
                        np.floor(pb * (n_eff + 1.0)).astype(int) - 1, 0, n_eff
                    )
                    pb = lv[j]
                else:
                    pb = np.minimum(1.0, pb + eps)
            p[m] = pb
            active[m] = pb < 1.0

        self._n_scored += s.size
        self._n_abstained += int((~active).sum())
        return np.clip(p, self.clip_min, 1.0), active

    @property
    def abstention_rate(self) -> float:
        """Fraction of scored frames on which the detector bet neutrally."""
        return self._n_abstained / self._n_scored if self._n_scored else 0.0

    @property
    def bin_sizes(self) -> dict:
        return {b: len(r["scores"]) for b, r in self._bins.items()}

    @property
    def inflations(self) -> dict:
        return {b: float(r["eps"]) for b, r in self._bins.items()}


# --------------------------------------------------------------------------- #
# Residual (context-normalised) conformal calibration
# --------------------------------------------------------------------------- #
class _RidgeBasis:
    """Quadratic feature map with an interaction block, plus ridge least squares."""

    def __init__(self, degree: int = 2, ridge: float = 1e-3) -> None:
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.gram_inv_: Optional[np.ndarray] = None

    def design(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        z = (x - self.mean_) / self.scale_
        parts = [np.ones((len(z), 1)), z]
        if self.degree >= 2:
            parts.append(z**2)
            d = z.shape[1]
            if d > 1:
                iu = np.triu_indices(d, k=1)
                parts.append(z[:, iu[0]] * z[:, iu[1]])
        return np.hstack(parts)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RidgeBasis":
        x = np.atleast_2d(np.asarray(x, dtype=float))
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        b = self.design(x)
        g = b.T @ b + self.ridge * len(b) * np.eye(b.shape[1])
        self.gram_inv_ = np.linalg.inv(g)
        self.coef_ = self.gram_inv_ @ (b.T @ np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.design(x) @ self.coef_

    def leverage(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis leverage of ``x`` in the fitted design space."""
        b = self.design(x)
        return np.einsum("ij,jk,ik->i", b, self.gram_inv_, b)


@dataclass
class ResidualConformalCalibrator:
    """Split conformal on context-normalised detector residuals.

    Binning the context (:class:`MondrianConformalCalibrator`) gives exact
    group-conditional validity but its bin count grows exponentially with the
    number of context variables, and a municipal camera has several that matter
    at once: clock, weather, ambient light and scene activity.  Splitting a
    realistic calibration set across a hundred bins leaves each of them too
    small for a useful tail level.

    This calibrator instead removes the predictable part of the score with an
    explicit model and conformalises what is left:

    .. math::
        \\hat g(x) = \\mathbb E[s \\mid x], \\quad
        \\hat\\sigma(x) = \\mathbb E\\bigl[|s - \\hat g(x)| \\mid x\\bigr], \\quad
        z = \\frac{s - \\hat g(x)}{\\hat\\sigma(x)},

    with :math:`\\hat g` and :math:`\\hat\\sigma` ridge regressions on a
    quadratic-with-interactions basis.  Both are fitted on one half of the
    calibration set and the conformal quantiles are taken on the *other* half,
    so exchangeability between the calibration residuals and a deployment
    residual is preserved exactly and the Beta levels of
    :func:`beta_calibration_levels` apply verbatim.  The whole held-out half
    backs every context value, instead of one bin's worth.

    Extrapolation is the obvious danger of a model-based correction, so it is
    guarded explicitly by :meth:`in_support`: frames whose context lies outside
    the region the calibration set covers are reported inactive and the detector
    bets neutrally on them, rather than trusting a regression beyond its data.

    Parameters
    ----------
    delta : float
        Calibration-conditional failure probability.
    degree : int
        Polynomial degree of the context basis (2 includes interactions).
    fit_frac : float
        Fraction of the calibration set spent on fitting the two models.
    support_quantile : float
        Tail quantile defining the per-coordinate calibrated box.
    support_margin : float
        Fractional widening of that box before a frame is called out-of-support.
    mahalanobis_slack : float
        Multiple of the calibration's 99.9th-percentile Mahalanobis distance
        beyond which a jointly novel context is declared out-of-support.
    """

    delta: float = 1e-3
    mode: str = "beta"
    degree: int = 2
    ridge: float = 1e-2
    fit_frac: float = 0.5
    scale_model: bool = True
    support_quantile: float = 0.001
    support_margin: float = 0.15
    mahalanobis_slack: float = 1.5
    clip_min: float = 1e-6
    random_state: int = 0

    _mean_model: Optional[_RidgeBasis] = field(default=None, repr=False)
    _scale_model: Optional[_RidgeBasis] = field(default=None, repr=False)
    _sorted_resid: Optional[np.ndarray] = field(default=None, repr=False)
    _levels: Optional[np.ndarray] = field(default=None, repr=False)
    _eps: float = field(default=0.0, repr=False)
    _box_lo: Optional[np.ndarray] = field(default=None, repr=False)
    _box_hi: Optional[np.ndarray] = field(default=None, repr=False)
    _ctx_mean: Optional[np.ndarray] = field(default=None, repr=False)
    _ctx_prec: Optional[np.ndarray] = field(default=None, repr=False)
    _maha_max: float = field(default=np.inf, repr=False)
    _n_scored: int = field(default=0, repr=False)
    _n_abstained: int = field(default=0, repr=False)

    # -- fitting ---------------------------------------------------------- #
    def fit(
        self, calibration_scores: Sequence[float], calibration_context: np.ndarray
    ) -> "ResidualConformalCalibrator":
        s = np.asarray(calibration_scores, dtype=float).ravel()
        x = np.atleast_2d(np.asarray(calibration_context, dtype=float))
        if len(x) != s.size:
            raise ValueError("context rows must match number of calibration scores")
        if s.size < 50:
            raise ValueError("residual conformal needs at least 50 calibration frames")

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(s.size)
        n_fit = int(round(self.fit_frac * s.size))
        fit_idx, cal_idx = perm[:n_fit], perm[n_fit:]

        self._mean_model = _RidgeBasis(self.degree, self.ridge).fit(x[fit_idx], s[fit_idx])
        resid_fit = s[fit_idx] - self._mean_model.predict(x[fit_idx])
        if self.scale_model:
            self._scale_model = _RidgeBasis(self.degree, self.ridge).fit(
                x[fit_idx], np.log(np.abs(resid_fit) + 1e-6)
            )
        else:
            self._scale_model = None

        z = self._residual(s[cal_idx], x[cal_idx])
        self._sorted_resid = np.sort(z)
        n = z.size
        if self.mode == "beta" and self.delta > 0:
            self._levels = beta_calibration_levels(n, self.delta)
            self._eps = float(self._levels[0] - 1.0 / (n + 1.0))
        elif self.mode == "dkw" and self.delta > 0:
            self._levels, self._eps = None, dkw_inflation(n, self.delta)
        else:
            self._levels, self._eps = None, 0.0

        self._fit_support(x)
        self._n_scored = self._n_abstained = 0
        return self

    # -- calibrated-support region ---------------------------------------- #
    def _fit_support(self, x: np.ndarray) -> None:
        """Record the region of context space the calibration set actually covers.

        Two complementary checks.  A per-coordinate box, widened by
        ``support_margin`` of the calibration range, catches a single variable
        drifting out of range (an activity level or an illumination never seen
        during labelling).  A Mahalanobis check on the joint context catches
        combinations that are individually ordinary but jointly novel --- bright
        *and* raining, say.  Extrapolating the residual model past either is
        exactly the failure mode a conformal guarantee is supposed to preclude,
        so such frames are declared inactive and bet on neutrally.
        """
        lo = np.quantile(x, self.support_quantile, axis=0)
        hi = np.quantile(x, 1.0 - self.support_quantile, axis=0)
        span = np.maximum(hi - lo, 1e-9)
        self._box_lo = lo - self.support_margin * span
        self._box_hi = hi + self.support_margin * span

        self._ctx_mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov) + 1e-8 * np.eye(x.shape[1])
        self._ctx_prec = np.linalg.inv(cov)
        d = self._mahalanobis(x)
        self._maha_max = float(self.mahalanobis_slack * np.quantile(d, 0.999))

    def _mahalanobis(self, x: np.ndarray) -> np.ndarray:
        c = x - self._ctx_mean
        return np.einsum("ij,jk,ik->i", c, self._ctx_prec, c)

    def in_support(self, context: np.ndarray) -> np.ndarray:
        """Boolean mask of frames whose context the calibration set covers."""
        x = np.atleast_2d(np.asarray(context, dtype=float))
        box = np.all((x >= self._box_lo) & (x <= self._box_hi), axis=1)
        return box & (self._mahalanobis(x) <= self._maha_max)

    def _residual(self, s: np.ndarray, x: np.ndarray) -> np.ndarray:
        r = s - self._mean_model.predict(x)
        if self._scale_model is not None:
            sd = np.exp(np.clip(self._scale_model.predict(x), -12.0, 3.0))
            r = r / np.maximum(sd, 1e-6)
        return r

    def _check_fitted(self) -> None:
        if self._sorted_resid is None:
            raise RuntimeError("ResidualConformalCalibrator.fit must be called first")

    # -- p-values --------------------------------------------------------- #
    def p_values(
        self, scores: Sequence[float], context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        s = np.asarray(scores, dtype=float).ravel()
        x = np.atleast_2d(np.asarray(context, dtype=float))
        if len(x) != s.size:
            raise ValueError("context rows must match number of scores")

        z = self._residual(s, x)
        cal = self._sorted_resid
        n = cal.size
        n_ge = n - np.searchsorted(cal, z, side="left")
        if self._levels is not None:
            p = self._levels[n_ge]
        else:
            p = np.minimum(1.0, (1.0 + n_ge) / (n + 1.0) + self._eps)

        active = self.in_support(x)
        p = np.where(active, p, 1.0)
        self._n_scored += s.size
        self._n_abstained += int((~active).sum())
        return np.clip(p, self.clip_min, 1.0), active

    @property
    def abstention_rate(self) -> float:
        return self._n_abstained / self._n_scored if self._n_scored else 0.0

    @property
    def inflation(self) -> float:
        self._check_fitted()
        return self._eps

    @property
    def n_conformal(self) -> int:
        """Number of held-out calibration residuals backing every p-value."""
        self._check_fitted()
        return int(self._sorted_resid.size)

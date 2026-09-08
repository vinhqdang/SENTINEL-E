"""Temporal decorrelation: turning a frame rate into an evidence rate.

Consecutive video frames are near-duplicates.  A backbone applied at 25 fps
produces scores with an autocorrelation of 0.9 or more at lag one, so a hundred
consecutive frames carry nowhere near a hundred frames' worth of independent
evidence.  Every sequential test --- CUSUM, Shiryaev--Roberts, and the betting
e-process alike --- assumes conditionally independent increments, and feeding it
raw frames inflates the realised false-alarm rate by orders of magnitude.  In
our simulations a nominal :math:`\\alpha = 0.01` becomes a realised 0.75.

The remedy adopted here is deliberately the simplest one that is also honest:
estimate the decorrelation lag from the *calibration* set alone and place a bet
only once per lag.  Betting on every twentieth frame at 25 fps still gives more
than one bet per second, which is far finer than the timescale of a dumping
event, while restoring near-independence.  It also cuts the guarantee layer's
already negligible cost by the same factor.

Two diagnostics are provided so the choice is auditable rather than assumed:
:func:`acf` and :func:`ljung_box`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["acf", "detrend", "estimate_decorrelation_lag", "thin_indices", "ljung_box"]


def acf(x: np.ndarray, n_lags: int = 100) -> np.ndarray:
    """Sample autocorrelation of ``x`` at lags ``0..n_lags`` (lag 0 equals one).

    Computed through the Wiener--Khinchin theorem: the autocovariance is the
    inverse transform of the periodogram, which costs one FFT instead of one
    dot product per lag.  On the calibration sets used here (hundreds of
    thousands of frames, two hundred lags) that is the difference between
    seconds and milliseconds, and it is the dominant cost of fitting a camera.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations")
    n_lags = int(min(n_lags, n - 1))
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return np.concatenate([[1.0], np.zeros(n_lags)])
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(xc, size)
    cov = np.fft.irfft(f * np.conjugate(f), size)[: n_lags + 1]
    out = cov / denom
    out[0] = 1.0
    return out


def detrend(scores: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove the context-predictable component of a score series.

    A surveillance score series carries two very different kinds of dependence:
    a slow, *predictable* drift driven by the clock and the weather, and a fast,
    *unpredictable* frame-to-frame correlation from the backbone smoothing
    near-duplicate images.  Only the second one is what thinning must remove ---
    the first is handled by conditioning on the context (see
    :class:`~sentinel_e.conformal.MondrianConformalCalibrator`).  Estimating the
    lag on the raw series conflates them and returns a uselessly large value, so
    the predictable part is projected out first with a quadratic least-squares
    fit on the context.
    """
    y = np.asarray(scores, dtype=float).ravel()
    if context is None:
        return y - y.mean()
    x = np.atleast_2d(np.asarray(context, dtype=float))
    if len(x) != y.size:
        raise ValueError("context rows must match number of scores")
    basis = np.hstack([np.ones((y.size, 1)), x, x**2])
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return y - basis @ coef


def estimate_decorrelation_lag(
    calibration_scores: np.ndarray,
    target: float = 0.05,
    max_lag: int = 200,
    min_lag: int = 1,
    context: Optional[np.ndarray] = None,
) -> int:
    """Smallest lag at which the calibration autocorrelation drops below ``target``.

    Estimated on confirmed no-dumping frames only, so it is a property of the
    null stream and introduces no dependence on the frames being monitored.
    When ``context`` is supplied the predictable drift is removed first (see
    :func:`detrend`).  Returns ``max_lag`` if the autocorrelation never falls
    below ``target``, which is itself a useful warning that the backbone output
    is extremely smooth and that bets should be placed sparingly.
    """
    resid = detrend(calibration_scores, context)
    r = acf(resid, n_lags=max_lag)

    below = np.flatnonzero(np.abs(r[1:]) < target)
    lag = int(below[0] + 1) if below.size else int(max_lag)
    return int(max(lag, min_lag))


def thin_indices(n_frames: int, lag: int, offset: int = 0) -> np.ndarray:
    """Frame indices retained when betting once per ``lag`` frames."""
    lag = int(max(lag, 1))
    return np.arange(int(offset) % lag, int(n_frames), lag)


def ljung_box(x: np.ndarray, n_lags: int = 20) -> Tuple[float, float]:
    """Ljung--Box portmanteau statistic and its asymptotic p-value.

    Used as a residual-dependence check on the *thinned* calibration series: a
    large p-value supports the conditional-independence premise of the e-process.
    The chi-square tail is evaluated with a series expansion so SciPy stays an
    optional dependency.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    r = acf(x, n_lags=n_lags)[1:]
    k = np.arange(1, r.size + 1)
    q = float(n * (n + 2) * np.sum(r**2 / (n - k)))
    return q, _chi2_sf(q, r.size)


def _chi2_sf(q: float, df: int) -> float:
    """Survival function of a chi-square distribution (regularised upper gamma)."""
    if q <= 0:
        return 1.0
    a, x = df / 2.0, q / 2.0
    if x < a + 1.0:
        # Series expansion for the lower regularised incomplete gamma.
        term = 1.0 / a
        total = term
        for k in range(1, 1000):
            term *= x / (a + k)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        log_p = -x + a * np.log(x) - _lgamma(a)
        return float(min(1.0, max(0.0, 1.0 - total * np.exp(log_p))))
    # Continued fraction for the upper regularised incomplete gamma.
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    log_q = -x + a * np.log(x) - _lgamma(a)
    return float(min(1.0, max(0.0, np.exp(log_q) * h)))


def _lgamma(z: float) -> float:
    """Lanczos approximation to ``log Gamma(z)`` for ``z > 0``."""
    g = [
        676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    z = float(z) - 1.0
    x = 0.99999999999980993
    for i, gi in enumerate(g):
        x += gi / (z + i + 1)
    t = z + len(g) - 0.5
    return float(0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x))

"""Fleet-level evidence combination with e-BH.

Each camera contributes its wealth :math:`W_{i,\\tau}` at the (possibly
data-dependent) inspection time :math:`\\tau`.  Optional stopping makes this a
valid e-value, :math:`\\mathbb E_\\infty[W_{i,\\tau}] \\le 1`, so the e-Benjamini--Hochberg
procedure applies directly.

Why e-BH rather than BH on p-values: the graph layer deliberately couples the
cameras (a neighbour's evidence steers this camera's bets), and adjacent
cameras see correlated weather, lighting and traffic.  BH needs PRDS or a
Benjamini--Yekutieli :math:`\\log K` penalty; e-BH controls FDR at level
:math:`\\alpha` under *arbitrary* dependence with no penalty at all.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["ebh", "ebh_rejections", "ebh_threshold", "global_e_merge"]


def ebh_threshold(e_values: Sequence[float], alpha: float) -> Tuple[int, float]:
    """Return ``(k_star, cutoff)`` for the e-BH procedure.

    ``k_star`` is the number of rejections and ``cutoff`` the smallest e-value
    that is rejected (``inf`` when there are no rejections).
    """
    e = np.asarray(e_values, dtype=float).ravel()
    if e.size == 0:
        return 0, float("inf")
    if np.any(e < 0):
        raise ValueError("e-values must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    K = e.size
    order = np.sort(e)[::-1]                       # descending
    ranks = np.arange(1, K + 1)
    ok = order >= K / (alpha * ranks)
    if not np.any(ok):
        return 0, float("inf")
    k_star = int(ranks[ok].max())
    return k_star, float(order[k_star - 1])


def ebh(e_values: Sequence[float], alpha: float = 0.1) -> np.ndarray:
    """Boolean rejection mask of the e-BH procedure at level ``alpha``."""
    e = np.asarray(e_values, dtype=float).ravel()
    k_star, cutoff = ebh_threshold(e, alpha)
    if k_star == 0:
        return np.zeros(e.size, dtype=bool)
    # Ties are broken so that exactly k_star hypotheses are rejected.
    order = np.argsort(-e, kind="stable")
    mask = np.zeros(e.size, dtype=bool)
    mask[order[:k_star]] = True
    return mask


def ebh_rejections(e_values: Sequence[float], alpha: float = 0.1) -> np.ndarray:
    """Indices rejected by e-BH."""
    return np.flatnonzero(ebh(e_values, alpha))


def global_e_merge(e_values: Sequence[float], method: str = "average") -> float:
    """Merge e-values into a single fleet-level e-value.

    ``average``
        Valid under arbitrary dependence (the arithmetic mean of e-values is an
        e-value).  This is the default for a camera fleet.
    ``product``
        Valid only when the cameras are independent under the null; included
        for ablations.
    """
    e = np.asarray(e_values, dtype=float).ravel()
    if e.size == 0:
        return 1.0
    if method == "average":
        return float(e.mean())
    if method == "product":
        return float(np.exp(np.sum(np.log(np.maximum(e, 1e-300)))))
    raise ValueError("method must be 'average' or 'product'")

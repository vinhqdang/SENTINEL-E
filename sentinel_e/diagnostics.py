"""Operator-facing diagnostics that sit outside the alarm guarantee.

The e-detector's false-alarm guarantee holds only while the conformal layer's
p-values are valid.  :func:`drift_monitor` is a slow, non-certified check on
the same p-values: it reports when they have drifted from super-uniform in a way
that suggests the calibration no longer describes deployment and a
recalibration is due.  It makes no time-uniform claim and does not change any
alarm.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def drift_monitor(
    p_values: np.ndarray,
    active: Optional[np.ndarray] = None,
    window: int = 1,
    q: float = 0.1,
    z: float = 4.0,
) -> Dict[str, object]:
    """Flag non-overlapping windows whose small-p-value share is too large.

    Parameters
    ----------
    p_values : array
        Conformal p-values on the betting grid, in time order.
    active : bool array, optional
        Only betting instants with ``active`` True are used.  ``None`` uses all.
    window : int
        Number of consecutive (active) p-values per window.  A natural choice is
        the number of betting instants in one simulated day, so that a dumping
        episode occupies a small share of the window.
    q : float
        Level at which the share of small p-values is compared against.
    z : float
        Number of binomial standard errors above ``q`` that triggers a flag.

    Returns
    -------
    dict with
        ``fraction``: per-window share of ``p <= q`` (length = number of windows),
        ``flags``: per-window boolean, ``fraction > q + z*sqrt(q(1-q)/window)``,
        ``threshold``: that cut-off,
        ``ends``: index into the *original* ``p_values`` of each window's last
        betting instant,
        ``first_flag``: index into the original array of the first flagged
        window's last betting instant (``-1`` when nothing is flagged),
        ``first_flag_window``: window number of the first flag (``-1`` if none).

    Under valid calibration p is super-uniform, so the share is at most ``q`` in
    expectation and a flag is a rare event.  A sustained rise in the share is
    evidence of drift and is not a valid alarm; it is the trigger for a human to
    recalibrate.  Windows that do not fit are dropped.
    """
    if window < 1:
        raise ValueError("window must be a positive integer")
    if not 0.0 < q < 1.0 or z <= 0.0:
        raise ValueError("q must lie in (0, 1) and z must be positive")
    p = np.asarray(p_values, dtype=float).ravel()
    if active is None:
        pos = np.arange(p.size)
    else:
        mask = np.asarray(active, dtype=bool).ravel()
        if mask.size != p.size:
            raise ValueError("active must have the same length as p_values")
        pos = np.flatnonzero(mask)
    pa = p[pos]
    n_win = pa.size // window
    threshold = q + z * np.sqrt(q * (1.0 - q) / window)
    if n_win == 0:
        return {
            "fraction": np.zeros(0), "flags": np.zeros(0, dtype=bool),
            "threshold": float(threshold), "ends": np.zeros(0, dtype=int),
            "first_flag": -1, "first_flag_window": -1,
        }
    blocks = pa[: n_win * window].reshape(n_win, window)
    fraction = np.mean(blocks <= q, axis=1)
    flags = fraction > threshold
    ends = pos[np.arange(1, n_win + 1) * window - 1]
    hit = np.flatnonzero(flags)
    return {
        "fraction": fraction,
        "flags": flags,
        "threshold": float(threshold),
        "ends": ends,
        "first_flag": int(ends[hit[0]]) if hit.size else -1,
        "first_flag_window": int(hit[0]) if hit.size else -1,
    }

"""Block p-values for betting once per block of consecutive frames.

Thinning (:func:`sentinel_e.temporal.thin_indices`) keeps one conformal p-value
every ``lag`` frames. On real video the kept p-values stay serially dependent
(the lag-one correlation of the thinned p-values is 0.76--0.99 on the real
footage used in the experiments), so the e-process premise, conditional
super-uniformity of each bet given the past, is not met. The block variant
instead bets once per block of ``B`` consecutive per-frame p-values and
combines each block into a single p-value.

Mathematics
-----------
Let ``p_1, ..., p_B`` be the per-frame conformal p-values of one block, with no
independence assumed among them. Under the null, each ``p_i`` is marginally
super-uniform, ``P(p_i <= u) <= u``. Two merging functions are valid for
*arbitrary* dependence:

* Bonferroni, ``p_B = min(1, B * min_i p_i)``. Then
  ``P(p_B <= u) = P(min_i p_i <= u / B) <= sum_i P(p_i <= u / B) <= B * (u / B) = u``.
* Twice the mean (Rueschendorf 1982; Vovk and Wang 2020),
  ``p_B = min(1, 2 * mean_i p_i)``, valid under arbitrary dependence.

So every block p-value is marginally super-uniform whatever the dependence
inside the block. Blocks are contiguous and non-overlapping, and ``gap`` frames
are skipped between consecutive blocks, so the block p-values are computed from
disjoint frames. What remains is the across-block assumption: conditional
super-uniformity of ``p_B(b)`` given the past blocks. This is an assumption, not
a theorem. It is plausibly closer to true when ``B + gap`` exceeds the
correlation length of the scores, because then consecutive blocks share few
frames. The experiments check this with the lag-one correlation and the
conditional mean ``E[p_b | p_{b-1} <= 0.1]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["BLOCK_COMBINERS", "combine_block", "block_starts", "block_pvalues"]

#: Supported block combiners. Both are valid under arbitrary within-block dependence.
BLOCK_COMBINERS = ("bonferroni", "mean2")


def combine_block(p_block: np.ndarray, combiner: str = "bonferroni") -> np.ndarray:
    """Merge the last axis of ``p_block`` into one p-value per block.

    Parameters
    ----------
    p_block : array of shape ``(..., B)``
        Per-frame p-values of each block.
    combiner : {'bonferroni', 'mean2'}
        ``'bonferroni'`` returns ``min(1, B * min_i p_i)``; ``'mean2'`` returns
        ``min(1, 2 * mean_i p_i)``.
    """
    P = np.asarray(p_block, dtype=float)
    if P.shape[-1] < 1:
        raise ValueError("a block must contain at least one p-value")
    B = P.shape[-1]
    if combiner == "bonferroni":
        return np.minimum(1.0, B * P.min(axis=-1))
    if combiner == "mean2":
        return np.minimum(1.0, 2.0 * P.mean(axis=-1))
    raise ValueError(f"block_combiner must be one of {BLOCK_COMBINERS}, got {combiner!r}")


def block_starts(n_frames: int, block_size: int, gap: int = 0) -> np.ndarray:
    """Start index of every complete block of ``block_size`` frames, ``gap`` frames apart."""
    B = int(block_size)
    G = int(gap)
    if B < 1:
        raise ValueError("block_size must be at least 1")
    if G < 0:
        raise ValueError("block_gap must be non-negative")
    if n_frames < B:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, int(n_frames) - B + 1, B + G, dtype=np.int64)


def block_pvalues(
    p: np.ndarray,
    block_size: int,
    gap: int = 0,
    combiner: str = "bonferroni",
    active: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One p-value per complete block of per-frame p-values.

    Returns ``(p_blocks, end_indices, active_blocks)``. ``end_indices`` holds the
    frame index of the last frame of each block: a block's evidence is available
    only once that frame has been observed, so an alarm is reported there. A
    block is active only if every one of its frames is active.
    """
    p = np.asarray(p, dtype=float).ravel()
    starts = block_starts(p.size, block_size, gap)
    if starts.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    B = int(block_size)
    idx = starts[:, None] + np.arange(B)[None, :]
    pb = combine_block(p[idx], combiner)
    if active is None:
        act = np.ones(starts.size, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool).ravel()[idx].all(axis=1)
    return pb, starts + B - 1, act

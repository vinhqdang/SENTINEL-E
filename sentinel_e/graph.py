"""Camera graph construction.

Cameras that watch the same street, the same access road or the same stretch of
canal share both the phenomenon (offenders relocate a few hundred metres after
being disturbed) and the nuisance (a rain shower or a sunset hits them all at
once).  The graph encodes that: an edge weight combines spatial proximity with
similarity of static context descriptors (mounting height, view type, road
class, illumination profile).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["CameraGraph"]


@dataclass
class CameraGraph:
    """Weighted, undirected graph over a camera fleet.

    Attributes
    ----------
    positions : ndarray, shape (K, 2)
        Planar camera coordinates in metres.
    weights : ndarray, shape (K, K)
        Symmetric non-negative adjacency with a zero diagonal.
    context : ndarray, shape (K, d) or None
        Static per-camera context descriptors used in the edge kernel.
    """

    positions: np.ndarray
    weights: np.ndarray
    context: Optional[np.ndarray] = None

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,
        context: Optional[np.ndarray] = None,
        length_scale: float = 300.0,
        k_neighbors: int = 4,
        context_scale: float = 1.0,
        min_weight: float = 1e-3,
    ) -> "CameraGraph":
        """Build a symmetric k-nearest-neighbour graph with a Gaussian kernel.

        Parameters
        ----------
        length_scale : float
            Spatial kernel bandwidth in metres.
        k_neighbors : int
            Each camera keeps its ``k`` nearest neighbours; the result is
            symmetrised by taking the elementwise maximum.
        context_scale : float
            Bandwidth of the context-similarity kernel.  Larger values make the
            graph purely geometric.
        """
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        K = pos.shape[0]
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2.0 * length_scale**2))

        if context is not None:
            ctx = np.atleast_2d(np.asarray(context, dtype=float))
            if len(ctx) != K:
                raise ValueError("context rows must match number of cameras")
            c2 = ((ctx[:, None, :] - ctx[None, :, :]) ** 2).sum(-1)
            w = w * np.exp(-c2 / (2.0 * context_scale**2))

        np.fill_diagonal(w, 0.0)
        if 0 < k_neighbors < K - 1:
            keep = np.zeros_like(w, dtype=bool)
            idx = np.argsort(-w, axis=1)[:, :k_neighbors]
            rows = np.repeat(np.arange(K), k_neighbors)
            keep[rows, idx.ravel()] = True
            keep = keep | keep.T
            w = np.where(keep, w, 0.0)
        w[w < min_weight] = 0.0
        w = 0.5 * (w + w.T)
        return cls(positions=pos, weights=w, context=context)

    # -- derived quantities ----------------------------------------------- #
    @property
    def n_cameras(self) -> int:
        return int(self.weights.shape[0])

    @property
    def degree(self) -> np.ndarray:
        return self.weights.sum(axis=1)

    def normalized_adjacency(self, add_self_loops: bool = True) -> np.ndarray:
        """Symmetrically normalised adjacency :math:`D^{-1/2} A D^{-1/2}`."""
        a = self.weights.copy()
        if add_self_loops:
            a = a + np.eye(self.n_cameras)
        deg = a.sum(axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        return a * dinv[:, None] * dinv[None, :]

    def neighbors(self, i: int) -> np.ndarray:
        """Indices of cameras adjacent to ``i``."""
        return np.flatnonzero(self.weights[i] > 0)

    def edge_list(self) -> np.ndarray:
        """``(E, 2)`` array of undirected edges, each listed once."""
        iu = np.triu_indices(self.n_cameras, k=1)
        mask = self.weights[iu] > 0
        return np.stack([iu[0][mask], iu[1][mask]], axis=1)

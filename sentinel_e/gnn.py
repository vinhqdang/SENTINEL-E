"""Layer 3a --- a graph neural network that supplies a *predictable* spatial prior.

Motivation
----------
Illegal dumping is not an isolated point event.  Offenders return to the same
lay-by, and when disturbed they move a few hundred metres and dump again; a
blocked bring-bank overflows onto every adjacent street at once.  A camera whose
neighbours are already accumulating evidence should therefore be *more* willing
to bet that a change is starting under its own view.

Validity
--------
The betting e-detector tolerates this only if the modulation is **predictable**:
if the hazard :math:`\\rho_{i,t}` and the stake scale :math:`m_{i,t}` are
measurable with respect to :math:`\\mathcal F_{t-1}`, then for every camera

.. math:: \\mathbb E\\bigl[f_{m_{i,t}}(p_{i,t}) \\mid \\mathcal F_{t-1}\\bigr] \\le 1,

and the telescoping identity :math:`Q_{t-1} = Q_t + w_t` still holds exactly with
:math:`w_t = \\rho_{i,t} Q_{t-1}`.  The wealth process therefore remains a
non-negative supermartingale *whatever the network outputs* --- even if the GNN
is badly trained, adversarially initialised, or fed a graph with the wrong
topology.  The network can only affect detection delay, never the false-alarm
guarantee.  This is why a learned component can be bolted onto a formal
guarantee here without any of the usual caveats.

To make that concrete the controller consumes only:

* each camera's own log-wealth **at the end of frame** :math:`t-1`;
* exponential moving averages of :math:`-\\log p` over three timescales, again
  updated only after the bet at :math:`t-1` was settled;
* static per-camera context and graph degree.

Training happens entirely offline on held-out simulated fleets, so the deployed
map from features to :math:`(\\rho, m)` is a fixed function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional: a heuristic controller covers the torch-free case
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    _HAS_TORCH = False

__all__ = [
    "SpatialPriorConfig",
    "FeatureTracker",
    "HeuristicSpatialPrior",
    "SpatialPriorGNN",
    "train_spatial_prior",
]

N_FEATURES = 9


@dataclass
class SpatialPriorConfig:
    """Ranges and shapes for the spatial-prior controller."""

    hidden: int = 32
    n_layers: int = 2
    rho_min: float = 1e-5
    rho_max: float = 5e-2
    stake_min: float = 0.05
    stake_max: float = 1.0
    ema_decays: Tuple[float, float, float] = (0.995, 0.95, 0.7)


# --------------------------------------------------------------------------- #
# Predictable feature tracker (shared by the heuristic and the GNN)
# --------------------------------------------------------------------------- #
class FeatureTracker:
    """Maintains the ``F``-dimensional predictable node features of a fleet.

    Call :meth:`features` *before* stepping the detectors at frame ``t`` and
    :meth:`update` *after*, so that everything the controller sees at ``t`` was
    already observable at ``t-1``.
    """

    def __init__(
        self,
        n_cameras: int,
        static_context: Optional[np.ndarray] = None,
        degree: Optional[np.ndarray] = None,
        config: Optional[SpatialPriorConfig] = None,
        log_threshold: float = np.log(100.0),
    ) -> None:
        self.K = int(n_cameras)
        self.cfg = config or SpatialPriorConfig()
        self.log_threshold = float(log_threshold)
        if static_context is None:
            static_context = np.zeros((self.K, 2))
        ctx = np.atleast_2d(np.asarray(static_context, dtype=float))
        if ctx.shape[1] < 2:
            ctx = np.pad(ctx, ((0, 0), (0, 2 - ctx.shape[1])))
        self.static = ctx[:, :2]
        deg = np.ones(self.K) if degree is None else np.asarray(degree, dtype=float)
        self.degree = deg / max(deg.max(), 1e-9)
        self.reset()

    def reset(self) -> None:
        self.log_wealth = np.zeros(self.K)
        self.posterior = np.zeros(self.K)
        self.ema = np.ones((self.K, 3))          # E[-log p] = 1 under uniform
        self.frames = 0

    def features(self) -> np.ndarray:
        """``(K, N_FEATURES)`` predictable node features.

        The episode posterior is the most useful of them: unlike log-wealth it
        is bounded in ``[0, 1]`` and directly comparable between cameras with
        different histories, so a message-passing layer can aggregate it across
        the graph without one long-running camera swamping its neighbours.
        """
        w = np.clip(self.log_wealth / max(self.log_threshold, 1e-9), -2.0, 2.0)
        return np.concatenate(
            [
                w[:, None],
                self.posterior[:, None],
                np.log1p(np.maximum(self.ema, 0.0)) - np.log(2.0),
                self.static,
                self.degree[:, None],
                np.full((self.K, 1), np.tanh(self.frames / 5_000.0)),
            ],
            axis=1,
        )

    def update(
        self,
        p_values: np.ndarray,
        log_wealth: np.ndarray,
        posterior: Optional[np.ndarray] = None,
    ) -> None:
        p = np.clip(np.asarray(p_values, dtype=float).ravel(), 1e-12, 1.0)
        nlp = -np.log(p)
        for j, d in enumerate(self.cfg.ema_decays):
            self.ema[:, j] = d * self.ema[:, j] + (1.0 - d) * nlp
        self.log_wealth = np.asarray(log_wealth, dtype=float).ravel()
        if posterior is not None:
            self.posterior = np.clip(
                np.asarray(posterior, dtype=float).ravel(), 0.0, 1.0
            )
        self.frames += 1


# --------------------------------------------------------------------------- #
# Heuristic controller (no learning; used as an ablation)
# --------------------------------------------------------------------------- #
class HeuristicSpatialPrior:
    """Hand-designed spatial prior: neighbours' wealth raises the local hazard.

    .. math:: \\rho_i = \\rho_0 \\exp\\bigl(\\gamma \\sum_j \\hat A_{ij}\\, \\Pi_j\\bigr),

    clipped to ``[rho_min, rho_max]``.  Serves both as a sanity check that the
    graph signal is real and as the ablation the learned controller must beat.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        rho0: float = 1e-3,
        gamma: float = 2.5,
        config: Optional[SpatialPriorConfig] = None,
    ) -> None:
        self.A = np.asarray(adjacency, dtype=float)
        self.rho0 = float(rho0)
        self.gamma = float(gamma)
        self.cfg = config or SpatialPriorConfig()

    def __call__(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Column 1 is the episode posterior: a neighbour that is probably inside
        # an episode right now raises this camera's onset hazard.
        post = features[:, 1]
        msg = self.A @ post
        rho = np.clip(
            self.rho0 * np.exp(self.gamma * msg), self.cfg.rho_min, self.cfg.rho_max
        )
        return rho, np.ones_like(post)


# --------------------------------------------------------------------------- #
# Learned controller
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    class SpatialPriorGNN(nn.Module):
        """Message-passing network mapping node features to ``(rho, stake)``.

        Two rounds of symmetric-normalised aggregation over the camera graph,
        then a per-node head.  Dense matrix multiplication is used because a
        municipal fleet has tens, not millions, of cameras; the whole forward
        pass costs a handful of small GEMMs per frame and is amortised over the
        fleet.
        """

        def __init__(self, config: Optional[SpatialPriorConfig] = None,
                     rho0: float = 1e-3) -> None:
            super().__init__()
            self.cfg = config or SpatialPriorConfig()
            self.rho0 = float(rho0)
            h = self.cfg.hidden
            self.inp = nn.Linear(N_FEATURES, h)
            self.self_layers = nn.ModuleList(
                [nn.Linear(h, h) for _ in range(self.cfg.n_layers)]
            )
            self.neigh_layers = nn.ModuleList(
                [nn.Linear(h, h) for _ in range(self.cfg.n_layers)]
            )
            self.head = nn.Linear(h, 2)
            # Initialise so that an untrained controller reproduces the
            # no-graph detector exactly: full stake, and the same episode-onset
            # hazard the ungoverned detector uses.  Starting from the default
            # sigmoid midpoint instead puts the controller at half stake and a
            # 25x hazard, which is far enough from the baseline that a short
            # training run cannot recover and the "learned" variant loses to no
            # graph at all -- a training artefact that looks like a finding.
            nn.init.normal_(self.head.weight, std=1e-2)
            with torch.no_grad():
                c = self.cfg
                span = max(c.rho_max - c.rho_min, 1e-12)
                frac = float(np.clip((rho0 - c.rho_min) / span, 1e-6, 1 - 1e-6))
                self.head.bias[0] = float(np.log(frac / (1 - frac)))
                sfrac = float(np.clip(
                    (1.0 - c.stake_min) / max(c.stake_max - c.stake_min, 1e-12),
                    1e-6, 1 - 1e-6))
                self.head.bias[1] = float(np.log(sfrac / (1 - sfrac)))

        def forward(self, x: "torch.Tensor", a_hat: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """``x``: ``(B, K, F)``; ``a_hat``: ``(B, K, K)``."""
            h = torch.tanh(self.inp(x))
            for lin_s, lin_n in zip(self.self_layers, self.neigh_layers):
                msg = torch.einsum("bij,bjh->bih", a_hat, h)
                h = torch.tanh(lin_s(h) + lin_n(msg))
            out = self.head(h)
            c = self.cfg
            rho = c.rho_min + (c.rho_max - c.rho_min) * torch.sigmoid(out[..., 0])
            stake = c.stake_min + (c.stake_max - c.stake_min) * torch.sigmoid(out[..., 1])
            return rho, stake

        # -- numpy convenience for deployment ---------------------------- #
        @torch.no_grad()
        def predict(self, features: np.ndarray, a_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            a = torch.as_tensor(a_hat, dtype=torch.float32).unsqueeze(0)
            rho, stake = self.forward(x, a)
            return rho.squeeze(0).numpy(), stake.squeeze(0).numpy()

else:  # pragma: no cover

    class SpatialPriorGNN:  # type: ignore
        """Placeholder raised when PyTorch is unavailable."""

        def __init__(self, *a, **kw):
            raise ImportError(
                "SpatialPriorGNN requires PyTorch; use HeuristicSpatialPrior instead"
            )


# --------------------------------------------------------------------------- #
# Offline training
# --------------------------------------------------------------------------- #
def _torch_features(
    log_wealth: "torch.Tensor",
    posterior: "torch.Tensor",
    ema: "torch.Tensor",
    static: "torch.Tensor",
    degree: "torch.Tensor",
    frames: int,
    log_threshold: float,
) -> "torch.Tensor":
    w = torch.clamp(log_wealth / log_threshold, -2.0, 2.0).unsqueeze(-1)
    pi = torch.clamp(posterior, 0.0, 1.0).unsqueeze(-1)
    e = torch.log1p(torch.clamp(ema, min=0.0)) - float(np.log(2.0))
    age = torch.full_like(w, float(np.tanh(frames / 5_000.0)))
    return torch.cat([w, pi, e, static, degree.unsqueeze(-1), age], dim=-1)


def train_spatial_prior(
    p_value_batches: Sequence[np.ndarray],
    adjacencies: Sequence[np.ndarray],
    static_contexts: Sequence[np.ndarray],
    degrees: Sequence[np.ndarray],
    change_points: Sequence[Sequence[Optional[int]]],
    alpha: float = 0.01,
    episode_prior=None,
    epochs: int = 30,
    lr: float = 3e-3,
    pre_change_penalty: float = 0.35,
    credit_window: int = 400,
    config: Optional[SpatialPriorConfig] = None,
    seed: int = 0,
    verbose: bool = False,
) -> "SpatialPriorGNN":
    """Fit the spatial prior by maximising post-change log-wealth growth.

    The wealth recursion is unrolled differentiably (it is all ``logaddexp``,
    ``log`` and ``log1p``), so gradients flow into both the hazard and the stake
    head.  The objective is the GROW / Kelly criterion restricted to cameras
    that truly experience an event,

    .. math:: \\max\\; \\frac1{|\\mathcal A|}\\sum_{i \\in \\mathcal A} \\log W_{i,T}
              \\;-\\; \\beta\\,\\frac1{|\\mathcal N|}\\sum_{i \\in \\mathcal N}
              \\bigl(\\log W_{i,T}\\bigr)_+,

    where :math:`\\mathcal A` are affected and :math:`\\mathcal N` unaffected
    cameras.  Maximising log-wealth growth after the change is the standard
    proxy for minimising detection delay, since the delay is essentially the
    time for the wealth to climb from one to :math:`1/\\alpha` at that rate.
    The penalty term keeps the network from burning wealth on quiet cameras,
    which would lengthen the delay of a later genuine event.

    Parameters
    ----------
    p_value_batches : sequence of ``(K, T)`` arrays
        Conformal p-values of the training fleets.
    episode_prior : EpisodePrior, optional
        The grid the deployed detector will use.  Training unrolls the same
        episodic recursion over the same grid, so the controller is fitted
        against the objective it will actually face rather than a proxy.
    adjacencies : sequence of ``(K, K)`` normalised adjacency matrices.
    change_points : sequence of length-``K`` sequences
        Event onsets per camera; ``None`` for unaffected cameras.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("train_spatial_prior requires PyTorch")

    from sentinel_e.episodic import EpisodePrior

    torch.manual_seed(seed)
    cfg = config or SpatialPriorConfig()
    model = SpatialPriorGNN(cfg, rho0=(episode_prior or EpisodePrior()).rho)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    P = torch.as_tensor(np.stack(p_value_batches), dtype=torch.float32)   # (B,K,T)
    A = torch.as_tensor(np.stack(adjacencies), dtype=torch.float32)       # (B,K,K)
    S = torch.as_tensor(np.stack(static_contexts), dtype=torch.float32)   # (B,K,2)
    D = torch.as_tensor(np.stack(degrees), dtype=torch.float32)           # (B,K)
    B, K, T = P.shape

    affected = torch.zeros(B, K)
    onset = torch.full((B, K), float(T))
    for b, cps in enumerate(change_points):
        for i, cp in enumerate(cps):
            if cp is not None:
                affected[b, i] = 1.0
                onset[b, i] = float(cp)

    # Product grid over betting aggressiveness and episode-end rate, matching
    # the deployed EpisodePrior.
    ep = episode_prior or EpisodePrior()
    kap, et, _pi, lp = ep.grid()
    grid = torch.as_tensor(kap, dtype=torch.float32)
    eta = torch.as_tensor(et, dtype=torch.float32)
    n_grid = int(grid.numel())
    log_prior = torch.as_tensor(lp, dtype=torch.float32)
    log_threshold = float(np.log(1.0 / alpha))
    decays = torch.as_tensor(cfg.ema_decays, dtype=torch.float32)
    nlp_all = -torch.log(torch.clamp(P, min=1e-12))

    history: List[float] = []
    for epoch in range(epochs):
        opt.zero_grad()
        # Episodic forward recursion, carried in scaled form exactly as the
        # deployed detector carries it, so the controller is trained against the
        # objective it will actually face.  a and q are renormalised at every step and
        # the scale accumulates in log_scale, which keeps the two-state simplex
        # from underflowing across a long unroll.
        a = torch.zeros(B, K, n_grid)
        q = torch.ones(B, K, n_grid)
        log_scale = torch.zeros(B, K, n_grid)
        ema = torch.ones(B, K, 3)
        log_W = torch.zeros(B, K)
        posterior = torch.zeros(B, K)
        credit = torch.zeros(())

        for t in range(T):
            feats = _torch_features(log_W, posterior, ema, S, D, t, log_threshold)
            rho, stake = model(feats, A)
            rho = torch.clamp(rho, cfg.rho_min, cfg.rho_max).unsqueeze(-1)

            p_t = torch.clamp(P[:, :, t], 1e-12, 1.0).unsqueeze(-1)
            # Power betting, interpolated towards the neutral bet kappa = 1 by
            # the predictable stake scale (see EDetector._modulated_kappa).
            kappa = torch.clamp(
                1.0 - stake.unsqueeze(-1) * (1.0 - grid), 1e-4, 1.0
            )
            f = torch.exp(torch.log(kappa) + (kappa - 1.0) * torch.log(p_t))

            a_new = ((1.0 - eta) * a + rho * q) * f
            q_new = eta * a + (1.0 - rho) * q
            c = torch.clamp(a_new + q_new, min=1e-30)
            a, q = a_new / c, q_new / c
            log_scale = log_scale + torch.log(c)

            w = log_prior + log_scale
            log_W = torch.logsumexp(w, dim=-1)
            mix = torch.softmax(w, dim=-1)
            posterior = (mix * a).sum(dim=-1)

            # Reward wealth *early* after the onset, not at the horizon.  The
            # detection delay is the time the wealth takes to reach 1/alpha, so
            # a camera that ends the stream rich but climbed slowly is no use;
            # scoring only the final wealth lets the controller trade delay for
            # terminal wealth, which is the opposite of what we want.  Credit is
            # therefore accumulated over a window that opens at each camera's
            # own onset.
            in_window = (
                (t >= onset) & (t < onset + credit_window) & (affected > 0)
            ).float()
            credit = credit + (log_W * in_window).sum() / float(credit_window)

            nlp = nlp_all[:, :, t].unsqueeze(-1)
            ema = decays * ema + (1.0 - decays) * nlp
            # Detach the carriers periodically: the controller only needs
            # short-horizon credit assignment and a full unroll would otherwise
            # build a T-deep graph.
            if (t + 1) % 200 == 0:
                a, q = a.detach(), q.detach()
                log_scale = log_scale.detach()
                ema = ema.detach()
                log_W = log_W.detach()
                posterior = posterior.detach()

        gain = credit / torch.clamp(affected.sum(), min=1.0)
        idle = ((torch.clamp(log_W, min=0.0)) * (1.0 - affected)).sum() / torch.clamp(
            (1.0 - affected).sum(), min=1.0
        )
        loss = -(gain - pre_change_penalty * idle)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history.append(float(loss.item()))
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={loss.item(): .4f}  "
                  f"gain={gain.item(): .3f}  idle={idle.item(): .3f}", flush=True)

    model.history = history  # type: ignore[attr-defined]
    model.eval()
    return model

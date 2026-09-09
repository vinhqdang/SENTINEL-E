"""Experiment 11 --- a single stream, worked through end to end.

Every other figure in this paper is an aggregate over hundreds of streams. This
one is not: it is a single simulated camera, chosen for the number of episodes
it happens to contain rather than for how favourably it behaves, run through
the full episodic pipeline, so a reader can see what the abstract quantities
(a p-value, a betting factor, accumulated wealth, an alarm) look like on an
actual trace rather than only in aggregate. It costs one stream simulation and
one pipeline run --- no Monte Carlo --- so it is cheap to regenerate and to
inspect by hand.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.common import FIGURES, save_json, setup_matplotlib
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator

FPS = 25.0
T = 30_000
ALPHA = 0.01
SEED = 12345


def run() -> Dict:
    cfg = StreamConfig(T=T, n_episodes=3, event_length=800, episode_gap=6_000,
                       event_intermittency=0.35, calibration_regime="representative")
    stream = StreamSimulator(cfg, seed=SEED).simulate_camera(change_point=8_000)
    model = SentinelE(alpha=ALPHA, calibration="residual", weighted=False,
                      detector="episodic")
    res = model.run_stream(stream)

    idx = np.flatnonzero(res.alarms)
    first_alarm = int(idx[0]) if idx.size else None
    payload = {
        "T": T, "alpha": ALPHA, "seed": SEED,
        "episodes": [list(e) for e in stream.episodes],
        "change_point": stream.change_point,
        "lag": res.lag,
        "abstention_rate": res.abstention_rate,
        "first_alarm": first_alarm,
        "n_episodes_in_window": len(stream.episodes),
    }
    make_figure(stream, res)
    save_json(payload, "exp11_worked_example")
    print("Experiment 11: worked example")
    print(f"  {len(stream.episodes)} episodes at {stream.episodes}")
    print(f"  decorrelation lag = {res.lag} frames")
    print(f"  first alarm at frame {first_alarm} "
          f"({None if first_alarm is None else first_alarm / FPS:.1f}s)"
          if first_alarm is not None else "  no alarm in this stream")
    print(" wrote results/figures/fig6_worked_example.pdf, "
          "results/exp11_worked_example.json")
    return payload


def make_figure(stream, res) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.0, 1.3]})
    t = np.arange(stream.T) / FPS

    def shade_episodes(ax):
        for j, (a, b) in enumerate(stream.episodes):
            ax.axvspan(a / FPS, b / FPS, color="#c1272d", alpha=0.12,
                      label="episode" if j == 0 else None)

    ax = axes[0]
    ax.plot(t, stream.scores, color="#555555", lw=0.5)
    shade_episodes(ax)
    ax.set_ylabel("backbone score $s_t$")
    ax.set_title("(a) raw detector score")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    p_shown = np.where(res.active, res.p_values, np.nan)
    ax.plot(t, p_shown, color="#0b3d91", lw=0.4, marker=".", ms=1, ls="none")
    shade_episodes(ax)
    ax.axhline(1.0 / 2001, color="k", ls=":", lw=0.7)
    ax.set_ylabel("conformal $p_t$")
    ax.set_yscale("log")
    ax.set_title("(b) conformal p-values at each bet (gaps: abstention/thinning)")

    ax = axes[2]
    ax.plot(t, res.log_wealth, color="#0b3d91", lw=1.0)
    shade_episodes(ax)
    ax.axhline(np.log(1.0 / ALPHA), color="k", ls="--", lw=1.0,
              label=r"$\log(1/\alpha)$")
    idx = np.flatnonzero(res.alarms)
    if idx.size:
        ax.scatter([idx[0] / FPS], [res.log_wealth[idx[0]]], color="#c1272d",
                  zorder=5, s=30, label="alarm")
    if stream.change_point is not None:
        ax.axvline(stream.change_point / FPS, color="#2e7d32", ls="-.", lw=0.8,
                  label="labelled onset")
    ax.set_ylabel(r"$\log W_t$")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) accumulated log-wealth, episodic e-process")
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig6_worked_example.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    run()

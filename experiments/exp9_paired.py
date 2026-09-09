"""Experiment 9 --- does the episode axis survive a paired test on a shared grid?

Experiment 8 compared ``EpisodicEDetector`` over a 6x4 (kappa, eta) grid with
``EDetector`` over 32 log-spaced kappa values and reported unpaired point
estimates.  Two things were wrong with that.  The arms did not share a betting
grid, so the measured gap confounds the episode axis with the grid; and the
differences were reported without uncertainty even though both arms are driven
by identical seeds, which is exactly the situation in which a paired analysis is
both available and far more powerful.

This experiment fixes both.  Every arm is the *same* class over the *same*
kappa grid; the arms differ only in the eta axis:

``cp``    eta = (0,)                     -- Proposition 3 boundary case, i.e. the
                                            changepoint mixture, 6 grid points.
``epi``   eta = (1e-4, 3e-3, 1e-2, 5e-2) -- the shipped episodic prior, 24 points.
``cp24``  eta = (0,) on a 24-point kappa grid -- grid-size-matched control that
                                            separates "more mixture components"
                                            from "an episode-end axis".

Differences are paired by seed and reported with a bootstrap interval; the miss
counts are compared with McNemar's exact test on the discordant pairs.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.common import fmt, write_latex_table
from experiments.runner import evaluate

REPS = 200
T = 90_000
CHANGE_POINT = 20_000
FPS = 25.0
ALPHA = 0.01
EPISODE_COUNTS = [1, 2, 4, 8]

KAPPA6 = (0.02, 0.05, 0.12, 0.28, 0.55, 0.85)
KAPPA24 = tuple(np.round(np.geomspace(0.02, 0.95, 24), 6).tolist())
ETA = (1e-4, 3e-3, 1e-2, 5e-2)

ARMS: Dict[str, Dict] = {
    "cp":   {"episode_prior_kw": (("kappa", KAPPA6), ("eta", (0.0,)))},
    "epi":  {"episode_prior_kw": (("kappa", KAPPA6), ("eta", ETA))},
    "cp24": {"episode_prior_kw": (("kappa", KAPPA24), ("eta", (0.0,)))},
}


def _censored(traces, horizon: int, change_point: int) -> np.ndarray:
    """Per-stream censored delay: misses are charged the remaining horizon.

    ``Trace.first_after_nu`` is an *absolute* frame index, not a delay, so the
    change point has to come off it.  Getting this wrong is invisible in the
    arm means -- the offset cancels -- but corrupts every paired difference in
    which one arm detects and the other does not.
    """
    out = []
    for t in traces:
        d = t.first_after_nu
        out.append(float(horizon - change_point) if d < 0
                   else float(d - change_point))
    return np.asarray(out)


def _missed(traces) -> np.ndarray:
    return np.asarray([t.first_after_nu < 0 for t in traces], dtype=bool)


def _paired_delay(a: np.ndarray, b: np.ndarray, boots: int = 20_000,
                  seed: int = 7) -> Dict:
    """Bootstrap the paired mean difference ``b - a`` over streams."""
    d = b - a
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(boots, n))
    reps = d[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    # two-sided bootstrap p for H0: mean difference = 0
    p = 2.0 * min((reps <= 0).mean(), (reps >= 0).mean())
    return {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(n)),
            "ci95": [float(lo), float(hi)], "p_boot": float(min(p, 1.0)), "n": int(n)}


def _mcnemar(a_miss: np.ndarray, b_miss: np.ndarray) -> Dict:
    """Exact McNemar on paired miss indicators (``b`` relative to ``a``)."""
    b01 = int(np.sum(~a_miss & b_miss))   # a detects, b misses
    b10 = int(np.sum(a_miss & ~b_miss))   # a misses, b detects
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_exact": 1.0}
    # exact two-sided binomial test at q = 1/2
    k = min(b01, b10)
    tail = sum(_choose(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return {"b01": b01, "b10": b10, "p_exact": float(min(1.0, 2.0 * tail))}


def _choose(n: int, k: int) -> float:
    from math import comb
    return float(comb(n, k))


def run(reps: int = REPS, counts: Sequence[int] = EPISODE_COUNTS) -> Dict:
    payload: Dict = {"reps": reps, "T": T, "change_point": CHANGE_POINT,
                     "alpha": ALPHA, "fps": FPS, "episode_counts": list(counts),
                     "by_episodes": {}}
    for n_ep in counts:
        cfg = dict(T=T, n_episodes=n_ep, event_length=800, episode_gap=6_000,
                   event_intermittency=0.35)
        arms = {}
        for name, opts in ARMS.items():
            out, _null, sig = evaluate("SENTINEL-E", ALPHA, cfg, reps,
                                       CHANGE_POINT, options=opts)
            delay = _censored(sig, T, CHANGE_POINT)
            # The per-stream vector must reproduce the harness aggregate
            # exactly.  If it does not, the pairing is against the wrong
            # object and every interval below is meaningless.
            assert abs(delay.mean() - out.add_censored) < 1e-6, (
                name, delay.mean(), out.add_censored)
            arms[name] = {
                "pfa": out.pfa, "miss_rate": out.miss_rate,
                "add_censored": out.add_censored,
                "delay": delay,
                "miss": _missed(sig),
            }
        row = {name: {k: v for k, v in a.items() if k not in ("delay", "miss")}
               for name, a in arms.items()}
        row["paired"] = {
            "epi_vs_cp": {
                "delay": _paired_delay(arms["epi"]["delay"], arms["cp"]["delay"]),
                "miss": _mcnemar(arms["epi"]["miss"], arms["cp"]["miss"]),
            },
            "epi_vs_cp24": {
                "delay": _paired_delay(arms["epi"]["delay"], arms["cp24"]["delay"]),
                "miss": _mcnemar(arms["epi"]["miss"], arms["cp24"]["miss"]),
            },
            "cp24_vs_cp": {
                "delay": _paired_delay(arms["cp24"]["delay"], arms["cp"]["delay"]),
                "miss": _mcnemar(arms["cp24"]["miss"], arms["cp"]["miss"]),
            },
        }
        payload["by_episodes"][str(n_ep)] = row
        _report(n_ep, row)
    return payload


def _report(n_ep: int, row: Dict) -> None:
    print(f"\n=== {n_ep} episode(s) ===")
    for name in ("epi", "cp", "cp24"):
        a = row[name]
        print(f"  {name:5s} pfa={a['pfa']:.3f} censored={a['add_censored']/FPS:8.1f}s "
              f"miss={a['miss_rate']:.3f}")
    for key, d in row["paired"].items():
        dl, mi = d["delay"], d["miss"]
        print(f"  {key:12s} delay diff (2nd - 1st) {dl['mean_diff']/FPS:+8.1f}s "
              f"[{dl['ci95'][0]/FPS:+.1f}, {dl['ci95'][1]/FPS:+.1f}] p={dl['p_boot']:.3f}"
              f" | miss discordant {mi['b01']}/{mi['b10']} p={mi['p_exact']:.3f}")


def make_table(payload: Dict) -> None:
    """Paired episodic-vs-changepoint comparison, on a shared 24-point kappa grid.

    Reports the ``epi_vs_cp24`` contrast: same class, same grid size, differing
    only in the eta axis, so a significant delay or miss difference here is not
    confounded with the grid-size effect the ablation shows can be comparably
    large (\\cref{sec:ablation}, "single bet aggressiveness").
    """
    rows = []
    for k in payload["episode_counts"] if "episode_counts" in payload else sorted(
            payload["by_episodes"], key=int):
        row = payload["by_episodes"][str(k)]
        d = row["paired"]["epi_vs_cp24"]["delay"]
        m = row["paired"]["epi_vs_cp24"]["miss"]
        rows.append([
            str(k),
            f"{d['mean_diff']/FPS:+.1f}",
            f"[{d['ci95'][0]/FPS:+.1f}, {d['ci95'][1]/FPS:+.1f}]",
            fmt(d["p_boot"], 3),
            f"{m['b01']}/{m['b10']}",
            fmt(m["p_exact"], 3),
        ])
    write_latex_table(
        rows,
        ["episodes", "mean diff.\\ (s)", "95\\% CI", "$p$ (boot.)",
         "miss discordant", "$p$ (McNemar)"],
        caption=(
            r"Paired episodic-vs-changepoint comparison on a shared "
            f"{len(KAPPA24)}-point $\\kappa$ grid ({payload['reps']} seeded "
            r"pairs per row): the changepoint-mixture minus episodic censored "
            r"delay, and the discordant miss counts (episodic-detects-only / "
            r"changepoint-detects-only) with an exact two-sided McNemar test. "
            r"A positive mean difference favours the episodic mixture. Unlike "
            r"\cref{tab:episodes}, which compares unpaired point estimates on "
            r"different grids, every row here isolates the episode-end axis "
            r"alone."
        ),
        label="tab:paired_episodes", name="tab13_paired_episodes", align="rrrrrr",
    )


if __name__ == "__main__":
    import sys
    from experiments.common import save_json
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else REPS
    out = run(reps=reps)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as fh:
            json.dump(out, fh, indent=1)
        print("\nwrote", sys.argv[2])
    else:
        save_json(out, "exp9_paired")
        print("\nwrote results/exp9_paired.json")
    make_table(out)
    print("wrote results/tables/tab13_paired_episodes.tex")

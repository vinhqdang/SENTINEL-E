"""Experiment 8 --- what the episodic mixture buys, and where it does not.

The claim under test is narrow and should be stated narrowly. The episodic
e-process generalises the changepoint mixture along one axis --- episode
duration --- and the generalisation can only matter when that axis is active.
Two regimes are therefore swept.

*Number of episodes.*  Illegal dumping recurs: a site used once is used again.
A changepoint-mixture detector that fails to fire during the first episode has
spent wealth betting on a change that has since reverted, and it carries that
loss into the second. The episodic process returns to its quiet state instead,
so evidence accumulates across episodes.

*Within-episode intermittency.*  Only a fraction of the frames inside an episode
show the act. The dilated emission says so explicitly; the undilated bet treats
every occluded frame as evidence against the change.

Where the classical model is right --- one long episode, no intermittency --- the
two should tie, and reporting that is part of the result: a strict
generalisation should cost nothing when its extra structure is absent.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.common import (
    FIGURES,
    fmt,
    save_json,
    setup_matplotlib,
    write_latex_table,
)
from experiments.runner import evaluate

REPS = 200
T = 90_000
CHANGE_POINT = 20_000
FPS = 25.0
ALPHA = 0.01

EPISODE_COUNTS = [1, 2, 4, 8]
INTERMITTENCY = [1.0, 0.6, 0.35, 0.2]

VARIANTS = {"episodic": {"detector": "episodic"},
            "changepoint": {"detector": "changepoint"}}


def _row(variant: str, opts: Dict, cfg: Dict, reps: int) -> Dict:
    out, null, _ = evaluate("SENTINEL-E", ALPHA, cfg, reps, CHANGE_POINT, options=opts)
    return {
        "variant": variant, "pfa": out.pfa, "arl0": out.arl0,
        "add": out.add, "add_se": out.add_se,
        "add_censored": out.add_censored,
        "add_censored_se": out.add_censored_se, "miss_rate": out.miss_rate,
    }


def sweep(axis: str, values: List, reps: int) -> Dict[str, List[Dict]]:
    res: Dict[str, List[Dict]] = {}
    for v in values:
        if axis == "episodes":
            cfg = dict(T=T, calibration_regime="representative",
                       n_episodes=int(v), event_length=800, episode_gap=6_000,
                       event_intermittency=0.35)
        else:
            cfg = dict(T=T, calibration_regime="representative",
                       n_episodes=4, event_length=800, episode_gap=6_000,
                       event_intermittency=float(v))
        res[str(v)] = [_row(k, o, cfg, reps) for k, o in VARIANTS.items()]
        for r in res[str(v)]:
            add = r["add"] if r["add"] is not None and np.isfinite(r["add"]) else float("nan")
            print(f"  {axis}={v:<5} {r['variant']:<12} PFA={r['pfa']:.3f} "
                  f"ADD={add:8.1f} miss={r['miss_rate']:.2f}", flush=True)
    return res


def make_figure(by_episodes, by_pi):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    colors = {"episodic": "#0b3d91", "changepoint": "#c1272d"}
    markers = {"episodic": "o", "changepoint": "v"}

    def series(data, keys, variant, field):
        out = []
        for k in keys:
            r = next(x for x in data[str(k)] if x["variant"] == variant)
            v = r[field]
            out.append(np.nan if v is None or not np.isfinite(v) else v)
        return np.array(out, dtype=float)

    ax = axes[0]
    for v in VARIANTS:
        ax.plot(EPISODE_COUNTS, series(by_episodes, EPISODE_COUNTS, v, "miss_rate"),
                label=v, color=colors[v], marker=markers[v])
    ax.set_xlabel("episodes per camera"); ax.set_ylabel("missed-detection rate")
    ax.set_title("(a) events that recur"); ax.set_xticks(EPISODE_COUNTS)
    ax.legend()

    ax = axes[1]
    for v in VARIANTS:
        ax.plot(EPISODE_COUNTS, series(by_episodes, EPISODE_COUNTS, v, "add_censored") / FPS,
                label=v, color=colors[v], marker=markers[v])
    ax.set_xlabel("episodes per camera"); ax.set_ylabel("censored delay (s)")
    ax.set_title("(b) delay"); ax.set_xticks(EPISODE_COUNTS)

    ax = axes[2]
    for v in VARIANTS:
        ax.plot(INTERMITTENCY, series(by_pi, INTERMITTENCY, v, "miss_rate"),
                label=v, color=colors[v], marker=markers[v])
    ax.set_xlabel(r"fraction of episode frames showing the act ($\pi$)")
    ax.set_ylabel("missed-detection rate")
    ax.set_title("(c) within-episode intermittency")
    ax.invert_xaxis()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig5_episodes.{ext}")
    plt.close(fig)


def make_table(by_episodes, by_pi):
    rows = []
    for k in EPISODE_COUNTS:
        for i, r in enumerate(by_episodes[str(k)]):
            rows.append([
                str(k) if i == 0 else "", r["variant"], fmt(r["pfa"], 3),
                fmt(None if r["add_censored"] is None else r["add_censored"] / FPS, 1),
                fmt(r["miss_rate"], 3),
            ])
    write_latex_table(
        rows, ["episodes", "mixture", "PFA", "cens. delay (s)", "miss rate"],
        caption=(
            r"Mixing over episodes against mixing over changepoints, as the "
            r"number of episodes at a camera grows ($\alpha=0.01$, "
            f"{REPS} null and {REPS} post-change streams per row, episodes of "
            r"800 frames at intermittency $\pi=0.35$). With one episode the two "
            r"agree, as a strict generalisation should; the gap opens as events "
            r"recur, which is what a dumping hotspot does."
        ),
        label="tab:episodes", name="tab11_episodes", align="rlrrr",
    )

    rows = []
    for k in INTERMITTENCY:
        for i, r in enumerate(by_pi[str(k)]):
            rows.append([
                fmt(k, 2) if i == 0 else "", r["variant"], fmt(r["pfa"], 3),
                fmt(None if r["add_censored"] is None else r["add_censored"] / FPS, 1),
                fmt(r["miss_rate"], 3),
            ])
    write_latex_table(
        rows, [r"$\pi$", "mixture", "PFA", "cens. delay (s)", "miss rate"],
        caption=(
            r"Effect of within-episode intermittency at four episodes per "
            r"camera. $\pi$ is the fraction of frames inside an episode that "
            r"actually show the act; smaller is harder. The dilated emission "
            r"$(1-\pi) + \pi f$ treats the occluded frames as expected rather "
            r"than as evidence against the episode."
        ),
        label="tab:intermittency", name="tab12_intermittency", align="rlrrr",
    )


def main(reps: int = REPS):
    print("Experiment 8: episodes versus changepoints")
    print(" sweeping episode count ...")
    by_episodes = sweep("episodes", EPISODE_COUNTS, reps)
    print(" sweeping intermittency ...")
    by_pi = sweep("pi", INTERMITTENCY, reps)
    save_json({"by_episodes": by_episodes, "by_intermittency": by_pi,
               "reps": reps, "T": T, "alpha": ALPHA,
               "episode_counts": EPISODE_COUNTS,
               "intermittency": INTERMITTENCY}, "exp8_episodes")
    make_figure(by_episodes, by_pi)
    make_table(by_episodes, by_pi)
    print(" wrote results/exp8_episodes.json, fig5_episodes, tab11/tab12")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    main(**vars(ap.parse_args()))

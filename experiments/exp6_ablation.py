"""Experiment 6 --- component ablation.

Each row removes or replaces exactly one design decision and reports what
breaks.  The interesting rows are the ones where a component that looks like an
implementation detail turns out to be load-bearing for the guarantee:

* betting on every frame instead of once per decorrelation lag;
* the additive DKW inflation instead of the exact order-statistic levels;
* treating the calibration frames as independent when they are consecutive video.

and the ones where a component only affects power, which is the right place for
a tuning knob to live: the betting family and the changepoint prior.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.common import fmt, fmt_int, save_json, write_latex_table
from experiments.runner import evaluate

REPS = 200
T = 90_000
CHANGE_POINT = 20_000
N_EPISODES = 4
FPS = 25.0
ALPHA = 0.01

BASE = dict(detector="episodic", calibration="residual", weighted=False,
            family="power",
            n_grid=32, rho=1e-3, prior_kind="geometric", delta=1e-3,
            thin_calibration=True)

ABLATIONS: List[Dict] = [
    {"name": "SENTINEL-E (full)", "group": "", "opts": {}},

    # --- the episodic core ------------------------------------------------ #
    {"name": "changepoint mixture ($\\eta = 0$)", "group": "e-process core",
     "opts": {"detector": "changepoint"}},
    {"name": "enable dilation ($\\pi$ grid)", "group": "e-process core",
     "opts": {"episode_prior_kw": (("pi", (0.25, 0.55, 1.0)),)}},
    {"name": "single episode length", "group": "e-process core",
     "opts": {"episode_prior_kw": (("eta", (3e-3,)),)}},
    {"name": "single bet aggressiveness", "group": "e-process core",
     "opts": {"episode_prior_kw": (("kappa", (0.28,)),)}},

    # --- temporal --------------------------------------------------------- #
    {"name": "bet on every frame", "group": "temporal",
     "opts": {"lag": 1}},
    {"name": "bet once per 5 frames", "group": "temporal",
     "opts": {"lag": 5}},
    {"name": "raw calibration (not thinned)", "group": "temporal",
     "opts": {"thin_calibration": False}},

    # --- calibration ------------------------------------------------------ #
    {"name": "pooled conformal", "group": "calibration",
     "opts": {"calibration": "pooled"}},
    {"name": "Mondrian conformal", "group": "calibration",
     "opts": {"calibration": "mondrian"}},
    {"name": "residual + LR weighting", "group": "calibration",
     "opts": {"weighted": True}},

    # --- calibration-conditional validity --------------------------------- #
    {"name": "DKW inflation", "group": "conditional validity",
     "opts": {"cc_mode": "dkw"}},
    {"name": "no correction ($\\delta=0$)", "group": "conditional validity",
     "opts": {"delta": 0.0}},

    # --- episode-onset prior ---------------------------------------------- #
    {"name": r"hazard $\rho=10^{-2}$", "group": "onset prior", "opts": {"rho": 1e-2}},
    {"name": r"hazard $\rho=10^{-4}$", "group": "onset prior", "opts": {"rho": 1e-4}},

    # --- betting family, evaluated on the changepoint core where it applies - #
    {"name": "linear betting (changepoint core)", "group": "betting",
     "opts": {"detector": "changepoint", "family": "linear"}},
    {"name": "adaptive ONS betting (changepoint core)", "group": "betting",
     "opts": {"detector": "changepoint", "family": "adaptive"}},
    {"name": "single bet, $K=1$ (changepoint core)", "group": "betting",
     "opts": {"detector": "changepoint", "n_grid": 1}},
    {"name": "fixed start (changepoint core)", "group": "betting",
     "opts": {"detector": "changepoint", "prior_kind": "point"}},
]


def run(reps: int = REPS) -> List[Dict]:
    cfg = dict(T=T, calibration_regime="representative",
               n_episodes=N_EPISODES, event_length=800, episode_gap=6_000,
               event_intermittency=0.35)
    rows = []
    for ab in ABLATIONS:
        opts = dict(BASE)
        opts.update(ab["opts"])
        out, null, _ = evaluate("SENTINEL-E", ALPHA, cfg, reps, CHANGE_POINT,
                                options=opts)
        rows.append({
            "name": ab["name"], "group": ab["group"],
            "pfa": out.pfa, "pfa_ci": list(out.pfa_ci), "arl0": out.arl0,
            "add": out.add, "add_se": out.add_se,
        "add_censored": out.add_censored,
        "add_censored_se": out.add_censored_se, "miss_rate": out.miss_rate,
            "lag": null[0].lag,
            "abstention": float(np.mean([t.abstention for t in null])),
        })
        print(f"  {ab['name']:34s} PFA={out.pfa:.3f} "
              f"ADD={out.add if np.isfinite(out.add) else float('nan'):8.1f} "
              f"miss={out.miss_rate:.2f} lag={null[0].lag}", flush=True)
    return rows


def make_table(rows):
    out, last_group = [], None
    for r in rows:
        g = r["group"]
        label = g if g and g != last_group else ""
        last_group = g if g else last_group
        valid = "\\checkmark" if r["pfa"] <= ALPHA + 1e-9 else "\\ding{55}"
        out.append([
            label, r["name"], fmt(r["pfa"], 3), valid, fmt_int(r["arl0"]),
            fmt(None if r["add_censored"] is None else r["add_censored"] / FPS, 1),
            fmt(r["miss_rate"], 2), fmt_int(r["lag"]),
        ])
    write_latex_table(
        out,
        ["group", "variant", "PFA", r"$\le\alpha$?", "ARL$_0$", "cens. delay (s)",
         "miss", "lag"],
        caption=(
            r"Component ablation at $\alpha=0.01$ over "
            f"{REPS} null and {REPS} post-change streams. A cross in the fourth "
            "column means the realised false-alarm rate exceeded the nominal "
            "level, i.e. the variant does not deliver the guarantee it claims. "
            "Note that the components which break validity are the temporal and "
            "calibration ones, not the betting or prior choices, which only move "
            "the delay."
        ),
        label="tab:ablation",
        name="tab8_ablation",
        align="llrcrrrr",
        notes=(
            r"``lag'' is the decorrelation lag chosen from the calibration "
            r"autocorrelation, in frames at 25\,fps."
        ),
    )


def main(reps: int = REPS):
    print("Experiment 6: component ablation")
    rows = run(reps)
    save_json({"rows": rows, "reps": reps, "alpha": ALPHA, "T": T,
               "n_episodes": N_EPISODES,
               "change_point": CHANGE_POINT}, "exp6_ablation")
    make_table(rows)
    print(" wrote results/exp6_ablation.json, tab8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    main(**vars(ap.parse_args()))

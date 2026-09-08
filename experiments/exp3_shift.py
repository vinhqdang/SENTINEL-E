"""Experiment 3 --- robustness to the gap between calibration and deployment.

Two axes of mismatch, both of them things that happen to a real camera between
the week its calibration frames were labelled and the month it is monitored.

*Coverage.*  ``representative`` calibration spans the diurnal cycle and both
weather regimes; ``full_cycle_benign`` spans the clock but was labelled during a
dry spell; ``daytime_only`` came from a single clean daytime window.

*Composition.*  ``activity_shift`` multiplies the deployment scene-activity
level relative to the labelling window --- a quiet week used for labelling, a
busy month in deployment.  Activity is *not* part of the operator's obvious
(clock, weather) taxonomy, which is exactly what makes it the interesting case.

The claim under test is not that detection power survives every mismatch --- it
demonstrably does not, and cannot: a regime that was never calibrated cannot be
certified.  The claim is that *false-alarm control* survives, and that the
degradation shows up as abstention and missed detections, which an operator can
see and act on, rather than as silent alarm inflation, which they cannot.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.common import (
    FIGURES,
    fmt,
    fmt_int,
    save_json,
    setup_matplotlib,
    write_latex_table,
)
from experiments.runner import evaluate

REPS = 200
T = 90_000
CHANGE_POINT = 20_000
FPS = 25.0

REGIMES = ["representative", "full_cycle_benign", "daytime_only"]
ACTIVITY_SHIFTS = [1.0, 1.5, 2.0, 3.0]

VARIANTS = {
    "pooled":            dict(calibration="pooled", weighted=False),
    "Mondrian":          dict(calibration="mondrian", weighted=False),
    "residual":          dict(calibration="residual", weighted=False),
    "residual + LR wts": dict(calibration="residual", weighted=True),
}

REGIME_LABEL = {
    "representative": "representative",
    "full_cycle_benign": "no adverse weather",
    "daytime_only": "daytime window only",
}


def _row(variant, opts, cfg, reps):
    out, null, _ = evaluate("SENTINEL-E", 0.01, cfg, reps, CHANGE_POINT, options=opts)
    return {
        "variant": variant,
        "pfa": out.pfa,
        "pfa_ci": list(out.pfa_ci),
        "arl0": out.arl0,
        "add": out.add,
        "add_censored": out.add_censored,
        "add_censored_se": out.add_censored_se,
        "miss_rate": out.miss_rate,
        "abstention": float(np.mean([t.abstention for t in null])),
    }


def sweep_coverage(reps: int = REPS) -> Dict[str, List[Dict]]:
    res = {}
    for regime in REGIMES:
        cfg = dict(T=T, calibration_regime=regime, n_episodes=4,
                   event_length=800, episode_gap=6_000,
                   event_intermittency=0.35)
        res[regime] = [_row(v, o, cfg, reps) for v, o in VARIANTS.items()]
        for r in res[regime]:
            print(f"  {regime:18s} {r['variant']:18s} PFA={r['pfa']:.3f} "
                  f"miss={r['miss_rate']:.2f} abstain={r['abstention']:.2f}", flush=True)
    return res


def sweep_activity(reps: int = REPS) -> Dict[str, List[Dict]]:
    res = {}
    for a in ACTIVITY_SHIFTS:
        cfg = dict(T=T, calibration_regime="representative", activity_shift=a,
                   n_episodes=4, event_length=800, episode_gap=6_000,
                   event_intermittency=0.35)
        res[str(a)] = [_row(v, o, cfg, reps) for v, o in VARIANTS.items()]
        for r in res[str(a)]:
            print(f"  activity x{a:<4} {r['variant']:18s} PFA={r['pfa']:.3f} "
                  f"miss={r['miss_rate']:.2f} abstain={r['abstention']:.2f}", flush=True)
    return res


def make_figure(activity):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    xs = np.array(ACTIVITY_SHIFTS)
    colors = {"pooled": "#c1272d", "Mondrian": "#e07b39",
              "residual": "#0b3d91", "residual + LR wts": "#3f7fd1"}
    markers = {"pooled": "v", "Mondrian": "D", "residual": "o", "residual + LR wts": "s"}

    for ax, key, ylabel, title in (
        (axes[0], "pfa", "realised false-alarm probability",
         "(a) validity under covariate shift"),
        (axes[1], "add_censored", "censored detection delay (s)",
         "(b) the price paid for it"),
    ):
        for v in VARIANTS:
            y = []
            for a in ACTIVITY_SHIFTS:
                r = next(x for x in activity[str(a)] if x["variant"] == v)
                val = r[key]
                y.append(np.nan if val is None or not np.isfinite(val) else val)
            y = np.array(y, dtype=float)
            if key == "add_censored":
                y = y / FPS
            ax.plot(xs, y, label=v, color=colors[v], marker=markers[v])
        ax.set_xlabel("deployment / calibration activity ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].axhline(0.01, color="k", ls="--", lw=1.0, label=r"$\alpha=0.01$")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].legend(loc="center left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig3_shift.{ext}")
    plt.close(fig)


def make_tables(coverage, activity):
    rows = []
    for regime in REGIMES:
        for i, r in enumerate(coverage[regime]):
            rows.append([
                REGIME_LABEL[regime] if i == 0 else "",
                r["variant"], fmt(r["pfa"], 3), fmt_int(r["arl0"]),
                fmt(None if r["add_censored"] is None else r["add_censored"] / FPS, 1),
                fmt(r["miss_rate"], 2), fmt(r["abstention"], 2),
            ])
    write_latex_table(
        rows,
        ["calibration coverage", "variant", "PFA", "ARL$_0$", "cens. delay (s)",
         "miss", "abstain"],
        caption=(
            r"Calibration coverage against detection performance at $\alpha=0.01$ "
            f"({REPS} null and {REPS} post-change streams per row). Pooled and "
            "Mondrian calibration lose false-alarm control; the residual "
            "construction never does. Under poor coverage its power degrades "
            "visibly through the abstention and miss columns rather than "
            "silently through the alarm rate."
        ),
        label="tab:coverage",
        name="tab4_coverage",
        align="llrrrrr",
    )

    rows = []
    for a in ACTIVITY_SHIFTS:
        for i, r in enumerate(activity[str(a)]):
            rows.append([
                f"$\\times {a}$" if i == 0 else "",
                r["variant"], fmt(r["pfa"], 3), fmt_int(r["arl0"]),
                fmt(None if r["add_censored"] is None else r["add_censored"] / FPS, 1),
                fmt(r["miss_rate"], 2), fmt(r["abstention"], 2),
            ])
    write_latex_table(
        rows,
        ["activity shift", "variant", "PFA", "ARL$_0$", "cens. delay (s)", "miss", "abstain"],
        caption=(
            r"Shift in an \emph{unbinned} covariate (scene activity) at "
            r"$\alpha=0.01$. Binning the context cannot help with a variable "
            r"that is not in the taxonomy, so Mondrian calibration fails with "
            r"the pooled baseline; conformalising the context-normalised "
            r"residual handles it."
        ),
        label="tab:activity",
        name="tab5_activity",
        align="llrrrrr",
    )


def main(reps: int = REPS):
    print("Experiment 3: robustness to calibration/deployment mismatch")
    print(" sweeping calibration coverage ...")
    coverage = sweep_coverage(reps)
    print(" sweeping activity composition ...")
    activity = sweep_activity(reps)
    save_json({"coverage": coverage, "activity": activity, "reps": reps,
               "T": T, "change_point": CHANGE_POINT}, "exp3_shift")
    make_figure(activity)
    make_tables(coverage, activity)
    print(" wrote results/exp3_shift.json, fig3_shift, tab4/tab5")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    main(**vars(ap.parse_args()))

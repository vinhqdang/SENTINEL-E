"""Experiment R1 --- time-uniform false-alarm control on real surveillance video.

The real-data analogue of ``experiments/exp1_validity.py``. Every stream here
is built out of real Farneback optical-flow scores from CUHK Avenue and UCSD
Ped2 (:mod:`sentinel_e.datasets.real_vad`), not the AR(1) simulator: real
per-frame noise, real train/test splits, real spliced anomalous runs.

This experiment reports the central real-data finding of the paper, not a
footnote to it: real per-frame anomaly scores are far more autocorrelated
than anything the simulator produced, and reconciling that with the brevity
of real anomalous episodes leaves only a narrow -- on Avenue, barely any --
band of usable operating points.

1. **Real autocorrelation is severe.** The full decorrelation lag (where the
   sample ACF first falls below 0.05) is ~357 frames on Ped2's official
   training split and ~207 on Avenue's -- two orders of magnitude past the
   simulator's AR(1) noise, and far longer than a single real anomalous
   episode (median 140 frames on Ped2, 29 on Avenue). Thinning to that lag,
   as the guarantee's i.i.d.-calibration assumption asks for, leaves at most
   one or two betting opportunities inside a typical real event -- too few
   for any sequential test to detect reliably, at any alpha.
2. **The lag sweep below is the reason, made precise.** Below a sharp
   threshold the realised false-alarm rate is uncontrolled (correlated bets
   are treated as independent evidence); above it, detection power collapses
   because episodes end before enough independent looks accumulate. The two
   curves cross in a narrow window -- roughly 5 frames on Ped2, 7 on Avenue
   -- rather than leaving the wide safe margin the simulated streams did.
   This experiment reports both curves directly, not just the one operating
   point inside the window, so the trade-off is visible rather than
   engineered away by the choice of what to report.
3. **The lag inside that window is what every other real-data result in
   this paper uses**, fixed once from this diagnostic and never re-tuned
   against a downstream PFA or delay number.
"""

from __future__ import annotations

import numpy as np

from experiments.common import fmt, fmt_int, save_json, setup_matplotlib, style_for, write_latex_table, FIGURES
from experiments.real_data.runner import evaluate_real

#: The practical operating lag used throughout the rest of the real-data
#: experiments: the smallest lag (from the sweep in ``sweep_lag`` below) at
#: which the realised false-alarm rate is controlled at alpha=0.1. Chosen
#: once from this diagnostic sweep, never re-tuned against a downstream
#: result.
DATASET_LAG = {"ped2": 5, "avenue": 7}
#: The fully rigorous decorrelation lag (sample ACF first below 0.05),
#: reported for contrast -- this is what the i.i.d.-calibration assumption
#: actually asks for, and it leaves essentially no power on either dataset.
DATASET_FULL_LAG = {"ped2": 357, "avenue": 207}

BASE = dict(n_frames=6_000, n_calibration=800, change_point=1_500, n_events=2)
CAL_OPTS = {"calibration": "pooled", "calibration_regime": "official"}
REPS = 400
LAG_REPS = 250
ALPHAS = [0.3, 0.2, 0.1, 0.05, 0.02]
LAG_GRID = {"ped2": [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20],
            "avenue": [1, 2, 3, 5, 6, 7, 8, 10, 15, 20]}


def sweep_lag(dataset: str, reps: int = LAG_REPS):
    rows = []
    for lag in LAG_GRID[dataset]:
        opts = dict(CAL_OPTS, lag=lag)
        out, null, sig = evaluate_real("SENTINEL-E", 0.1, dataset, reps, options=opts, **BASE)
        rows.append({"lag": lag, "pfa": out.pfa, "pfa_ci": list(out.pfa_ci),
                     "miss_rate": out.miss_rate})
        print(f"  {dataset:7s} lag={lag:<4d} PFA={out.pfa:.3f} miss={out.miss_rate:.3f}",
              flush=True)
    return rows


def sweep_alpha(dataset: str, reps: int = REPS):
    opts = dict(CAL_OPTS, lag=DATASET_LAG[dataset])
    rows = []
    for a in ALPHAS:
        se, null, sig = evaluate_real("SENTINEL-E", a, dataset, reps, options=opts, **BASE)
        rows.append({
            "alpha": a, "pfa": se.pfa, "pfa_ci": list(se.pfa_ci), "arl0": se.arl0,
            "add": se.add, "add_censored": se.add_censored,
            "add_censored_se": se.add_censored_se, "miss_rate": se.miss_rate,
            "lag": null[0].lag,
        })
        print(f"  {dataset:7s} alpha={a:<6} PFA={se.pfa:.3f} "
              f"miss={se.miss_rate:.3f} lag={null[0].lag}", flush=True)
    return rows


def make_figure(lag_rows, alpha_rows):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = {"ped2": "#0b3d91", "avenue": "#c1272d"}
    labels = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}

    ax = axes[0]
    for name, rows in lag_rows.items():
        lag = np.array([r["lag"] for r in rows])
        pfa = np.array([max(r["pfa"], 2e-3) for r in rows])
        miss = np.array([r["miss_rate"] for r in rows])
        ax.plot(lag, pfa, color=colors[name], marker="o", label=f"{labels[name]} PFA")
        ax.plot(lag, miss, color=colors[name], marker="s", ls="--",
                label=f"{labels[name]} miss rate")
        ax.axvline(DATASET_LAG[name], color=colors[name], lw=0.8, alpha=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("betting-grid thinning lag (frames)")
    ax.set_ylabel("rate")
    ax.set_title("(a) validity vs power: a narrow crossing, not a margin")
    ax.legend(loc="center right", fontsize=6.5)

    ax = axes[1]
    FLOOR = 2e-3
    for name, rows in alpha_rows.items():
        a = np.array([r["alpha"] for r in rows])
        pfa = np.array([r["pfa"] for r in rows], dtype=float)
        ci_hi = np.array([r["pfa_ci"][1] for r in rows], dtype=float)
        obs = pfa > 0
        st = {"color": colors[name], "marker": "o", "zorder": 4}
        if obs.any():
            ax.plot(a[obs], pfa[obs], label=labels[name], **st)
        if (~obs).any():
            ax.plot(a[~obs], np.maximum(ci_hi[~obs], FLOOR), linestyle="none",
                    marker="v", markerfacecolor="white", markeredgecolor=st["color"],
                    markersize=6, label=None if obs.any() else labels[name])
            ax.plot(a, np.where(obs, pfa, np.maximum(ci_hi, FLOOR)),
                    color=st["color"], lw=1.0, ls=":")
    ax.plot(ALPHAS, ALPHAS, "k--", lw=1.0, label=r"nominal $\alpha$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.7, 1.6)
    ax.set_xlabel(r"target level $\alpha$")
    ax.set_ylabel("realised false-alarm rate")
    ax.set_title("(b) validity at the fixed operating lag")
    ax.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_r1_validity.{ext}")
    plt.close(fig)


def make_table(lag_rows, alpha_rows, reps=REPS, lag_reps=LAG_REPS):
    labels = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}
    rows = []
    for name, rows_l in lag_rows.items():
        for r in rows_l:
            marker = r"$\leftarrow$" if r["lag"] == DATASET_LAG[name] else ""
            rows.append([labels[name], fmt_int(r["lag"]), fmt(r["pfa"], 3),
                        fmt(r["miss_rate"], 3), marker])
    write_latex_table(
        rows,
        ["dataset", "lag", "PFA", "miss rate", ""],
        caption=(
            "Validity and power against the betting-grid thinning lag at "
            f"$\\alpha=0.1$ ({lag_reps} null and {lag_reps} signal streams per "
            "point). Real per-frame autocorrelation forces a narrow crossing "
            "between an uncontrolled false-alarm rate and near-total misses, "
            "not the wide margin the simulated streams left. The arrow marks "
            "the operating lag used by every other real-data result."
        ),
        label="tab:real_lag_sweep",
        name="tabr1_lag_sweep",
        align="lrrrc",
    )
    rows2 = []
    for name, rs in alpha_rows.items():
        for r in rs:
            rows2.append([
                labels[name], fmt(r["alpha"], 2), fmt(r["pfa"], 3),
                f"[{fmt(r['pfa_ci'][0], 3)}, {fmt(r['pfa_ci'][1], 3)}]",
                fmt_int(r["arl0"]), fmt(r["miss_rate"], 2), fmt_int(r["lag"]),
            ])
    write_latex_table(
        rows2,
        ["dataset", r"$\alpha$", "PFA", "95\\% CI", "ARL$_0$", "miss rate", "lag"],
        caption=(
            "Time-uniform false-alarm control on real surveillance video at "
            f"the fixed operating lag ({reps} Monte-Carlo streams per level; "
            "official-training-split calibration, real Farneback-flow "
            "backbone scores, real spliced test-split events)."
        ),
        label="tab:real_validity",
        name="tabr1b_validity",
        align="lrrlrrr",
        notes=(
            r"``miss rate'' is over the two real spliced events per stream. "
            r"PFA is the probability a null stream raises at least one alarm."
        ),
    )


def main(reps: int = REPS, lag_reps: int = LAG_REPS):
    print("Experiment R1: time-uniform false-alarm control, real video")
    lag_rows = {}
    for dataset in ("ped2", "avenue"):
        print(f" sweeping lag on {dataset} ({lag_reps} streams per point) ...")
        lag_rows[dataset] = sweep_lag(dataset, lag_reps)
    alpha_rows = {}
    for dataset in ("ped2", "avenue"):
        print(f" sweeping alpha on {dataset} at lag={DATASET_LAG[dataset]} "
              f"({reps} streams per point) ...")
        alpha_rows[dataset] = sweep_alpha(dataset, reps)
    save_json({"lag_sweep": lag_rows, "alpha_sweep": alpha_rows,
               "reps": reps, "lag_reps": lag_reps, "base": BASE,
               "dataset_lag": DATASET_LAG, "dataset_full_lag": DATASET_FULL_LAG,
               "cal_opts": CAL_OPTS}, "exp_r1_validity")
    make_figure(lag_rows, alpha_rows)
    make_table(lag_rows, alpha_rows, reps, lag_reps)
    print(" wrote results/exp_r1_validity.json, fig_r1_validity, tabr1_lag_sweep, tabr1b_validity")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--lag-reps", type=int, default=LAG_REPS, dest="lag_reps")
    main(**vars(ap.parse_args()))

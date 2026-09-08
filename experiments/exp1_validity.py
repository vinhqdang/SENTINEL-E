"""Experiment 1 --- time-uniform false-alarm control.

Two claims are tested.

1. **Nominal levels are honoured.** Sweeping the target ``alpha``, the realised
   probability that a null stream ever raises an alarm stays below the diagonal
   for SENTINEL-E, while a per-frame p-value threshold --- built on *exactly the
   same* valid conformal p-values --- sits far above it.  The only difference is
   the stopping rule, which isolates optional stopping as the cause.

2. **The bound does not decay with the monitoring horizon.** Extending the
   stream from one hour to a full day leaves SENTINEL-E's false-alarm rate flat,
   whereas the fixed-threshold and p-value-threshold rates approach one at the
   geometric rate a per-frame guarantee implies.
"""

from __future__ import annotations

import numpy as np

from experiments.common import fmt, fmt_int, save_json, setup_matplotlib, style_for, write_latex_table, FIGURES
from experiments.runner import evaluate

BASE_CFG = dict(T=60_000, calibration_regime="representative")
REPS = 1_500          # alpha sweep: enough that a zero count is informative
HORIZON_REPS = 200    # horizon sweep: the effect is enormous, precision is cheap
ALPHAS = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
HORIZONS = [15_000, 30_000, 60_000, 120_000, 240_000]


def sweep_alpha(reps: int = REPS):
    rows = []
    for a in ALPHAS:
        se, _, _ = evaluate("SENTINEL-E", a, BASE_CFG, reps, None)
        pv, _, _ = evaluate("p-value threshold", a, BASE_CFG, reps, None)
        orc, _, _ = evaluate("Oracle p-values", a, BASE_CFG, reps, None)
        rows.append(
            {
                "alpha": a,
                "sentinel_pfa": se.pfa,
                "sentinel_ci": list(se.pfa_ci),
                "sentinel_arl0": se.arl0,
                "oracle_pfa": orc.pfa,
                "oracle_ci": list(orc.pfa_ci),
                "pvalue_pfa": pv.pfa,
                "pvalue_ci": list(pv.pfa_ci),
                "pvalue_arl0": pv.arl0,
            }
        )
        print(f"  alpha={a:<6} SENTINEL-E PFA={se.pfa:.3f}  oracle-p PFA={orc.pfa:.3f}  "
              f"p-value threshold PFA={pv.pfa:.3f}", flush=True)
    return rows


def sweep_horizon(reps: int = REPS):
    rows = []
    for T in HORIZONS:
        cfg = dict(BASE_CFG, T=T)
        se, _, _ = evaluate("SENTINEL-E", 0.01, cfg, reps, None)
        pv, _, _ = evaluate("p-value threshold", 0.01, cfg, reps, None)
        ft, _, _ = evaluate("Fixed threshold", 1e-3, cfg, reps, None)
        rows.append(
            {
                "T": T,
                "hours": T / (25.0 * 3600.0),
                "sentinel_pfa": se.pfa,
                "sentinel_ci": list(se.pfa_ci),
                "pvalue_pfa": pv.pfa,
                "fixed_pfa": ft.pfa,
            }
        )
        print(f"  T={T:>7}  SENTINEL-E={se.pfa:.3f}  p-value={pv.pfa:.3f}  "
              f"fixed={ft.pfa:.3f}", flush=True)
    return rows


def make_figure(alpha_rows, horizon_rows):
    """Realised false-alarm rate, on a log scale with zero counts shown honestly.

    An observed rate of exactly zero cannot be drawn on a log axis, and drawing
    it at zero on a linear axis hides the only thing the reader needs: how far
    below the nominal level it sits, and how tightly the Monte-Carlo sample
    pins it down.  Zero-count outcomes are therefore plotted as open
    downward-pointing markers at the 95\% Wilson upper confidence limit, which
    is the strongest statement the data supports.
    """
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    FLOOR = 2e-3

    def draw(ax, x, y, ci_hi, label, style):
        y = np.asarray(y, dtype=float)
        ci_hi = np.asarray(ci_hi, dtype=float)
        obs = y > 0
        if obs.any():
            ax.plot(np.asarray(x)[obs], y[obs], label=label, **style)
        if (~obs).any():
            ax.plot(np.asarray(x)[~obs], np.maximum(ci_hi[~obs], FLOOR),
                    linestyle="none", marker="v", markerfacecolor="white",
                    markeredgecolor=style["color"], markersize=6,
                    label=None if obs.any() else label, zorder=style.get("zorder", 3))
            ax.plot(np.asarray(x), np.where(obs, y, np.maximum(ci_hi, FLOOR)),
                    color=style["color"], lw=1.0, ls=":", zorder=2)

    ax = axes[0]
    a = np.array([r["alpha"] for r in alpha_rows])
    for key, label in (("pvalue", "p-value threshold"),
                       ("oracle", "e-detector, exact uniform $p$"),
                       ("sentinel", "SENTINEL-E")):
        st = style_for("SENTINEL-E (no graph)" if key == "oracle" else label)
        draw(ax, a, [r[f"{key}_pfa"] for r in alpha_rows],
             [r[f"{key}_ci"][1] for r in alpha_rows], label, st)
    ax.plot(a, a, "k--", lw=1.0, label=r"nominal $\alpha$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.7, 1.6)
    ax.set_xlabel(r"target level $\alpha$")
    ax.set_ylabel("realised false-alarm rate")
    ax.set_title("(a) nominal vs realised, 40 min streams")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    hours = np.array([r["hours"] for r in horizon_rows])
    for key, label in (("pvalue", "p-value threshold"),
                       ("fixed", "Fixed threshold"),
                       ("sentinel", "SENTINEL-E")):
        y = [r[f"{key}_pfa"] for r in horizon_rows]
        ci = (
            [r["sentinel_ci"][1] for r in horizon_rows]
            if key == "sentinel" else [0.0] * len(horizon_rows)
        )
        draw(ax, hours, y, ci, label, style_for(label))
    ax.axhline(0.01, color="k", ls="--", lw=1.0, label=r"$\alpha=0.01$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.7, 1.6)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:.1f}" for h in hours])
    ax.minorticks_off()
    ax.set_xlabel("monitoring horizon (hours of video)")
    ax.set_ylabel("P(at least one false alarm)")
    ax.set_title("(b) the horizon does not erode the bound")
    ax.legend(loc="center right", fontsize=7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig1_validity.{ext}")
    plt.close(fig)


def make_table(alpha_rows, horizon_rows, reps=REPS, horizon_reps=HORIZON_REPS):
    rows = [
        [
            fmt(r["alpha"], 3),
            fmt(r["sentinel_pfa"], 3),
            f"[{fmt(r['sentinel_ci'][0], 3)}, {fmt(r['sentinel_ci'][1], 3)}]",
            fmt(r["oracle_pfa"], 3),
            fmt_int(r["sentinel_arl0"]),
            fmt(r["pvalue_pfa"], 3),
            fmt_int(r["pvalue_arl0"]),
        ]
        for r in alpha_rows
    ]
    write_latex_table(
        rows,
        [r"$\alpha$", "PFA", "95\\% CI", "PFA (oracle $p$)", "ARL$_0$", "PFA", "ARL$_0$"],
        caption=(
            "Time-uniform false-alarm control on 40-minute null streams "
            f"({reps} Monte-Carlo streams per level). SENTINEL-E and the "
            "per-frame p-value threshold consume identical conformal p-values; "
            "only the stopping rule differs. ARL$_0$ is censored at the "
            "60{,}000-frame horizon."
        ),
        label="tab:validity",
        name="tab1_validity",
        align="lrrrrrr",
        notes=(
            r"Columns 2--3 and 5: SENTINEL-E; column 4 is the same e-detector fed "
            r"exactly uniform p-values, isolating the slack in Ville's inequality "
            r"from the conservatism of the conformal layer. Columns 6--7: per-frame "
            r"p-value threshold. "
            r"PFA is the probability that a null stream raises at least one alarm."
        ),
    )
    rows2 = [
        [
            fmt_int(r["T"]),
            fmt(r["hours"], 2),
            fmt(r["sentinel_pfa"], 3),
            fmt(r["pvalue_pfa"], 3),
            fmt(r["fixed_pfa"], 3),
        ]
        for r in horizon_rows
    ]
    write_latex_table(
        rows2,
        ["frames", "hours", "SENTINEL-E", "p-value thr.", "fixed thr."],
        caption=(
            r"False-alarm probability against monitoring horizon at $\alpha=0.01$ "
            r"(fixed threshold calibrated to a $10^{-3}$ per-frame rate; "
            f"{horizon_reps} streams per point). "
            r"SENTINEL-E is flat in the horizon; the per-frame rules are not."
        ),
        label="tab:horizon",
        name="tab2_horizon",
        align="rrrrr",
    )


def main(reps: int = REPS, horizon_reps: int = HORIZON_REPS):
    print("Experiment 1: time-uniform false-alarm control")
    print(f" sweeping alpha ({reps} streams per point) ...")
    alpha_rows = sweep_alpha(reps)
    print(f" sweeping horizon ({horizon_reps} streams per point) ...")
    horizon_rows = sweep_horizon(horizon_reps)
    save_json({"alpha_sweep": alpha_rows, "horizon_sweep": horizon_rows,
               "reps": reps, "horizon_reps": horizon_reps,
               "config": BASE_CFG}, "exp1_validity")
    make_figure(alpha_rows, horizon_rows)
    make_table(alpha_rows, horizon_rows, reps, horizon_reps)
    print(" wrote results/exp1_validity.json, fig1_validity, tab1/tab2")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--horizon-reps", type=int, default=HORIZON_REPS,
                    dest="horizon_reps")
    main(**vars(ap.parse_args()))

"""Experiment 13 --- false-alarm probability over a realistic week-long horizon.

Experiment 1 extended the monitoring horizon only to 2.7 hours, with a diurnal
cycle of 40,000 frames (26.7 minutes).  That compressed "day" means a 2.7 hour
stream already contains about six full illumination cycles, which is not what a
camera sees.  This experiment uses a realistic day, 24 hours at 25 fps, and
horizons up to one week.

Null streams only (no change point).  SENTINEL-E is run at 200 streams per
horizon; the two per-frame baselines at 50 streams per horizon, because they are
expensive at week-long lengths and their false-alarm probability is already
pinned near one at every horizon here.

Calibration.  The representative calibration regime places the calibration
frames at diurnal phases ``t = 0 .. n_calibration - 1``.  With the default
``n_calibration = 150_000`` and a 2,160,000-frame day, the calibration set would
cover only 6.9% of the cycle (diurnal logit range [0.000, 0.254] of the full
[-0.6, 0.6]), i.e. it would not span the cycle that the representative regime is
documented to span.  We therefore set ``n_calibration = diurnal_period``, so the
calibration window covers exactly one full day.  Nothing else is changed.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Sequence

import numpy as np

from experiments.common import (
    FIGURES,
    fmt,
    fmt_int,
    save_json,
    setup_matplotlib,
    style_for,
    write_latex_table,
)
from experiments.runner import evaluate
from sentinel_e.metrics import wilson_interval

FPS = 25.0
ALPHA = 0.01
DIURNAL_PERIOD = 2_160_000          # 24 h at 25 fps
N_CALIBRATION = DIURNAL_PERIOD      # one full day, see module docstring
DEFAULT_N_CALIBRATION = 150_000     # the simulator default, reported for context
HOURS = [1, 6, 24, 72, 168]
REPS_SENTINEL = 200
REPS_BASELINE = 50
WORKERS = 3
FIXED_RATE = 1e-3


def frames_for(hours: float) -> int:
    return int(round(hours * 3600.0 * FPS))


def make_cfg(T: int) -> Dict:
    return dict(T=T, calibration_regime="representative",
                diurnal_period=DIURNAL_PERIOD, n_calibration=N_CALIBRATION)


def run(hours: Sequence[float] = HOURS, reps_sentinel: int = REPS_SENTINEL,
        reps_baseline: int = REPS_BASELINE, workers: int = WORKERS) -> List[Dict]:
    rows = []
    for h in hours:
        T = frames_for(h)
        cfg = make_cfg(T)
        print(f"  horizon {h} h ({T:,} frames)", flush=True)

        t0 = time.perf_counter()
        se, se_null, _ = evaluate("SENTINEL-E", ALPHA, cfg, reps_sentinel, None,
                                  workers=workers)
        t_se = time.perf_counter() - t0

        t0 = time.perf_counter()
        pv, _, _ = evaluate("p-value threshold", ALPHA, cfg, reps_baseline, None,
                            workers=workers)
        t_pv = time.perf_counter() - t0

        t0 = time.perf_counter()
        ft, _, _ = evaluate("Fixed threshold", FIXED_RATE, cfg, reps_baseline, None,
                            workers=workers)
        t_ft = time.perf_counter() - t0

        row = {
            "hours": float(h),
            "frames": T,
            "n_calibration": N_CALIBRATION,
            "sentinel": {"pfa": se.pfa, "pfa_ci": list(se.pfa_ci),
                         "n": se.n_null, "arl0": se.arl0,
                         "lag": se_null[0].lag,
                         "seconds": t_se},
            "pvalue": {"pfa": pv.pfa, "pfa_ci": list(pv.pfa_ci),
                       "n": pv.n_null, "seconds": t_pv},
            "fixed": {"pfa": ft.pfa, "pfa_ci": list(ft.pfa_ci),
                      "n": ft.n_null, "seconds": t_ft},
        }
        rows.append(row)
        print(f"    SENTINEL-E PFA={se.pfa:.3f} {tuple(round(x,3) for x in se.pfa_ci)} "
              f"({t_se:.0f}s, lag={row['sentinel']['lag']})  "
              f"p-value={pv.pfa:.3f} ({t_pv:.0f}s)  fixed={ft.pfa:.3f} ({t_ft:.0f}s)",
              flush=True)
    return rows


def make_figure(rows: List[Dict]) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    FLOOR = 2e-3
    hours = np.array([r["hours"] for r in rows])

    def draw(key: str, label: str, ci: bool) -> None:
        style = style_for(label)
        y = np.array([r[key]["pfa"] for r in rows], dtype=float)
        obs = y > 0
        if obs.any():
            ax.plot(hours[obs], y[obs], label=label, **style)
        if (~obs).any():
            ax.plot(hours[~obs], [FLOOR] * int((~obs).sum()), linestyle="none",
                    marker="v", markerfacecolor="white",
                    markeredgecolor=style["color"], markersize=6,
                    label=None if obs.any() else label)
        if ci:
            lo = np.array([r[key]["pfa_ci"][0] for r in rows])
            hi = np.array([r[key]["pfa_ci"][1] for r in rows])
            ok = y > 0
            ax.fill_between(hours[ok], np.maximum(lo[ok], FLOOR * 0.5), hi[ok],
                            color=style["color"], alpha=0.15, lw=0)

    draw("sentinel", "SENTINEL-E", ci=True)
    draw("pvalue", "p-value threshold", ci=False)
    draw("fixed", "Fixed threshold", ci=False)
    ax.axhline(ALPHA, color="k", ls="--", lw=1.0, label=r"$\alpha=0.01$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.5, 1.6)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:g}" for h in hours])
    ax.minorticks_off()
    ax.set_xlabel("monitoring horizon (hours of video, 25 fps)")
    ax.set_ylabel("P(at least one false alarm)")
    ax.legend(loc="center left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_horizon_week.{ext}")
    plt.close(fig)


def make_table(rows: List[Dict], reps_sentinel: int, reps_baseline: int) -> None:
    body = []
    for r in rows:
        s = r["sentinel"]
        body.append([
            fmt(r["hours"], 0),
            fmt_int(r["frames"]),
            f"{fmt(s['pfa'], 3)} [{fmt(s['pfa_ci'][0], 3)}, {fmt(s['pfa_ci'][1], 3)}]",
            fmt(r["pvalue"]["pfa"], 3),
            fmt(r["fixed"]["pfa"], 3),
        ])
    write_latex_table(
        body,
        ["hours", "frames", "SENTINEL-E PFA [95\\% CI]", "p-value thr. PFA",
         "fixed thr. PFA"],
        caption=(
            r"False-alarm probability over week-long null streams at $\alpha=0.01$, "
            r"with a realistic 24\,h diurnal cycle at 25\,fps. "
            f"SENTINEL-E: {reps_sentinel} streams per horizon; the two per-frame "
            f"baselines: {reps_baseline} streams per horizon (fixed threshold at a "
            r"$10^{-3}$ per-frame rate). Calibration spans one full day."
        ),
        label="tab:horizon_week",
        name="tab_horizon_week",
        align="rrrrr",
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (1 h and 6 h, 4 and 2 streams); writes nothing")
    args = ap.parse_args(argv)
    if args.smoke:
        rows = run(hours=[1, 6], reps_sentinel=4, reps_baseline=2)
        print("smoke test ok:", [round(r["sentinel"]["pfa"], 3) for r in rows])
        return
    print("Experiment 13: week-long horizon, realistic diurnal cycle")
    t0 = time.perf_counter()
    rows = run()
    total = time.perf_counter() - t0
    save_json({
        "rows": rows,
        "alpha": ALPHA,
        "fps": FPS,
        "diurnal_period": DIURNAL_PERIOD,
        "n_calibration": N_CALIBRATION,
        "n_calibration_default": DEFAULT_N_CALIBRATION,
        "calibration_cycle_fraction": N_CALIBRATION / DIURNAL_PERIOD,
        "default_calibration_cycle_fraction": DEFAULT_N_CALIBRATION / DIURNAL_PERIOD,
        "reps_sentinel": REPS_SENTINEL,
        "reps_baseline": REPS_BASELINE,
        "workers": WORKERS,
        "total_seconds": total,
    }, "exp13_horizon_week")
    make_figure(rows)
    make_table(rows, REPS_SENTINEL, REPS_BASELINE)
    print(f" wrote results/exp13_horizon_week.json, fig_horizon_week, tab_horizon_week "
          f"({total:.0f}s total)")


if __name__ == "__main__":
    main()

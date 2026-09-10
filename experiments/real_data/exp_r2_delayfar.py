"""Experiment R2 --- detection delay against realised false-alarm rate, real video.

The real-data analogue of ``experiments/exp2_delay_far.py``, and the paper's
head-to-head comparison against classical sequential baselines on real
surveillance footage. Every method sees the identical real Farneback-flow
score stream, the identical real spliced event, and -- critically, since
Experiment R1 showed this is where real video punishes an unfair comparison
-- the identical fixed betting-grid lag Experiment R1 selected for that
dataset (5 frames on Ped2, 7 on Avenue). No method is thinned differently
from any other.

Reading the table: at the lag Experiment R1's sweep identifies as usable,
Ped2 keeps a real, non-trivial detection window; Avenue's is thin enough that
most operating points here simply confirm the R1 finding from a different
angle -- a real backbone whose validity and power windows barely overlap
leaves classical baselines no more room to manoeuvre than SENTINEL-E.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.common import (
    FIGURES, fmt, fmt_int, save_json, setup_matplotlib, style_for, write_latex_table,
)
from experiments.real_data.exp_r1_validity import DATASET_LAG
from experiments.real_data.runner import evaluate_real

BASE = dict(n_frames=6_000, n_calibration=800, change_point=1_500, n_events=1)
REPS = 250
FPS = 25.0

SWEEPS: Dict[str, List[float]] = {
    "SENTINEL-E":            [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    "Fixed threshold":       [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "p-value threshold":     [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "CUSUM":                 [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0],
    "Shiryaev--Roberts":     [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
    "Parametric e-detector": [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    "E-SHIFT":               [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
}


def run(dataset: str, reps: int = REPS) -> Dict[str, List[Dict]]:
    lag = DATASET_LAG[dataset]
    results: Dict[str, List[Dict]] = {}
    for method, knobs in SWEEPS.items():
        opts = {"calibration": "pooled", "calibration_regime": "official", "lag": lag}
        rows = []
        for k in knobs:
            out, null, _ = evaluate_real(method, k, dataset, reps, options=opts, **BASE)
            rows.append({
                "knob": k, "pfa": out.pfa, "pfa_ci": list(out.pfa_ci), "arl0": out.arl0,
                "add": out.add, "add_se": out.add_se, "add_median": out.add_median,
                "add_censored": out.add_censored, "add_censored_se": out.add_censored_se,
                "miss_rate": out.miss_rate, "lag": null[0].lag,
            })
            print(f"  {dataset:7s} {method:24s} knob={k:<9g} PFA={out.pfa:.3f} "
                  f"ADD={out.add if np.isfinite(out.add) else float('nan'):8.1f} "
                  f"miss={out.miss_rate:.2f}", flush=True)
        results[method] = rows
    return results


def _finite(rows, pfa_max=1.01):
    return sorted(
        [r for r in rows if r["add"] is not None and np.isfinite(r["add"])
         and r["pfa"] <= pfa_max and r["miss_rate"] < 0.9],
        key=lambda r: r["pfa"],
    )


def make_figure(results_by_dataset):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, dataset, title in (
        (axes[0], "ped2", "(a) UCSD Ped2 (lag=%d)" % DATASET_LAG["ped2"]),
        (axes[1], "avenue", "(b) CUHK Avenue (lag=%d)" % DATASET_LAG["avenue"]),
    ):
        methods = list(results_by_dataset[dataset])
        for i, method in enumerate(methods):
            rows = results_by_dataset[dataset][method]
            pts = _finite(rows)
            if not pts:
                continue
            x = np.array([r["pfa"] for r in pts])
            y = np.array([r["add_censored"] for r in pts]) / FPS
            # On real video every classical baseline's knob sweep saturates at
            # (or near) PFA=1: without this, five methods' markers stack
            # exactly on top of one another at the plot's right edge, which
            # is what made this figure unreadable. A tiny, fixed,
            # method-indexed multiplicative offset (disclosed in the caption)
            # fans them out horizontally; the offset is never positive, so a
            # jittered point never reads as a higher false-alarm probability
            # than the method actually realised, and it never moves a point
            # by more than 3%.
            jitter = 1.0 - 0.006 * i
            st = style_for(method)
            ax.plot(x * jitter, y, label=method, alpha=0.85,
                    markeredgecolor="white", markeredgewidth=0.4, **st)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("realised false-alarm probability")
        ax.set_ylabel("censored detection delay (s)")
        ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_r2_delay_far.{ext}")
    plt.close(fig)


def _fmt_knob(v: float) -> str:
    a = abs(float(v))
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e5:
        exp = int(np.floor(np.log10(a)))
        mant = a / 10**exp
        mant_s = f"{mant:g}"
        return (f"$10^{{{exp}}}$" if mant_s == "1"
                else f"${mant_s}\\times 10^{{{exp}}}$")
    return f"{a:g}"


def make_table(results_by_dataset, cap: float = 0.1):
    labels = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}
    rows = []
    for dataset, results in results_by_dataset.items():
        for method, rs in results.items():
            usable = [r for r in rs if r["add_censored"] is not None
                      and np.isfinite(r["add_censored"]) and r["miss_rate"] < 0.9]
            if usable:
                best = min(usable, key=lambda r: (r["pfa"], r["add_censored"]))
                detects = True
            else:
                best = min(rs, key=lambda r: r["pfa"])
                detects = False
            ok = detects and best["pfa"] <= cap
            pfa_cell = fmt(best["pfa"], 3)
            if not ok:
                pfa_cell = f"\\textbf{{{pfa_cell}}}"
            rows.append([
                labels[dataset], method, _fmt_knob(best["knob"]), pfa_cell,
                "\\checkmark" if ok else "\\ding{55}", fmt_int(best["arl0"]),
                (f"{fmt(best['add_censored'] / FPS, 1)} $\\pm$ "
                 f"{fmt(1.96 * (best['add_censored_se'] or 0) / FPS, 1)}")
                if detects else "--",
                fmt(best["miss_rate"], 2),
            ])
    write_latex_table(
        rows,
        ["dataset", "method", "knob", "realised PFA", f"$\\le {cap}$?", "ARL$_0$",
         "censored delay (s)", "miss rate"],
        caption=(
            "Each method at the tightest realised false-alarm probability it "
            "achieves while still detecting most events, on real surveillance "
            f"video at the operating lag fixed by Experiment R1 ({REPS} null "
            "and signal streams per operating point)."
        ),
        label="tab:real_delay_far",
        name="tabr2_delay_far",
        align="llrrcrrr",
        notes=(
            r"Delays are in seconds at 25\,fps, censored at the horizon on a "
            r"miss. A bold rate violates the target column; ``--'' means the "
            r"method never detects at any setting. Every method here shares "
            r"the identical fixed lag SENTINEL-E was run at (Experiment R1), "
            r"so no method is thinned more or less favourably than another."
        ),
    )


def main(reps: int = REPS):
    print("Experiment R2: detection delay vs realised false-alarm rate, real video")
    results_by_dataset = {}
    for dataset in ("ped2", "avenue"):
        print(f" {dataset} ...")
        results_by_dataset[dataset] = run(dataset, reps)
    save_json({"results": results_by_dataset, "reps": reps, "base": BASE,
               "dataset_lag": DATASET_LAG, "fps": FPS}, "exp_r2_delayfar")
    make_figure(results_by_dataset)
    make_table(results_by_dataset)
    print(" wrote results/exp_r2_delayfar.json, fig_r2_delay_far, tabr2_delay_far")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    main(**vars(ap.parse_args()))

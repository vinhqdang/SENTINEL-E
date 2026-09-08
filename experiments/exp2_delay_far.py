"""Experiment 2 --- detection delay against realised false-alarm rate.

The headline result.  Every method is swept over its own tuning knob, and each
operating point is plotted at the false-alarm rate it *actually achieves* rather
than the one it nominally targets.  This is the only fair comparison: a method
whose nominal level is meaningless (because its modelling assumptions fail on
real detector scores) would otherwise appear to dominate simply by lying about
its own error rate.

Reading the figure: lower and further left is better.  A method is only
admissible in the operating region a municipality can live with --- roughly one
false dispatch per week of monitoring or fewer --- which is the left-hand edge.
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
    style_for,
    write_latex_table,
)
from experiments.runner import evaluate

BASE_CFG = dict(T=90_000, calibration_regime="representative",
                n_episodes=4, event_length=800, episode_gap=6_000,
                event_intermittency=0.35)
CHANGE_POINT = 20_000
REPS = 200
FPS = 25.0

#: Each method's knob sweep, chosen to span the same realised-PFA range.
SWEEPS: Dict[str, List[float]] = {
    "SENTINEL-E":            [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    "Fixed threshold":       [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "p-value threshold":     [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "CUSUM":                 [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0],
    "Shiryaev--Roberts":     [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
    "Parametric e-detector": [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    "E-SHIFT":               [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
}


def run(reps: int = REPS) -> Dict[str, List[Dict]]:
    results: Dict[str, List[Dict]] = {}
    for method, knobs in SWEEPS.items():
        rows = []
        for k in knobs:
            out, null, _ = evaluate(method, k, BASE_CFG, reps, CHANGE_POINT)
            rows.append(
                {
                    "knob": k,
                    "pfa": out.pfa,
                    "pfa_ci": list(out.pfa_ci),
                    "arl0": out.arl0,
                    "add": out.add,
                    "add_se": out.add_se,
                    "add_median": out.add_median,
                    "add_censored": out.add_censored,
                    "add_censored_se": out.add_censored_se,
                    "miss_rate": out.miss_rate,
                    "lag": null[0].lag,
                }
            )
            print(f"  {method:24s} knob={k:<9g} PFA={out.pfa:.3f} "
                  f"ADD={out.add if np.isfinite(out.add) else float('nan'):8.1f} "
                  f"miss={out.miss_rate:.2f}", flush=True)
        results[method] = rows
    return results


def _finite(rows, pfa_max=1.01):
    """Operating points with a measurable delay, sorted by realised PFA."""
    pts = [
        r for r in rows
        if r["add"] is not None and np.isfinite(r["add"]) and r["pfa"] <= pfa_max
        and r["miss_rate"] < 0.9
    ]
    return sorted(pts, key=lambda r: r["pfa"])


def make_figure(results):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for ax, xkey, xlabel, title in (
        (axes[0], "pfa", "realised false-alarm probability over 40 min",
         "(a) delay vs false-alarm probability"),
        (axes[1], "arl0", r"realised ARL$_0$ (frames to a false alarm)",
         "(b) delay vs mean time between false alarms"),
    ):
        for method, rows in results.items():
            pts = _finite(rows)
            if not pts:
                continue
            x = np.array([r[xkey] for r in pts])
            y = np.array([r["add_censored"] for r in pts]) / FPS
            se = np.array(
                [r["add_censored_se"] if r["add_censored_se"] else 0.0 for r in pts]
            ) / FPS
            st = style_for(method)
            if xkey == "arl0":
                order = np.argsort(x)
                x, y, se = x[order], y[order], se[order]
            ax.plot(x, y, label=method, **st)
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se,
                            color=st["color"], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("censored detection delay (s)")
        ax.set_title(title)
    axes[0].axvline(0.01, color="k", ls=":", lw=1.0)
    axes[0].text(0.011, axes[0].get_ylim()[1] * 0.95, r"$\alpha=0.01$",
                 fontsize=7, va="top")
    axes[0].legend(loc="upper right", ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig2_delay_far.{ext}")
    plt.close(fig)


def _fmt_knob(v: float) -> str:
    """Knobs span 1e-5 to 1e8, so pick a readable form per magnitude."""
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


def make_table(results, cap: float = 0.05):
    """One row per method at the tightest false-alarm rate it can still detect at.

    Two selection rules are wrong here and worth naming. Minimising delay
    outright picks every baseline's loosest knob, which buys a five-second delay
    at a false-alarm probability of one --- true, but not an operating point any
    municipality would run. Requiring the cap and blanking the rest hides how
    the baselines fail. So each method is shown at the smallest realised
    false-alarm probability it achieves while still detecting most events, which
    is the question an operator actually asks: if I insist on few false alarms,
    what does this method cost me in delay? The full trade-off is in
    \\cref{fig:delayfar}.
    """
    rows = []
    for method, rs in results.items():
        usable = [r for r in rs
                  if r["add_censored"] is not None
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
            method,
            _fmt_knob(best["knob"]),
            pfa_cell,
            "\\checkmark" if ok else "\\ding{55}",
            fmt_int(best["arl0"]),
            fmt(best["add"] / FPS, 1) if detects else "--",
            (f"{fmt(best['add_censored'] / FPS, 1)} $\\pm$ "
             f"{fmt(1.96 * (best['add_censored_se'] or 0) / FPS, 1)}")
            if detects else "--",
            fmt(best["miss_rate"], 2),
        ])
    write_latex_table(
        rows,
        ["method", "knob", "realised PFA", f"$\\le {cap}$?", "ARL$_0$",
         "cond. delay (s)", "censored delay (s)", "miss rate"],
        caption=(
            "Each method at the tightest realised false-alarm probability it "
            "achieves while still detecting most events, over a 60-minute stream "
            f"({REPS} null and {REPS} post-change streams per operating point). "
            "This is the operator's question: insisting on few false alarms, what "
            "does the method cost in delay? SENTINEL-E is the only one that "
            f"reaches a rate of at most {cap} at all. The classical procedures "
            "can be made fast, but only at a false-alarm probability near one; "
            "no setting of their knob moves them into the operating region. The "
            "full trade-off is in Figure~\\ref{fig:delayfar}."
        ),
        label="tab:delay_far",
        name="tab3_delay_far",
        align="llrcrrrr",
        notes=(
            r"Delays are in seconds at 25\,fps. The conditional delay averages "
            r"only over runs that detect, so it is computed on a different "
            r"subset for each method and flatters those that miss the hard "
            r"events; the censored delay charges a miss the whole remaining "
            r"horizon and is the comparable figure. A bold rate violates the "
            r"constraint in the fourth column; ``--'' means the method never "
            r"detects at any setting."
        ),
    )


def main(reps: int = REPS):
    print("Experiment 2: detection delay vs realised false-alarm rate")
    results = run(reps)
    save_json({"results": results, "reps": reps, "config": BASE_CFG,
               "change_point": CHANGE_POINT, "fps": FPS}, "exp2_delay_far")
    make_figure(results)
    make_table(results)
    print(" wrote results/exp2_delay_far.json, fig2_delay_far, tab3")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    main(**vars(ap.parse_args()))

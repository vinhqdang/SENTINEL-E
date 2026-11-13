"""Experiment R2 (protocol v2) --- detection delay against false-alarm rate, real video.

The v2 counterpart of the head-to-head comparison with classical sequential
baselines. Every method runs at the lag L* that Experiment R1 selected for
that dataset from *training* videos only, on the same real streams:

* null streams: permutations of the held-out training group ``null``;
* test streams: permutations of the real labelled test videos, every video
  once per stream, scored with :func:`protocol_v2.score_labelled_stream`.

Knob sweeps are the same as in the v1 file. For each method, the table reports
the knob with the smallest null false-alarm probability among those whose
test episode detection rate is at least 0.10 (ties in null PFA broken by the
higher detection rate). If no knob reaches 0.10 detection, the knob with the
smallest null PFA is reported and its delay is shown as ``--``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import numpy as np

from experiments.common import (
    FIGURES, fmt, fmt_int, load_json, parallel_map, save_json, setup_matplotlib,
    style_for, write_latex_table,
)
from experiments.real_data.protocol_v2 import (
    FPS, GRACE_FRAMES, make_stream, rate, rng_for, run_alarms, score_labelled_stream,
    state, test_summary,
)
from sentinel_e.datasets.real_vad import concat_whole_videos

DATASETS = ("ped2", "avenue")
LABELS = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}
REPS = 200
DETECT_FLOOR = 0.10
#: Floor used to draw zero false-alarm rates on the log-scale figure.
FA_FLOOR = 0.1

SWEEPS: Dict[str, List[float]] = {
    "SENTINEL-E":            [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    "Fixed threshold":       [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "p-value threshold":     [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "CUSUM":                 [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0],
    "Shiryaev--Roberts":     [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
    "Parametric e-detector": [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    "E-SHIFT":               [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
}


def _job(args):
    """One stream of one method at one knob. Returns a bool (null) or a summary dict (test)."""
    dataset, method, knob, lag, kind, rep = args
    s = state(dataset)
    if kind == "null":
        scores, _, _ = concat_whole_videos(s["groups"]["null"], rng_for(dataset, "null", rep))
        st = make_stream(scores, np.zeros(scores.size, dtype=bool), [], s["cal"])
        return bool(run_alarms(method, knob, st, lag).size > 0)
    scores, labels, eps = concat_whole_videos(
        [v for v, _ in s["tv"]], rng_for(dataset, "test", rep), [l for _, l in s["tv"]],
    )
    st = make_stream(scores, labels, eps, s["cal"])
    idx = run_alarms(method, knob, st, lag)
    return score_labelled_stream(idx, labels, eps, GRACE_FRAMES, FPS)


def run_dataset(dataset: str, lag: int, reps: int) -> Dict[str, List[Dict]]:
    state(dataset)  # warm the cache in the parent process
    jobs, keys = [], []
    for method, knobs in SWEEPS.items():
        for k in knobs:
            for r in range(reps):
                jobs.append((dataset, method, k, lag, "null", r))
                keys.append((method, k, "null"))
                jobs.append((dataset, method, k, lag, "test", r))
                keys.append((method, k, "test"))
    print(f"  {dataset}: {len(jobs)} stream runs at lag={lag}", flush=True)
    out = parallel_map(_job, jobs, desc=f"R2 {dataset}")

    results: Dict[str, List[Dict]] = {}
    for method, knobs in SWEEPS.items():
        rows = []
        for k in knobs:
            nulls = [o for o, kk in zip(out, keys) if kk == (method, k, "null")]
            tests = [o for o, kk in zip(out, keys) if kk == (method, k, "test")]
            null_rate = rate(sum(nulls), len(nulls))
            summ = test_summary(tests)
            rows.append({"knob": k, "null_pfa": null_rate["pfa"], "null_ci": null_rate["ci"],
                         "null_k": null_rate["k"], "null_n": null_rate["n"], **summ})
            print(f"  {dataset:7s} {method:22s} knob={k:<9g} nullPFA={null_rate['pfa']:.3f} "
                  f"det={summ['episode_detection_rate']:.3f} "
                  f"FA/h={summ['false_alarms_per_hour']:.1f}", flush=True)
        results[method] = rows
    return results


def select_knob(rows: List[Dict]) -> Dict:
    """Smallest null PFA among knobs with test detection >= DETECT_FLOOR (ties: higher detection)."""
    usable = [r for r in rows if np.isfinite(r["episode_detection_rate"])
              and r["episode_detection_rate"] >= DETECT_FLOOR]
    if usable:
        best = min(usable, key=lambda r: (r["null_pfa"], -r["episode_detection_rate"]))
        return {**best, "detects": True}
    best = min(rows, key=lambda r: (r["null_pfa"], -(r["episode_detection_rate"] or 0.0)))
    return {**best, "detects": False}


def make_figure(payload: Dict) -> None:
    plt = setup_matplotlib()
    datasets = list(payload["results"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.1), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for method, rows in payload["results"][ds].items():
            pts = [r for r in rows if np.isfinite(r["false_alarms_per_hour"])]
            if not pts:
                continue
            x = np.array([max(r["false_alarms_per_hour"], FA_FLOOR) for r in pts])
            y = np.array([r["episode_detection_rate"] for r in pts])
            ax.plot(x, y, label=method, alpha=0.9, markeredgecolor="white",
                    markeredgewidth=0.4, **style_for(method))
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(DETECT_FLOOR, color="k", lw=0.7, ls=":")
        ax.set_xlabel(f"false alarms per hour, real normal footage (zero drawn at {FA_FLOOR:g})")
        ax.set_ylabel("episode detection rate")
        ax.set_title(f"{LABELS.get(ds, ds)} (lag $L^\\ast$={payload['lag'][ds]})")
    axes[0][0].legend(loc="lower right", fontsize=6.3)
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
        return f"$10^{{{exp}}}$" if mant_s == "1" else f"${mant_s}\\times 10^{{{exp}}}$"
    return f"{a:g}"


def make_table(payload: Dict) -> None:
    rows = []
    for ds, res in payload["results"].items():
        for method, rs in res.items():
            best = select_knob(rs)
            ok = best["null_pfa"] <= 0.1
            pfa_cell = fmt(best["null_pfa"], 3)
            if not ok:
                pfa_cell = f"\\textbf{{{pfa_cell}}}"
            det = best["detects"]
            rows.append([
                LABELS.get(ds, ds), method, _fmt_knob(best["knob"]), pfa_cell,
                "\\checkmark" if ok else "\\ding{55}",
                fmt(best["episode_detection_rate"], 2),
                fmt(best["mean_delay_s"], 1) if det else "--",
                fmt(best["false_alarms_per_hour"], 2),
            ])
    write_latex_table(
        rows,
        ["dataset", "method", "knob", "null PFA", r"$\le 0.1$?", "episode det.",
         "mean delay (s)", "FA / hour"],
        caption=(
            "Each method at the knob with the smallest null false-alarm "
            "probability among knobs that detect at least 10\\% of real test "
            f"episodes, at the lag $L^\\ast$ selected on training videos in "
            f"Experiment R1 ({REPS} null and {REPS} test streams per knob)."
        ),
        label="tab:r2_delay_far", name="tabr2_delay_far", align="llrrcrrr",
        notes=(
            r"Delays are means over detected test episodes. ``--'' in the delay column means the method reached no knob with "
            r"detection $\geq 0.10$; its row then shows the smallest-PFA knob. "
            r"Ties in null PFA are broken by the higher detection rate. Null "
            r"PFA is over held-out training-video streams; detection and "
            r"false-alarm rate are over the real labelled test split. Bold PFA "
            r"exceeds 0.1."
        ),
    )


def main(reps: int = REPS, datasets: Sequence[str] = DATASETS):
    print("Experiment R2 (v2): delay vs false-alarm rate on real video")
    r1 = load_json("exp_r1_validity")
    lag = {ds: int(r1["datasets"][ds]["selection"]["selected_lag"]) for ds in datasets}
    results = {}
    for ds in datasets:
        print(f" {ds} (L*={lag[ds]}) ...", flush=True)
        results[ds] = run_dataset(ds, lag[ds], reps)
    payload = {"reps": reps, "lag": lag, "fps": FPS, "grace_frames": GRACE_FRAMES,
               "detect_floor": DETECT_FLOOR, "fa_floor": FA_FLOOR, "sweeps": SWEEPS,
               "results": results}
    save_json(payload, "exp_r2_delayfar")
    make_figure(payload)
    make_table(payload)
    print(" wrote results/exp_r2_delayfar.json, fig_r2_delay_far, tabr2_delay_far")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    main(reps=a.reps, datasets=tuple(x.strip() for x in a.datasets.split(",") if x.strip()))

"""Experiment R3b (descriptive) --- detection-versus-false-alarm frontier on the test videos.

Companion to :mod:`exp_r3_per_video`. The selected-knob comparison of R3 mixes
power and false alarms: a baseline that detects more also raises more false
alarms. This script removes the selection step and reads every method's whole
operating curve off the TEST videos, so the methods can be compared at matched
realised per-video false-alarm levels.

IMPORTANT: the knobs here are read off the test set. They are not a deployable
selection. The deployable comparison is the test table of exp_r3 (``tab:r3_test``).

Protocol (identical to the test protocol of R3, one video is one monitored stream):

* Calibration for test is all training videos (as in R3).
* Baselines run at every knob of :data:`exp_r2_delayfar.SWEEPS` at the lag
  ``L*(0.1)`` selected on training videos by LOVO (or 10-fold for Mivia-IWDD),
  exactly as R3 does for every method.
* SENTINEL-E runs at every knob of its sweep, which is its alpha grid, each at its
  own lag ``L*(alpha)``. The lag is selected on training videos only, by the same
  LOVO rule as R3. The grid includes the R3 alphas 0.05, 0.1 and 0.2.
* Per-video false-alarm proportion is the null-trial proportion of R3
  (``null_primary``); detection is the episode proportion (``episodes``).

Every unit is cached under ``results/cache/exp_r3b/``. A unit whose spec is
already cached by R3 (``results/cache/exp_r3/``, same key, same code) is copied
across instead of recomputed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.common import (
    FIGURES, RESULTS, TABLES, fmt, parallel_map, save_json, setup_matplotlib,
    style_for, write_latex_table,
)
from experiments.real_data import exp_r3_per_video as r3
from experiments.real_data.exp_r2_delayfar import SWEEPS, _fmt_knob
from sentinel_e.datasets.real_vad import DATA_ROOT

#: Datasets in the order of the R3 table, plus mivia_supervised when its data exist.
DATASETS = list(r3.DATASETS) + (["mivia_supervised"]
                                if (DATA_ROOT / "mivia_supervised" / "backbone_auc.json").exists() else [])
LABELS = dict(r3.LABELS, mivia_supervised="Mivia-IWDD (supervised)")
BASELINES = list(r3.BASELINES)
SENT = "SENTINEL-E"
#: SENTINEL-E frontier knobs: its alpha grid (includes R3 alphas 0.05, 0.1, 0.2).
SENT_ALPHAS = sorted(set(SWEEPS[SENT]) | set(r3.ALPHAS))
HEADLINE = r3.HEADLINE
#: Upper limit of the partial area under the detection-versus-FA curve.
PAUC_UPPER = 0.2
#: Per-video FA cap for the highest-detection summary.
FA_CAP = 0.1
CACHE_DIR = RESULTS / "cache" / "exp_r3b"
R3_CACHE = RESULTS / "cache" / "exp_r3"
TABLE_NAME = "tabr3b_frontier"
FIG_NAME = "fig_r3b_frontier"


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def best_at_fa(points: Sequence[Dict], cap: float = FA_CAP) -> Optional[Dict]:
    """Highest-detection point with per-video FA proportion at most ``cap``.

    Ties in detection go to the point with the lower FA proportion, then to the
    earlier knob. Returns None when no point is admissible.
    """
    ok = [p for p in points if p["fa"] is not None and p["fa"] <= cap + 1e-9]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["det"], -p["fa"], -p["order"]))


def envelope_area(points: Sequence[Dict], upper: float = PAUC_UPPER) -> float:
    """Normalised area under the empirical detection-versus-FA envelope on [0, upper].

    The envelope is the step function ``D(x) = max{det_i : fa_i <= x}`` over the
    method's knobs, with ``D(x) = 0`` where no knob reaches FA ``x``. The area is
    divided by ``upper`` (so the result lies in [0, 1]). Because the knobs are
    discrete, this is a lower bound on the area under the continuous frontier.
    """
    pts = [(p["fa"], p["det"]) for p in points if p["fa"] is not None and p["fa"] <= upper + 1e-12]
    if not pts:
        return 0.0
    xs = sorted({round(fa, 12) for fa, _ in pts})
    area = 0.0
    for i, x in enumerate(xs):
        nxt = xs[i + 1] if i + 1 < len(xs) else upper
        d = max(det for fa, det in pts if fa <= x + 1e-12)
        area += d * (min(nxt, upper) - x)
    return float(area / upper)


def dominators(point: Dict, others: Sequence[Dict]) -> List[Dict]:
    """Points with FA at most ``point``'s FA and detection strictly above it."""
    return [o for o in others
            if o["fa"] is not None and point["fa"] is not None
            and o["fa"] <= point["fa"] + 1e-12 and o["det"] > point["det"] + 1e-12]


def frontier_gap(point: Dict, others: Sequence[Dict]) -> float:
    """Best detection of ``others`` at FA at most ``point``'s FA, minus ``point``'s detection.

    Positive: the point lies below the empirical frontier of ``others``.
    Zero or negative: the point is on or above it.
    """
    best = [o["det"] for o in others
            if o["fa"] is not None and o["fa"] <= point["fa"] + 1e-12]
    return (max(best) if best else 0.0) - point["det"]


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
def seed_from_r3(specs: Sequence[Dict]) -> int:
    """Copy R3 cache files for specs R3 already computed (same key, same code)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for s in specs:
        key = r3._key(s)
        dst, src = CACHE_DIR / f"{key}.json", R3_CACHE / f"{key}.json"
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def run_units(specs: Sequence[Dict], desc: str, workers: int) -> Dict[str, Dict]:
    unique: Dict[str, Dict] = {}
    for s in specs:
        unique.setdefault(r3._key(s), s)
    seeded = seed_from_r3(list(unique.values()))
    keys = list(unique)
    args = [(str(CACHE_DIR), k, unique[k]) for k in keys]
    print(f"  {desc}: {len(keys)} units ({seeded} copied from R3 cache)", flush=True)
    outs = parallel_map(r3._unit, args, workers=workers, desc=desc)
    return dict(zip(keys, outs))


def lovo_curve(ds: r3.DS, R: Dict, alpha: float) -> Dict[int, Tuple[int, int]]:
    return {L: r3.lovo_prop(ds, R, SENT, alpha, L)[:2] for L in r3.LAG_GRID}


# --------------------------------------------------------------------------- #
# Main computation
# --------------------------------------------------------------------------- #
def point(ds: r3.DS, R: Dict, method: str, knob: float, lag: int, order: int) -> Dict:
    m = r3.test_metrics(ds, R[r3._key(r3.test_spec(ds.name, method, knob, lag))])
    fa = m["null_primary"]
    det = m["episodes"]
    return {
        "method": method, "knob": float(knob), "lag": int(lag), "order": order,
        "fa": fa["pfa"], "fa_k": fa["k"], "fa_n": fa["n"], "fa_ci": fa["ci"],
        "det": det["pfa"], "det_k": det["k"], "det_n": det["n"], "det_ci": det["ci"],
        "fa_per_hour": m["fa_per_hour"], "fa_per_hour_ci": m["fa_per_hour_ci"],
        "median_delay_s": m["delay"]["median_s"],
        "null_no_anomaly": m["null_no_anomaly"], "null_pre_onset": m["null_pre_onset"],
    }


def run(names: Sequence[str], workers: int) -> Dict:
    t0 = time.time()
    names, skipped = r3.available(names)
    DSS = {n: r3.get_ds(n) for n in names}
    print(f"Datasets: {names}", flush=True)

    # Phase A: SENTINEL-E LOVO over the alpha grid and the lag grid (training only).
    specs = [r3.cv_spec(n, SENT, a, L, f)
             for n in names for a in SENT_ALPHAS for L in r3.LAG_GRID for f in range(len(DSS[n].folds))]
    R = run_units(specs, "A: SENTINEL-E LOVO", workers)
    lag_star: Dict[Tuple[str, float], int] = {}
    selection: Dict[str, Dict] = {}
    for n in names:
        selection[n] = {}
        for a in SENT_ALPHAS:
            curve = lovo_curve(DSS[n], R, a)
            Lst, failed = r3.select_lag(curve, a, r3.LAG_GRID)
            lag_star[(n, a)] = Lst
            selection[n][str(a)] = {"lag": int(Lst), "selection_failed": bool(failed),
                                    "lovo_curve": {str(L): list(curve[L]) for L in r3.LAG_GRID}}

    # Phase B: test units. Baselines at L*(0.1); SENTINEL-E at each alpha's L*(alpha).
    specs = []
    for n in names:
        for a in SENT_ALPHAS:
            specs.append(r3.test_spec(n, SENT, a, lag_star[(n, a)]))
        for m in BASELINES:
            for kn in SWEEPS[m]:
                specs.append(r3.test_spec(n, m, kn, lag_star[(n, HEADLINE)]))
    R.update(run_units(specs, "B: test", workers))

    # Points per dataset and method.
    points: Dict[str, Dict[str, List[Dict]]] = {}
    for n in names:
        ds = DSS[n]
        points[n] = {}
        points[n][SENT] = [point(ds, R, SENT, a, lag_star[(n, a)], i)
                           for i, a in enumerate(SWEEPS[SENT])]
        for m in BASELINES:
            points[n][m] = [point(ds, R, m, kn, lag_star[(n, HEADLINE)], i)
                            for i, kn in enumerate(SWEEPS[m])]

    # Summaries.
    summary: Dict[str, Dict] = {}
    for n in names:
        summary[n] = {"methods": {}, "sentinel_0.1": {}}
        all_base = [p for m in BASELINES for p in points[n][m]]
        for m in [SENT] + BASELINES:
            pts = points[n][m]
            b = best_at_fa(pts)
            summary[n]["methods"][m] = {
                "best_at_fa_cap": b, "pauc_0.2": envelope_area(pts),
                "n_knobs": len(pts), "n_knobs_fa_le_cap": sum(p["fa"] <= FA_CAP + 1e-9 for p in pts),
            }
        s_pts = [p for p in points[n][SENT] if abs(p["knob"] - HEADLINE) < 1e-12][0]
        dom = dominators(s_pts, all_base)
        summary[n]["sentinel_0.1"] = {
            "point": s_pts,
            "dominated": bool(dom),
            "dominators": [{"method": d["method"], "knob": d["knob"], "fa": d["fa"], "det": d["det"]} for d in dom],
            "frontier_gap_vs_baselines": frontier_gap(s_pts, all_base),
        }
        summary[n]["sentinel_all_alphas"] = [
            {"knob": p["knob"], "lag": p["lag"], "fa": p["fa"], "det": p["det"],
             "dominated_by_baseline_knob": bool(dominators(p, all_base)),
             "frontier_gap_vs_baselines": frontier_gap(p, all_base)}
            for p in points[n][SENT]]

    # Cross-check against the R3 test table (same specs, same code).
    crosscheck = {}
    r3_payload = json.loads((RESULTS / "exp_r3_per_video.json").read_text())["payload"]
    for n in names:
        if n not in r3_payload["test"]:
            continue
        for a in r3.ALPHAS:
            ours = [p for p in points[n][SENT] if abs(p["knob"] - a) < 1e-12][0]
            ref = r3_payload["test"][n][str(a)][SENT]["null_primary"]
            crosscheck[f"{n}|{a}"] = {
                "ours_fa": [ours["fa_k"], ours["fa_n"]], "r3_fa": [ref["k"], ref["n"]],
                "lag_ours": ours["lag"], "lag_r3": r3_payload["selection"][n][str(a)]["lag"],
                "match": bool(ours["fa_k"] == ref["k"] and ours["fa_n"] == ref["n"]
                              and ours["lag"] == r3_payload["selection"][n][str(a)]["lag"]),
            }

    meta = {}
    for n in names:
        ds = DSS[n]
        meta[n] = {
            "label": LABELS.get(n, n), "family": ds.family, "fps": ds.fps,
            "n_train_videos": len(ds.train), "n_test_videos": len(ds.test),
            "n_null_trials": points[n][SENT][0]["fa_n"],
            "n_test_episodes": sum(len(v.episodes) for v in ds.test),
        }
    return {"meta": meta, "skipped": skipped, "datasets": names, "selection": selection,
            "points": points, "summary": summary, "crosscheck_r3": crosscheck,
            "runtime_s": time.time() - t0}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _knob_tex(v: float) -> str:
    """A knob as LaTeX math (``_fmt_knob`` already returns math for scientific values)."""
    s = _fmt_knob(v)
    return s if "$" in s else f"${s}$"


def _ci(ci, nd: int = 2) -> str:
    return f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"


def write_table(payload: Dict) -> Path:
    names, S, meta = payload["datasets"], payload["summary"], payload["meta"]
    rows = []
    for n in names:
        for m in [SENT] + BASELINES:
            b = S[n]["methods"][m]["best_at_fa_cap"]
            if b is None:
                knob_cell, fa_cell, det_cell = "none", "--", "--"
            else:
                knob_cell = f"{_knob_tex(b['knob'])} ($L={b['lag']}$)"
                fa_cell = f"{fmt(b['fa'], 3)} ({b['fa_k']}/{b['fa_n']})"
                det_cell = f"{fmt(b['det'], 2)} {_ci(b['det_ci'])} ({b['det_k']}/{b['det_n']})"
            if m == SENT:
                sd = S[n]["sentinel_0.1"]
                if sd["dominated"]:
                    d = sd["dominators"][0]
                    flag = (f"yes ({len(sd['dominators'])} knob(s), e.g.\\ {d['method']} "
                            f"{_knob_tex(d['knob'])})")
                else:
                    flag = "no"
                dom_cell = flag
            else:
                dom_cell = "--"
            rows.append([meta[n]["label"], m, knob_cell, fa_cell, det_cell,
                         fmt(S[n]["methods"][m]["pauc_0.2"], 3), dom_cell])
    path = write_latex_table(
        rows,
        ["dataset", "method", "knob with highest detection at FA $\\le 0.1$", "test per-video FA (k/n)",
         "detection [95\\% CI] (k/n)", r"pAUC$_{0.2}$", r"SENTINEL-E dominated at $\alpha=0.1$?"],
        caption=(
            "Descriptive detection--false-alarm frontier on the test videos (each video monitored alone; "
            "calibration on all training videos). "
            "The knobs here are read off the TEST set and are NOT a deployable selection: the deployable "
            "comparison is Table~\\ref{tab:r3_test}. "
            "Baselines run at $L^\\ast(0.1)$; SENTINEL-E runs at each of its alphas at its own $L^\\ast(\\alpha)$, "
            "with $L^\\ast$ chosen on training videos. "
            "UCSD Ped2 has only 3 null test segments (pre-onset segments of the 12 test videos), so its per-video FA "
            "proportion takes the values 0, 1/3, 2/3 or 1, and FA $\\le 0.1$ means zero false alarms. "
            "pAUC$_{0.2}$ is the area under the empirical detection-versus-FA envelope on FA $\\in[0,0.2]$, "
            "divided by 0.2; it is 0 where no knob reaches a given FA."
        ),
        label="tab:r3b_frontier", name=TABLE_NAME, align="llllllc",
        notes=(
            "Per-video FA is over no-anomaly test videos and the pre-onset segments (at least 2\\,s) of anomalous test "
            "videos for UCSD Ped2 and CUHK Avenue, and over the negative test videos for Mivia-IWDD, as in Table~\\ref{tab:r3_test}. "
            "Detection is the proportion of labelled episodes detected (one per positive video for Mivia-IWDD). "
            "The highest-detection knob is chosen among knobs with test FA $\\le 0.1$; ties go to lower FA. "
            "Dominated: some baseline knob at $L^\\ast(0.1)$ has test FA no higher and detection strictly higher than "
            "SENTINEL-E at $\\alpha=0.1$. The Ped2 FA column counts 3 segments, so it is coarse."
        ),
    )
    # Long table: float it on its own page, in a smaller font.
    txt = path.read_text()
    txt = txt.replace("\\begin{table}[t]", "\\begin{table}[p]", 1).replace("\\centering", "\\centering\n\\footnotesize", 1)
    path.write_text(txt)
    return path


def write_figure(payload: Dict) -> None:
    plt = setup_matplotlib()
    names, P = payload["datasets"], payload["points"]
    cols = min(3, len(names))
    nrows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.5 * cols, 3.25 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    handles = {}
    for ax, n in zip(axes.ravel(), names):
        ax.axvline(FA_CAP, color="#999999", lw=0.8, ls=":", zorder=1)
        for m in BASELINES + [SENT]:
            pts = sorted(P[n][m], key=lambda p: (p["fa"], p["det"]))
            x = [p["fa"] for p in pts]
            y = [p["det"] for p in pts]
            st = style_for(m)
            if m == SENT:
                h, = ax.plot(x, y, color=st["color"], marker=st["marker"], lw=2.2, ms=5.5,
                             markeredgecolor="white", markeredgewidth=0.5, zorder=6, label=m)
                for p in P[n][m]:
                    ax.annotate(f"{p['knob']:g}", (p["fa"], p["det"]), textcoords="offset points",
                                xytext=(4, -8), fontsize=5.5, color=st["color"])
            else:
                h, = ax.plot(x, y, color=st["color"], marker=st["marker"], lw=1.0, ms=3.2,
                             alpha=0.85, zorder=st["zorder"], label=m)
            handles.setdefault(m, h)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("test per-video FA proportion (null trials)")
        ax.set_ylabel("episode detection proportion")
        ns = payload["meta"][n]["n_null_trials"]
        title = f"{payload['meta'][n]['label']}"
        if n == "ped2" or n == "ped2_patchcore":
            title += f"\n(only $n$={ns} null segments)"
        else:
            title += f"\n($n$={ns} null trials)"
        ax.set_title(title)
    order = [m for m in BASELINES + [SENT] if m in handles]
    fig.legend([handles[m] for m in order], order, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.text(0.5, -0.045 - 0.0 * nrows,
             "Each marker is one knob (labels: SENTINEL-E alpha). Baselines at L*(0.1); SENTINEL-E at L*(alpha). "
             "Knobs read off the test set: descriptive only. Dotted line: FA = 0.1.",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08 + 0.02 * nrows, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{FIG_NAME}.{ext}")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> Optional[Dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma-separated; default is the R3 datasets plus mivia_supervised when its data exist")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    names = [x.strip() for x in a.datasets.split(",") if x.strip()]
    print(f"Experiment R3b (descriptive frontier, test knobs)  workers={a.workers}  cache={CACHE_DIR}", flush=True)
    payload = run(names, a.workers)
    payload["protocol"] = {
        "sentinel_alphas": SENT_ALPHAS, "baseline_lag": "L*(0.1) selected on training videos",
        "sentinel_lag": "L*(alpha) selected on training videos by LOVO / 10-fold, as in R3",
        "lag_grid": list(r3.LAG_GRID), "headline_alpha": HEADLINE, "fa_cap": FA_CAP,
        "pauc_upper": PAUC_UPPER, "sweeps": SWEEPS, "min_null_s": r3.MIN_NULL_S,
        "calibration_test": "all training videos",
        "caveat": "knobs read off the test set; descriptive only, not a deployable selection",
    }
    bad = [k for k, v in payload["crosscheck_r3"].items() if not v["match"]]
    print(f"  cross-check vs R3 test table: {len(payload['crosscheck_r3']) - len(bad)} match, {len(bad)} differ {bad}",
          flush=True)
    save_json(payload, "exp_r3b_frontier")
    write_table(payload)
    write_figure(payload)
    print(f" wrote results/exp_r3b_frontier.json, tables/{TABLE_NAME}.tex, figures/{FIG_NAME}; "
          f"runtime {payload['runtime_s']:.0f}s; skipped={payload['skipped']}", flush=True)
    return payload


if __name__ == "__main__":
    main()

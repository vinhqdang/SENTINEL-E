"""Experiment 3c --- a p-value drift monitor as a recalibration request.

Experiments 3b and 3c-gate showed that a context two-sample gate is confounded
with benign context drift: the deployment context differs from calibration in
the weather chain and in diurnal coverage even under no shift, and the residual
normalisation absorbs that difference.  A gate on context therefore fires on
benign shift, whichever threshold rule it uses.

This script instead evaluates a slow monitor on the output of the default
pipeline: the unweighted residual detector (``shift_gate`` fixed, no weights).
:func:`sentinel_e.diagnostics.drift_monitor` looks at non-overlapping windows of
one simulated day of betting-instant p-values and flags a window whose share of
``p <= q`` exceeds ``q + z * sqrt(q (1 - q) / W)``.  The monitor sits outside the
alarm guarantee: it does not change any alarm, it tells the operator when the
conformal assumption looks broken and recalibration is due.

Per shift the script reports, for the unweighted arm with the same configs and
seeds as experiment 3b (1500 null streams, 200 post-change streams):

* PFA and miss rate, which must reproduce experiment 3b's ``residual`` arm;
* the fraction of null streams with at least one drift flag;
* the median time (s) to the first flag on null streams;
* among null streams with a false alarm, the fraction whose first flag comes
  before that false alarm;
* the fraction of post-change streams with at least one flag (flags caused by
  dumping episodes are false recalibration requests).

Units are cached under ``results/cache/exp3c/``.  Run with ``--smoke`` for a
tiny check that writes nothing.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from experiments.checkpoint import UnitCache
from experiments.common import fmt, load_json, parallel_map, save_json, write_latex_table
from experiments.exp3_shift import CHANGE_POINT, FPS, T
from experiments.exp3b_shift_default import (
    ACTIVITY_SHIFTS,
    ALPHA,
    NULL_SEED0,
    REPS_NULL,
    REPS_SIGNAL,
    SIGNAL_SEED0,
    base_cfg,
    verdict,
)
from experiments.runner import Trace, _to_alarm_array
from sentinel_e.diagnostics import drift_monitor
from sentinel_e.metrics import summarize_runs
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator

WORKERS = 2
Q = 0.1
Z = 4.0
DIURNAL_PERIOD = StreamConfig().diurnal_period  # frames per simulated day
ARM = "residual (drift monitor)"
SOURCE = "results/exp3b_shift_default.json"
SMOKE_NULL_FLAG_STOP = 0.2


def _stream_unit(args):
    """One stream of the unweighted residual arm plus its drift monitor."""
    seed, cfg, change_point = args
    st = StreamSimulator(StreamConfig(**cfg), seed=seed).simulate_camera(
        change_point=change_point)
    model = SentinelE(alpha=ALPHA, calibration="residual", weighted=False, restart=True)
    res = model.run_stream(st)
    alarms = np.flatnonzero(res.alarms)
    nu = change_point if change_point is not None else 0
    after = alarms[alarms >= nu]
    # Drift monitor on the betting-instant p-values, active instants only.
    bet = res.bet_index
    window = max(1, int(round(DIURNAL_PERIOD / res.lag)))
    out = drift_monitor(res.p_values[bet], active=res.active[bet],
                        window=window, q=Q, z=Z)
    first_flag = int(bet[out["first_flag"]]) if out["first_flag"] >= 0 else -1
    return {
        "T": int(st.T),
        "first_alarm": int(alarms[0]) if alarms.size else -1,
        "first_after_nu": int(after[0]) if after.size else -1,
        "abstention": float(res.abstention_rate),
        "lag": int(res.lag),
        "n_alarms": int(alarms.size),
        "window": int(window),
        "n_windows": int(out["fraction"].size),
        "n_flags": int(out["flags"].sum()),
        "first_flag": first_flag,
    }


def part_units(shift: float, part: str, reps: int, workers: int) -> Dict:
    """Unit (shift, part): all streams of one part, each with its drift result."""
    cfg = base_cfg(shift)
    if part == "null":
        cp, seed0 = None, NULL_SEED0
    else:
        cp, seed0 = CHANGE_POINT, SIGNAL_SEED0 + 500_000
    rows = parallel_map(_stream_unit,
                        [(seed0 + i, cfg, cp) for i in range(reps)], workers=workers)
    return {"rows": rows}


def _outcome(rows: List[Dict], change_point):
    """Operational metrics, computed exactly as runner.evaluate does."""
    traces = [Trace(r["T"], r["first_alarm"], r["first_after_nu"], r["abstention"],
                    r["lag"], r["n_alarms"]) for r in rows]
    return summarize_runs([_to_alarm_array(t) for t in traces], [], [])


def _signal_outcome(rows: List[Dict]):
    traces = [Trace(r["T"], r["first_alarm"], r["first_after_nu"], r["abstention"],
                    r["lag"], r["n_alarms"]) for r in rows]
    return summarize_runs([], [_to_alarm_array(t) for t in traces],
                          [CHANGE_POINT] * len(rows))


def drift_stats(null_rows: List[Dict], sig_rows: List[Dict]) -> Dict:
    flagged_null = [r for r in null_rows if r["first_flag"] >= 0]
    fa_rows = [r for r in null_rows if r["first_alarm"] >= 0]
    before = [r for r in fa_rows if 0 <= r["first_flag"] < r["first_alarm"]]
    times = [r["first_flag"] / FPS for r in flagged_null]
    return {
        "null_streams": len(null_rows),
        "null_flag_frac": float(np.mean([r["first_flag"] >= 0 for r in null_rows])),
        "null_mean_flags_per_stream": float(np.mean([r["n_flags"] for r in null_rows])),
        "median_t_first_flag_s": float(np.median(times)) if times else None,
        "n_false_alarm_streams": len(fa_rows),
        "fa_flagged_before_frac": (len(before) / len(fa_rows)) if fa_rows else None,
        "signal_streams": len(sig_rows),
        "signal_flag_frac": float(np.mean([r["first_flag"] >= 0 for r in sig_rows])),
        "windows_per_stream": float(np.mean([r["n_windows"] for r in null_rows])),
        "window_frames_median": float(np.median([r["window"] for r in null_rows])),
    }


def cell(null_rows: List[Dict], sig_rows: List[Dict]) -> Dict:
    out_n = _outcome(null_rows, None)
    out_s = _signal_outcome(sig_rows)
    lo, hi = out_n.pfa_ci
    cens = out_s.add_censored
    return {
        "pfa": out_n.pfa,
        "pfa_ci": [lo, hi],
        "verdict": verdict(lo, hi),
        "n_null": out_n.n_null,
        "miss_rate": out_s.miss_rate,
        "add_censored_s": None if cens is None else float(cens) / FPS,
        "n_signal": out_s.n_signal,
        "drift": drift_stats(null_rows, sig_rows),
    }


def make_table(cells: Dict, reps_null: int, reps_sig: int, check: Dict) -> None:
    rows = []
    for a in ACTIVITY_SHIFTS:
        c = cells[str(a)]
        d = c["drift"]
        lo, hi = c["pfa_ci"]
        rows.append([
            f"$\\times {a}$",
            fmt(c["pfa"], 4),
            f"[{lo:.4f}, {hi:.4f}]",
            fmt(c["miss_rate"], 3),
            fmt(d["null_flag_frac"], 3),
            fmt(d["median_t_first_flag_s"], 0),
            fmt(d["fa_flagged_before_frac"], 3),
            fmt(d["signal_flag_frac"], 3),
        ])
    notes = (
        f"Unweighted residual detector (default pipeline), alpha = {ALPHA}; "
        f"{reps_null} null and {reps_sig} post-change streams per row. PFA and miss "
        f"rate are reproduced from experiment 3b's residual arm (maximum absolute "
        f"difference {check['max_abs_diff']:.2e}). Drift monitor on betting-instant "
        f"p-values: windows of one simulated day (diurnal period / lag), flag when "
        f"the share of $p \\le {Q}$ exceeds ${Q} + {Z}\\sqrt{{{Q}(1-{Q})/W}}$. The "
        "monitor is outside the alarm guarantee and does not change any alarm. "
        "Columns: null streams with at least one flag; median time to the first flag "
        "on flagged null streams; among null streams with a false alarm, the fraction "
        "flagged before it; post-change streams with at least one flag (false "
        "recalibration requests caused by dumping episodes)."
    )
    write_latex_table(
        rows,
        ["activity shift", "PFA", "95\\% CI", "miss rate", "null flag frac",
         "median first flag (s)", "FA flagged before", "signal flag frac"],
        caption=(
            r"Residual detector at $\alpha=0.01$ with a slow p-value drift monitor "
            r"as a recalibration request, across activity shifts."
        ),
        label="tab:shift_monitor",
        name="tab_shift_monitor",
        align="lrlrrrrr",
        notes=notes,
    )


def main(reps_null: int, reps_sig: int, workers: int, smoke: bool, no_cache: bool = False):
    t0 = time.time()
    print(f"Experiment 3c: drift monitor on the unweighted residual arm (null={reps_null}, "
          f"signal={reps_sig}, workers={workers}, smoke={smoke})", flush=True)
    cache = UnitCache("exp3c", enabled=not (no_cache or smoke))
    ref = None if smoke else load_json("exp3b_shift_default")
    cells: Dict[str, Dict] = {}
    for a in ACTIVITY_SHIFTS:
        base = dict(shift=a, arm=ARM, cfg=base_cfg(a), q=Q, z=Z, diurnal=DIURNAL_PERIOD)
        null = cache.get_or_compute(
            f"x{a}__null", dict(base, part="null", reps=reps_null, seed0=NULL_SEED0),
            lambda: part_units(a, "null", reps_null, workers))
        sig = cache.get_or_compute(
            f"x{a}__signal", dict(base, part="signal", reps=reps_sig, seed0=SIGNAL_SEED0),
            lambda: part_units(a, "signal", reps_sig, workers))
        cells[str(a)] = c = cell(null["rows"], sig["rows"])
        d = c["drift"]
        print(f"  x{a:<4} PFA={c['pfa']:.4f} miss={c['miss_rate']:.3f} | null flag frac="
              f"{d['null_flag_frac']:.3f} median t={d['median_t_first_flag_s']} "
              f"FA before={d['fa_flagged_before_frac']} signal flag frac="
              f"{d['signal_flag_frac']:.3f}", flush=True)

    # Cross-check: the unweighted arm must reproduce experiment 3b's residual arm.
    check = {"max_abs_diff": None, "compared": False}
    if ref is not None and reps_null == REPS_NULL and reps_sig == REPS_SIGNAL:
        diffs = []
        for a in ACTIVITY_SHIFTS:
            r = ref["cells"][str(a)]["residual"]
            diffs += [abs(cells[str(a)]["pfa"] - r["pfa"]),
                      abs(cells[str(a)]["miss_rate"] - r["miss_rate"])]
        check = {"max_abs_diff": float(max(diffs)), "compared": True}
        print(f" exp3b cross-check: max |diff| = {check['max_abs_diff']:.2e}", flush=True)
        assert check["max_abs_diff"] <= 1e-9, "unweighted arm does not reproduce exp3b"

    runtime = time.time() - t0
    print(f" units computed={cache.computed} loaded from cache={cache.hits}", flush=True)
    if smoke:
        x10 = cells["1.0"]["drift"]["null_flag_frac"]
        print(f" SMOKE x1.0 null flag fraction = {x10:.3f} "
              f"(stop threshold {SMOKE_NULL_FLAG_STOP}); smoke run finished in "
              f"{runtime:.1f}s; nothing written", flush=True)
        return
    payload = {
        "alpha": ALPHA, "T": T, "change_point": CHANGE_POINT, "fps": FPS,
        "reps_null": reps_null, "reps_signal": reps_sig,
        "seed0_null": NULL_SEED0, "seed0_signal": SIGNAL_SEED0,
        "workers": workers,
        "arm": {"name": ARM, "calibration": "residual", "weighted": False,
                "shift_gate": "fixed"},
        "monitor": {"q": Q, "z": Z, "window_rule": "diurnal_period / lag per stream",
                    "diurnal_period": DIURNAL_PERIOD},
        "exp3b_check": check,
        "cells": cells,
        "runtime_s": runtime,
    }
    save_json(payload, "exp3c_shift_gate")
    make_table(cells, reps_null, reps_sig, check)
    print(f" wrote results/exp3c_shift_gate.json and results/tables/tab_shift_monitor.tex "
          f"in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-null", type=int, default=REPS_NULL)
    ap.add_argument("--reps-sig", type=int, default=REPS_SIGNAL)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny counts (40 null, 20 signal unless set); writes nothing "
                         "and uses no unit cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the unit cache (neither read nor write it)")
    a = ap.parse_args()
    if a.smoke:
        a.reps_null = min(a.reps_null, 40)
        a.reps_sig = min(a.reps_sig, 20)
    main(a.reps_null, a.reps_sig, a.workers, a.smoke, a.no_cache)

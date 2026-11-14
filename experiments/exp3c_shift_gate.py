"""Experiment 3c --- a stricter support region as the remedy for x3 false alarms.

Experiment 3b shows that the unweighted residual detector's false-alarm
probability (PFA) exceeds alpha at activity shift x3.0 (0.062), while it is
valid at x1.0.  Diagnosis showed that the marginal p-values stay conservative
at every shift, that false alarms at x3 are preceded by runs of small p-values,
and that alarms cluster just after abstention runs at the calibrated-support
edge.

The remedy evaluated here is a stricter calibrated support region, via the
residual calibrator's Mahalanobis abstention threshold
(``support_slack = 0.5`` instead of the default ``1.5``; the threshold is the
calibration set's 99.9th-percentile Mahalanobis distance times this factor).
Frames whose context lies outside the region are declared inactive and carry no
evidence.  The decision is a deterministic function of the context and the
calibration set, not of the scores, so the e-process argument is unchanged.

Arms (same configurations, seeds, alpha and replicate counts as experiment 3b:
1500 null and 200 post-change streams per shift):

* ``residual`` and ``residual + LR wts`` are loaded from
  ``results/exp3b_shift_default.json`` and are not recomputed;
* ``residual (support slack 0.5)`` is computed here and cached under
  ``results/cache/exp3c/``.

Run with ``--smoke`` for a tiny check that writes nothing.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

from experiments.checkpoint import UnitCache, safe_name
from experiments.common import fmt, load_json, save_json, write_latex_table
from experiments.exp3_shift import CHANGE_POINT, FPS, T
from experiments.exp3b_shift_default import (
    ACTIVITY_SHIFTS,
    ALPHA,
    NULL_SEED0,
    REPS_NULL,
    REPS_SIGNAL,
    SIGNAL_SEED0,
    assemble_cell,
    base_cfg,
    null_part,
    signal_part,
)

WORKERS = 2
SUPPORT_SLACK = 0.5
NEW_VARIANT = "residual (support slack 0.5)"
NEW_OPTS = dict(calibration="residual", weighted=False, support_slack=SUPPORT_SLACK)
LOADED_VARIANTS = ("residual", "residual + LR wts")
SOURCE = "results/exp3b_shift_default.json"


def make_table(cells: Dict, reps_null: int, reps_sig: int) -> None:
    variants = list(LOADED_VARIANTS) + [NEW_VARIANT]
    rows = []
    for a in ACTIVITY_SHIFTS:
        for i, v in enumerate(variants):
            c = cells[str(a)][v]
            lo, hi = c["pfa_ci"]
            rows.append([
                f"$\\times {a}$" if i == 0 else "",
                v,
                fmt(c["pfa"], 4),
                f"[{lo:.4f}, {hi:.4f}]",
                c["verdict"],
                fmt(c["miss_rate"], 3),
                fmt(c["add_censored_s"], 1),
                fmt(c["abstention_null"], 3),
            ])
    notes = (
        f"Null PFA over {reps_null} streams and post-change metrics over {reps_sig} "
        f"streams per row, alpha = {ALPHA}. Verdict: valid if the Wilson 95\\% upper "
        "bound is at most alpha; invalid if the lower bound exceeds alpha; otherwise "
        "unresolved. The first two rows are taken unchanged from experiment 3b. The "
        f"support-slack row sets the residual calibrator's Mahalanobis abstention "
        f"multiplier to {SUPPORT_SLACK} (default 1.5), which abstains on frames near "
        "the calibrated-support edge. Abstention is the fraction of null betting "
        "instants declared inactive."
    )
    write_latex_table(
        rows,
        ["activity shift", "variant", "PFA", "95\\% CI", "verdict", "miss rate",
         "censored delay (s)", "abstention (null)"],
        caption=(
            r"Activity shift at $\alpha=0.01$: the unweighted residual detector with a "
            r"stricter calibrated support region, against the two experiment 3b arms."
        ),
        label="tab:shift_remedy",
        name="tab_shift_remedy",
        align="llrlllrr",
        notes=notes,
    )


def main(reps_null: int, reps_sig: int, workers: int, smoke: bool, no_cache: bool = False):
    t0 = time.time()
    print(f"Experiment 3c: support-slack remedy (null={reps_null}, signal={reps_sig}, "
          f"workers={workers}, smoke={smoke})", flush=True)
    cache = UnitCache("exp3c", enabled=not (no_cache or smoke))
    ref = None if smoke else load_json("exp3b_shift_default")
    cells: Dict[str, Dict] = {}
    for a in ACTIVITY_SHIFTS:
        key_a = str(a)
        cells[key_a] = {}
        if ref is not None:
            for v in LOADED_VARIANTS:
                cells[key_a][v] = ref["cells"][key_a][v]
        base = dict(shift=a, variant=NEW_VARIANT, opts=NEW_OPTS, cfg=base_cfg(a))
        nkey = f"x{a}__{safe_name(NEW_VARIANT)}__null"
        skey = f"x{a}__{safe_name(NEW_VARIANT)}__signal"
        null = cache.get_or_compute(
            nkey, dict(base, part="null", reps=reps_null, seed0=NULL_SEED0),
            lambda: null_part(a, NEW_OPTS, reps_null, workers))
        sig = cache.get_or_compute(
            skey, dict(base, part="signal", reps=reps_sig, seed0=SIGNAL_SEED0),
            lambda: signal_part(a, NEW_OPTS, reps_sig, workers))
        cells[key_a][NEW_VARIANT] = c = assemble_cell(null, sig)
        print(f"  x{a:<4} {NEW_VARIANT:30s} PFA={c['pfa']:.4f} "
              f"CI=[{c['pfa_ci'][0]:.4f},{c['pfa_ci'][1]:.4f}] {c['verdict']:10s} "
              f"miss={c['miss_rate']:.3f} cens_s={c['add_censored_s']} "
              f"abst={c['abstention_null']:.3f}", flush=True)

    runtime = time.time() - t0
    print(f" units computed={cache.computed} loaded from cache={cache.hits}", flush=True)
    if smoke:
        print(f" smoke run finished in {runtime:.1f}s; nothing written", flush=True)
        return
    payload = {
        "alpha": ALPHA, "T": T, "change_point": CHANGE_POINT, "fps": FPS,
        "reps_null": reps_null, "reps_signal": reps_sig,
        "seed0_null": NULL_SEED0, "seed0_signal": SIGNAL_SEED0,
        "workers": workers,
        "new_arm": {"name": NEW_VARIANT, "options": NEW_OPTS},
        "loaded_from": {"file": SOURCE, "variants": list(LOADED_VARIANTS)},
        "cells": cells,
        "runtime_s": runtime,
    }
    save_json(payload, "exp3c_shift_gate")
    make_table(cells, reps_null, reps_sig)
    print(f" wrote results/exp3c_shift_gate.json and results/tables/tab_shift_remedy.tex "
          f"in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-null", type=int, default=REPS_NULL)
    ap.add_argument("--reps-sig", type=int, default=REPS_SIGNAL)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny counts (20 null, 10 signal unless set); writes nothing and "
                         "uses no unit cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the unit cache (neither read nor write it)")
    a = ap.parse_args()
    if a.smoke:
        a.reps_null = min(a.reps_null, 20)
        a.reps_sig = min(a.reps_sig, 10)
    main(a.reps_null, a.reps_sig, a.workers, a.smoke, a.no_cache)

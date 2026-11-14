"""Experiment 6b --- high-replicate re-run of the statistically load-bearing ablation rows.

At 200 replicates, 14 of the 19 rows of exp6 cannot be resolved: the Wilson
interval on their realised PFA straddles alpha = 0.01.  This script re-runs the
seven rows that carry the paper's validity claims with 1,500 null streams (for
PFA) and 200 post-change streams (for miss rate and censored delay).

The variant definitions, the base options and the simulator configuration are
imported from :mod:`experiments.exp6_ablation`, so the two experiments cannot
drift apart.  Seeds follow :func:`experiments.runner.evaluate`: null stream ``i``
uses seed ``10000 + i`` and post-change stream ``i`` uses ``510000 + i``, so the
first 200 post-change streams are exactly those exp6 used.  Null and
post-change streams are produced by separate calls because :func:`evaluate`
always runs both sides at the same replicate count.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import numpy as np

from experiments.checkpoint import UnitCache, safe_name
from experiments.common import fmt, fmt_int, parallel_map, save_json, write_latex_table
from experiments.exp6_ablation import (
    ABLATIONS,
    ALPHA,
    BASE,
    CHANGE_POINT,
    FPS,
    N_EPISODES,
    T,
)
from experiments.exp9_paired import _censored, _missed
from experiments.runner import Spec, evaluate, run_spec
from sentinel_e.metrics import wilson_interval

NULL_REPS = 1_500
SIGNAL_REPS = 200
WORKERS = 3
SEED0 = 10_000

#: The rows whose validity or power claims the paper rests on.
ROW_NAMES = (
    "SENTINEL-E (full)",
    "changepoint mixture ($\\eta = 0$)",
    "raw calibration (not thinned)",
    "bet on every frame",
    "pooled conformal",
    "DKW inflation",
    "no correction ($\\delta=0$)",
)

CFG = dict(T=T, calibration_regime="representative", n_episodes=N_EPISODES,
           event_length=800, episode_gap=6_000, event_intermittency=0.35)


def _signal_traces(opts: Dict, reps: int):
    """Post-change traces with exp6's seeds (``510000 + i``), no null side."""
    opt_items = tuple(sorted(opts.items()))
    specs = [Spec("SENTINEL-E", ALPHA, SEED0 + 500_000 + i, CHANGE_POINT, CFG, opt_items)
             for i in range(reps)]
    return parallel_map(run_spec, specs, WORKERS)


def verdict(k: int, n: int, alpha: float = ALPHA) -> str:
    """Three-state validity verdict from the Wilson interval on the realised PFA."""
    lo, hi = wilson_interval(k, n)
    if hi <= alpha + 1e-12:
        return "valid"
    if lo > alpha + 1e-12:
        return "invalid"
    return "unresolved"


def _null_unit(opts: Dict, null_reps: int) -> Dict:
    """Unit (row, "null"): null streams for one ablation row."""
    t0 = time.perf_counter()
    null_out, null, _ = evaluate("SENTINEL-E", ALPHA, CFG, null_reps, None,
                                 options=opts, workers=WORKERS)
    return {"pfa": null_out.pfa, "pfa_ci": list(null_out.pfa_ci),
            "arl0": null_out.arl0, "lag": null[0].lag,
            "seconds": time.perf_counter() - t0}


def _signal_unit(opts: Dict, signal_reps: int) -> Dict:
    """Unit (row, "signal"): post-change streams for one ablation row."""
    t0 = time.perf_counter()
    sig = _signal_traces(opts, signal_reps)
    delay = _censored(sig, T, CHANGE_POINT)
    missed = _missed(sig)
    return {"miss_rate": float(missed.mean()),
            "censored_delay_s": float(delay.mean()) / FPS,
            "seconds": time.perf_counter() - t0}


def run(null_reps: int = NULL_REPS, signal_reps: int = SIGNAL_REPS,
        cache: Optional[UnitCache] = None) -> List[Dict]:
    cache = cache if cache is not None else UnitCache("exp6b_ablation_highrep")
    rows = []
    for ab in ABLATIONS:
        if ab["name"] not in ROW_NAMES:
            continue
        opts = dict(BASE)
        opts.update(ab["opts"])
        row_key = safe_name(ab["name"])
        null = cache.get_or_compute(
            f"{row_key}__null",
            dict(row=ab["name"], part="null", opts=opts, cfg=CFG, reps=null_reps,
                 seed0=SEED0),
            lambda: _null_unit(opts, null_reps))
        sig = cache.get_or_compute(
            f"{row_key}__signal",
            dict(row=ab["name"], part="signal", opts=opts, cfg=CFG, reps=signal_reps,
                 seed0=SEED0 + 500_000, change_point=CHANGE_POINT),
            lambda: _signal_unit(opts, signal_reps))
        k = int(round(null["pfa"] * null_reps))
        row = {
            "name": ab["name"], "group": ab["group"],
            "null_reps": null_reps, "signal_reps": signal_reps,
            "pfa": null["pfa"], "pfa_ci": null["pfa_ci"],
            "false_alarms": k,
            "verdict": verdict(k, null_reps),
            "arl0": null["arl0"],
            "miss_rate": sig["miss_rate"],
            "censored_delay_s": sig["censored_delay_s"],
            "lag": null["lag"],
            "seconds": null["seconds"] + sig["seconds"],
        }
        rows.append(row)
        print(f"  {ab['name']:34s} PFA={row['pfa']:.4f} "
              f"[{row['pfa_ci'][0]:.4f}, {row['pfa_ci'][1]:.4f}] {row['verdict']:10s} "
              f"miss={row['miss_rate']:.3f} delay={row['censored_delay_s']:.1f}s "
              f"lag={row['lag']} ({row['seconds']:.0f}s)", flush=True)
    print(f" units computed={cache.computed} loaded from cache={cache.hits}", flush=True)
    return rows


def make_table(rows: List[Dict], null_reps: int, signal_reps: int) -> None:
    body = []
    for r in rows:
        lo, hi = r["pfa_ci"]
        body.append([
            r["name"],
            f"{fmt(r['pfa'], 4)} [{fmt(lo, 4)}, {fmt(hi, 4)}]",
            r["verdict"],
            fmt_int(r["arl0"]),
            fmt(r["censored_delay_s"], 1),
            fmt(r["miss_rate"], 3),
            fmt_int(r["lag"]),
        ])
    n_unres = sum(1 for r in rows if r["verdict"] == "unresolved")
    write_latex_table(
        body,
        ["variant", "PFA [95\\% CI]", "verdict", "ARL$_0$", "cens. delay (s)",
         "miss", "lag"],
        caption=(
            r"High-replicate re-run of the load-bearing ablation rows at "
            r"$\alpha=0.01$: "
            f"{null_reps:,} null streams per row for the realised PFA "
            f"and {signal_reps} post-change streams for the miss rate and "
            r"censored delay. A row is \emph{valid} when the Wilson upper bound "
            r"is at or below nominal, \emph{invalid} when the lower bound exceeds "
            f"it, and \\emph{{unresolved}} otherwise ({n_unres} of {len(rows)} "
            r"rows unresolved)."
        ),
        label="tab:ablation_highrep",
        name="tab_ablation_highrep",
        align="lrrrrrr",
        notes=(
            r"Same configuration, options and seeds as \cref{tab:ablation}; the "
            r"first 200 post-change streams are identical to exp6's."
        ),
    )


def main(argv=None) -> None:
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (8 null, 4 signal per row); writes nothing; "
                         "uses results/cache/exp6b_ablation_highrep_smoke")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the unit cache (neither read nor write it)")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="process-pool size (does not change any result)")
    args = ap.parse_args(argv)
    WORKERS = args.workers
    cache = UnitCache("exp6b_ablation_highrep", enabled=not args.no_cache,
                      smoke=args.smoke)
    if args.smoke:
        rows = run(null_reps=8, signal_reps=4, cache=cache)
        print("smoke test ok:", [r["verdict"] for r in rows])
        return
    print(f"Experiment 6b: {NULL_REPS} null / {SIGNAL_REPS} signal streams per row")
    t0 = time.perf_counter()
    rows = run(cache=cache)
    total = time.perf_counter() - t0
    save_json({"rows": rows, "null_reps": NULL_REPS, "signal_reps": SIGNAL_REPS,
               "alpha": ALPHA, "T": T, "change_point": CHANGE_POINT,
               "n_episodes": N_EPISODES, "cfg": CFG, "total_seconds": total},
              "exp6b_ablation_highrep")
    make_table(rows, NULL_REPS, SIGNAL_REPS)
    print(f" wrote results/exp6b_ablation_highrep.json, tab_ablation_highrep ({total:.0f}s)")


if __name__ == "__main__":
    main()

"""Experiment 3c --- a phase-invariant shift gate for likelihood-ratio weighting.

Experiment 3b shows that the held-out two-sample gate in front of the
likelihood-ratio (LR) weights fires on every null stream, because the
calibration set spans a whole simulated day while one deployment block covers a
narrow slice of it: the classifier separates them by clock and luminance alone.
The weights then also condition on the clock, the effective sample size
collapses, and detection power is lost.

This script evaluates one new arm: residual calibration with LR weights whose
gate and density ratio both use only context columns [1, 3] (weather and scene
activity, i.e. no clock or luminance).  The two arms from experiment 3b
(``residual`` and ``residual + LR wts`` with all columns) are loaded from
``results/exp3b_shift_default.json`` and are not recomputed.

Same configurations, seeds, alpha, replicate counts and verdict rule as
experiment 3b (see that module's docstring).  Units are cached under
``results/cache/exp3c/`` so an interrupted run resumes.

Run with ``--smoke`` for a tiny check that writes nothing (no results files and
no unit cache).
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from experiments.checkpoint import UnitCache, safe_name
from experiments.common import fmt, load_json, parallel_map, save_json, write_latex_table
from experiments.exp3_shift import CHANGE_POINT, FPS, T
from experiments.exp3b_shift_default import (
    ACTIVITY_SHIFTS,
    ALPHA,
    GATE_STREAMS,
    NULL_SEED0,
    REPS_NULL,
    REPS_SIGNAL,
    SIGNAL_SEED0,
    assemble_cell,
    base_cfg,
    null_part,
    signal_part,
    verdict,
)
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices

WORKERS = 2
SHIFT_FEATURES = (1, 3)  # weather and scene activity; clock and luminance excluded
NEW_VARIANT = "residual + LR wts (phase-invariant gate)"
NEW_OPTS = dict(calibration="residual", weighted=True, shift_features=SHIFT_FEATURES)
# Arms carried over from experiment 3b, in table order.
LOADED_VARIANTS = ("residual", "residual + LR wts")
SOURCE = "results/exp3b_shift_default.json"


def _gate_stream(args):
    """Replay the shift-gate loop of SentinelE.p_values on one null stream.

    Identical to exp3b's gate replay except that the gate and the density ratio
    use ``SHIFT_FEATURES`` only.
    """
    seed, cfg = args
    st = StreamSimulator(StreamConfig(**cfg), seed=seed).simulate_camera(change_point=None)
    model = SentinelE(alpha=ALPHA, calibration="residual", weighted=True, restart=True,
                      shift_features=SHIFT_FEATURES)
    model.fit(st.calibration_scores, st.calibration_context)
    cal = model.calibrator
    idx = thin_indices(st.T, model.decorrelation_lag)
    ctx = np.atleast_2d(np.asarray(st.context, dtype=float))[idx]
    blk = model.shift_block
    n_blocks = n_trig = 0
    for start in range(blk, ctx.shape[0], blk):
        cal.update_shift(ctx[start - blk:start])
        n_blocks += 1
        if cal.shift_auc >= cal.shift_auc_threshold:
            n_trig += 1
    return n_blocks, n_trig


def gate_stats(shift: float, n_streams: int, workers: int) -> Dict:
    cfg = base_cfg(shift)
    res = parallel_map(_gate_stream, [(NULL_SEED0 + i, cfg) for i in range(n_streams)],
                       workers=workers)
    n_blocks = np.array([r[0] for r in res], dtype=float)
    n_trig = np.array([r[1] for r in res], dtype=float)
    frac_per_stream = np.divide(n_trig, n_blocks, out=np.zeros_like(n_trig),
                                where=n_blocks > 0)
    return {
        "streams": int(n_streams),
        "frac_streams_triggered": float(np.mean(n_trig >= 1)),
        "mean_frac_blocks_triggered": float(np.mean(frac_per_stream)),
        "mean_blocks_per_stream": float(np.mean(n_blocks)),
    }


def _gate_text(g: Dict) -> str:
    return f"{g['frac_streams_triggered']:.2f} / {g['mean_frac_blocks_triggered']:.2f}"


def make_table(cells: Dict, gates: Dict, reps_null: int, reps_sig: int) -> None:
    variants = list(LOADED_VARIANTS) + [NEW_VARIANT]
    rows = []
    for a in ACTIVITY_SHIFTS:
        for i, v in enumerate(variants):
            c = cells[str(a)][v]
            lo, hi = c["pfa_ci"]
            if v == "residual":
                gate_txt = "n/a (off)"
            else:
                gate_txt = _gate_text(gates[str(a)][v])
            rows.append([
                f"$\\times {a}$" if i == 0 else "",
                v,
                fmt(c["pfa"], 4),
                f"[{lo:.4f}, {hi:.4f}]",
                c["verdict"],
                fmt(c["miss_rate"], 3),
                fmt(c["add_censored_s"], 1),
                gate_txt,
            ])
    notes = (
        f"Null PFA over {reps_null} streams and signal metrics over {reps_sig} "
        f"post-change streams per row, alpha = {ALPHA}. Verdict: valid if the Wilson "
        "95\\% upper bound is at most alpha; invalid if the lower bound exceeds alpha; "
        "otherwise unresolved. Gate trigger rate: fraction of null streams with at "
        "least one shift block whose held-out AUC reaches the threshold / mean "
        f"fraction of triggered blocks ({GATE_STREAMS} streams per shift). "
        "The first two arms are taken unchanged from experiment 3b. The phase-invariant "
        f"gate restricts the classifier and the density ratio to context columns "
        f"{list(SHIFT_FEATURES)} (weather and scene activity)."
    )
    write_latex_table(
        rows,
        ["activity shift", "variant", "PFA", "95\\% CI", "verdict", "miss rate",
         "censored delay (s)", "gate trigger rate"],
        caption=(
            r"Activity shift at $\alpha=0.01$: residual calibration with likelihood-ratio "
            r"weights, with the held-out gate on all context columns and on the "
            r"phase-invariant columns $\{1,3\}$."
        ),
        label="tab:shift_gate",
        name="tab_shift_gate",
        align="llrlllrr",
        notes=notes,
    )


def main(reps_null: int, reps_sig: int, gate_streams: int, workers: int, smoke: bool,
         no_cache: bool = False):
    t0 = time.time()
    print(f"Experiment 3c: phase-invariant shift gate (null={reps_null}, signal={reps_sig}, "
          f"gate={gate_streams}, workers={workers}, smoke={smoke})", flush=True)
    # Smoke runs write nothing, including the unit cache.
    cache = UnitCache("exp3c", enabled=not (no_cache or smoke))

    ref = load_json("exp3b_shift_default")
    cells: Dict[str, Dict] = {}
    gates: Dict[str, Dict] = {}
    for a in ACTIVITY_SHIFTS:
        key_a = str(a)
        cells[key_a] = {}
        gates[key_a] = {}
        for v in LOADED_VARIANTS:
            cells[key_a][v] = ref["cells"][key_a][v]
            if v != "residual":
                gates[key_a][v] = ref["gate"][key_a]

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
        print(f"  x{a:<4} {NEW_VARIANT:40s} PFA={c['pfa']:.4f} "
              f"CI=[{c['pfa_ci'][0]:.4f},{c['pfa_ci'][1]:.4f}] {c['verdict']:10s} "
              f"miss={c['miss_rate']:.3f} cens_s={c['add_censored_s']}", flush=True)

        gates[key_a][NEW_VARIANT] = cache.get_or_compute(
            f"x{a}__gate_phase_invariant",
            dict(shift=a, part="gate", features=list(SHIFT_FEATURES),
                 streams=gate_streams, seed0=NULL_SEED0, cfg=base_cfg(a)),
            lambda: gate_stats(a, gate_streams, workers))
        print(f"  x{a:<4} gate (phase-invariant): streams>=1 triggered="
              f"{gates[key_a][NEW_VARIANT]['frac_streams_triggered']:.3f} "
              f"mean block frac={gates[key_a][NEW_VARIANT]['mean_frac_blocks_triggered']:.3f}",
              flush=True)

    print(f" units computed={cache.computed} loaded from cache={cache.hits}", flush=True)
    runtime = time.time() - t0
    if smoke:
        print(f" smoke run finished in {runtime:.1f}s; nothing written", flush=True)
        return
    payload = {
        "alpha": ALPHA,
        "T": T,
        "change_point": CHANGE_POINT,
        "fps": FPS,
        "reps_null": reps_null,
        "reps_signal": reps_sig,
        "gate_streams": gate_streams,
        "seed0_null": NULL_SEED0,
        "seed0_signal": SIGNAL_SEED0,
        "workers": workers,
        "shift_features": list(SHIFT_FEATURES),
        "variants": {NEW_VARIANT: NEW_OPTS},
        "loaded_from": {"file": SOURCE, "variants": list(LOADED_VARIANTS)},
        "cells": cells,
        "gate": gates,
        "runtime_s": runtime,
    }
    save_json(payload, "exp3c_shift_gate")
    make_table(cells, gates, reps_null, reps_sig)
    print(f" wrote results/exp3c_shift_gate.json and results/tables/tab_shift_gate.tex "
          f"in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-null", type=int, default=REPS_NULL)
    ap.add_argument("--reps-sig", type=int, default=REPS_SIGNAL)
    ap.add_argument("--gate-streams", type=int, default=GATE_STREAMS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny replicate counts (set here unless overridden); writes "
                         "nothing and uses no unit cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the unit cache (neither read nor write it)")
    a = ap.parse_args()
    if a.smoke:
        # Tiny counts unless the caller set them explicitly.
        a.reps_null = min(a.reps_null, 20)
        a.reps_sig = min(a.reps_sig, 10)
        a.gate_streams = min(a.gate_streams, 3)
    main(a.reps_null, a.reps_sig, a.gate_streams, a.workers, a.smoke, a.no_cache)

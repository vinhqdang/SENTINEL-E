"""Experiment 3b --- should likelihood-ratio shift weighting be the default?

Experiment 3 shows that the shipped ``residual`` calibration keeps false-alarm
control under a x3 activity shift only as far as its Monte-Carlo resolution
allows, and that the weighted variant's safeguard (a held-out two-sample test
that gates the likelihood-ratio reweighting) is off by default.  This script
re-runs the activity sweep at a larger replicate count, for the two candidate
defaults, and records how often the safeguard actually fires.

Pre-specified decision rule (fixed before any result was seen, applied
mechanically):

    Recommend "residual + LR wts" as the default if and only if
    (a) its verdict is not "invalid" at every activity shift level, and
    (b) at activity shift 1.0 its miss rate exceeds the unweighted "residual"
        miss rate by at most 0.05 (absolute).

Verdicts use the Wilson 95% interval of the null false-alarm probability over
1500 null streams: "valid" if the interval's upper end is <= alpha (0.01),
"invalid" if its lower end is > alpha, otherwise "unresolved".

Run with ``--smoke`` for a tiny check that writes nothing.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from experiments.checkpoint import UnitCache, safe_name
from experiments.common import fmt, parallel_map, save_json, write_latex_table
from experiments.exp3_shift import CHANGE_POINT, FPS, T
from experiments.runner import evaluate
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices

ALPHA = 0.01
REPS_NULL = 1500
REPS_SIGNAL = 200
GATE_STREAMS = 100
NULL_SEED0 = 10_000
SIGNAL_SEED0 = 900_000
WORKERS = 2
ACTIVITY_SHIFTS = [1.0, 1.5, 2.0, 3.0]

# Same two options dicts as exp3_shift.VARIANTS (the two candidate defaults).
VARIANTS = {
    "residual":          dict(calibration="residual", weighted=False),
    "residual + LR wts": dict(calibration="residual", weighted=True),
}

DECISION_RULE = (
    'Recommend making "residual + LR wts" the default iff (a) its verdict is '
    'not "invalid" at every activity shift level, and (b) at activity shift 1.0 '
    "its miss rate exceeds the unweighted residual miss rate by at most 0.05 "
    "(absolute). Otherwise keep the unweighted residual default."
)


def base_cfg(shift: float) -> Dict:
    """Base stream configuration, identical to exp3_shift.sweep_activity."""
    return dict(T=T, calibration_regime="representative", activity_shift=shift,
                n_episodes=4, event_length=800, episode_gap=6_000,
                event_intermittency=0.35)


def verdict(lo: float, hi: float, alpha: float = ALPHA) -> str:
    if hi <= alpha:
        return "valid"
    if lo > alpha:
        return "invalid"
    return "unresolved"


def _gate_stream(args):
    """Replay the pipeline's shift-gate loop on one null stream.

    Mirrors SentinelE.p_values: thin the deployment context by the decorrelation
    lag, then call update_shift on each preceding shift_block window and record
    whether the held-out AUC reached the threshold.
    """
    seed, cfg = args
    st = StreamSimulator(StreamConfig(**cfg), seed=seed).simulate_camera(change_point=None)
    model = SentinelE(alpha=ALPHA, calibration="residual", weighted=True, restart=True)
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


def null_part(shift: float, opts: Dict, reps_null: int, workers: int) -> Dict:
    """Unit (shift, variant, "null"): null streams for one variant."""
    cfg = base_cfg(shift)
    out_n, null_tr, _ = evaluate("SENTINEL-E", ALPHA, cfg, reps_null, None,
                                 options=opts, seed0=NULL_SEED0, workers=workers)
    return {
        "pfa": out_n.pfa,
        "pfa_ci": list(out_n.pfa_ci),
        "n_null": out_n.n_null,
        "arl0": out_n.arl0,
        "abstention_null": float(np.mean([t.abstention for t in null_tr])) if null_tr else None,
    }


def signal_part(shift: float, opts: Dict, reps_sig: int, workers: int) -> Dict:
    """Unit (shift, variant, "signal"): post-change streams for one variant."""
    cfg = base_cfg(shift)
    out_s, _, sig_tr = evaluate("SENTINEL-E", ALPHA, cfg, reps_sig, CHANGE_POINT,
                                options=opts, seed0=SIGNAL_SEED0, workers=workers)
    cens = out_s.add_censored
    return {
        "miss_rate": out_s.miss_rate,
        "add_censored_frames": cens,
        "add_censored_s": None if cens is None else float(cens) / FPS,
        "abstention_signal": float(np.mean([t.abstention for t in sig_tr])) if sig_tr else None,
        "n_signal": out_s.n_signal,
    }


def assemble_cell(null: Dict, sig: Dict) -> Dict:
    """Combine a variant's null and signal units into one cell (same keys as before)."""
    lo, hi = null["pfa_ci"]
    return {
        "pfa": null["pfa"],
        "pfa_ci": [lo, hi],
        "verdict": verdict(lo, hi),
        "n_null": null["n_null"],
        "arl0": null["arl0"],
        "miss_rate": sig["miss_rate"],
        "add_censored_frames": sig["add_censored_frames"],
        "add_censored_s": sig["add_censored_s"],
        "abstention_signal": sig["abstention_signal"],
        "abstention_null": null["abstention_null"],
        "n_signal": sig["n_signal"],
    }


def decide(cells: Dict) -> Dict:
    """Apply the pre-specified rule mechanically to the computed cells."""
    w, u = "residual + LR wts", "residual"
    cond_a = all(cells[str(a)][w]["verdict"] != "invalid" for a in ACTIVITY_SHIFTS)
    mu, mw = cells["1.0"][u]["miss_rate"], cells["1.0"][w]["miss_rate"]
    if mu is None or mw is None or not np.isfinite(mu) or not np.isfinite(mw):
        cond_b, delta = False, None
    else:
        delta = float(mw - mu)
        cond_b = delta <= 0.05
    return {
        "rule": DECISION_RULE,
        "condition_a_no_invalid_verdict_weighted": bool(cond_a),
        "verdicts_weighted": {str(a): cells[str(a)][w]["verdict"] for a in ACTIVITY_SHIFTS},
        "condition_b_miss_delta_at_1.0": delta,
        "condition_b_threshold": 0.05,
        "condition_b_holds": bool(cond_b),
        "recommend_weighted_default": bool(cond_a and cond_b),
    }


def make_table(cells: Dict, gates: Dict, decision: Dict, reps_null: int, reps_sig: int) -> None:
    rows = []
    for a in ACTIVITY_SHIFTS:
        for i, v in enumerate(VARIANTS):
            c = cells[str(a)][v]
            lo, hi = c["pfa_ci"]
            if v == "residual":
                gate_txt = "n/a (off)"
            else:
                g = gates[str(a)]
                gate_txt = (f"{g['frac_streams_triggered']:.2f} / "
                            f"{g['mean_frac_blocks_triggered']:.2f}")
            rows.append([
                f"$\\times {a}$" if i == 0 else "",
                v,
                fmt(c["pfa"], 4),
                f"[{lo:.4f}, {hi:.4f}]",
                c["verdict"],
                fmt(c["miss_rate"], 3),
                fmt(c["add_censored_s"], 1),
                fmt(c["abstention_signal"], 3),
                gate_txt,
            ])
    d = decision
    notes = (
        f"Null PFA over {reps_null} streams and signal metrics over {reps_sig} "
        f"post-change streams per row, alpha = {ALPHA}. Verdict: valid if the Wilson "
        "95\\% upper bound is at most alpha; invalid if the lower bound exceeds alpha; "
        "otherwise unresolved. Gate trigger rate: fraction of null streams with at "
        "least one shift block whose held-out AUC reaches the threshold / mean "
        f"fraction of triggered blocks ({GATE_STREAMS} streams per shift). "
        f"Decision rule: {DECISION_RULE} "
        f"Outcome: condition (a) {'holds' if d['condition_a_no_invalid_verdict_weighted'] else 'fails'}; "
        f"condition (b) miss-rate delta at x1.0 = "
        f"{'n/a' if d['condition_b_miss_delta_at_1.0'] is None else format(d['condition_b_miss_delta_at_1.0'], '.3f')}, "
        f"{'holds' if d['condition_b_holds'] else 'fails'}. "
        f"Recommendation: {'make residual + LR wts the default' if d['recommend_weighted_default'] else 'keep the unweighted residual default'}."
    )
    write_latex_table(
        rows,
        ["activity shift", "variant", "PFA", "95\\% CI", "verdict", "miss rate",
         "censored delay (s)", "abstention", "gate trigger rate"],
        caption=(
            r"Activity shift at $\alpha=0.01$ for the two candidate defaults, with the "
            r"held-out gate on and off, at the replicate count needed to resolve the "
            r"false-alarm level."
        ),
        label="tab:shift_default",
        name="tab_shift_default",
        align="llrlllrrrr",
        notes=notes,
    )


def main(reps_null: int, reps_sig: int, gate_streams: int, workers: int, smoke: bool,
         no_cache: bool = False):
    t0 = time.time()
    print(f"Experiment 3b: default-setting check (null={reps_null}, signal={reps_sig}, "
          f"gate={gate_streams}, workers={workers}, smoke={smoke})", flush=True)
    cache = UnitCache("exp3b_shift_default", enabled=not no_cache, smoke=smoke)
    cells: Dict[str, Dict] = {}
    gates: Dict[str, Dict] = {}
    for a in ACTIVITY_SHIFTS:
        cells[str(a)] = {}
        for v, opts in VARIANTS.items():
            base = dict(shift=a, variant=v, opts=opts, cfg=base_cfg(a))
            nkey = f"x{a}__{safe_name(v)}__null"
            skey = f"x{a}__{safe_name(v)}__signal"
            null = cache.get_or_compute(
                nkey, dict(base, part="null", reps=reps_null, seed0=NULL_SEED0),
                lambda: null_part(a, opts, reps_null, workers))
            sig = cache.get_or_compute(
                skey, dict(base, part="signal", reps=reps_sig, seed0=SIGNAL_SEED0),
                lambda: signal_part(a, opts, reps_sig, workers))
            cells[str(a)][v] = c = assemble_cell(null, sig)
            print(f"  x{a:<4} {v:18s} PFA={c['pfa']:.4f} CI=[{c['pfa_ci'][0]:.4f},"
                  f"{c['pfa_ci'][1]:.4f}] {c['verdict']:10s} miss={c['miss_rate']:.3f} "
                  f"cens_s={c['add_censored_s']} abst={c['abstention_signal']:.3f}",
                  flush=True)
        gates[str(a)] = cache.get_or_compute(
            f"x{a}__gate",
            dict(shift=a, part="gate", streams=gate_streams, seed0=NULL_SEED0,
                 cfg=base_cfg(a)),
            lambda: gate_stats(a, gate_streams, workers))
        g = gates[str(a)]
        print(f"  x{a:<4} gate: streams>=1 triggered={g['frac_streams_triggered']:.3f} "
              f"mean block frac={g['mean_frac_blocks_triggered']:.3f}", flush=True)
    print(f" units computed={cache.computed} loaded from cache={cache.hits}", flush=True)
    decision = decide(cells)
    runtime = time.time() - t0
    print(f" decision: {decision}", flush=True)
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
        "variants": VARIANTS,
        "cells": cells,
        "gate": gates,
        "decision": decision,
        "runtime_s": runtime,
    }
    save_json(payload, "exp3b_shift_default")
    make_table(cells, gates, decision, reps_null, reps_sig)
    print(f" wrote results/exp3b_shift_default.json and results/tables/tab_shift_default.tex "
          f"in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-null", type=int, default=REPS_NULL)
    ap.add_argument("--reps-sig", type=int, default=REPS_SIGNAL)
    ap.add_argument("--gate-streams", type=int, default=GATE_STREAMS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--smoke", action="store_true",
                    help="run with the given counts, writes nothing; uses the "
                         "results/cache/exp3b_shift_default_smoke cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the unit cache (neither read nor write it)")
    a = ap.parse_args()
    main(a.reps_null, a.reps_sig, a.gate_streams, a.workers, a.smoke, a.no_cache)

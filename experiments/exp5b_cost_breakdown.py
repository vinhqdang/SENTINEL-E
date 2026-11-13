"""Experiment 5b --- where the per-frame cost of the guarantee layer comes from.

Experiment 5 reported a headline end-to-end cost (about 0.6 microseconds per
frame) and, separately, a per-bet cost for the e-detector step (about 18
microseconds for the 32-point changepoint mixture).  Those two numbers come from
different code paths and cannot be derived from one another.  This script
measures the components of the *default* pipeline on one machine, single
thread, and checks whether the amortised prediction matches the measured
end-to-end cost.

Pipeline facts this relies on (``sentinel_e.pipeline.SentinelE.run``):

* the decorrelation lag ``L`` is estimated from calibration;
* conformal p-values are computed only at the thinned betting instants
  ``thin_indices(T, L)``, not on every frame;
* the episodic detector runs its batch recursion over those betting instants;
* the result is then expanded back onto the full ``T``-frame grid, which costs
  O(T) and is not per-bet work.

So the predicted amortised per-frame cost is
``(p-value cost per bet + episodic step cost per bet) / L``.

Excluded from every number here: context-feature extraction (the backbone
or the context model that produces the four context columns), video decoding,
the frozen backbone itself, and peak memory.  The backbone per-frame time is
taken from ``results/exp5_compute.json``.
"""

from __future__ import annotations

import os

# Single-thread BLAS/OpenMP, set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import platform
import statistics
import time
from typing import Dict, List

import numpy as np

from experiments.common import fmt, load_json, save_json, write_latex_table
from sentinel_e.edetector import EDetector
from sentinel_e.episodic import EpisodePrior, EpisodicEDetector
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices

FPS = 25.0
T_END_TO_END = 200_000
SEED = 0
ALPHA = 0.01


def _times(fn, repeats: int, warmup: int = 1) -> List[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _summ(xs: List[float]) -> Dict[str, float]:
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs),
            "repeats": len(xs)}


def measure(T: int = T_END_TO_END, repeats: int = 5, scalar_n: int = 2_000) -> Dict:
    load_before = os.getloadavg()
    # Identical to exp5's end-to-end benchmark: same stream, seed and calibration.
    st = StreamSimulator(StreamConfig(T=T, calibration_regime="representative"),
                         seed=SEED).simulate_camera()
    model = SentinelE(alpha=ALPHA, calibration="residual", weighted=False)
    model.fit(st.calibration_scores, st.calibration_context)
    lag = model.decorrelation_lag
    idx = thin_indices(T, lag)
    n_bets = int(idx.size)
    s_thin = st.scores[idx]
    c_thin = st.context[idx]

    # (1) conformal p-value cost, on the pipeline's own path, per bet.
    t_p = _times(lambda: model.p_values(s_thin, c_thin), repeats)
    p_thin, active_thin = model.p_values(s_thin, c_thin)
    p_cost_per_bet = _summ(t_p)
    p_cost_per_bet = {k: (v / n_bets if k != "repeats" else v)
                      for k, v in p_cost_per_bet.items()}

    # (2) episodic e-detector: batch path (what the pipeline calls) and the
    # streaming single-step path, both per bet, with the shipped prior.
    det = EpisodicEDetector(alpha=ALPHA, prior=EpisodePrior(rho=1e-3), restart=False)
    def batch():
        det.reset()
        return det.run(p_thin, active=active_thin)
    t_batch = _times(batch, repeats)
    det.reset()
    _, al_batch = det.run(p_thin, active=active_thin)
    alarms_on_null = int(np.sum(al_batch))
    step_batch = {k: (v / n_bets if k != "repeats" else v)
                  for k, v in _summ(t_batch).items()}

    short = p_thin[:scalar_n]
    def stream():
        det.reset()
        for x in short:
            det.step(float(x))
    t_stream = _times(stream, 3)
    step_stream = {k: (v / short.size if k != "repeats" else v)
                   for k, v in _summ(t_stream).items()}

    # Reference: the changepoint mixture with 32 grid points, same streaming step.
    cp = EDetector(alpha=ALPHA, family="power", n_grid=32, restart=False)
    def cp_stream():
        cp.reset()
        for x in short:
            cp.step(float(x))
    t_cp = _times(cp_stream, 3)
    cp_step_stream = {k: (v / short.size if k != "repeats" else v)
                      for k, v in _summ(t_cp).items()}

    # (3) + (4) predicted amortised per-frame cost from the components.
    pred_batch_us = 1e6 * (p_cost_per_bet["median"] + step_batch["median"]) / lag
    pred_stream_us = 1e6 * (p_cost_per_bet["median"] + step_stream["median"]) / lag

    # (5) measured end-to-end monitor time, as exp5 does it.
    t_run = _times(lambda: model.run(st.scores, st.context), repeats)
    e2e = _summ(t_run)
    e2e_us = {k: (1e6 * v / T if k != "repeats" else v) for k, v in e2e.items()}

    # Decomposition of the end-to-end run: per-bet work versus O(T) expansion.
    def per_bet_only():
        p, a = model.p_values(st.scores[thin_indices(T, lag)],
                              st.context[thin_indices(T, lag)])
        det.reset()
        det.run(p, active=a)
    t_perbet = _times(per_bet_only, repeats)
    def expansion_only():
        pos = np.searchsorted(idx, np.arange(T), side="right") - 1
        p_full = np.ones(T)
        act_full = np.zeros(T, dtype=bool)
        lw_full = np.zeros(T)
        al_full = np.zeros(T, dtype=bool)
        p_full[idx] = p_thin
        act_full[idx] = active_thin
        al_full[idx] = al_thin_cache
        lw_full[pos >= 0] = lw_cache[pos[pos >= 0]]
        return p_full
    det.reset()
    lw_cache, al_thin_cache = det.run(p_thin, active=active_thin)
    t_exp = _times(expansion_only, repeats)
    expansion_us = {k: (1e6 * v / T if k != "repeats" else v)
                    for k, v in _summ(t_exp).items()}

    load_after = os.getloadavg()
    return {
        "frames": T,
        "calibration_frames": int(st.calibration_scores.size),
        "decorrelation_lag": int(lag),
        "betting_steps": n_bets,
        "alarms_on_null_stream": alarms_on_null,
        "p_value_sec_per_bet": p_cost_per_bet,
        "episodic_step_sec_per_bet_batch": step_batch,
        "episodic_step_sec_per_bet_streaming": step_stream,
        "changepoint32_step_sec_per_bet_streaming": cp_step_stream,
        "predicted_amortised_us_per_frame_batch": pred_batch_us,
        "predicted_amortised_us_per_frame_streaming": pred_stream_us,
        "measured_end_to_end_run_sec": e2e,
        "measured_end_to_end_us_per_frame": e2e_us,
        "per_bet_work_sec": _summ(t_perbet),
        "expansion_us_per_frame": expansion_us,
        "loadavg_before": list(load_before),
        "loadavg_after": list(load_after),
    }


def backbone_us_per_frame() -> float:
    bb = load_json("exp5_compute")["backbone"]["seconds_per_frame"]
    return 1e6 * float(bb)


def make_table(res: Dict, bb_us: float) -> None:
    lag = res["decorrelation_lag"]
    p_us = 1e6 * res["p_value_sec_per_bet"]["median"]
    sb_us = 1e6 * res["episodic_step_sec_per_bet_batch"]["median"]
    ss_us = 1e6 * res["episodic_step_sec_per_bet_streaming"]["median"]
    cp_us = 1e6 * res["changepoint32_step_sec_per_bet_streaming"]["median"]
    pred = res["predicted_amortised_us_per_frame_batch"]
    e2e = res["measured_end_to_end_us_per_frame"]
    exp_us = res["expansion_us_per_frame"]["median"]
    rows = [
        ["conformal p-value (per bet, pipeline path)", fmt(p_us, 2), "per bet", "--"],
        ["episodic e-detector step, batch (per bet)", fmt(sb_us, 2), "per bet", "--"],
        ["episodic e-detector step, streaming (per bet)", fmt(ss_us, 2), "per bet", "--"],
        ["changepoint 32-point step, streaming (reference)", fmt(cp_us, 2), "per bet", "--"],
        ["decorrelation lag $L$", str(lag), "frames", "--"],
        ["predicted amortised: $(p + \\text{step})/L$ (batch)",
         fmt(pred, 3), "per frame",
         fmt(100 * pred / bb_us, 4)],
        ["O(T) expansion onto frame grid (measured)", fmt(exp_us, 3), "per frame", "--"],
        ["measured end-to-end monitor ({:,} frames)".format(res["frames"]),
         fmt(e2e["median"], 3), "per frame",
         fmt(100 * e2e["median"] / bb_us, 4)],
        ["frozen backbone (reference, from exp5)", fmt(bb_us / 1e3, 2) + " ms", "per frame", "100"],
    ]
    write_latex_table(
        rows,
        ["component", "cost ($\\mu$s)", "unit", "\\% of backbone"],
        caption=(
            r"Cost breakdown of the default SENTINEL-E pipeline, single-thread, measured "
            r"on one machine. Per-bet components are timed on the pipeline's own code "
            r"path; the amortised prediction divides their sum by the decorrelation "
            r"lag, because p-values and the e-detector step run only on betting "
            r"instants. Excluded: context-feature extraction, decoding, the backbone, "
            r"peak memory."
        ),
        label="tab:cost_breakdown",
        name="tab_cost_breakdown",
        align="lrlr",
        notes=(
            r"Median of repeated timings; the measured end-to-end row is "
            r"\texttt{SentinelE.run} on a 200{,}000-frame stream, as in exp5. The "
            r"gap between the predicted and measured rows is the O(T) expansion of "
            r"the betting-grid output onto the frame grid."
        ),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="short run (20,000 frames, 1 repeat); writes nothing")
    args = ap.parse_args(argv)
    if args.smoke:
        res = measure(T=20_000, repeats=1, scalar_n=200)
        print("smoke test ok: lag", res["decorrelation_lag"],
              "e2e us/frame", round(res["measured_end_to_end_us_per_frame"]["median"], 3))
        return
    print("Experiment 5b: cost breakdown (single thread)")
    res = measure()
    bb_us = backbone_us_per_frame()
    print(f"  lag={res['decorrelation_lag']} bets={res['betting_steps']} "
          f"p/bet={1e6*res['p_value_sec_per_bet']['median']:.2f}us "
          f"step/bet(batch)={1e6*res['episodic_step_sec_per_bet_batch']['median']:.2f}us "
          f"step/bet(stream)={1e6*res['episodic_step_sec_per_bet_streaming']['median']:.2f}us")
    print(f"  predicted={res['predicted_amortised_us_per_frame_batch']:.3f} us/frame  "
          f"measured={res['measured_end_to_end_us_per_frame']['median']:.3f} us/frame  "
          f"backbone={bb_us/1e3:.2f} ms/frame")
    payload = {
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor() or "unknown",
                    "threads": {v: os.environ[v] for v in
                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                 "OPENBLAS_NUM_THREADS")}},
        "backbone_us_per_frame": bb_us,
        "source_backbone": "results/exp5_compute.json",
        "excluded": ["context-feature extraction", "video decoding",
                     "frozen backbone inference", "peak memory"],
        "measurement": res,
    }
    save_json(payload, "exp5b_cost_breakdown")
    make_table(res, bb_us)
    print(" wrote results/exp5b_cost_breakdown.json, tab_cost_breakdown")


if __name__ == "__main__":
    main()

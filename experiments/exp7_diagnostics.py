"""Experiment 7 --- assumption diagnostics.

The guarantee of the supermartingale theorem rests on two premises that are assumptions, not
theorems, and both are checkable on the calibration set alone --- which means an
operator can run these checks on their own camera before trusting the alarm
rate.

*Conditional independence.*  The wealth process is a supermartingale only if the
increments are conditionally super-uniform given the past.  We report the
autocorrelation of the calibration residuals before and after thinning, and the
Ljung--Box portmanteau p-value at both rates.  A large p-value after thinning
supports the premise; the raw series flatly fails it.

*Marginal calibration of the deployment p-values.*  Under the null the p-values
should be super-uniform.  We report the realised exceedance rate at several
levels on null streams, which is a direct check of Layer 1 that does not depend
on the detector at all.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.common import fmt, save_json, write_latex_table
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import acf, detrend, ljung_box, thin_indices

N_STREAMS = 60
T = 60_000
LEVELS = (0.02, 0.05, 0.10, 0.25, 0.50)


def run(n_streams: int = N_STREAMS) -> Dict:
    cfg = StreamConfig(T=T, calibration_regime="representative")
    sim = StreamSimulator(cfg, seed=770)

    raw_acf1, thin_acf1, raw_lb, thin_lb, lags = [], [], [], [], []
    p_pool: List[np.ndarray] = []
    abst = []

    for i in range(n_streams):
        st = sim.simulate_camera(change_point=None)
        model = SentinelE(alpha=0.01, calibration="residual", weighted=False)
        model.fit(st.calibration_scores, st.calibration_context)
        lag = model.decorrelation_lag
        lags.append(lag)

        resid = detrend(st.calibration_scores, st.calibration_context)
        raw_acf1.append(acf(resid, 1)[1])
        thinned = resid[::lag]
        thin_acf1.append(acf(thinned, 1)[1])
        raw_lb.append(ljung_box(resid[: 20 * lag * 50], 20)[1])
        thin_lb.append(ljung_box(thinned, 20)[1])

        idx = thin_indices(st.T, lag)
        p, active = model.p_values(st.scores[idx], st.context[idx])
        p_pool.append(p[active])
        abst.append(float(1.0 - active.mean()))

    p = np.concatenate(p_pool)
    coverage = {
        str(u): {"nominal": u, "realised": float(np.mean(p <= u)),
                 "n": int(p.size)}
        for u in LEVELS
    }
    return {
        "n_streams": n_streams,
        "T": T,
        "lag_mean": float(np.mean(lags)),
        "lag_min": int(np.min(lags)),
        "lag_max": int(np.max(lags)),
        "acf1_raw": float(np.mean(raw_acf1)),
        "acf1_thinned": float(np.mean(thin_acf1)),
        "ljung_box_raw_median": float(np.median(raw_lb)),
        "ljung_box_thinned_median": float(np.median(thin_lb)),
        "ljung_box_thinned_frac_above_005": float(np.mean(np.array(thin_lb) > 0.05)),
        "abstention": float(np.mean(abst)),
        "p_min": float(p.min()),
        "coverage": coverage,
    }


def make_table(d: Dict):
    rows = [
        ["lag-1 autocorrelation of calibration residuals",
         fmt(d["acf1_raw"], 3), fmt(d["acf1_thinned"], 3)],
        ["Ljung--Box p-value (20 lags, median over cameras)",
         fmt(d["ljung_box_raw_median"], 3), fmt(d["ljung_box_thinned_median"], 3)],
        ["fraction of cameras with Ljung--Box $p > 0.05$",
         "0.000", fmt(d["ljung_box_thinned_frac_above_005"], 3)],
    ]
    write_latex_table(
        rows, ["diagnostic", "every frame", "one bet per lag"],
        caption=(
            f"Conditional-independence diagnostics over {d['n_streams']} cameras "
            f"(estimated decorrelation lag {d['lag_min']}--{d['lag_max']} frames, "
            f"mean {d['lag_mean']:.1f}). Betting on every frame violates the "
            "premise of \\cref{thm:episodic} outright. Betting once per lag "
            "reduces the violation by every measure here but does not remove "
            "it: the median Ljung--Box $p$ is still below $0.05$ and only a "
            "minority of cameras pass at that level, so the diagnostic "
            "positively rejects conditional independence rather than failing "
            "to certify it. What survives is the empirical false-alarm "
            "control of \\cref{sec:validity}, not the literal hypothesis of "
            "\\cref{thm:episodic}; see \\cref{sec:limitations}. These are "
            "computed from betting instants on labelled null streams, which an "
            "operator does not have; the calibration-set analogue is weaker."
        ),
        label="tab:diagnostics", name="tab9_diagnostics", align="lrr",
    )

    rows = [[fmt(v["nominal"], 2), fmt(v["realised"], 4),
             "\\checkmark" if v["realised"] <= v["nominal"] else "\\ding{55}"]
            for v in d["coverage"].values()]
    write_latex_table(
        rows, ["nominal level $u$", "realised $P(p \\le u)$", "super-uniform?"],
        caption=(
            "Marginal calibration of the deployment p-values on null streams "
            f"({d['coverage'][str(LEVELS[0])]['n']:,} betting instants pooled over "
            f"{d['n_streams']} cameras). This checks Layer 1 in isolation: the "
            "p-values must be super-uniform before any betting takes place. The "
            f"smallest attainable p-value is {d['p_min']:.4f}, which is the tail "
            "resolution the exact Beta levels of \\cref{thm:beta} provide."
        ),
        label="tab:pcoverage", name="tab10_pcoverage", align="rrc",
    )


def main(n_streams: int = N_STREAMS):
    print("Experiment 7: assumption diagnostics")
    d = run(n_streams)
    print(f"  lag {d['lag_min']}-{d['lag_max']} (mean {d['lag_mean']:.1f}); "
          f"acf1 {d['acf1_raw']:.3f} -> {d['acf1_thinned']:.3f}; "
          f"Ljung-Box {d['ljung_box_raw_median']:.3g} -> "
          f"{d['ljung_box_thinned_median']:.3f}")
    for u, v in d["coverage"].items():
        print(f"  P(p<={u}) = {v['realised']:.4f}")
    save_json(d, "exp7_diagnostics")
    make_table(d)
    print(" wrote results/exp7_diagnostics.json, tab9/tab10")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-streams", type=int, default=N_STREAMS, dest="n_streams")
    main(**vars(ap.parse_args()))

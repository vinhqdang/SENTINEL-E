"""Extend the exp2 knob sweeps and merge, so 'cannot' is not 'did not try'.

Several baselines never reach a tolerable false-alarm rate anywhere in their
original sweep.  That is only a fair claim if the sweep went far enough, so this
script pushes each knob well past the point of diminishing returns and merges
the extra operating points into the existing payload.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.common import load_json, save_json
from experiments.exp2_delay_far import (
    BASE_CFG,
    CHANGE_POINT,
    make_figure,
    make_table,
)
from experiments.runner import evaluate

EXTRA: Dict[str, List[float]] = {
    "Fixed threshold":       [3e-6, 1e-6, 1e-7],
    "CUSUM":                 [26.0, 34.0, 45.0, 60.0],
    "Shiryaev--Roberts":     [1e9, 1e10, 1e12, 1e14],
    "Parametric e-detector": [1e-3, 1e-4, 1e-6, 1e-8],
    "E-SHIFT":               [1e-3, 1e-4, 1e-6, 1e-8],
    "p-value threshold":     [3e-6, 1e-6],
    "SENTINEL-E":            [1e-3, 1e-4],
}


def main(reps: int = 200):
    payload = load_json("exp2_delay_far")
    results = payload["results"]
    for method, knobs in EXTRA.items():
        seen = {r["knob"] for r in results.get(method, [])}
        for k in knobs:
            if k in seen:
                continue
            out, null, _ = evaluate(method, k, BASE_CFG, reps, CHANGE_POINT)
            results.setdefault(method, []).append({
                "knob": k, "pfa": out.pfa, "pfa_ci": list(out.pfa_ci),
                "arl0": out.arl0, "add": out.add, "add_se": out.add_se,
                "add_median": out.add_median,
                "add_censored": out.add_censored,
                "add_censored_se": out.add_censored_se,
                "miss_rate": out.miss_rate, "lag": null[0].lag,
            })
            add = out.add if out.add is not None and np.isfinite(out.add) else float("nan")
            print(f"  {method:24s} knob={k:<9g} PFA={out.pfa:.3f} "
                  f"ADD={add:8.1f} miss={out.miss_rate:.2f}", flush=True)
        results[method].sort(key=lambda r: r["pfa"])

    payload["results"] = results
    payload["extended"] = True
    save_json(payload, "exp2_delay_far")
    make_figure(results)
    make_table(results)
    print(" merged; regenerated fig2 and tab3")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    main(**vars(ap.parse_args()))

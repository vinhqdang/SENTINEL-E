"""Run the complete SENTINEL-E experimental protocol.

    python experiments/run_all.py            # full protocol (~30 min on 4 cores)
    python experiments/run_all.py --quick    # reduced replication, for a smoke test

Every experiment writes a JSON payload to ``results/``, figures to
``results/figures/`` and LaTeX tables to ``results/tables/``.  Results are
deterministic given the seeds fixed in each script.
"""

from __future__ import annotations

import argparse
import time

from experiments import (
    exp1_validity,
    exp2_delay_far,
    exp3_shift,
    exp4_network,
    exp5_compute,
    exp6_ablation,
    exp7_diagnostics,
)


def main(quick: bool = False, only: str = ""):
    reps = 40 if quick else None
    plan = [
        ("exp1", lambda: exp1_validity.main(reps or exp1_validity.REPS)),
        ("exp2", lambda: exp2_delay_far.main(reps or exp2_delay_far.REPS)),
        ("exp3", lambda: exp3_shift.main(reps or exp3_shift.REPS)),
        ("exp4", lambda: exp4_network.main(
            n_train=3 if quick else exp4_network.N_TRAIN_FLEETS,
            n_eval=6 if quick else exp4_network.N_EVAL_FLEETS,
            epochs=3 if quick else 25,
        )),
        ("exp5", exp5_compute.main),
        ("exp6", lambda: exp6_ablation.main(reps or exp6_ablation.REPS)),
        ("exp7", lambda: exp7_diagnostics.main(
            10 if quick else exp7_diagnostics.N_STREAMS)),
    ]
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    for name, fn in plan:
        if wanted and name not in wanted:
            continue
        t0 = time.time()
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}", flush=True)
        fn()
        print(f"[{name} finished in {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced replication for a smoke test")
    ap.add_argument("--only", default="",
                    help="comma-separated subset, e.g. 'exp2,exp4'")
    main(**vars(ap.parse_args()))

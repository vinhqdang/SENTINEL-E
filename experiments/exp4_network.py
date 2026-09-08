"""Experiment 4 --- the camera graph: spatial prior and fleet-level FDR.

Two distinct questions, often conflated in multi-camera work.

*Does the graph shorten detection delay?*  Offenders relocate, so evidence
accumulating at one camera is genuine prior information about its neighbours.
The spatial prior acts only through a **predictable** hazard and stake
modulation, so by Theorem 2 it cannot damage the false-alarm guarantee whatever
it outputs.  That is checked here empirically as well as proved: the
all-null fleet columns must stay at or below the nominal level for the learned
controller exactly as they do without it.

*How should fleet-wide decisions be made?*  Adjacent cameras share weather,
lighting and traffic, and the graph layer deliberately couples them further, so
independence is untenable.  e-BH controls the false-discovery rate under
arbitrary dependence with no penalty, which is what the comparison against
per-camera thresholding and against BH-on-p is there to show.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from experiments.common import (
    FIGURES,
    fmt,
    fmt_int,
    save_json,
    setup_matplotlib,
    write_latex_table,
)
from sentinel_e.ebh import ebh
from sentinel_e.gnn import HeuristicSpatialPrior, SpatialPriorConfig, train_spatial_prior
from sentinel_e.metrics import detection_delay, fdr_power, first_alarm_index
from sentinel_e.pipeline import FleetSentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator

FPS = 25.0
ALPHA = 0.01
FDR_LEVEL = 0.1
T_TRAIN = 30_000
T_EVAL = 60_000
N_TRAIN_FLEETS = 10
N_EVAL_FLEETS = 40
N_CAMERAS = 12


def _cfg(T: int) -> StreamConfig:
    return StreamConfig(T=T, n_cameras=N_CAMERAS, calibration_regime="representative")


def build_training_set(n_fleets: int = N_TRAIN_FLEETS, seed0: int = 900):
    """Conformal p-values and graphs for the fleets the controller learns from."""
    p_batches, adjs, statics, degs, cps = [], [], [], [], []
    for i in range(n_fleets):
        fleet = StreamSimulator(_cfg(T_TRAIN), seed=seed0 + i).simulate_fleet(n_events=3)
        det = FleetSentinelE(alpha=ALPHA, calibration="residual", weighted=False)
        p, active, idx = det.compute_p_values(fleet)
        # Abstentions become neutral bets, which is a p-value of one.
        p = np.where(active, p, 1.0)
        p_batches.append(p)
        adjs.append(fleet.graph.normalized_adjacency())
        ctx = fleet.graph.context
        statics.append(np.zeros((N_CAMERAS, 2)) if ctx is None else np.asarray(ctx)[:, :2])
        deg = fleet.graph.degree
        degs.append(deg / max(deg.max(), 1e-9))
        # Change points must be expressed on the betting grid the model sees.
        cps.append([
            None if cp is None else int(np.searchsorted(idx, cp))
            for cp in fleet.change_points
        ])
        print(f"    fleet {i}: {p.shape[1]} betting steps, "
              f"{sum(c is not None for c in cps[-1])} affected cameras", flush=True)
    # Trim to a common horizon so the batch stacks.
    B = min(p.shape[1] for p in p_batches)
    p_batches = [p[:, :B] for p in p_batches]
    return p_batches, adjs, statics, degs, cps


def evaluate_fleets(controller_name: str, model=None, n_fleets: int = N_EVAL_FLEETS,
                    seed0: int = 4_000) -> Dict:
    """Run a controller over held-out fleets, with and without real events."""
    delays: List[int] = []
    misses = 0
    fdps, powers, n_rej = [], [], []
    null_fleet_alarm = 0
    null_fleet_rejections = 0

    for i in range(n_fleets):
        for null_only in (False, True):
            fleet = StreamSimulator(
                _cfg(T_EVAL), seed=seed0 + i + (500_000 if null_only else 0)
            ).simulate_fleet(n_events=0 if null_only else 3)

            kw = dict(alpha=ALPHA, calibration="residual", weighted=False,
                      family="linear", n_grid=16, fdr_level=FDR_LEVEL, restart=False)
            if controller_name == "none":
                det = FleetSentinelE(controller=None, **kw)
            elif controller_name == "heuristic":
                prior = HeuristicSpatialPrior(fleet.graph.normalized_adjacency())
                det = FleetSentinelE(controller=prior, **kw)
            else:
                det = FleetSentinelE.from_gnn(model, fleet.graph, **kw)
            res = det.run(fleet)

            truth = fleet.affected
            if null_only:
                null_fleet_alarm += int(res.alarms.any())
                null_fleet_rejections += int(res.ebh_rejected.sum())
                continue

            for k in range(fleet.n_cameras):
                cp = fleet.change_points[k]
                if cp is None:
                    continue
                # Convert the onset to the betting grid, then delays back to frames.
                cp_bet = int(np.searchsorted(res.bet_index, cp))
                d = detection_delay(res.alarms[k], min(cp_bet, res.alarms.shape[1] - 1))
                if d is None:
                    misses += 1
                else:
                    delays.append(d * (res.bet_index[1] - res.bet_index[0]))
            m = fdr_power(res.ebh_rejected, truth)
            fdps.append(m["fdp"])
            powers.append(m["power"])
            n_rej.append(m["n_rejected"])

    n_events = len(delays) + misses
    return {
        "controller": controller_name,
        "add": float(np.mean(delays)) if delays else float("nan"),
        "add_se": float(np.std(delays, ddof=1) / np.sqrt(len(delays)))
        if len(delays) > 1 else float("nan"),
        "add_median": float(np.median(delays)) if delays else float("nan"),
        "miss_rate": misses / n_events if n_events else float("nan"),
        "n_events": n_events,
        "fdr": float(np.mean(fdps)) if fdps else float("nan"),
        "power": float(np.nanmean(powers)) if powers else float("nan"),
        "mean_rejections": float(np.mean(n_rej)) if n_rej else float("nan"),
        "null_fleet_alarm_rate": null_fleet_alarm / n_fleets,
        "null_fleet_false_rejections": null_fleet_rejections / n_fleets,
    }


def make_figure(rows):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    labels = {"none": "no graph", "heuristic": "heuristic prior", "gnn": "learned GNN prior"}
    names = [labels[r["controller"]] for r in rows]
    colors = ["#7ab3f5", "#3f7fd1", "#0b3d91"]
    x = np.arange(len(rows))

    ax = axes[0]
    y = np.array([r["add"] for r in rows]) / FPS
    e = np.array([r["add_se"] for r in rows]) / FPS * 1.96
    ax.bar(x, y, yerr=e, color=colors, capsize=3, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("detection delay (s)"); ax.set_title("(a) delay")

    ax = axes[1]
    ax.bar(x, [r["power"] for r in rows], color=colors, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("e-BH power"); ax.set_ylim(0, 1.05); ax.set_title("(b) fleet power")

    ax = axes[2]
    ax.bar(x, [r["fdr"] for r in rows], color=colors, width=0.6)
    ax.axhline(FDR_LEVEL, color="k", ls="--", lw=1.0, label=f"target {FDR_LEVEL}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("realised FDR"); ax.set_ylim(0, max(0.15, FDR_LEVEL * 1.4))
    ax.set_title("(c) fleet FDR"); ax.legend()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig4_network.{ext}")
    plt.close(fig)


def make_table(rows):
    labels = {"none": "independent detectors", "heuristic": "heuristic spatial prior",
              "gnn": "learned GNN spatial prior"}
    out = []
    for r in rows:
        out.append([
            labels[r["controller"]],
            f"{fmt(r['add'] / FPS, 1)} $\\pm$ {fmt(1.96 * r['add_se'] / FPS, 1)}",
            fmt(r["miss_rate"], 2),
            fmt(r["power"], 3),
            fmt(r["fdr"], 3),
            fmt(r["null_fleet_alarm_rate"], 3),
            fmt(r["null_fleet_false_rejections"], 3),
        ])
    write_latex_table(
        out,
        ["spatial prior", "delay (s)", "miss", "e-BH power", "e-BH FDR",
         "null fleet PFA", "null rejections"],
        caption=(
            f"Fleet of {N_CAMERAS} cameras, {N_EVAL_FLEETS} held-out fleets with "
            f"events and {N_EVAL_FLEETS} all-null fleets, "
            r"$\alpha=0.01$ per camera and e-BH at level "
            f"{FDR_LEVEL}. The last two columns are the guarantee check: whatever "
            "the learned controller outputs, an all-null fleet must not alarm "
            "more often than the nominal level allows, because the modulation is "
            "predictable (Theorem 2)."
        ),
        label="tab:network",
        name="tab6_network",
        align="lrrrrrr",
    )


def main(n_train: int = N_TRAIN_FLEETS, n_eval: int = N_EVAL_FLEETS, epochs: int = 25):
    print("Experiment 4: camera graph, spatial prior and fleet-level FDR")
    print(" building training fleets ...")
    p_batches, adjs, statics, degs, cps = build_training_set(n_train)
    print(" training the spatial prior ...")
    model = train_spatial_prior(
        p_batches, adjs, statics, degs, cps,
        alpha=ALPHA, n_grid=16, epochs=epochs, verbose=True,
        config=SpatialPriorConfig(hidden=32, n_layers=2),
    )
    rows = []
    for name, m in (("none", None), ("heuristic", None), ("gnn", model)):
        print(f" evaluating controller: {name} ...", flush=True)
        r = evaluate_fleets(name, m, n_eval)
        rows.append(r)
        print(f"   delay={r['add'] / FPS:.1f}s power={r['power']:.3f} "
              f"FDR={r['fdr']:.3f} null-PFA={r['null_fleet_alarm_rate']:.3f}", flush=True)

    save_json({"rows": rows, "n_train": n_train, "n_eval": n_eval,
               "alpha": ALPHA, "fdr_level": FDR_LEVEL, "n_cameras": N_CAMERAS,
               "T_eval": T_EVAL, "train_loss": list(getattr(model, "history", []))},
              "exp4_network")
    make_figure(rows)
    make_table(rows)
    print(" wrote results/exp4_network.json, fig4_network, tab6")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=N_TRAIN_FLEETS, dest="n_train")
    ap.add_argument("--n-eval", type=int, default=N_EVAL_FLEETS, dest="n_eval")
    ap.add_argument("--epochs", type=int, default=25)
    main(**vars(ap.parse_args()))

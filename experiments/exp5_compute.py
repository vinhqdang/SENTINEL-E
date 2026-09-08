"""Experiment 5 --- computational cost of the guarantee layer.

SENTINEL-E is a wrapper: the backbone is frozen and untouched, and the guarantee
layer only ever sees one float per frame.  This experiment measures what that
wrapper actually costs, in absolute terms and relative to the detector it wraps,
so the edge-deployment claim rests on a measurement rather than on an argument
from asymptotics.

The reference backbone is a depthwise-separable convolutional stack of the size
typically deployed on municipal edge hardware.  It is a *stand-in*, timed on the
same CPU as everything else, not a reproduction of any published detector; the
point of the comparison is the order of magnitude, and that conclusion is robust
to a factor of several either way.
"""

from __future__ import annotations

import platform
import time
from typing import Dict, List

import numpy as np

from experiments.common import fmt, save_json, write_latex_table
from sentinel_e.conformal import ResidualConformalCalibrator
from sentinel_e.edetector import EDetector
from sentinel_e.graph import CameraGraph
from sentinel_e.streams import StreamConfig, StreamSimulator
from sentinel_e.temporal import thin_indices

FPS = 25.0


def _time(fn, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def reference_backbone_cost(n_frames: int = 30) -> Dict[str, float]:
    """Time a lightweight depthwise-separable backbone on one 320x320 frame."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return {"available": 0.0}

    torch.set_num_threads(1)

    def block(cin, cout, stride):
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        block(32, 64, 1), block(64, 128, 2), block(128, 128, 1),
        block(128, 256, 2), block(256, 256, 1), block(256, 512, 2),
        *[block(512, 512, 1) for _ in range(5)],
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1),
    ).eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        per_frame = _time(lambda: net(x), n_frames)
    n_params = sum(p.numel() for p in net.parameters())
    return {"available": 1.0, "seconds_per_frame": per_frame, "parameters": float(n_params)}


def layer_costs(n_grid_values=(8, 16, 32, 64), n_steps: int = 20_000) -> List[Dict]:
    """Per-betting-step cost of the e-detector as the mixture grid grows."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=n_steps)
    rows = []
    for k in n_grid_values:
        det = EDetector(alpha=0.01, family="power", n_grid=k, restart=False)
        det.reset()
        t_vec = _time(lambda: (det.reset(), det.run(p))[1], 3) / n_steps
        det.reset()
        # Scalar path: the per-frame cost of a genuine streaming deployment,
        # which cannot batch because the frame has not happened yet.
        short = p[:2000]
        def scalar():
            det.reset()
            for x in short:
                det.step(x)
        t_scalar = _time(scalar, 3) / short.size
        rows.append({"n_grid": k, "sec_per_step_vectorised": t_vec,
                     "sec_per_step_streaming": t_scalar})
    return rows


def conformal_cost(n_cal_values=(2_000, 10_000, 50_000), n_test: int = 20_000) -> List[Dict]:
    rng = np.random.default_rng(1)
    rows = []
    for n in n_cal_values:
        x = rng.normal(size=(n, 4))
        s = x[:, 0] * 0.5 + rng.normal(size=n)
        cal = ResidualConformalCalibrator(delta=1e-3)
        t_fit = _time(lambda: cal.fit(s, x), 3)
        cal.fit(s, x)
        xt = rng.normal(size=(n_test, 4))
        st = rng.normal(size=n_test)
        t_score = _time(lambda: cal.p_values(st, xt), 5) / n_test
        rows.append({"n_calibration": n, "fit_seconds": t_fit,
                     "sec_per_frame": t_score})
    return rows


def fleet_cost(sizes=(8, 16, 32, 64, 128), n_steps: int = 500) -> List[Dict]:
    """Cost of the graph controller per betting step as the fleet grows."""
    try:
        import torch
        from sentinel_e.gnn import SpatialPriorGNN
    except Exception:
        return []
    torch.set_num_threads(1)
    rng = np.random.default_rng(2)
    model = SpatialPriorGNN()
    rows = []
    for K in sizes:
        g = CameraGraph.from_positions(rng.uniform(0, 2000, size=(K, 2)), k_neighbors=4)
        a = g.normalized_adjacency()
        feats = rng.normal(size=(K, 8))
        t = _time(lambda: model.predict(feats, a), 30)
        rows.append({"n_cameras": K, "sec_per_step": t, "sec_per_camera_step": t / K})
    return rows


def end_to_end(T: int = 200_000) -> Dict:
    """Wall-clock cost of monitoring a real-length stream, end to end."""
    from sentinel_e.pipeline import SentinelE

    st = StreamSimulator(StreamConfig(T=T, calibration_regime="representative"),
                         seed=0).simulate_camera()
    model = SentinelE(alpha=0.01, calibration="residual", weighted=False)
    t_fit = _time(lambda: model.fit(st.calibration_scores, st.calibration_context), 1, 0)
    model.fit(st.calibration_scores, st.calibration_context)
    t_run = _time(lambda: model.run(st.scores, st.context), 3)
    lag = model.decorrelation_lag
    n_bets = thin_indices(T, lag).size
    return {
        "frames": T,
        "video_seconds": T / FPS,
        "calibration_fit_seconds": t_fit,
        "monitor_seconds": t_run,
        "decorrelation_lag": lag,
        "betting_steps": int(n_bets),
        "realtime_factor": (T / FPS) / t_run,
        "microseconds_per_frame": 1e6 * t_run / T,
    }


def main():
    print("Experiment 5: computational cost")
    backbone = reference_backbone_cost()
    print(f"  reference backbone: {backbone.get('seconds_per_frame', float('nan')) * 1e3:.1f} ms/frame")
    layers = layer_costs()
    conf = conformal_cost()
    fleet = fleet_cost()
    e2e = end_to_end()
    print(f"  end-to-end: {e2e['microseconds_per_frame']:.2f} us/frame, "
          f"{e2e['realtime_factor']:.0f}x real time")

    payload = {"machine": {"platform": platform.platform(),
                           "processor": platform.processor() or "unknown"},
               "backbone": backbone, "detector": layers, "conformal": conf,
               "fleet": fleet, "end_to_end": e2e}
    save_json(payload, "exp5_compute")

    bb = backbone.get("seconds_per_frame")
    rows = [
        ["frozen backbone (reference, 320$\\times$320)",
         fmt(bb * 1e3, 2) if bb else "--", "--"],
        ["conformal p-value (per frame scored)",
         fmt(conf[-1]["sec_per_frame"] * 1e3, 5),
         fmt(100 * conf[-1]["sec_per_frame"] / bb, 4) if bb else "--"],
        ["e-detector step, 32-point mixture (per bet)",
         fmt(layers[2]["sec_per_step_streaming"] * 1e3, 4),
         fmt(100 * layers[2]["sec_per_step_streaming"] / bb, 3) if bb else "--"],
        ["graph controller, 32 cameras (per camera-bet)",
         fmt(fleet[2]["sec_per_camera_step"] * 1e3, 4) if fleet else "--",
         fmt(100 * fleet[2]["sec_per_camera_step"] / bb, 3) if (fleet and bb) else "--"],
        ["\\textbf{SENTINEL-E, amortised per video frame}",
         "\\textbf{" + fmt(e2e["microseconds_per_frame"] / 1e3, 5) + "}",
         "\\textbf{" + (fmt(100 * e2e["microseconds_per_frame"] * 1e-6 / bb, 4) if bb else "--") + "}"],
    ]
    write_latex_table(
        rows,
        ["component", "cost (ms)", "\\% of backbone"],
        caption=(
            "Single-thread CPU cost of each layer, measured on the machine that "
            "produced every other result in this paper. The guarantee layer runs "
            "once per decorrelation lag rather than once per frame, so its "
            "amortised cost is the last row: about "
            f"{100 * e2e['microseconds_per_frame'] * 1e-6 / bb:.4f}\\% of the "
            "backbone it wraps. Deploying SENTINEL-E on hardware that already "
            "runs the detector in real time therefore requires no additional "
            "compute budget."
        ) if bb else "Single-thread CPU cost of each layer.",
        label="tab:compute",
        name="tab7_compute",
        align="lrr",
        notes=(
            r"The reference backbone is a depthwise-separable stack of the scale "
            r"used on edge hardware, timed on the same CPU; it stands in for a "
            r"deployed detector rather than reproducing a specific published one."
        ),
    )
    print(" wrote results/exp5_compute.json, tab7")


if __name__ == "__main__":
    main()

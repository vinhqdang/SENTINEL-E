"""Qualitative figure: real frames, real optical flow, real scores.

Every number in \\cref{sec:real-data} comes from a scalar per-frame score, and
a reader has no way to see what that scalar is actually measuring. This
script extracts one genuinely normal and one genuinely anomalous example from
each real corpus (CUHK Avenue, UCSD Ped2), computes the identical Farneback
optical-flow magnitude :func:`download_and_extract.compute_scores` uses on
the identical 64x64 arrays that pipeline runs on, and renders the frame pair
alongside the flow-magnitude field and the resulting scalar score -- so
``fig_r0_qualitative`` in the manuscript is a picture of the actual pipeline
input, not an illustration of it.

The score and the flow-magnitude heatmap are always computed from the same
64x64, backbone-resolution arrays the rest of this paper's pipeline consumes
-- never from a higher-resolution source. The "frame" panel shown next to
each is, purely for legibility in print, re-read from the dataset's original
higher-resolution release (a 360x240 UCSD Ped2 .tif, a native-resolution
Avenue .avi frame) at the identical frame index; this is disclosed in the
figure caption and affects only what a human looking at the figure sees, not
any number this paper reports.

Only a handful of frames are ever written to disk under
``data/real_vad/<name>/qualitative_examples.npz`` (two 64x64 pairs plus two
native-resolution stills per dataset); the full raw frame cache and original
dataset releases this script reads from are the same large,
deliberately-uncommitted ``/tmp/data/cache/<name>`` and ``/tmp/data/{ucsd,avenue}``
trees ``download_and_extract.py`` produces, so re-running this script from
scratch needs that script run first. Once the small example file exists, the
figure regenerates from it alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.common import FIGURES, save_json, setup_matplotlib
from experiments.real_data.download_and_extract import RAW_ROOT

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_vad"

#: One normal and one anomalous frame index, chosen from a labelled run well
#: inside a single test video (never at a label boundary), so the pair is
#: unambiguously an example of that class rather than a transition frame.
EXAMPLES = {
    "ped2": {"video": "Test001", "normal": 30, "anomalous": 120},
    "avenue": {"video": "01.avi", "normal": 750, "anomalous": 580},
}


def _flow_and_score(prev_f: np.ndarray, cur_f: np.ndarray):
    """Identical recipe to download_and_extract.compute_scores.flow_scores."""
    import cv2

    prev = (prev_f * 255).astype(np.uint8)
    cur = (cur_f * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return mag, float(mag.mean())


def _native_ped2_frame(frame_idx0: int) -> np.ndarray:
    """The original 360x240 .tif for Ped2/Test001, display only."""
    import cv2

    d = RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Test" / "Test001"
    files = sorted(f for f in os.listdir(d) if f.endswith(".tif"))
    im = cv2.imread(str(d / files[frame_idx0]), cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32) / 255.0


def _native_avenue_frame(frame_idx0: int) -> np.ndarray:
    """The original-resolution Avenue test video frame, display only."""
    import cv2

    path = RAW_ROOT / "avenue" / "extracted" / "Avenue Dataset" / "testing_videos" / "01.avi"
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"could not read frame {frame_idx0} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


_NATIVE_READERS = {"ped2": _native_ped2_frame, "avenue": _native_avenue_frame}


def extract_examples(name: str) -> Path:
    """Pull the (prev, cur) 64x64 pairs plus a native-resolution still for
    both examples out of the raw releases, and save just those few frames --
    never the full frame array or original video."""
    out_path = DATA_ROOT / name / "qualitative_examples.npz"
    cache = RAW_ROOT / "cache" / name
    frames_path = cache / "test_frames.npy"
    meta_path = cache / "test_meta.json"
    if not frames_path.exists():
        raise FileNotFoundError(
            f"{frames_path} not found. Run "
            "experiments/real_data/download_and_extract.py first (its raw "
            f"frame cache under {cache} is what this script reads the "
            "example frames from; only the handful it selects is ever "
            "written to the repository)."
        )
    meta = json.loads(meta_path.read_text())
    idx = next(i for i, m in enumerate(meta) if m["name"] == EXAMPLES[name]["video"])
    lo = sum(m["n_frames"] for m in meta[:idx])
    frames = np.load(frames_path, mmap_mode="r")

    payload = {}
    for kind in ("normal", "anomalous"):
        frame_idx0 = EXAMPLES[name][kind]
        t = lo + frame_idx0
        payload[f"{kind}_prev"] = np.array(frames[t - 1])
        payload[f"{kind}_cur"] = np.array(frames[t])
        payload[f"{kind}_native"] = _NATIVE_READERS[name](frame_idx0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def load_examples(name: str) -> dict:
    path = DATA_ROOT / name / "qualitative_examples.npz"
    if not path.exists():
        path = extract_examples(name)
    return dict(np.load(path))


def make_figure():
    plt = setup_matplotlib()
    labels = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}
    fig, axes = plt.subplots(2, 4, figsize=(8.5, 4.4))
    scores = {}

    for row, name in enumerate(("ped2", "avenue")):
        ex = load_examples(name)
        scores[name] = {}
        for col, kind in enumerate(("normal", "anomalous")):
            prev_f, cur_f = ex[f"{kind}_prev"], ex[f"{kind}_cur"]
            native = ex[f"{kind}_native"]
            mag, score = _flow_and_score(prev_f, cur_f)
            scores[name][kind] = score
            ax_frame = axes[row, 2 * col]
            ax_flow = axes[row, 2 * col + 1]
            ax_frame.imshow(native, cmap="gray", vmin=0, vmax=1)
            ax_frame.set_title(f"{kind} frame", fontsize=8)
            ax_flow.imshow(mag, cmap="inferno")
            ax_flow.set_title(f"flow magnitude\nscore={score:.3f}", fontsize=8)
            for ax in (ax_frame, ax_flow):
                ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(labels[name], fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_r0_qualitative.{ext}")
    plt.close(fig)
    return scores


def main():
    for name in ("ped2", "avenue"):
        path = DATA_ROOT / name / "qualitative_examples.npz"
        if not path.exists():
            print(f" extracting {name} example frames from the raw releases ...")
            extract_examples(name)
        else:
            print(f" {name}: using cached {path}")
    scores = make_figure()
    save_json({"examples": EXAMPLES, "scores": scores}, "exp_r0_qualitative")
    print(" wrote results/figures/fig_r0_qualitative.{pdf,png}, "
          "results/exp_r0_qualitative.json")


if __name__ == "__main__":
    main()

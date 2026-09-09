"""Download CUHK Avenue and UCSD Ped2, and compute real per-frame backbone scores.

This is the one script in the paper that touches pixels. Everything else --
every table, every figure, every claim about SENTINEL-E's own behaviour --
consumes only the scalar outputs this script writes to
``data/real_vad/{ped2,avenue}/``. Run it once:

    python -m experiments.real_data.download_and_extract

It downloads the official releases directly (no account, no request form,
~1.5GB total):

  Avenue   http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
           http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip
  Ped2     http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz

Backbone. Neither dataset ships per-frame detector scores, only raw video, so
a real backbone has to be chosen. We use Farneback dense optical-flow
magnitude (``cv2.calcOpticalFlowFarneback``), averaged over each frame,
computed at 64x64 grayscale resolution. This is a deliberately simple,
classical, unsupervised choice -- not a claim of state-of-the-art detection
-- for three reasons the manuscript's data section states explicitly: (i) it
needs no GPU and no training, so it is reproducible on any machine within
minutes; (ii) it is a real, if weak, per-frame signal (frame-level AUC
0.76 on Ped2, 0.78 on Avenue -- see ``backbone_auc.json`` in each dataset's
folder), which is the regime this paper's whole argument is about: a signal
too weak to threshold safely on its own, that a sequential detector must
accumulate; (iii) SENTINEL-E is backbone-agnostic by construction (it
consumes a frozen detector's scalar output and never touches pixels), so
nothing about the sequential-detection results below depends on this
particular choice -- a stronger backbone would only widen the margin.

A necessary post-processing step, disclosed rather than silently applied:
each dataset's official *training* split contains a handful of isolated
single-frame score spikes (one frame's Farneback magnitude 10-15x the
surrounding frames', immediately followed and preceded by ordinary values --
plausibly a compression or motion-blur artifact rather than a real behavioural
event). Left in, these spikes alone make an otherwise unremarkable training
video look, to the calibration layer, like it contains an undisclosed anomaly,
which is large enough to break false-alarm control regardless of the alarm
threshold used (Section 5's real-validity results report this as a finding in
its own right). We apply a 3-frame median filter, computed independently
within each video (never crossing a video boundary), before saving scores.
This is a standard, minimal denoising step for optical-flow magnitude, applied
identically to training and test frames, and disclosed here rather than
tuned to any downstream result.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAW_ROOT = Path("/tmp/data")  # large raw video/frame caches: not committed
OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_vad"
SIZE = 64  # resized frame side, pixels

AVENUE_URL = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip"
AVENUE_GT_URL = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip"
UCSD_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"


def _sh(cmd: str) -> None:
    print(f"  $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def download() -> None:
    (RAW_ROOT / "avenue").mkdir(parents=True, exist_ok=True)
    (RAW_ROOT / "ucsd").mkdir(parents=True, exist_ok=True)
    targets = [
        (AVENUE_URL, RAW_ROOT / "avenue" / "Avenue_Dataset.zip"),
        (AVENUE_GT_URL, RAW_ROOT / "avenue" / "ground_truth_demo.zip"),
        (UCSD_URL, RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.tar.gz"),
    ]
    for url, dest in targets:
        if dest.exists():
            print(f"  already have {dest}")
            continue
        _sh(f'curl -sS -o "{dest}" "{url}"')
    if not (RAW_ROOT / "avenue" / "extracted").exists():
        _sh(f'unzip -q "{RAW_ROOT / "avenue" / "Avenue_Dataset.zip"}" -d "{RAW_ROOT / "avenue" / "extracted"}"')
    if not (RAW_ROOT / "avenue" / "gt_extracted").exists():
        _sh(f'unzip -q "{RAW_ROOT / "avenue" / "ground_truth_demo.zip"}" -d "{RAW_ROOT / "avenue" / "gt_extracted"}"')
    if not (RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2").exists():
        _sh(f'tar xzf "{RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.tar.gz"}" -C "{RAW_ROOT / "ucsd"}"')


def extract_ped2() -> None:
    import cv2

    root = RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    out = RAW_ROOT / "cache" / "ped2"
    out.mkdir(parents=True, exist_ok=True)

    def load_video(d: Path) -> np.ndarray:
        files = sorted(f for f in os.listdir(d) if f.endswith(".tif"))
        frames = [
            cv2.resize(cv2.imread(str(d / f), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE),
                      interpolation=cv2.INTER_AREA)
            for f in files
        ]
        return np.stack(frames).astype(np.float32) / 255.0

    def load_gt(d: Path) -> np.ndarray:
        files = sorted(f for f in os.listdir(d) if f.endswith(".bmp"))
        return np.array(
            [1 if (im := cv2.imread(str(d / f), cv2.IMREAD_GRAYSCALE)) is not None
                  and im.max() > 0 else 0 for f in files], dtype=np.int64)

    train_dirs = sorted(d for d in os.listdir(root / "Train")
                        if d.startswith("Train") and not d.startswith("."))
    test_dirs = sorted(d for d in os.listdir(root / "Test")
                       if d.startswith("Test") and d[4:].isdigit())

    train_meta, train_frames = [], []
    for d in train_dirs:
        fr = load_video(root / "Train" / d)
        train_meta.append({"name": d, "n_frames": int(fr.shape[0])})
        train_frames.append(fr)
    test_meta, test_frames, test_labels = [], [], []
    for d in test_dirs:
        fr = load_video(root / "Test" / d)
        gt = load_gt(root / "Test" / f"{d}_gt")
        n = min(fr.shape[0], gt.shape[0])
        test_meta.append({"name": d, "n_frames": int(n)})
        test_frames.append(fr[:n])
        test_labels.append(gt[:n])

    np.save(out / "train_frames.npy", np.concatenate(train_frames))
    np.save(out / "test_frames.npy", np.concatenate(test_frames))
    np.save(out / "test_labels.npy", np.concatenate(test_labels))
    (out / "train_meta.json").write_text(json.dumps(train_meta, indent=1))
    (out / "test_meta.json").write_text(json.dumps(test_meta, indent=1))
    print(f"  ped2: {sum(m['n_frames'] for m in train_meta)} train frames, "
          f"{sum(m['n_frames'] for m in test_meta)} test frames")


def extract_avenue() -> None:
    import cv2
    import scipy.io as sio

    root = RAW_ROOT / "avenue" / "extracted" / "Avenue Dataset"
    gtroot = RAW_ROOT / "avenue" / "gt_extracted" / "ground_truth_demo" / "testing_label_mask"
    out = RAW_ROOT / "cache" / "avenue"
    out.mkdir(parents=True, exist_ok=True)

    def load_video(path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        cap.release()
        return np.stack(frames).astype(np.float32) / 255.0

    train_files = sorted(f for f in os.listdir(root / "training_videos") if f.endswith(".avi"))
    test_files = sorted(f for f in os.listdir(root / "testing_videos") if f.endswith(".avi"))

    train_meta, train_frames = [], []
    for f in train_files:
        fr = load_video(root / "training_videos" / f)
        train_meta.append({"name": f, "n_frames": int(fr.shape[0])})
        train_frames.append(fr)
    test_meta, test_frames, test_labels = [], [], []
    for f in test_files:
        idx = int(f.split(".")[0])
        fr = load_video(root / "testing_videos" / f)
        vol = sio.loadmat(gtroot / f"{idx}_label.mat")["volLabel"][0]
        labels = np.array([1 if m.max() > 0 else 0 for m in vol], dtype=np.int64)
        n = min(fr.shape[0], labels.shape[0])
        test_meta.append({"name": f, "n_frames": int(n)})
        test_frames.append(fr[:n])
        test_labels.append(labels[:n])

    np.save(out / "train_frames.npy", np.concatenate(train_frames))
    np.save(out / "test_frames.npy", np.concatenate(test_frames))
    np.save(out / "test_labels.npy", np.concatenate(test_labels))
    (out / "train_meta.json").write_text(json.dumps(train_meta, indent=1))
    (out / "test_meta.json").write_text(json.dumps(test_meta, indent=1))
    print(f"  avenue: {sum(m['n_frames'] for m in train_meta)} train frames, "
          f"{sum(m['n_frames'] for m in test_meta)} test frames")


def compute_scores(name: str) -> None:
    """Farneback optical-flow magnitude, per video, then a 3-frame median filter."""
    import cv2
    from scipy.signal import medfilt

    cache = RAW_ROOT / "cache" / name

    def flow_scores(frames: np.ndarray) -> np.ndarray:
        mags = np.zeros(frames.shape[0], dtype=np.float32)
        prev = (frames[0] * 255).astype(np.uint8)
        for i in range(1, frames.shape[0]):
            cur = (frames[i] * 255).astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mags[i] = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            prev = cur
        mags[0] = mags[1] if mags.size > 1 else mags[0]
        return mags

    def per_video(frames: np.ndarray, meta) -> np.ndarray:
        parts, idx = [], 0
        for m in meta:
            n = m["n_frames"]
            parts.append(flow_scores(frames[idx:idx + n]))
            idx += n
        return np.concatenate(parts)

    def median_filter_per_video(scores: np.ndarray, meta, k: int = 3) -> np.ndarray:
        out, idx = np.empty_like(scores), 0
        for m in meta:
            n = m["n_frames"]
            kk = k if n >= k else (1 if n % 2 == 1 else max(1, n - 1))
            out[idx:idx + n] = medfilt(scores[idx:idx + n], kernel_size=kk)
            idx += n
        return out

    train_meta = json.loads((cache / "train_meta.json").read_text())
    test_meta = json.loads((cache / "test_meta.json").read_text())
    train_frames = np.load(cache / "train_frames.npy")
    test_frames = np.load(cache / "test_frames.npy")

    t0 = time.time()
    train_scores = median_filter_per_video(per_video(train_frames, train_meta), train_meta)
    test_scores = median_filter_per_video(per_video(test_frames, test_meta), test_meta)
    print(f"  {name}: flow+filter in {time.time() - t0:.1f}s")

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "calibration_scores.npy", train_scores)
    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "test_labels.npy", np.load(cache / "test_labels.npy"))
    (out_dir / "test_meta.json").write_text(json.dumps(test_meta, indent=1))
    (out_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=1))

    from sklearn.metrics import roc_auc_score
    labels = np.load(cache / "test_labels.npy")
    auc = float(roc_auc_score(labels, test_scores))
    (out_dir / "backbone_auc.json").write_text(json.dumps({
        "auc": auc, "n_test": int(labels.size), "n_anomalous": int(labels.sum()),
        "note": "Farneback optical-flow magnitude, 3-frame per-video median filter",
    }, indent=1))
    print(f"  {name}: frame-level AUC = {auc:.4f}")


def main() -> None:
    print("1/3 downloading...")
    download()
    print("2/3 extracting frames + real ground truth...")
    extract_ped2()
    extract_avenue()
    print("3/3 computing real per-frame backbone scores...")
    for name in ("ped2", "avenue"):
        compute_scores(name)
    print("done: data/real_vad/{ped2,avenue}/*.npy")


if __name__ == "__main__":
    main()

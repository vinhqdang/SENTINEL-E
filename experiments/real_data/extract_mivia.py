"""Per-frame Farneback flow scores for the Mivia-IWDD illegal-dumping benchmark.

Source (public): https://github.com/MiviaLab/IllegalWasteDumping. Raw videos are
read from a local download (``MIVIA_ROOT``) and are never copied into the repo.

Run from the repository root::

    python3 -m experiments.real_data.extract_mivia

Sampling. Every video is sampled at 10 fps by time: sampled frame k is the
native frame at index ``round(k * fps / 10)``. Videos with native fps below 10
use every native frame. Frames are grayscale, resized to 64x64 (INTER_AREA),
scaled to [0, 1]. Scores are the same Farneback magnitude used for Ped2/Avenue
(``experiments/real_data/download_and_extract.py``), followed by a per-video
3-frame median filter.

Labels. Positive videos carry an onset time in seconds. A sampled frame is
labelled 1 when its timestamp (k/10 s, or native_index/fps for low-fps videos)
is at or after onset, and 0 otherwise. Negative videos are all 0.

Calibration pool: training videos labelled "false", sorted by video_id.
Test set: all 100 test videos, sorted by video_id.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import numpy as np

MIVIA_ROOT = Path("/home/user/mivia_raw")
OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_vad"
TARGET_FPS = 10.0
FLOW_SIZE = 64
SOURCE_URL = "https://github.com/MiviaLab/IllegalWasteDumping"


@dataclass
class Sample:
    k: int          # index among sampled frames
    index: int      # native frame index
    t: float        # timestamp in seconds
    fps: float      # native fps
    bgr: np.ndarray


def roc_auc(y, s) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties (no sklearn dependency)."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(s, dtype=np.float64))
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_split(split: str) -> List[dict]:
    """Read the per-video label JSONs of one split, sorted by video_id."""
    rows = []
    for jf in sorted((MIVIA_ROOT / split / "labels").glob("*.json")):
        d = json.loads(jf.read_text())
        md = d["metadata"]
        rows.append({
            "video_id": md["video_id"],
            "path": MIVIA_ROOT / split / "videos" / md["video_id"],
            "label": d["label"],
            "timestamp": float(d["timestamp"]),
            "event_type": md.get("event_type"),
            "meta_fps": float(md.get("frame_rate") or 0.0),
        })
    rows.sort(key=lambda r: r["video_id"])
    return rows


def sample_frames(path: Path, meta_fps: float = 0.0) -> Iterator[Sample]:
    """Stream the 10 fps sampled frames of one video (all frames if fps < 10)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not fps > 0:
            fps = meta_fps
        if not fps > 0:
            raise RuntimeError(f"no fps for {path}")
        low = fps < TARGET_FPS
        i, k, target = 0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if low or i == target:
                t = i / fps if low else k / TARGET_FPS
                yield Sample(k, i, t, fps, frame)
                k += 1
                if not low:
                    target = round(k * fps / TARGET_FPS)
            i += 1
    finally:
        cap.release()


def gray64(bgr: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (FLOW_SIZE, FLOW_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def flow_magnitudes(frames: np.ndarray) -> np.ndarray:
    """Mean Farneback magnitude per frame; first frame copies the second."""
    import cv2

    mags = np.zeros(frames.shape[0], dtype=np.float32)
    prev = (frames[0] * 255).astype(np.uint8)
    for i in range(1, frames.shape[0]):
        cur = (frames[i] * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        mags[i] = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        prev = cur
    mags[0] = mags[1] if mags.size > 1 else mags[0]
    return mags


def median3(x: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.signal import medfilt

    kk = k if x.size >= k else (1 if x.size % 2 == 1 else max(1, x.size - 1))
    return medfilt(x, kernel_size=kk)


def process_video(row: dict) -> dict:
    frames, ts, idxs = [], [], []
    fps = None
    for s in sample_frames(row["path"], row["meta_fps"]):
        frames.append(gray64(s.bgr))
        ts.append(s.t)
        idxs.append(s.index)
        fps = s.fps
    if not frames:
        raise RuntimeError(f"no frames decoded from {row['video_id']}")
    raw = flow_magnitudes(np.stack(frames))
    return {
        "scores": median3(raw),
        "t": np.asarray(ts),
        "native_fps": round(float(fps), 2),
        "n_frames": len(frames),
    }


def main() -> None:
    import cv2

    cv2.setNumThreads(2)
    out_dir = OUT_ROOT / "mivia_flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_rows = [r for r in load_split("train") if r["label"] == "false"]
    print(f"calibration pool: {len(train_rows)} normal train videos", flush=True)
    cal_parts, train_meta = [], []
    for j, r in enumerate(train_rows):
        res = process_video(r)
        cal_parts.append(res["scores"])
        train_meta.append({"name": r["video_id"], "n_frames": res["n_frames"]})
        if j % 20 == 0:
            print(f"  train {j + 1}/{len(train_rows)} ({time.time() - t0:.0f}s)", flush=True)

    test_rows = load_split("test")
    print(f"test: {len(test_rows)} videos", flush=True)
    test_parts, label_parts, test_meta = [], [], []
    for j, r in enumerate(test_rows):
        positive = r["label"] == "true"
        if positive and r["timestamp"] < 0:
            raise ValueError(f"{r['video_id']}: positive video with no onset time")
        res = process_video(r)
        lab = np.zeros(res["n_frames"], dtype=np.int64)
        if positive:
            lab = (res["t"] >= r["timestamp"]).astype(np.int64)
        test_parts.append(res["scores"])
        label_parts.append(lab)
        test_meta.append({
            "name": r["video_id"],
            "n_frames": res["n_frames"],
            "event_type": r["event_type"],
            "native_fps": res["native_fps"],
        })
        if j % 20 == 0:
            print(f"  test {j + 1}/{len(test_rows)} ({time.time() - t0:.0f}s)", flush=True)

    cal = np.concatenate(cal_parts).astype(np.float64)
    test = np.concatenate(test_parts).astype(np.float64)
    labels = np.concatenate(label_parts)
    np.save(out_dir / "calibration_scores.npy", cal)
    np.save(out_dir / "test_scores.npy", test)
    np.save(out_dir / "test_labels.npy", labels)
    (out_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=1))
    (out_dir / "test_meta.json").write_text(json.dumps(test_meta, indent=1))

    auc = float(roc_auc(labels, test))
    (out_dir / "backbone_auc.json").write_text(json.dumps({
        "auc": auc,
        "note": ("Farneback optical-flow magnitude at 64x64 gray, 3-frame per-video median "
                 "filter; Mivia-IWDD test set, labels from timestamp onset"),
        "fps": 10.0,
        "sampled_fps_note": ("Frames sampled at 10 fps by time; videos with native fps below 10 "
                             "use every native frame (their timestamps are native_index/fps)."),
        "n_test": int(labels.size),
        "n_anomalous": int(labels.sum()),
    }, indent=1))
    print(f"mivia_flow: AUC={auc:.4f} calib={cal.size} test={test.size} "
          f"runtime={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

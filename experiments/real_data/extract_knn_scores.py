"""Patch-kNN per-frame anomaly scores (backbone "knn", PatchCore-style) for Ped2, Avenue and Mivia-IWDD.

Run from the repository root, one dataset at a time::

    python3 -m experiments.real_data.extract_knn_scores --dataset ped2
    python3 -m experiments.real_data.extract_knn_scores --dataset avenue
    python3 -m experiments.real_data.extract_knn_scores --dataset mivia

Configuration (fixed for all datasets; recorded verbatim in each backbone_auc.json):
ResNet-18 (ImageNet, IMAGENET1K_V1, eval, no grad). Frames are RGB, resized to 224x224
(INTER_AREA), ImageNet-normalised. Layer2 (128x28x28) and layer3 (256x14x14) are each
3x3 average-pooled (stride 1, pad 1); layer3 is bilinearly upsampled to 28x28 and
concatenated with layer2 -> 384-d, 784 patches per frame. A fixed Gaussian random
projection (seed 0) maps patches to 128-d. The memory bank is a greedy k-center coreset
of normal training patches at 1% of all normal training patches, capped at 60,000.
Frame score = max over patches of the min L2 distance to the bank (k = 1). Calibration
frames of training video v use the bank minus v's own patches (leave-one-video-out).
Test frames use the full bank. A 3-frame per-video median filter follows.

Frames are identical to the flow backbone's: every frame for Ped2/Avenue, the 10 fps
samples for Mivia. Per-video frame counts are asserted against the flow metas.

Deviation (documented): the greedy coreset runs over a candidate pool of at most 200,000
patches, drawn uniformly at random (seed 0) with an equal quota per training video, not
over all normal patches. The coreset size is still 1% of all normal training patches
(capped at 60,000).

Passes: A featurises training videos, caches float16 features in the scratchpad (not the
repo) and draws the pool. The coreset is built. B scores cached training features
leave-one-video-out. C featurises and scores the test videos.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from experiments.real_data.extract_mivia import (
    MIVIA_ROOT, OUT_ROOT, load_split, median3, sample_frames,
)

RAW_ROOT = Path("/tmp/data")
CACHE_ROOT = Path("/tmp/sentinel_e_scratch/knn_pc_cache")
COREST_FRAC = 0.01
COREST_CAP = 60000
POOL_TARGET = 200000
PATCH_DIM_PROJ = 128
N_PATCH = 784  # 28 x 28
BATCH = 32
QCHUNK = 1024
SEED = 0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIG = {
    "backbone": "torchvision ResNet-18, ImageNet IMAGENET1K_V1, eval, no grad; input 224x224 RGB",
    "features": ("layer2 (128x28x28) and layer3 (256x14x14), each 3x3 avg-pool stride 1 pad 1; "
                 "layer3 bilinear-upsampled to 28x28 and concatenated -> 384-d, 784 patches/frame"),
    "projection": "fixed Gaussian random projection 384 -> 128, torch seed 0, scaled by 1/sqrt(128)",
    "coreset": (f"greedy k-center over a candidate pool (<= {POOL_TARGET} patches, uniform random "
                f"with equal quota per training video, seed 0); size = min(round({COREST_FRAC} x all "
                f"normal training patches), {COREST_CAP})"),
    "score": "frame = max over 784 patches of min L2 distance to the bank (k=1)",
    "calibration": "leave-one-video-out: training frames of video v use bank minus v's patches",
    "smoothing": "3-frame per-video median filter",
}

def roc_auc(y, s) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties (no sklearn dependency)."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(s, dtype=np.float64))
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ----------------------------------------------------------------- sources
def ped2_dirs(split: str) -> List[str]:
    root = RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / split
    if split == "Train":
        return sorted(d for d in os.listdir(root) if d.startswith("Train") and not d.startswith("."))
    return sorted(d for d in os.listdir(root) if d.startswith("Test") and d[4:].isdigit())


def ped2_frames(split: str, d: str, n_limit: Optional[int]) -> Iterator[np.ndarray]:
    import cv2

    root = RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / split / d
    files = sorted(f for f in os.listdir(root) if f.endswith(".tif"))
    if n_limit is not None:
        files = files[:n_limit]
    for f in files:
        g = cv2.imread(str(root / f), cv2.IMREAD_GRAYSCALE)
        yield gray_to_rgb224(g)


def ped2_gt_len(d: str) -> int:
    root = RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Test" / f"{d}_gt"
    return sum(1 for f in os.listdir(root) if f.endswith(".bmp"))


def avenue_subdir(split: str) -> str:
    return "training_videos" if split == "Train" else "testing_videos"


def avenue_files(split: str) -> List[str]:
    root = RAW_ROOT / "avenue" / "extracted" / "Avenue Dataset" / avenue_subdir(split)
    return sorted(f for f in os.listdir(root) if f.endswith(".avi"))


def avenue_frames(split: str, f: str, n_limit: Optional[int]) -> Iterator[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(RAW_ROOT / "avenue" / "extracted" / "Avenue Dataset" / avenue_subdir(split) / f))
    count = 0
    try:
        while n_limit is None or count < n_limit:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            yield bgr_to_rgb224(frame)
    finally:
        cap.release()


def avenue_gt_len(f: str) -> int:
    import scipy.io as sio

    idx = int(f.split(".")[0])
    gt = RAW_ROOT / "avenue" / "gt_extracted" / "ground_truth_demo" / "testing_label_mask" / f"{idx}_label.mat"
    return len(sio.loadmat(gt)["volLabel"][0])


def build_sources(dataset: str):
    """Return (train_items, test_items); each item = (name, expected_len_or_None, make_frames)."""
    if dataset == "ped2":
        tr, te = [], []
        for d in ped2_dirs("Train"):
            meta_n = None
            tr.append((d, None, lambda d=d: ped2_frames("Train", d, None)))
        for d in ped2_dirs("Test"):
            n_lim = min(len([f for f in os.listdir(RAW_ROOT / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Test" / d) if f.endswith(".tif")]), ped2_gt_len(d))
            te.append((d, n_lim, lambda d=d, n=n_lim: ped2_frames("Test", d, n)))
        return tr, te
    if dataset == "avenue":
        tr, te = [], []
        for f in avenue_files("Train"):
            tr.append((f, None, lambda f=f: avenue_frames("Train", f, None)))
        for f in avenue_files("Test"):
            n_lim = avenue_gt_len(f)
            te.append((f, n_lim, lambda f=f, n=n_lim: avenue_frames("Test", f, n)))
        return tr, te
    if dataset == "mivia":
        def mk(r):
            return lambda: (bgr_to_rgb224(s.bgr) for s in sample_frames(r["path"], r["meta_fps"]))
        tr = [(r["video_id"], None, mk(r)) for r in load_split("train") if r["label"] == "false"]
        te = [(r["video_id"], None, mk(r)) for r in load_split("test")]
        return tr, te
    raise ValueError(dataset)


# ----------------------------------------------------------------- model
def build_model():
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2).eval()
    return stem, m.layer3.eval()


def projection():
    import torch

    g = torch.Generator().manual_seed(SEED)
    return torch.randn(384, PATCH_DIM_PROJ, generator=g) / math.sqrt(PATCH_DIM_PROJ)


def featurize(stem, layer3, proj, imgs: List[np.ndarray]):
    """(B,224,224,3) RGB uint8 -> (B,784,128) float32 projected PatchCore-style patches."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)
    with torch.no_grad():
        f2 = stem(x)
        f3 = layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=(28, 28), mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)  # B,384,28,28
        f = f.flatten(2).transpose(1, 2)  # B,784,384
        return f @ proj  # B,784,128


def batched_features(stem, layer3, proj, frames: Iterator[np.ndarray]):
    buf = []
    for img in frames:
        buf.append(img)
        if len(buf) == BATCH:
            yield featurize(stem, layer3, proj, buf)
            buf = []
    if buf:
        yield featurize(stem, layer3, proj, buf)


def greedy_coreset(X, m: int, seed: int = SEED):
    """Greedy k-center (farthest-first) selection of m rows of X (n,d)."""
    import torch

    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=g))
    sel = torch.empty(m, dtype=torch.long)
    sel[0] = first
    sq = (X * X).sum(1)
    mind = sq - 2 * (X @ X[first]) + sq[first]
    for t in range(1, m):
        i = int(torch.argmax(mind))
        sel[t] = i
        d = sq - 2 * (X @ X[i]) + sq[i]
        mind = torch.minimum(mind, d)
        mind[i] = 0.0
    return sel


def frame_scores(feats, bank, bank_mask=None):
    """feats (c,784,128) float32 -> (c,) max over patches of min L2 distance to bank."""
    import torch

    c = feats.shape[0]
    q = feats.reshape(-1, feats.shape[-1])
    out = torch.empty(q.shape[0])
    for s in range(0, q.shape[0], QCHUNK):
        d = torch.cdist(q[s:s + QCHUNK], bank)
        if bank_mask is not None:
            d.masked_fill_(bank_mask.view(1, -1), float("inf"))
        out[s:s + QCHUNK] = d.min(dim=1).values
    return out.view(c, N_PATCH).max(dim=1).values.numpy().astype(np.float64)


# ----------------------------------------------------------------- main
def main() -> None:
    import torch
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ped2", "avenue", "mivia"])
    args = ap.parse_args()
    ds = args.dataset

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    t0 = time.time()

    if ds == "mivia":
        n_json = len(list((MIVIA_ROOT / "train" / "labels").glob("*.json")))
        n_mp4 = len(list((MIVIA_ROOT / "train" / "videos").glob("*.mp4")))
        assert n_json == 400 and n_mp4 == 400, f"Mivia train incomplete: json={n_json} mp4={n_mp4}"

    ref_dir = OUT_ROOT / {"ped2": "ped2", "avenue": "avenue", "mivia": "mivia_flow"}[ds]
    ref_train = json.loads((ref_dir / "train_meta.json").read_text())
    ref_test = json.loads((ref_dir / "test_meta.json").read_text())
    ref_test_labels = np.load(ref_dir / "test_labels.npy")

    train_items, test_items = build_sources(ds)
    assert len(train_items) == len(ref_train), (len(train_items), len(ref_train))
    assert len(test_items) == len(ref_test), (len(test_items), len(ref_test))
    cache = CACHE_ROOT / ds
    cache.mkdir(parents=True, exist_ok=True)
    stem, layer3 = build_model()
    proj = projection()

    # Pass A: featurise training videos, cache float16, draw the candidate pool.
    rng = np.random.default_rng(SEED)
    quota = int(math.ceil(POOL_TARGET / len(train_items)))
    pool_parts, pool_src, n_patches_total = [], [], 0
    for j, (name, _, make) in enumerate(train_items):
        feats = torch.cat(list(batched_features(stem, layer3, proj, make())))  # (n,784,128)
        n = feats.shape[0]
        if ref_train[j]["name"] != name or ref_train[j]["n_frames"] != n:
            raise AssertionError(f"train mismatch {name}: decoded {n}, meta {ref_train[j]}")
        n_patches_total += n * N_PATCH
        flat = feats.reshape(-1, PATCH_DIM_PROJ)
        pick = torch.from_numpy(rng.choice(flat.shape[0], size=min(quota, flat.shape[0]), replace=False))
        pool_parts.append(flat[pick].clone())
        pool_src.append(np.full(pick.shape[0], j, dtype=np.int64))
        np.save(cache / f"train_{j:04d}.npy", feats.half().numpy())
        if j % 5 == 0:
            print(f"[A] train {j + 1}/{len(train_items)} {name} n={n} ({time.time() - t0:.0f}s)", flush=True)
    pool = torch.cat(pool_parts)
    pool_src_t = torch.from_numpy(np.concatenate(pool_src))
    m = min(int(round(COREST_FRAC * n_patches_total)), COREST_CAP)
    print(f"[A] normal patches={n_patches_total} pool={pool.shape[0]} coreset target={m}", flush=True)

    # Coreset.
    sel = greedy_coreset(pool, m)
    bank = pool[sel].contiguous()
    bank_src = pool_src_t[sel]
    print(f"[A] coreset built: {bank.shape[0]} patches ({time.time() - t0:.0f}s)", flush=True)

    # Pass B: leave-one-video-out scores for calibration frames.
    cal_parts = []
    for j in range(len(train_items)):
        feats = torch.from_numpy(np.load(cache / f"train_{j:04d}.npy").astype(np.float32))
        mask = bank_src == j
        raws = [frame_scores(feats[i:i + BATCH], bank, mask) for i in range(0, feats.shape[0], BATCH)]
        cal_parts.append(median3(np.concatenate(raws)))
        if j % 5 == 0:
            print(f"[B] calib {j + 1}/{len(train_items)} ({time.time() - t0:.0f}s)", flush=True)

    # Pass C: test frames, full bank.
    test_parts = []
    for j, (name, n_lim, make) in enumerate(test_items):
        raws = []
        for feats in batched_features(stem, layer3, proj, make()):
            raws.append(frame_scores(feats, bank, None))
        raw = np.concatenate(raws)
        n = raw.shape[0]
        if ref_test[j]["name"] != name or ref_test[j]["n_frames"] != n:
            raise AssertionError(f"test mismatch {name}: decoded {n}, meta {ref_test[j]}")
        test_parts.append(median3(raw))
        if j % 5 == 0:
            print(f"[C] test {j + 1}/{len(test_items)} {name} n={n} ({time.time() - t0:.0f}s)", flush=True)

    cal = np.concatenate(cal_parts).astype(np.float64)
    test = np.concatenate(test_parts).astype(np.float64)
    assert test.size == ref_test_labels.size
    out_name = {"ped2": "ped2_knn", "avenue": "avenue_knn", "mivia": "mivia_knn"}[ds]
    out_dir = OUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "calibration_scores.npy", cal)
    np.save(out_dir / "test_scores.npy", test)
    np.save(out_dir / "test_labels.npy", ref_test_labels)
    (out_dir / "train_meta.json").write_text(json.dumps(ref_train, indent=1))
    (out_dir / "test_meta.json").write_text(json.dumps(ref_test, indent=1))
    auc = roc_auc(ref_test_labels, test)
    fps = 10.0 if ds == "mivia" else 25.0
    (out_dir / "backbone_auc.json").write_text(json.dumps({
        "auc": auc,
        "fps": fps,
        "n_test": int(ref_test_labels.size),
        "n_anomalous": int(ref_test_labels.sum()),
        "n_normal_patches": int(n_patches_total),
        "coreset_size": int(bank.shape[0]),
        "pool_size": int(pool.shape[0]),
        "config": CONFIG,
        "note": ("PatchCore-style patch-kNN on ResNet-18 layer2+layer3 features (see config); "
                 "frame score = max over patches of min L2 distance to a greedy coreset of normal "
                 "training patches, leave-one-video-out for calibration; 3-frame per-video median filter"),
    }, indent=1))
    print(f"{out_name}: AUC={auc:.4f} calib={cal.size} test={test.size} "
          f"runtime={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

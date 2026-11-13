"""Patch-kNN per-frame anomaly scores (backbone "knn") for Ped2, Avenue and Mivia-IWDD.

Run from the repository root, one dataset at a time::

    python3 -m experiments.real_data.extract_knn_scores --dataset ped2
    python3 -m experiments.real_data.extract_knn_scores --dataset avenue
    python3 -m experiments.real_data.extract_knn_scores --dataset mivia

Backbone. torchvision ResNet-18 with ImageNet weights (IMAGENET1K_V1), eval
mode, no grad, truncated after layer3. Each frame is RGB (grayscale frames are
replicated to 3 channels), resized to 224x224 (INTER_AREA), ImageNet-normalised.
The layer3 map (256 x 14 x 14) gives 196 patch vectors of dimension 256.

Score. Memory bank = about 20,000 patch vectors sampled (seed 0) across the
normal training videos, with each bank patch tagged by its source video. A
frame's score is the maximum over its 196 patches of the minimum L2 distance
to the bank. Calibration frames of training video v are scored with the bank
minus v's own patches (leave-one-video-out), so their scores are not biased
low by near-duplicate patches from the same video. Test frames use the full
bank. Every score sequence then gets the same per-video 3-frame median filter
as the flow backbone.

Frames are identical to the flow backbone's: every frame for Ped2/Avenue, the
10 fps samples for Mivia. Frame counts are asserted against the flow metas.

Pass structure. Pass A featurises the training videos and caches their
float16 patch features in the scratchpad (not in the repo). The bank is built
from them. Pass B scores the cached training features leave-one-video-out.
Pass C featurises and scores the test videos.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from experiments.real_data.extract_mivia import (
    MIVIA_ROOT, OUT_ROOT, load_split, median3, sample_frames,
)

RAW_ROOT = Path("/tmp/data")
CACHE_ROOT = Path("/tmp/sentinel_e_scratch/knn_cache")
BANK_TARGET = 20000
BATCH = 32
SCORE_CHUNK = 8
SEED = 0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rgb224(img_rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)


def gray_to_rgb224(gray: np.ndarray) -> np.ndarray:
    import cv2

    return rgb224(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def bgr_to_rgb224(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return rgb224(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


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
def build_trunk():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    return torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3).eval()


def featurize(trunk, imgs: List[np.ndarray]):
    """(B,224,224,3) RGB uint8 -> (B,196,256) float32 layer3 patch vectors."""
    import torch

    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)
    with torch.no_grad():
        y = trunk(x)
    return y.flatten(2).transpose(1, 2).contiguous()


def batched_features(trunk, frames: Iterator[np.ndarray]):
    """Yield (B,196,256) feature blocks of up to BATCH frames each."""
    buf = []
    for img in frames:
        buf.append(img)
        if len(buf) == BATCH:
            yield featurize(trunk, buf)
            buf = []
    if buf:
        yield featurize(trunk, buf)


def max_min_dist(feats, bank, exclude_mask=None) -> np.ndarray:
    """feats (c,196,256); bank (M,256). Per frame: max over patches of min L2 to bank."""
    import torch

    c = feats.shape[0]
    d = torch.cdist(feats.reshape(-1, feats.shape[-1]), bank)
    if exclude_mask is not None:
        d.masked_fill_(exclude_mask.view(1, -1), float("inf"))
    pm = d.min(dim=1).values.view(c, feats.shape[1])
    return pm.max(dim=1).values.numpy().astype(np.float64)


def score_block(feats, bank, exclude_mask=None) -> np.ndarray:
    out = []
    for i in range(0, feats.shape[0], SCORE_CHUNK):
        out.append(max_min_dist(feats[i:i + SCORE_CHUNK], bank, exclude_mask))
    return np.concatenate(out)


# ----------------------------------------------------------------- main
def main() -> None:
    import cv2
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ped2", "avenue", "mivia"])
    args = ap.parse_args()
    ds = args.dataset

    torch.set_num_threads(2)
    cv2.setNumThreads(2)

    # Mivia: the train download must be complete before reading it.
    if ds == "mivia":
        n_json = len(list((MIVIA_ROOT / "train" / "labels").glob("*.json")))
        n_mp4 = len(list((MIVIA_ROOT / "train" / "videos").glob("*.mp4")))
        assert n_json == 400 and n_mp4 == 400, f"Mivia train incomplete: json={n_json} mp4={n_mp4}"

    # Reference metas: the flow backbone's metas define the frame layout.
    ref_dir = OUT_ROOT / {"ped2": "ped2", "avenue": "avenue", "mivia": "mivia_flow"}[ds]
    ref_train = json.loads((ref_dir / "train_meta.json").read_text())
    ref_test = json.loads((ref_dir / "test_meta.json").read_text())
    ref_test_labels = np.load(ref_dir / "test_labels.npy")

    train_items, test_items = build_sources(ds)
    assert len(train_items) == len(ref_train), (len(train_items), len(ref_train))
    assert len(test_items) == len(ref_test), (len(test_items), len(ref_test))
    cache = CACHE_ROOT / ds
    cache.mkdir(parents=True, exist_ok=True)
    trunk = build_trunk()
    t0 = time.time()

    # Pass A: featurise training videos, cache, sample the bank.
    rng = np.random.default_rng(SEED)
    quota = int(round(BANK_TARGET / len(train_items)))
    bank_parts, src_parts, train_counts = [], [], []
    for j, (name, _, make) in enumerate(train_items):
        feats = torch.cat(list(batched_features(trunk, make())))  # (n,196,256)
        n = feats.shape[0]
        train_counts.append(n)
        if ref_train[j]["name"] != name or ref_train[j]["n_frames"] != n:
            raise AssertionError(f"train mismatch {name}: decoded {n}, meta {ref_train[j]}")
        flat = feats.reshape(-1, feats.shape[-1])
        pick = torch.from_numpy(rng.choice(flat.shape[0], size=min(quota, flat.shape[0]), replace=False))
        bank_parts.append(flat[pick].clone())
        src_parts.append(np.full(pick.shape[0], j, dtype=np.int64))
        np.save(cache / f"train_{j:04d}.npy", feats.half().numpy())
        if j % 5 == 0:
            print(f"[A] train {j + 1}/{len(train_items)} {name} n={n} ({time.time() - t0:.0f}s)", flush=True)
    bank = torch.cat(bank_parts)
    bank_src = torch.from_numpy(np.concatenate(src_parts))
    print(f"[A] bank: {bank.shape[0]} patches from {len(train_items)} videos (quota {quota})", flush=True)

    # Pass B: leave-one-video-out scores for the calibration frames.
    cal_parts = []
    for j in range(len(train_items)):
        feats = torch.from_numpy(np.load(cache / f"train_{j:04d}.npy").astype(np.float32))
        mask = bank_src == j
        raw = score_block(feats, bank, mask)
        cal_parts.append(median3(raw))
        if j % 5 == 0:
            print(f"[B] calib {j + 1}/{len(train_items)} ({time.time() - t0:.0f}s)", flush=True)

    # Pass C: test frames, full bank.
    test_parts, test_counts = [], []
    for j, (name, n_lim, make) in enumerate(test_items):
        raws = []
        for feats in batched_features(trunk, make()):
            raws.append(score_block(feats, bank, None))
        raw = np.concatenate(raws)
        n = raw.shape[0]
        test_counts.append(n)
        if ref_test[j]["name"] != name or ref_test[j]["n_frames"] != n:
            raise AssertionError(f"test mismatch {name}: decoded {n}, meta {ref_test[j]}")
        test_parts.append(median3(raw))
        if j % 5 == 0:
            print(f"[C] test {j + 1}/{len(test_items)} {name} n={n} ({time.time() - t0:.0f}s)", flush=True)

    cal = np.concatenate(cal_parts).astype(np.float64)
    test = np.concatenate(test_parts).astype(np.float64)
    out_name = {"ped2": "ped2_knn", "avenue": "avenue_knn", "mivia": "mivia_knn"}[ds]
    out_dir = OUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    assert test.size == ref_test_labels.size
    np.save(out_dir / "calibration_scores.npy", cal)
    np.save(out_dir / "test_scores.npy", test)
    np.save(out_dir / "test_labels.npy", ref_test_labels)
    (out_dir / "train_meta.json").write_text(json.dumps(ref_train, indent=1))
    (out_dir / "test_meta.json").write_text(json.dumps(ref_test, indent=1))
    auc = float(roc_auc(ref_test_labels, test))
    fps = 10.0 if ds == "mivia" else 25.0
    (out_dir / "backbone_auc.json").write_text(json.dumps({
        "auc": auc,
        "fps": fps,
        "n_test": int(ref_test_labels.size),
        "n_anomalous": int(ref_test_labels.sum()),
        "note": ("ResNet-18 (ImageNet, layer3) 196 patch vectors per 224x224 frame; frame score = "
                 "max over patches of min L2 distance to a ~20k-patch bank of normal training "
                 "patches (leave-one-video-out for calibration); 3-frame per-video median filter"),
    }, indent=1))
    print(f"{out_name}: AUC={auc:.4f} calib={cal.size} test={test.size} "
          f"runtime={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

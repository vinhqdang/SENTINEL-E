"""Assemble a downloaded PatchCore run into the standard real-VAD layout.

Input is the ``.npz`` written by ``experiments/real_data/patchcore_remote.py`` (raw per-frame
scores, one array per split). This script applies the same 3-frame per-video median filter as
``download_and_extract.compute_scores``, copies labels and metadata from the flow dataset
directory, computes the frame-level test ROC-AUC and writes::

    data/real_vad/<dataset>_patchcore/{calibration_scores,test_scores,test_labels}.npy
    data/real_vad/<dataset>_patchcore/{train_meta,test_meta}.json, backbone_auc.json

Run from the repository root::

    python3 -m experiments.real_data.assemble_patchcore --dataset ped2 --npz /path/ped2_pc.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import roc_auc_score

OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_vad"
#: Flow dataset directory supplying labels and metadata for each dataset.
SOURCE_DIR = {"ped2": "ped2", "avenue": "avenue", "mivia": "mivia_flow"}
FPS = {"ped2": 25.0, "avenue": 25.0, "mivia": 10.0}


def median_filter_per_video(scores: np.ndarray, meta, k: int = 3) -> np.ndarray:
    """Identical to ``download_and_extract.compute_scores``: never crosses a video boundary."""
    out, idx = np.empty_like(scores), 0
    for m in meta:
        n = m["n_frames"]
        kk = k if n >= k else (1 if n % 2 == 1 else max(1, n - 1))
        out[idx:idx + n] = medfilt(scores[idx:idx + n], kernel_size=kk)
        idx += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SOURCE_DIR))
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    ds = args.dataset
    src = args.out_root / SOURCE_DIR[ds]
    z = np.load(args.npz, allow_pickle=False)
    train_raw = z["train_scores"].astype(np.float64)
    test_raw = z["test_scores"].astype(np.float64)
    config = json.loads(str(z["config"]))

    train_meta = json.loads((src / "train_meta.json").read_text())
    test_meta = json.loads((src / "test_meta.json").read_text())
    labels = np.load(src / "test_labels.npy").astype(np.int64)

    if config["dataset"] != ds:
        raise AssertionError(f"npz was produced for {config['dataset']!r}, not {ds!r}")
    if not config["complete"]:
        raise AssertionError("npz is a partial (--max-videos) run; refusing to assemble it")
    if config["train_names"] != [m["name"] for m in train_meta]:
        raise AssertionError("training video order differs from the flow dataset metadata")
    if config["test_names"] != [m["name"] for m in test_meta]:
        raise AssertionError("test video order differs from the flow dataset metadata")
    n_train = sum(m["n_frames"] for m in train_meta)
    n_test = sum(m["n_frames"] for m in test_meta)
    assert train_raw.size == n_train, (train_raw.size, n_train)
    assert test_raw.size == n_test == labels.size, (test_raw.size, n_test, labels.size)

    cal = median_filter_per_video(train_raw, train_meta)
    test = median_filter_per_video(test_raw, test_meta)
    auc = float(roc_auc_score(labels, test))

    out = args.out_root / f"{ds}_patchcore"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "calibration_scores.npy", cal)
    np.save(out / "test_scores.npy", test)
    np.save(out / "test_labels.npy", labels)
    shutil.copyfile(src / "train_meta.json", out / "train_meta.json")
    shutil.copyfile(src / "test_meta.json", out / "test_meta.json")
    (out / "backbone_auc.json").write_text(json.dumps({
        "auc": auc,
        "fps": FPS[ds],
        "n_test": int(labels.size),
        "n_anomalous": int(labels.sum()),
        "note": ("PatchCore (Roth et al. 2022): torchvision wide_resnet50_2 layer2+layer3, 1% "
                 "approximate greedy coreset of every-4th-frame training patches, leave-one-video-out "
                 "calibration, 3-frame per-video median filter"),
        "config": config,
    }, indent=1))
    print(f"{ds}_patchcore: AUC={auc:.4f} calib={cal.size} test={test.size} "
          f"positives={int(labels.sum())} -> {out}")


if __name__ == "__main__":
    main()

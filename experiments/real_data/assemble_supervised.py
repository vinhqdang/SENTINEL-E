"""Supervised per-frame detector for Mivia-IWDD, trained on the 400 training videos only.

Input: per-video causal clip features written by ``supervised_features_remote.py`` (one
``train_<video>.npy`` / ``test_<video>.npy`` per video, plus ``*_idx.npy`` with the native frame
index and timestamp of every sampled frame) and ``labels.json`` (compact label rows exported from
the raw label JSONs: ``{"train": [...], "test": [...]}`` with ``video_id``, ``label``, ``timestamp``).

Frame labels: 1 for sampled frames at or after the onset of a positive video, 0 otherwise.

Protocol (all choices use training videos only; the test set is read once, at the end):
  * Folds. The 400 training videos are split into 5 folds, stratified by class (positives and
    negatives each assigned round-robin in video_id order). A test video is scored by the fold model
    with index md5(video_id) mod 5. Test scores come from one fold model, never an ensemble.
  * Model. Standardised features (scaler fitted on the training part of each fold) and an L2-regularised
    logistic regression. The regularisation strength C is chosen from a fixed grid by the pooled
    out-of-fold log-loss over the 5 training folds. Score = decision function (logit).
  * Cross-fitting. Each training video is scored by the fold model that did not train on it; these
    out-of-fold scores are the calibration pool for the 200 normal training videos.
  * Smoothing. The same 3-frame per-video median filter as the PatchCore and flow datasets.
  * Freezing. After C is fixed and all scores are computed, the test AUC is computed once.

Output ``data/real_vad/mivia_supervised/`` in the layout of ``mivia_patchcore``::

    calibration_scores.npy, test_scores.npy, test_labels.npy, train_meta.json, test_meta.json,
    backbone_auc.json (auc, fps 10.0, config)

Run from the repository root::

    python3 -m experiments.real_data.assemble_supervised --features-dir /home/user/sup_feats/mvit_v1_b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from experiments.real_data.assemble_patchcore import median_filter_per_video, roc_auc_score

OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_vad"
PATCHCORE_DIR = OUT_ROOT / "mivia_patchcore"
FLOW_DIR = OUT_ROOT / "mivia_flow"
N_FOLDS = 5
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
FPS = 10.0


def fold_of_test(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % N_FOLDS


def load_video(features_dir: Path, split: str, name: str):
    feats = np.load(features_dir / f"{split}_{name}.npy").astype(np.float32)
    idx = np.load(features_dir / f"{split}_{name}_idx.npy")
    if idx.shape != (feats.shape[0], 2):
        raise AssertionError(f"{split} {name}: index array {idx.shape} vs features {feats.shape}")
    return feats, idx[:, 0].astype(np.int64), idx[:, 1]


def frame_labels(row: dict, t: np.ndarray) -> np.ndarray:
    if row["label"] == "true":
        if row["timestamp"] < 0:
            raise ValueError(f"{row['video_id']}: positive video with no onset time")
        return (t >= row["timestamp"]).astype(np.int64)
    return np.zeros(t.size, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=Path("/home/user/sup_feats/labels.json"))
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--dataset", default="mivia_supervised")
    args = ap.parse_args()

    labels = json.loads(args.labels.read_text())
    tr_rows = sorted(labels["train"], key=lambda r: r["video_id"])
    te_rows = sorted(labels["test"], key=lambda r: r["video_id"])
    tr_meta = json.loads((PATCHCORE_DIR / "train_meta.json").read_text())
    te_meta = json.loads((PATCHCORE_DIR / "test_meta.json").read_text())
    normal_names = [m["name"] for m in tr_meta]
    if normal_names != [r["video_id"] for r in tr_rows if r["label"] == "false"]:
        raise AssertionError("calibration video order differs from mivia_patchcore/train_meta.json")
    if [m["name"] for m in te_meta] != [r["video_id"] for r in te_rows]:
        raise AssertionError("test video order differs from mivia_patchcore/test_meta.json")

    # ---- training data: all 400 videos
    tr = []
    for r in tr_rows:
        X, idx, t = load_video(args.features_dir, "train", r["video_id"])
        y = frame_labels(r, t)
        tr.append({"name": r["video_id"], "X": X, "y": y, "t": t, "pos": r["label"] == "true"})
    # ---- test data
    te = []
    for r in te_rows:
        X, idx, t = load_video(args.features_dir, "test", r["video_id"])
        te.append({"name": r["video_id"], "X": X, "y": frame_labels(r, t), "t": t,
                   "fold": fold_of_test(r["video_id"])})
    flow_labels = np.load(FLOW_DIR / "test_labels.npy").astype(np.int64)
    test_labels = np.concatenate([v["y"] for v in te])
    if test_labels.size != flow_labels.size or not np.array_equal(test_labels, flow_labels):
        raise AssertionError("test labels from the label JSONs and the sampled timestamps do not "
                             "match mivia_flow/test_labels.npy")
    for m, v in zip(te_meta, te):
        if m["n_frames"] != v["X"].shape[0]:
            raise AssertionError(f"{v['name']}: {v['X'].shape[0]} frames vs metadata {m['n_frames']}")
    for m, v in zip(tr_meta, [v for v in tr if not v["pos"]]):
        if m["n_frames"] != v["X"].shape[0]:
            raise AssertionError(f"{v['name']}: {v['X'].shape[0]} frames vs metadata {m['n_frames']}")
    # Positive training videos: frame counts and onset indices must match the flow pipeline's metadata.
    pos_meta = {m["name"]: m for m in json.loads((FLOW_DIR / "train_positive_meta.json").read_text())}
    mismatches = []
    for v in tr:
        if not v["pos"]:
            continue
        m = pos_meta[v["name"]]
        onset_idx = int(np.argmax(v["y"] == 1)) if v["y"].any() else -1
        if m["n_frames"] != v["X"].shape[0] or onset_idx != m["onset_frame"]:
            mismatches.append({"name": v["name"], "n_frames": [m["n_frames"], int(v["X"].shape[0])],
                               "onset_frame": [m["onset_frame"], onset_idx]})
    if mismatches:
        raise AssertionError(f"positive training videos disagree with train_positive_meta.json: {mismatches[:5]}")
    print(f"positive training videos: frame counts and onset frames match train_positive_meta.json "
          f"({len(pos_meta)} videos)", flush=True)

    # ---- folds (stratified by class, video_id order)
    fold = {}
    for pos in (True, False):
        group = [v for v in tr if v["pos"] == pos]
        for i, v in enumerate(group):
            fold[v["name"]] = i % N_FOLDS
    for v in tr:
        v["fold"] = fold[v["name"]]

    def fit(X, y, C):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(sc.transform(X), y)
        return sc, clf

    def score(model, X):
        sc, clf = model
        return clf.decision_function(sc.transform(X))

    # ---- choose C by pooled out-of-fold log-loss on training videos only
    search = {}
    for C in C_GRID:
        oof_s, oof_y, models = [], [], {}
        for k in range(N_FOLDS):
            trn = [v for v in tr if v["fold"] != k]
            hold = [v for v in tr if v["fold"] == k]
            models[k] = fit(np.concatenate([v["X"] for v in trn]), np.concatenate([v["y"] for v in trn]), C)
            for v in hold:
                oof_s.append(score(models[k], v["X"]))
                oof_y.append(v["y"])
        s = np.concatenate(oof_s)
        y = np.concatenate(oof_y)
        p = 1.0 / (1.0 + np.exp(-s))
        ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
        search[C] = {"oof_logloss": ll, "oof_auc_raw": float(roc_auc_score(y, s))}
        print(f"C={C:g}: OOF logloss={ll:.4f} OOF frame AUC(raw)={search[C]['oof_auc_raw']:.4f}", flush=True)
    C_star = min(C_GRID, key=lambda c: search[c]["oof_logloss"])
    print(f"selected C={C_star:g} (lowest out-of-fold log-loss)", flush=True)

    # ---- freeze: cross-fitted training scores and fold models for test
    models = {k: fit(np.concatenate([v["X"] for v in tr if v["fold"] != k]),
                     np.concatenate([v["y"] for v in tr if v["fold"] != k]), C_star)
              for k in range(N_FOLDS)}
    for v in tr:
        v["s"] = score(models[v["fold"]], v["X"])
    for v in te:
        v["s"] = score(models[v["fold"]], v["X"])

    cal_raw = np.concatenate([v["s"] for v in tr if not v["pos"]])
    cal = median_filter_per_video(cal_raw, tr_meta)
    tr_all_raw = np.concatenate([v["s"] for v in tr])
    tr_all_y = np.concatenate([v["y"] for v in tr])
    tr_all_meta = [{"n_frames": v["X"].shape[0]} for v in tr]
    oof_auc_smoothed = float(roc_auc_score(tr_all_y, median_filter_per_video(tr_all_raw, tr_all_meta)))
    oof_auc_raw = float(roc_auc_score(tr_all_y, tr_all_raw))

    test_raw = np.concatenate([v["s"] for v in te])
    test = median_filter_per_video(test_raw, te_meta)
    auc = float(roc_auc_score(test_labels, test))
    auc_raw = float(roc_auc_score(test_labels, test_raw))

    out = args.out_root / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "calibration_scores.npy", cal.astype(np.float64))
    np.save(out / "test_scores.npy", test.astype(np.float64))
    np.save(out / "test_labels.npy", test_labels)
    shutil.copyfile(PATCHCORE_DIR / "train_meta.json", out / "train_meta.json")
    shutil.copyfile(PATCHCORE_DIR / "test_meta.json", out / "test_meta.json")
    config = {
        "features_dir": str(args.features_dir),
        "arch_feature_dim": int(te[0]["X"].shape[1]),
        "folds": N_FOLDS,
        "fold_rule_train": "class-stratified round-robin in video_id order",
        "fold_rule_test": "md5(video_id) mod 5",
        "C_grid": list(C_GRID),
        "C_selected": C_star,
        "C_search_oof": {str(k): v for k, v in search.items()},
        "score": "logit (decision function), 3-frame per-video median filter",
        "train_videos": len(tr),
        "train_positive_videos": int(sum(v["pos"] for v in tr)),
        "train_frames": int(tr_all_y.size),
        "train_frames_positive": int(tr_all_y.sum()),
        "oof_auc_training_smoothed": oof_auc_smoothed,
        "oof_auc_training_raw": oof_auc_raw,
        "auc_test_raw": auc_raw,
        "calibration": "out-of-fold scores of the 200 normal training videos",
    }
    (out / "backbone_auc.json").write_text(json.dumps({
        "auc": auc,
        "fps": FPS,
        "n_test": int(test_labels.size),
        "n_anomalous": int(test_labels.sum()),
        "note": ("Supervised per-frame logistic regression on causal 16-frame clip features from a "
                 "Kinetics-400 pretrained video backbone, trained on the 400 Mivia training videos; "
                 "5-fold cross-fitting by video; 3-frame per-video median filter"),
        "config": config,
    }, indent=1))
    print(f"{args.dataset}: test AUC={auc:.4f} (raw {auc_raw:.4f}) OOF train AUC={oof_auc_smoothed:.4f} "
          f"calib={cal.size} test={test.size} positives={int(test_labels.sum())} -> {out}", flush=True)


if __name__ == "__main__":
    main()

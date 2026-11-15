"""Causal clip features for the Mivia-IWDD supervised detector, self-contained for Google Colab.

This file has no repository imports. Upload it alone to a Colab VM and run it on the raw
video tree. It needs only numpy, opencv-python-headless and torch / torchvision.

Example (Colab GPU)::

    !python supervised_features_remote.py --raw-root /content/raw/mivia \
        --arch mvit_v1_b --out-dir /content/work/out_sup_ckpt \
        --check-indices /content/work/meta/sampled_indices.json --device cuda

``--raw-root`` holds ``{train,test}/videos/*.mp4`` as the local download lays them out.
Labels are not read here; the local assembly step attaches them from the label JSONs.

Sampling (identical to experiments/real_data/extract_mivia.py::sample_frames): every video is
sampled at 10 fps by time, sampled frame k is the native frame at index ``round(k * fps / 10)``.
Videos with native fps below 10 use every native frame. The sampled frame indices are checked
against ``--check-indices`` (sampled_indices.json of mivia_flow) for every video listed there.

Features (causal): for every sampled frame t, the clip is the last 16 sampled frames ending at t
(positions t-15..t; positions before the first sampled frame repeat the first frame). The clip is
passed through a Kinetics-400 pretrained torchvision video backbone with its classification head
replaced by an identity, so the stored vector is the pooled penultimate-layer feature.

Input: RGB, resized (squashed, no crop) to the backbone's resolution with INTER_AREA, normalised
with the Kinetics mean and std. fp16 autocast on CUDA.

Checkpoints (``--out-dir``): one ``<split>_<video>.npy`` (float32, shape (n_sampled, D)) and one
``<split>_<video>_idx.npy`` (float64, shape (n_sampled, 2): native frame index, timestamp in s) per
video, both written atomically. A video is skipped when both files exist and have the right shape.
``DONE`` (json) is written when every video is finished. ``config.json`` records the settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

TARGET_FPS = 10.0
CLIP_LEN = 16
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

# arch -> (builder name, weights name, input side)
ARCHS = {
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights", 112),
    "swin3d_t": ("swin3d_t", "Swin3D_T_Weights", 224),
    "mvit_v1_b": ("mvit_v1_b", "MViT_V1_B_Weights", 224),
}


def build_backbone(arch: str, device: torch.device) -> nn.Module:
    name, wname, _ = ARCHS[arch]
    weights = getattr(torchvision.models.video, wname).KINETICS400_V1
    m = getattr(torchvision.models.video, name)(weights=weights)
    # Penultimate pooled feature: replace the classifier with identity.
    if arch == "r2plus1d_18":
        m.fc = nn.Identity()
    else:
        m.head = nn.Identity()
    return m.eval().to(device)


def sample_frames(path: Path, side: int):
    """(frames uint8 (T,side,side,3) RGB, native indices, timestamps, native fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames, idxs, ts = [], [], []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not fps > 0:
            raise RuntimeError(f"no fps for {path}")
        low = fps < TARGET_FPS
        i, k, target = 0, 0, 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if low or i == target:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA))
                idxs.append(int(i))
                ts.append(i / fps if low else k / TARGET_FPS)
                k += 1
                if not low:
                    target = round(k * fps / TARGET_FPS)
            i += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames), np.asarray(idxs, dtype=np.int64), np.asarray(ts, dtype=np.float64), fps


def clip_positions(T: int, end: np.ndarray) -> np.ndarray:
    """Positions of the 16-frame causal clip ending at each index (clamped at 0)."""
    offs = np.arange(-(CLIP_LEN - 1), 1)
    return np.clip(end[:, None] + offs[None, :], 0, T - 1)


@torch.no_grad()
def features_for_video(model, frames: np.ndarray, device: torch.device, batch: int, use_amp: bool,
                       max_clips: int | None = None) -> np.ndarray:
    T = frames.shape[0]
    fr = torch.from_numpy(frames).to(device)                      # (T,S,S,3) uint8
    mean, std = KINETICS_MEAN.to(device), KINETICS_STD.to(device)
    n = T if max_clips is None else min(T, max_clips)
    out = []
    for s in range(0, n, batch):
        end = np.arange(s, min(s + batch, n))
        pos = torch.from_numpy(clip_positions(T, end)).to(device)  # (B,16)
        x = fr[pos]                                                # (B,16,S,S,3)
        x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)           # (B,3,16,S,S)
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="mvit_v1_b")
    ap.add_argument("--check-indices", type=Path, default=None,
                    help="sampled_indices.json (mivia_flow): asserted for every listed video")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--bench", type=int, default=0,
                    help="benchmark only: time this many clips of the first video and exit")
    ap.add_argument("--max-videos", type=int, default=None, help="smoke test: first N videos per split")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(2)
    _, _, side = ARCHS[args.arch]
    model = build_backbone(args.arch, device)
    checks = json.loads(args.check_indices.read_text()) if args.check_indices else {}

    if args.bench:
        names = sorted(p.name for p in (args.raw_root / "train" / "videos").glob("*.mp4"))
        frames, _, _, _ = sample_frames(args.raw_root / "train" / "videos" / names[0], side)
        n = min(args.bench, frames.shape[0])
        features_for_video(model, frames, device, args.batch_size, use_amp, max_clips=args.batch_size)
        torch.cuda.synchronize()
        t0 = time.time()
        features_for_video(model, frames, device, args.batch_size, use_amp, max_clips=n)
        torch.cuda.synchronize()
        rate = n / (time.time() - t0)
        print(json.dumps({"arch": args.arch, "side": side, "clips_per_s": round(rate, 1)}))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {"arch": args.arch, "side": side, "clip_len": CLIP_LEN, "target_fps": TARGET_FPS,
              "weights": f"torchvision {ARCHS[args.arch][1]}.KINETICS400_V1",
              "feature": "pooled penultimate (identity head), fp16 autocast" if use_amp else "fp32",
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "device": str(device), "batch_size": args.batch_size}
    _atomic_text(args.out_dir / "config.json", json.dumps(config, indent=1))
    (args.out_dir / "DONE").unlink(missing_ok=True)

    t_all = time.time()
    done_videos = 0
    for split in args.splits.split(","):
        vids = sorted((args.raw_root / split / "videos").glob("*.mp4"))
        if args.max_videos is not None:
            vids = vids[:args.max_videos]
        for j, vp in enumerate(vids):
            name = vp.name
            stem = f"{split}_{safe_name(name)}"
            f_path = args.out_dir / f"{stem}.npy"
            i_path = args.out_dir / f"{stem}_idx.npy"
            if f_path.exists() and i_path.exists():
                try:
                    if np.load(f_path, mmap_mode="r").ndim == 2 and np.load(i_path).shape[1] == 2:
                        done_videos += 1
                        continue
                except Exception:
                    pass  # torn file: recompute
            t0 = time.time()
            frames, idxs, ts, fps = sample_frames(vp, side)
            if name in checks and [int(v) for v in checks[name]] != idxs.tolist():
                raise AssertionError(f"{split} {name}: sampled indices differ from sampled_indices.json")
            feats = features_for_video(model, frames, device, args.batch_size, use_amp)
            assert feats.shape[0] == frames.shape[0]
            _atomic_save(f_path, feats)
            _atomic_save(i_path, np.stack([idxs.astype(np.float64), ts], axis=1))
            done_videos += 1
            print(f"[{split} {j + 1}/{len(vids)}] {name}: T={feats.shape[0]} D={feats.shape[1]} "
                  f"fps={fps:.2f} {time.time() - t0:.1f}s total={time.time() - t_all:.0f}s", flush=True)

    _atomic_text(args.out_dir / "DONE", json.dumps({"videos": done_videos,
                                                   "seconds": round(time.time() - t_all, 1)}))
    print(f"DONE {done_videos} videos in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()

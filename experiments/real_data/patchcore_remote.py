"""PatchCore (Roth et al., CVPR 2022) per-frame anomaly scores, self-contained for Google Colab.

This file has no repository imports. Upload it alone to a Colab VM (or run it from a
checkout) and run one dataset at a time. It needs only numpy, scipy, opencv-python-headless,
torch, torchvision and gdown.

Example (Colab GPU)::

    !python patchcore_remote.py --dataset ped2   --raw-root /content/raw \
        --meta-dir /content/meta/ped2   --out /content/out/ped2_pc.npz
    !python patchcore_remote.py --dataset avenue --raw-root /content/raw \
        --meta-dir /content/meta/avenue --out /content/out/avenue_pc.npz
    !python patchcore_remote.py --dataset mivia  --raw-root /content/raw \
        --meta-dir /content/meta/mivia_flow --out /content/out/mivia_pc.npz

``--raw-root`` is a directory laid out as the downloads place it::

    <raw-root>/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped2/{Train,Test}/...   (ped2)
    <raw-root>/avenue/extracted/Avenue Dataset/{training,testing}_videos/ (avenue)
    <raw-root>/avenue/gt_extracted/ground_truth_demo/testing_label_mask/  (avenue)
    <raw-root>/mivia/{train,test}/{videos,labels}/                        (mivia)

Missing raw files are downloaded into that layout: ped2/avenue from the official URLs,
mivia from the two Google Drive folders with per-file retries.

``--meta-dir`` holds the uploaded frame metadata: ``train_meta.json`` and ``test_meta.json``
(for mivia also ``sampled_indices.json``), copied from the local ``data/real_vad/<dataset>``
directory (``mivia_flow`` for mivia). Per-video frame counts are asserted against them.

Method (fixed, published defaults; not tuned on test results):
  * torchvision wide_resnet50_2, ImageNet IMAGENET1K_V1 weights, eval, no grad, fp32.
  * Input: native-resolution frame in RGB (gray replicated to 3 channels), INTER_AREA to
    224x224, ImageNet mean/std.
  * layer2 (512x28x28) and layer3 (1024x14x14), each 3x3 average pooling (stride 1, pad 1);
    layer3 bilinearly upsampled to 28x28; concatenated (1536 ch); reduced per location to
    1024 dims by adaptive average pooling over the channel axis (PatchCore reference).
  * Memory bank candidates: patches of every 4th sampled frame of each training video.
  * Coreset: approximate greedy k-center (PatchCore ApproximateGreedyCoresetSampler) in a
    fixed 128-d Gaussian projection (torch seed 0), random start (seed 0), 1% of candidates.
  * Frame score: max over the 784 patches of the min L2 distance (1024-d, fp32) to the bank.
  * Training frames (ALL of them): leave-one-video-out, i.e. bank entries from the frame's own
    video are excluded. Test frames: full bank.
  * Raw scores only. The 3-frame per-video median filter is applied in assemble_patchcore.py.

Output ``.npz``: ``train_scores`` (train_meta order), ``test_scores`` (test_meta order),
``config`` (JSON string with every setting, counts and names) and ``timing`` (JSON string).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

UCSD_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
AVENUE_URL = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip"
AVENUE_GT_URL = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip"
MIVIA_DRIVE = {
    "train": "https://drive.google.com/drive/folders/1DgXQnvK3keGP9BKB-cqpixnKaNLl2746",
    "test": "https://drive.google.com/drive/folders/1OnA3rHn5uKAMEuHsQc11oxueTT-TQQwJ",
}

IMG = 224
FEAT_HW = 28
N_PATCH = FEAT_HW * FEAT_HW  # 784
PATCH_DIM = 1024
CONCAT_DIM = 512 + 1024      # layer2 + layer3 channels
TRAIN_STRIDE = 4             # memory bank uses every 4th sampled training frame
CORESET_FRAC = 0.01
PROJ_DIM = 128
SEED = 0
QCHUNK = 2048                # query patches per cdist call
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------- raw layout
def ped2_root(raw: Path) -> Path:
    return raw / "ucsd" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"


def avenue_root(raw: Path) -> Path:
    return raw / "avenue" / "extracted" / "Avenue Dataset"


def avenue_gt_root(raw: Path) -> Path:
    return raw / "avenue" / "gt_extracted" / "ground_truth_demo" / "testing_label_mask"


def mivia_root(raw: Path, split: str) -> Path:
    return raw / "mivia" / split


def _fetch(url: str, dest: Path, tries: int = 3) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            part.rename(dest)
            return
        except OSError as e:
            print(f"  download failed ({attempt}/{tries}): {url}: {e}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def ensure_ped2(raw: Path) -> None:
    if ped2_root(raw).exists():
        return
    base = raw / "ucsd"
    base.mkdir(parents=True, exist_ok=True)
    tgz = base / "UCSD_Anomaly_Dataset.tar.gz"
    if not tgz.exists():
        _fetch(UCSD_URL, tgz)
    with tarfile.open(tgz) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(base, filter="data")
        else:
            t.extractall(base)
    if not ped2_root(raw).exists():
        raise RuntimeError("UCSD archive did not contain UCSDped2")


def ensure_avenue(raw: Path) -> None:
    base = raw / "avenue"
    if not avenue_root(raw).exists():
        z = base / "Avenue_Dataset.zip"
        if not z.exists():
            _fetch(AVENUE_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(base / "extracted")
    if not avenue_gt_root(raw).exists():
        z = base / "ground_truth_demo.zip"
        if not z.exists():
            _fetch(AVENUE_GT_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(base / "gt_extracted")
    if not avenue_root(raw).exists() or not avenue_gt_root(raw).exists():
        raise RuntimeError("Avenue archives did not contain the expected folders")


def ensure_mivia(raw: Path, needed: dict) -> None:
    """Download the Mivia Drive folders file by file; ``needed`` maps split -> video names."""
    import gdown

    for split in ("train", "test"):
        root = mivia_root(raw, split)
        missing = [n for n in needed[split] if not (root / "videos" / n).exists()]
        if not missing:
            continue
        listing = gdown.download_folder(url=MIVIA_DRIVE[split], output=str(root),
                                        skip_download=True, quiet=True)
        print(f"  mivia {split}: {len(listing)} files listed on Drive", flush=True)
        for item in listing:
            dest = Path(item.local_path)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, 5):
                try:
                    out = gdown.download(id=item.id, output=str(dest), quiet=True)
                    if out and dest.exists() and dest.stat().st_size > 0:
                        break
                except Exception as e:  # gdown raises on quota / network errors
                    print(f"  {item.path}: attempt {attempt} failed: {e}", flush=True)
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"mivia {split}: could not download {item.path}")
        still = [n for n in needed[split] if not (root / "videos" / n).exists()]
        if still:
            raise RuntimeError(f"mivia {split}: videos still missing after download: {still[:5]}")


# ----------------------------------------------------------------- frame sources
def video_source(ds: str, raw: Path, split: str, name: str, limit, indices):
    """Yield (k, native frame) for one video, in the local pipeline's frame order.

    ped2 / avenue: every frame, truncated to ``limit`` frames (the label count for test).
    mivia: exactly the native frames listed in ``indices`` (sampled_indices.json).
    """
    if ds == "ped2":
        d = ped2_root(raw) / ("Train" if split == "train" else "Test") / name
        files = sorted(f for f in os.listdir(d) if f.endswith(".tif"))
        if limit is not None:
            files = files[:limit]
        for k, f in enumerate(files):
            img = cv2.imread(str(d / f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"unreadable frame {d / f}")
            yield k, img
    elif ds == "avenue":
        sub = "training_videos" if split == "train" else "testing_videos"
        cap = cv2.VideoCapture(str(avenue_root(raw) / sub / name))
        try:
            k = 0
            while limit is None or k < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                yield k, frame
                k += 1
        finally:
            cap.release()
    elif ds == "mivia":
        cap = cv2.VideoCapture(str(mivia_root(raw, split) / "videos" / name))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {name}")
        try:
            wanted = list(indices)
            ptr, i = 0, 0
            while ptr < len(wanted):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{name}: video ended before sampled frame {wanted[ptr]}")
                if i == wanted[ptr]:
                    yield ptr, frame
                    ptr += 1
                i += 1
        finally:
            cap.release()
    else:
        raise ValueError(ds)


def to_model_input(native: np.ndarray) -> np.ndarray:
    if native.ndim == 2:
        rgb = cv2.cvtColor(native, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(native, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)


def strided_natives(src, stride: int, seen: list):
    """Yield the native frames whose index is a multiple of ``stride``; ``seen[0]`` counts all."""
    for k, nat in src:
        seen[0] = k + 1
        if k % stride == 0:
            yield nat


def only_natives(src, seen: list):
    for k, nat in src:
        seen[0] = k + 1
        yield nat


# ----------------------------------------------------------------- model
def build_backbone(device: torch.device):
    from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def patch_features(m, images: list, device: torch.device) -> torch.Tensor:
    """List of (224,224,3) uint8 RGB -> (B, 784, 1024) fp32 PatchCore features."""
    x = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - MEAN.to(device)) / STD.to(device)
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer1(x)
    l2 = m.layer2(x)                      # B,512,28,28
    l3 = m.layer3(l2)                     # B,1024,14,14
    l2 = F.avg_pool2d(l2, 3, 1, 1)
    l3 = F.avg_pool2d(l3, 3, 1, 1)
    l3 = F.interpolate(l3, size=(FEAT_HW, FEAT_HW), mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], dim=1)        # B,1536,28,28
    b = f.shape[0]
    f = f.permute(0, 2, 3, 1).reshape(b * N_PATCH, CONCAT_DIM).unsqueeze(1)
    f = F.adaptive_avg_pool1d(f, PATCH_DIM).squeeze(1)   # B*784,1024
    return f.view(b, N_PATCH, PATCH_DIM)


def feature_batches(m, natives, device: torch.device, batch: int):
    buf = []
    for nat in natives:
        buf.append(to_model_input(nat))
        if len(buf) == batch:
            yield patch_features(m, buf, device)
            buf = []
    if buf:
        yield patch_features(m, buf, device)


# ----------------------------------------------------------------- memory bank
@torch.no_grad()
def approx_greedy_coreset(P: torch.Tensor, m: int, seed: int = SEED) -> np.ndarray:
    """Greedy k-center in the projected space (PatchCore's approximate greedy sampler).

    Uses squared distances internally: argmax over squared minimum distances equals argmax
    over the Euclidean minimum distances used by the reference implementation.
    """
    n = P.shape[0]
    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=g))
    sq = (P * P).sum(dim=1)
    sel = torch.empty(m, dtype=torch.long)
    sel[0] = first
    mind = (sq - 2.0 * (P @ P[first]) + sq[first]).clamp_min_(0)
    mind[first] = 0
    for t in range(1, m):
        i = int(torch.argmax(mind))
        sel[t] = i
        d = (sq - 2.0 * (P @ P[i]) + sq[i]).clamp_min_(0)
        torch.minimum(mind, d, out=mind)
        mind[i] = 0
    return sel.numpy()


@torch.no_grad()
def frame_scores(feats: torch.Tensor, bank: torch.Tensor, excl: torch.Tensor | None = None) -> np.ndarray:
    """(B,784,1024) -> (B,) max over patches of min L2 distance to the bank.

    ``excl`` is an optional boolean mask over bank rows that must not be used (leave-one-video-out).
    """
    b = feats.shape[0]
    q = feats.reshape(-1, feats.shape[-1])
    out = torch.empty(q.shape[0], device=q.device)
    for s in range(0, q.shape[0], QCHUNK):
        d = torch.cdist(q[s:s + QCHUNK], bank)
        if excl is not None:
            d.masked_fill_(excl.view(1, -1), float("inf"))
        out[s:s + QCHUNK] = d.min(dim=1).values
    return out.view(b, N_PATCH).max(dim=1).values.double().cpu().numpy()


# ----------------------------------------------------------------- plan
def load_meta(meta_dir: Path):
    train_meta = json.loads((meta_dir / "train_meta.json").read_text())
    test_meta = json.loads((meta_dir / "test_meta.json").read_text())
    sampled = None
    if (meta_dir / "sampled_indices.json").exists():
        sampled = json.loads((meta_dir / "sampled_indices.json").read_text())
    return train_meta, test_meta, sampled


def build_plan(ds: str, raw: Path, train_meta, test_meta, sampled):
    """Per video: name, frame-count limit for the source, and sampled native indices (mivia)."""
    train, test = [], []
    if ds == "ped2":
        root = ped2_root(raw)
        train_names = sorted(d for d in os.listdir(root / "Train")
                             if d.startswith("Train") and not d.startswith("."))
        test_names = sorted(d for d in os.listdir(root / "Test")
                            if d.startswith("Test") and d[4:].isdigit())
        for name in train_names:
            train.append({"name": name, "limit": None, "indices": None})
        for name in test_names:
            gt = root / "Test" / f"{name}_gt"
            limit = sum(1 for f in os.listdir(gt) if f.endswith(".bmp"))
            test.append({"name": name, "limit": limit, "indices": None})
    elif ds == "avenue":
        import scipy.io as sio

        train_names = sorted(f for f in os.listdir(avenue_root(raw) / "training_videos") if f.endswith(".avi"))
        test_names = sorted(f for f in os.listdir(avenue_root(raw) / "testing_videos") if f.endswith(".avi"))
        for name in train_names:
            train.append({"name": name, "limit": None, "indices": None})
        for name in test_names:
            idx = int(name.split(".")[0])
            vol = sio.loadmat(avenue_gt_root(raw) / f"{idx}_label.mat")["volLabel"][0]
            test.append({"name": name, "limit": len(vol), "indices": None})
    else:  # mivia: the uploaded metadata defines the videos and the sampled frames
        if sampled is None:
            raise FileNotFoundError("mivia needs sampled_indices.json in --meta-dir")
        for m in train_meta:
            train.append({"name": m["name"], "limit": None, "indices": sampled[m["name"]]})
        for m in test_meta:
            test.append({"name": m["name"], "limit": None, "indices": sampled[m["name"]]})
    return train, test


def check_plan_against_meta(ds: str, plan_items, meta, split: str):
    names = [p["name"] for p in plan_items]
    meta_names = [m["name"] for m in meta]
    if ds != "mivia" and names != meta_names:
        raise AssertionError(f"{ds} {split}: video names differ from the uploaded metadata "
                             f"({names[:3]}... vs {meta_names[:3]}...)")
    if ds == "mivia":
        for p, m in zip(plan_items, meta):
            if len(p["indices"]) != m["n_frames"]:
                raise AssertionError(f"mivia {split} {p['name']}: sampled_indices has "
                                     f"{len(p['indices'])} frames, metadata {m['n_frames']}")


def _check_count(ds: str, split: str, name: str, got: int, expected: int) -> None:
    if got != expected:
        raise AssertionError(f"{ds} {split} {name}: decoded {got} frames, metadata says {expected}")


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ped2", "avenue", "mivia"])
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--meta-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-videos", type=int, default=None,
                    help="process only the first N training and first N test videos (smoke tests)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (torch and OpenCV)")
    args = ap.parse_args()
    if args.out.suffix != ".npz":
        raise SystemExit("--out must end in .npz")

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds, raw = args.dataset, args.raw_root
    timing: dict = {}
    t_all = time.time()

    train_meta_all, test_meta_all, sampled = load_meta(args.meta_dir)
    # Acquire raw data if absent. ped2/avenue plans read the raw folders, so acquire first;
    # mivia plans come from the uploaded metadata, so they are built first.
    t0 = time.time()
    if ds == "mivia":
        plan_tr_all, plan_te_all = build_plan(ds, raw, train_meta_all, test_meta_all, sampled)
        ensure_mivia(raw, {"train": [p["name"] for p in plan_tr_all],
                           "test": [p["name"] for p in plan_te_all]})
    else:
        if ds == "ped2":
            ensure_ped2(raw)
        else:
            ensure_avenue(raw)
        plan_tr_all, plan_te_all = build_plan(ds, raw, train_meta_all, test_meta_all, sampled)
    timing["acquire_s"] = round(time.time() - t0, 1)
    # Every video's name and frame count is checked against the uploaded metadata.
    check_plan_against_meta(ds, plan_tr_all, train_meta_all, "train")
    check_plan_against_meta(ds, plan_te_all, test_meta_all, "test")
    n_max = args.max_videos
    plan_tr, plan_te = plan_tr_all[:n_max], plan_te_all[:n_max]
    train_meta, test_meta = train_meta_all[:n_max], test_meta_all[:n_max]

    model = build_backbone(device)
    g = torch.Generator().manual_seed(SEED)
    proj = torch.randn(PATCH_DIM, PROJ_DIM, generator=g).to(device)
    bs = args.batch_size

    # Pass A: candidate patches of every 4th training frame, projected to 128-d.
    t0 = time.time()
    cand_proj, cand_src = [], []
    n_cand_frames = 0
    for j, p in enumerate(plan_tr):
        seen = [0]
        natives = strided_natives(video_source(ds, raw, "train", p["name"], None, p["indices"]),
                                  TRAIN_STRIDE, seen)
        for feats in feature_batches(model, natives, device, bs):
            flat = feats.reshape(-1, PATCH_DIM)
            cand_proj.append(flat @ proj)
            cand_src.append(torch.full((flat.shape[0],), j, dtype=torch.int32, device=device))
            n_cand_frames += feats.shape[0]
        _check_count(ds, "train", p["name"], seen[0], train_meta[j]["n_frames"])
    P_all = torch.cat(cand_proj)
    src_all = torch.cat(cand_src)
    del cand_proj, cand_src
    n_cand = int(P_all.shape[0])
    timing["candidates_s"] = round(time.time() - t0, 1)

    # Coreset of 1% of the candidate patches.
    t0 = time.time()
    m = max(1, int(round(CORESET_FRAC * n_cand)))
    sel = np.sort(approx_greedy_coreset(P_all, m))
    sel_t = torch.from_numpy(sel).to(device)
    bank_src = src_all[sel_t].clone()
    del P_all, src_all
    timing["coreset_s"] = round(time.time() - t0, 1)

    # Pass B: re-featurise the same training frames and keep the selected patches (full 1024-d).
    t0 = time.time()
    bank = torch.empty(m, PATCH_DIM, device=device)
    frames_done = 0
    for p in plan_tr:
        natives = strided_natives(video_source(ds, raw, "train", p["name"], None, p["indices"]),
                                  TRAIN_STRIDE, [0])
        for feats in feature_batches(model, natives, device, bs):
            nfr = feats.shape[0]
            lo, hi = frames_done * N_PATCH, (frames_done + nfr) * N_PATCH
            a, b = int(np.searchsorted(sel, lo)), int(np.searchsorted(sel, hi))
            if b > a:
                local = torch.from_numpy(sel[a:b] - lo).to(device)
                bank[a:b] = feats.reshape(-1, PATCH_DIM)[local]
            frames_done += nfr
    if frames_done != n_cand_frames:
        raise AssertionError("pass B did not revisit the same frames as pass A")
    timing["bank_gather_s"] = round(time.time() - t0, 1)
    print(f"[{ds}] candidates={n_cand} bank={m} ({timing['candidates_s']}s, coreset "
          f"{timing['coreset_s']}s)", flush=True)

    # Pass C: ALL training frames, leave-one-video-out.
    t0 = time.time()
    train_parts = []
    for j, p in enumerate(plan_tr):
        excl = bank_src == j
        if bool(excl.all()):
            raise AssertionError("leave-one-video-out would exclude the whole bank")
        seen = [0]
        parts = []
        for feats in feature_batches(model, only_natives(video_source(ds, raw, "train", p["name"],
                                                                      None, p["indices"]), seen),
                                     device, bs):
            parts.append(frame_scores(feats, bank, excl))
        _check_count(ds, "train", p["name"], seen[0], train_meta[j]["n_frames"])
        s = np.concatenate(parts)
        if not np.all(s > 0):
            raise AssertionError(f"{p['name']}: non-positive leave-one-video-out score")
        train_parts.append(s)
        print(f"  [C] train {j + 1}/{len(plan_tr)} {p['name']}: n={s.size} "
              f"mean={s.mean():.4f} n_excluded={int(excl.sum())}", flush=True)
    timing["train_scores_s"] = round(time.time() - t0, 1)

    # Pass D: test frames, full bank.
    t0 = time.time()
    test_parts = []
    for j, p in enumerate(plan_te):
        seen = [0]
        parts = []
        for feats in feature_batches(model, only_natives(video_source(ds, raw, "test", p["name"],
                                                                      p["limit"], p["indices"]), seen),
                                     device, bs):
            parts.append(frame_scores(feats, bank, None))
        _check_count(ds, "test", p["name"], seen[0], test_meta[j]["n_frames"])
        test_parts.append(np.concatenate(parts))
        print(f"  [D] test {j + 1}/{len(plan_te)} {p['name']}: n={seen[0]}", flush=True)
    timing["test_scores_s"] = round(time.time() - t0, 1)

    train_scores = np.concatenate(train_parts).astype(np.float64)
    test_scores = np.concatenate(test_parts).astype(np.float64)
    timing["total_s"] = round(time.time() - t_all, 1)
    if device.type == "cuda":
        timing["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    config = {
        "dataset": ds,
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "backbone": "torchvision wide_resnet50_2, IMAGENET1K_V1, eval, no grad, fp32",
        "input": "native frame, RGB (gray replicated), INTER_AREA to 224x224, ImageNet mean/std",
        "features": ("layer2 (512x28x28) and layer3 (1024x14x14), each 3x3 avg pool stride 1 pad 1; "
                     "layer3 bilinear to 28x28; concat 1536; adaptive_avg_pool1d to 1024 per location"),
        "patches_per_frame": N_PATCH,
        "memory_bank_candidates": f"patches of every {TRAIN_STRIDE}th sampled frame of each training video",
        "candidate_patches": n_cand,
        "bank_size": m,
        "coreset": (f"approximate greedy k-center, fraction {CORESET_FRAC} of candidates, "
                    f"{PROJ_DIM}-d fixed Gaussian projection (unscaled, torch seed {SEED}), "
                    f"random start (seed {SEED})"),
        "score": "frame = max over 784 patches of min L2 distance to bank (k=1), fp32",
        "calibration": "ALL training frames, leave-one-video-out (own-video bank entries excluded)",
        "test": "full bank",
        "smoothing": "none in this file; 3-frame per-video median filter in assemble_patchcore.py",
        "train_stride": TRAIN_STRIDE,
        "batch_size": args.batch_size,
        "max_videos": args.max_videos,
        "complete": args.max_videos is None,
        "train_names": [p["name"] for p in plan_tr],
        "test_names": [p["name"] for p in plan_te],
        "train_n_frames": [int(s.size) for s in train_parts],
        "test_n_frames": [int(s.size) for s in test_parts],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, train_scores=train_scores, test_scores=test_scores,
             config=np.array(json.dumps(config)), timing=np.array(json.dumps(timing)))
    print(f"[{ds}] train_scores {train_scores.shape} test_scores {test_scores.shape} "
          f"-> {args.out} ({timing})", flush=True)


if __name__ == "__main__":
    main()

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

The figure also has two Mivia-IWDD rows (one static and one dynamic dumping
test video). Their stills are read from the raw Mivia-IWDD release with
OpenCV, which is run with ``python3 -I`` and never executes anything from the
release. Set ``SENTINEL_MIVIA_RAW`` to its root (containing ``test/videos`` and
``test/labels``). Only two downscaled stills per video are written to
``data/real_vad/mivia_patchcore/qualitative_examples.npz``. The PatchCore score
trace is read from the committed per-frame scores.
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

#: Mivia-IWDD: raw release root (not committed). Override with SENTINEL_MIVIA_RAW.
MIVIA_RAW = Path(os.environ.get("SENTINEL_MIVIA_RAW", str(Path("/tmp/data") / "mivia")))
MIVIA_DIR = DATA_ROOT / "mivia_patchcore"
MIVIA_FLOW_DIR = DATA_ROOT / "mivia_flow"
MIVIA_FPS = 10.0            # sampling rate of the committed Mivia scores
MIVIA_PRE_S = 1.0           # pre-onset still: this many seconds before onset
MIVIA_POST_S = 2.0          # post-onset still: this many seconds after onset
MIVIA_MIN_PRE_S = 3.0       # candidate videos need this much footage before onset
MIVIA_STILL_WIDTH = 384     # stored still width in pixels (display only)
#: (panel key, event type in test_meta.json). The first qualifying positive video of
#: each type, in video-id order, is used. No selection on the score values.
MIVIA_ROLES = (("static", "static dumping"), ("dynamic", "dynamic dumping"))


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


def _mivia_offsets(meta) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([m["n_frames"] for m in meta])])


def _choose_mivia(meta, labels, offs, event_type):
    """First positive test video of ``event_type`` with enough footage on both sides of onset."""
    for i, m in enumerate(meta):
        if m["event_type"] != event_type:
            continue
        lab = labels[offs[i]:offs[i + 1]]
        if not lab.any():
            continue
        k0 = int(np.argmax(lab))
        k_pre = k0 - int(round(MIVIA_PRE_S * MIVIA_FPS))
        k_post = k0 + int(round(MIVIA_POST_S * MIVIA_FPS))
        if k0 >= MIVIA_MIN_PRE_S * MIVIA_FPS and k_post < lab.size:
            return i, k0, k_pre, k_post
    raise RuntimeError(f"no positive {event_type!r} test video with enough footage")


def extract_mivia_examples() -> Path:
    """Read two frames per Mivia example from the raw release with OpenCV only.

    The release is untrusted data: it is only opened as video and label files,
    and nothing in it is executed. Stills are stored downscaled, for display.
    """
    import cv2

    out_path = MIVIA_DIR / "qualitative_examples.npz"
    if not MIVIA_RAW.is_dir():
        raise FileNotFoundError(
            f"{MIVIA_RAW} not found. Set SENTINEL_MIVIA_RAW to the Mivia-IWDD raw root "
            "(the folder that contains test/videos and test/labels).")
    meta = json.loads((MIVIA_DIR / "test_meta.json").read_text())
    labels = np.load(MIVIA_DIR / "test_labels.npy").astype(bool)
    sampled = json.loads((MIVIA_FLOW_DIR / "sampled_indices.json").read_text())
    offs = _mivia_offsets(meta)
    payload, info = {}, {}
    for kind, etype in MIVIA_ROLES:
        i, k0, k_pre, k_post = _choose_mivia(meta, labels, offs, etype)
        name = meta[i]["name"]
        label_file = json.loads((MIVIA_RAW / "test" / "labels" / f"{Path(name).stem}.json").read_text())
        onset_s = float(label_file["timestamp"])
        if not (onset_s - 1e-9 <= k0 / MIVIA_FPS < onset_s + 1.0 / MIVIA_FPS):
            raise RuntimeError(f"{name}: first positive sample {k0} does not match onset {onset_s} s")
        native = {"pre": int(sampled[name][k_pre]), "post": int(sampled[name][k_post])}
        want = {native["pre"]: "pre", native["post"]: "post"}
        cap = cv2.VideoCapture(str(MIVIA_RAW / "test" / "videos" / name))
        got, j, last = {}, 0, max(want)
        while j <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if j in want:
                got[want[j]] = frame
            j += 1
        cap.release()
        if set(got) != {"pre", "post"}:
            raise RuntimeError(f"could not read both stills from {name}")
        for role in ("pre", "post"):
            rgb = cv2.cvtColor(got[role], cv2.COLOR_BGR2RGB)
            h = int(round(rgb.shape[0] * MIVIA_STILL_WIDTH / rgb.shape[1]))
            payload[f"{kind}_{role}"] = cv2.resize(rgb, (MIVIA_STILL_WIDTH, h), interpolation=cv2.INTER_AREA)
        info[kind] = {"video": name, "event_type": etype, "onset_s": onset_s,
                      "k_onset": k0, "k_pre": k_pre, "k_post": k_post,
                      "native_pre": native["pre"], "native_post": native["post"]}
    payload["info"] = np.array(json.dumps(info))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def load_mivia_examples():
    path = MIVIA_DIR / "qualitative_examples.npz"
    if not path.exists():
        path = extract_mivia_examples()
    z = np.load(path)
    frames = {k: z[k] for k in z.files if k != "info"}
    return frames, json.loads(str(z["info"]))


def _mivia_trace(name: str) -> np.ndarray:
    """PatchCore per-frame scores of one Mivia test video (committed, 10 fps)."""
    meta = json.loads((MIVIA_DIR / "test_meta.json").read_text())
    offs = _mivia_offsets(meta)
    scores = np.load(MIVIA_DIR / "test_scores.npy").astype(float)
    i = [m["name"] for m in meta].index(name)
    return scores[offs[i]:offs[i + 1]]


def make_figure(mivia_frames: dict, mivia_info: dict):
    plt = setup_matplotlib()
    labels = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}
    fig = plt.figure(figsize=(8.5, 8.8))
    gs = fig.add_gridspec(4, 4, hspace=0.62, wspace=0.34, left=0.08, right=0.98,
                          top=0.97, bottom=0.04)
    scores = {}

    for row, name in enumerate(("ped2", "avenue")):
        ex = load_examples(name)
        scores[name] = {}
        for col, kind in enumerate(("normal", "anomalous")):
            prev_f, cur_f = ex[f"{kind}_prev"], ex[f"{kind}_cur"]
            native = ex[f"{kind}_native"]
            mag, score = _flow_and_score(prev_f, cur_f)
            scores[name][kind] = score
            ax_frame = fig.add_subplot(gs[row, 2 * col])
            ax_flow = fig.add_subplot(gs[row, 2 * col + 1])
            ax_frame.imshow(native, cmap="gray", vmin=0, vmax=1)
            ax_frame.set_title(f"{kind} frame", fontsize=8)
            ax_flow.imshow(mag, cmap="inferno")
            ax_flow.set_title(f"flow magnitude\nscore={score:.3f}", fontsize=8)
            for ax in (ax_frame, ax_flow):
                ax.set_xticks([]); ax.set_yticks([])
        fig.text(0.02, 0.97 - row * (0.93 / 4) - 0.1, labels[name], rotation=90,
                 va="center", ha="center", fontsize=9)

    scores["mivia"] = {}
    for row, (kind, etype) in enumerate(MIVIA_ROLES, start=2):
        info = mivia_info[kind]
        trace = _mivia_trace(info["video"])
        t = np.arange(trace.size) / MIVIA_FPS
        t_pre, t_post = info["k_pre"] / MIVIA_FPS, info["k_post"] / MIVIA_FPS
        scores["mivia"][kind] = {"video": info["video"], "onset_s": info["onset_s"],
                                 "score_pre": float(trace[info["k_pre"]]),
                                 "score_post": float(trace[info["k_post"]])}
        for col, role, t_role in ((0, "pre", t_pre), (1, "post", t_post)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(mivia_frames[f"{kind}_{role}"])
            ax.set_title(f"{role}-onset frame ({t_role:.1f} s)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        ax_tr = fig.add_subplot(gs[row, 2:4])
        ax_tr.plot(t, trace, color="#333333", lw=0.9)
        ax_tr.axvline(info["onset_s"], color="#c0392b", ls="--", lw=0.9)
        ax_tr.plot([t_pre], [trace[info["k_pre"]]], "o", ms=3.5, color="#2c7fb8")
        ax_tr.plot([t_post], [trace[info["k_post"]]], "o", ms=3.5, color="#41ab5d")
        ax_tr.text(info["onset_s"], ax_tr.get_ylim()[1], " onset", color="#c0392b",
                   fontsize=7, va="top", ha="left")
        ax_tr.set_title(f"PatchCore score, {etype} ({info['video']})", fontsize=8)
        ax_tr.set_xlabel("time (s)", fontsize=7)
        ax_tr.set_ylabel("PatchCore score", fontsize=7, labelpad=2)
        ax_tr.tick_params(labelsize=7)
        fig.text(0.02, 0.97 - row * (0.93 / 4) - 0.1, f"Mivia {kind}", rotation=90,
                 va="center", ha="center", fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_r0_qualitative.{ext}", dpi=200)
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
    path = MIVIA_DIR / "qualitative_examples.npz"
    if not path.exists():
        print(f" extracting Mivia-IWDD example stills from {MIVIA_RAW} ...")
    else:
        print(f" mivia_patchcore: using cached {path}")
    mivia_frames, mivia_info = load_mivia_examples()
    scores = make_figure(mivia_frames, mivia_info)
    save_json({"examples": EXAMPLES, "scores": scores, "mivia_examples": mivia_info},
              "exp_r0_qualitative")
    print(" wrote results/figures/fig_r0_qualitative.{pdf,png}, "
          "results/exp_r0_qualitative.json")


if __name__ == "__main__":
    main()

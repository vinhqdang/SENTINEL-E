"""Experiment R4 --- per-camera warm-up normalisation on Mivia-IWDD (one video = one stream).

Problem. Under protocol v3 every Mivia-IWDD video is a different camera, yet the
calibration pool of :mod:`exp_r3_per_video` concatenates the raw scores of all 200
normal training videos. Scores from one camera are not exchangeable with another
camera's, so a pooled threshold can be miscalibrated for the cameras it is applied to.

Deployment assumption (stated, not tested here). The first W seconds of every
monitored video, and of every training video, are footage of the same camera taken
at installation time and are known to be normal (no dumping in progress).

Warm-up normalisation. For a video with per-frame scores ``s`` and warm-up length
``n_W = round(W * fps)`` frames, with ``warm = s[:n_W]``:

    med = median(warm),   mad = median(|warm - med|),
    z_t = (s_t - med) / (mad + EPS)     for t >= n_W,      EPS = 1e-6 (fixed, score units).

Training normal videos are standardised with their own warm-up. The calibration pool is
the concatenation of their standardised post-warm-up scores. Each test video is monitored
only after its own warm-up, with a fresh detector, exactly as in :mod:`exp_r3_per_video`
(same thinning lag search, same methods and knob sweeps, same LOVO / 10-fold selection by
video on training videos for L* and the baseline knobs, same Clopper--Pearson and Poisson
intervals).

Arms (same populations, same selection machinery):
* ``warm``: warm-up normalised scores (the method under test);
* ``raw``: post-warm-up raw scores, no standardisation (a control that isolates the effect
  of the normalisation from the effect of dropping the warm-up footage);
* pooled, full videos, raw scores: the protocol v3 numbers, read unchanged from
  ``results/exp_r3_per_video.json``. Its population differs (no warm-up removed).

Usability. A video (training or test) contributes a null trial only if its post-warm-up
part lasts at least ``MIN_NULL_S`` = 2 s; otherwise it is excluded for that W and counted.
A positive video (training or test) is evaluated only if its onset is at or after
``W + 2`` s; positives with an earlier onset are excluded for that W and counted.

Pre-registered rule for the warm-up length W (written before any test-set evaluation
was run; training data only; separately for each dataset and each alpha):

1. For each W in {5, 10, 20} s, select SENTINEL-E's lag L*_W(alpha) on the warm arm
   exactly as in exp_r3 (smallest grid lag whose LOVO / 10-fold per-video false-alarm
   proportion on training normal videos is at most alpha; SENTINEL-E knob = alpha).
2. W is feasible iff that selection did not fail, i.e. the training-normal per-video
   false-alarm proportion at L*_W(alpha) is at most alpha.
3. Among feasible W, choose the one that maximises SENTINEL-E's episode detection on the
   TRAINING POSITIVE videos (post-warm-up part, onset at or after W + 2 s, calibration on
   all usable training normal videos, lag L*_W(alpha)).
4. Ties, and the case where the training-positive detection is not evaluable for any
   feasible W, are broken by the smallest W.
5. If no W is feasible, the warm arm reports no admissible W and has no test result.

Test evaluation (once, after 1--5). Per-video false alarms on the negative test videos
(post-warm-up) and episode detection on the positive test videos with onset at or after
W + 2 s, with static / dynamic dumping split. Baseline knobs are selected on training
videos as in exp_r3 at the selected W and the arm's own L*. Other W values are reported
as a sensitivity analysis, labelled as not selected on training data.

Training positives. The Mivia training split has 200 positive videos (labelled "true")
that were not scored by the protocol v3 extraction. :func:`extract_train_positives`
scores them with the same flow backbone and writes
``data/real_vad/mivia_flow/train_positive_scores.npy`` and ``train_positive_meta.json``.
No PatchCore scores exist for them; for ``mivia_patchcore`` the rule's step 3 is therefore
not evaluable and step 4 applies (documented in the outputs).

Every unit is cached in ``results/cache/exp_r4/`` as it finishes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.common import TABLES, fmt, load_json, parallel_map, save_json, write_latex_table
from experiments.real_data.exp_r2_delayfar import SWEEPS, _fmt_knob
from experiments.real_data.exp_r3_per_video import (
    ALPHAS, BASELINES, DS, HEADLINE, LABELS, LAG_GRID, MIN_NULL_S, SMOKE_LAG_GRID,
    TestVideo, _runs, _slug, _stream, get_ds, select_knob, select_lag, test_metrics,
)
from experiments.real_data.protocol_v2 import run_alarms, score_labelled_stream
from sentinel_e.datasets.real_vad import DATA_ROOT, causal_local_context

W_GRID = (5, 10, 20)
EPS = 1e-6
ARMS = ("warm", "raw")
ARM_LABELS = {"warm": "warm-up normalised", "raw": "post-warm-up raw (control)"}
CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "cache" / "exp_r4"
SMOKE_DIR = Path("/tmp/claude-0/-home-user-SENTINEL-E/11c4e690-66a8-5971-aa35-0c968d71c4a8/scratchpad/exp_r4_smoke")
BASE_DATASETS = ("mivia_flow", "mivia_patchcore")
OPTIONAL_DATASETS = ("mivia_supervised",)
POS_SCORES = "train_positive_scores.npy"
POS_META = "train_positive_meta.json"


# --------------------------------------------------------------------------- #
# Pure helpers (exercised by tests/test_exp_r4_warmup.py)
# --------------------------------------------------------------------------- #
def warmup_normalise(scores, n_warm: int, eps: float = EPS) -> np.ndarray:
    """Standardise the post-warm-up part of one video by its own warm-up.

    Returns ``(s[n_warm:] - median(warm)) / (MAD(warm) + eps)`` where ``warm = s[:n_warm]``.
    """
    s = np.asarray(scores, float)
    if not 1 <= int(n_warm) < s.size:
        raise ValueError(f"a warm-up of {n_warm} frames leaves no post-warm-up part of a "
                         f"{s.size}-frame video")
    warm = s[:n_warm]
    med = float(np.median(warm))
    mad = float(np.median(np.abs(warm - med)))
    return (s[n_warm:] - med) / (mad + eps)


def post_warmup(scores, n_warm: int, arm: str, eps: float = EPS) -> np.ndarray:
    """Post-warm-up stream of one video under ``arm`` (``warm`` or ``raw``)."""
    if arm == "warm":
        return warmup_normalise(scores, n_warm, eps)
    if arm == "raw":
        s = np.asarray(scores, float)
        if not 1 <= int(n_warm) < s.size:
            raise ValueError("warm-up does not leave a post-warm-up part")
        return s[n_warm:].copy()
    raise ValueError(f"unknown arm {arm!r}")


def usable_null(post_size: int, min_null_frames: int) -> bool:
    """A video contributes a null trial only if its post-warm-up part is long enough."""
    return int(post_size) >= int(min_null_frames)


def onset_eligible(onset_frame: Optional[int], n_warm: int, min_null_frames: int) -> bool:
    """A positive is evaluated only if its onset is at or after W + 2 s (video-local frames)."""
    return onset_frame is not None and int(onset_frame) >= int(n_warm) + int(min_null_frames)


def choose_w(feasible: Dict[int, bool], detection: Dict[int, Optional[float]]) -> Tuple[Optional[int], str]:
    """Step 2--4 of the pre-registered rule for one dataset and alpha.

    ``feasible[W]``: the training-normal FA constraint holds at L*_W. ``detection[W]``:
    SENTINEL-E detection on training positives, or None when not evaluable.
    Returns ``(W or None, reason)``.
    """
    cands = sorted(W for W, ok in feasible.items() if ok)
    if not cands:
        return None, "no admissible W (no W meets the training-normal FA constraint)"
    if all(detection.get(W) is None for W in cands):
        return cands[0], "tie-break: training-positive detection not evaluable; smallest feasible W"
    best = max(cands, key=lambda W: (-1.0 if detection.get(W) is None else detection[W]))
    return best, "maximum training-positive detection among feasible W; ties to smallest W"


def _pool(train: Sequence[Optional[np.ndarray]], exclude: Sequence[int] = ()) -> np.ndarray:
    ex = set(int(e) for e in exclude)
    parts = [np.asarray(v, float) for j, v in enumerate(train) if v is not None and j not in ex]
    if not parts:
        raise ValueError("no usable calibration videos")
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Training positives (flow backbone only)
# --------------------------------------------------------------------------- #
def extract_train_positives(out_dir: Optional[Path] = None) -> Path:
    """Score the 200 labelled-positive training videos with the flow backbone.

    Uses :func:`extract_mivia.process_video` unchanged, so the scores are the same
    quantity as the test and normal-training scores. Writes concatenated scores and a
    per-video metadata file (``onset_frame`` is the index among sampled frames of the
    first frame at or after the labelled onset, -1 if none).
    """
    import cv2
    from experiments.real_data.extract_mivia import load_split, process_video

    cv2.setNumThreads(2)
    out_dir = out_dir or (DATA_ROOT / "mivia_flow")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_split("train") if r["label"] == "true"]
    parts, meta = [], []
    t0 = time.time()
    for j, r in enumerate(rows):
        res = process_video(r)
        onset = -1
        if r["timestamp"] >= 0:
            hit = np.flatnonzero(res["t"] >= r["timestamp"])
            onset = int(hit[0]) if hit.size else -1
        meta.append({"name": r["video_id"], "n_frames": int(res["n_frames"]), "onset_frame": onset,
                     "onset_s": float(r["timestamp"]), "event_type": r["event_type"],
                     "native_fps": res["native_fps"]})
        parts.append(np.asarray(res["scores"], np.float64))
        if j % 20 == 0:
            print(f"  train positive {j + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(out_dir / POS_SCORES, np.concatenate(parts))
    (out_dir / POS_META).write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(meta)} training positives to {out_dir}", flush=True)
    return out_dir / POS_META


def load_train_positives(name: str) -> Optional[List[Dict]]:
    meta_p = DATA_ROOT / name / POS_META
    sc_p = DATA_ROOT / name / POS_SCORES
    if not (meta_p.exists() and sc_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    flat = np.load(sc_p)
    out, off = [], 0
    for m in meta:
        n = int(m["n_frames"])
        out.append({"name": m["name"], "scores": flat[off:off + n].copy(),
                    "onset": int(m["onset_frame"]), "event_type": m.get("event_type")})
        off += n
    if off != flat.size:
        raise ValueError(f"{name}: training-positive metadata covers {off} frames, file has {flat.size}")
    return out


# --------------------------------------------------------------------------- #
# Per-(dataset, arm, W) preparation
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    ds: DS
    arm: str
    W: int
    n_warm: int
    train: List[Optional[np.ndarray]]    # usable training normal post-warm-up streams, else None
    train_pos: List[TestVideo]           # included training positives
    test: List[TestVideo]                # included test videos (negatives and positives)
    info: Dict


_PREP: Dict[Tuple[str, str, int], Prepared] = {}


def prepare(name: str, arm: str, W: int) -> Prepared:
    key = (name, arm, int(W))
    if key in _PREP:
        return _PREP[key]
    ds = get_ds(name)
    n_warm = int(round(W * ds.fps))
    minN = ds.min_null_frames
    train: List[Optional[np.ndarray]] = []
    for v in ds.train:
        if usable_null(v.size - n_warm, minN):
            train.append(post_warmup(v, n_warm, arm))
        else:
            train.append(None)
    n_train_usable = sum(t is not None for t in train)

    pos_excl = {"no_onset": 0, "onset_before_W_plus_2s": 0}
    train_pos: List[TestVideo] = []
    tp = load_train_positives(name)
    for p in tp or []:
        if p["onset"] < 0:
            pos_excl["no_onset"] += 1
            continue
        if not onset_eligible(p["onset"], n_warm, minN):
            pos_excl["onset_before_W_plus_2s"] += 1
            continue
        post = post_warmup(p["scores"], n_warm, arm)
        onset = p["onset"] - n_warm
        lab = np.zeros(post.size, bool)
        lab[onset:] = True
        train_pos.append(TestVideo(p["name"], post, lab, [(onset, post.size)], onset, p["event_type"]))

    neg_excl = 0
    pos_test_excl = {"onset_before_W_plus_2s": 0}
    test: List[TestVideo] = []
    for v in ds.test:
        if v.onset is None:
            if not usable_null(v.scores.size - n_warm, minN):
                neg_excl += 1
                continue
            post = post_warmup(v.scores, n_warm, arm)
            test.append(TestVideo(v.name, post, np.zeros(post.size, bool), [], None, v.event_type))
        else:
            if not onset_eligible(v.onset, n_warm, minN):
                pos_test_excl["onset_before_W_plus_2s"] += 1
                continue
            post = post_warmup(v.scores, n_warm, arm)
            lab = np.asarray(v.labels[n_warm:], bool)
            eps = _runs(lab)
            test.append(TestVideo(v.name, post, lab, eps, eps[0][0], v.event_type))

    info = {
        "W_s": int(W), "n_warm_frames": n_warm, "fps": ds.fps,
        "train_videos": len(ds.train), "train_usable": int(n_train_usable),
        "train_excluded_short": int(len(ds.train) - n_train_usable),
        "train_positives_available": tp is not None,
        "train_positives_total": len(tp) if tp is not None else None,
        "train_positives_included": len(train_pos),
        "train_positives_excluded": pos_excl,
        "test_negative_included": sum(v.onset is None for v in test),
        "test_negative_excluded_short": int(neg_excl),
        "test_positive_included": sum(v.onset is not None for v in test),
        "test_positive_excluded": pos_test_excl,
    }
    P = Prepared(ds=ds, arm=arm, W=int(W), n_warm=n_warm, train=train, train_pos=train_pos,
                 test=test, info=info)
    _PREP[key] = P
    return P


# --------------------------------------------------------------------------- #
# Units (cached on disk, keyed by spec)
# --------------------------------------------------------------------------- #
def _key(spec: Dict) -> str:
    parts = [spec["ds"], spec["arm"], f"W{spec['W']}", spec["kind"]]
    if "method" in spec:
        parts += [_slug(spec["method"]), f"k{spec['knob']:.6g}"]
    parts.append(f"L{spec['lag']}")
    if "fold" in spec:
        parts.append(f"f{spec['fold']}")
    return "__".join(parts)


def cv_spec(ds, arm, W, method, knob, lag, fold) -> Dict:
    return {"ds": ds, "arm": arm, "W": int(W), "kind": "cv", "method": method,
            "knob": float(knob), "lag": int(lag), "fold": int(fold)}


def run_spec(ds, arm, W, kind, method, knob, lag) -> Dict:
    """kind: 'test' (test videos) or 'pos' (training positives)."""
    return {"ds": ds, "arm": arm, "W": int(W), "kind": kind, "method": method,
            "knob": float(knob), "lag": int(lag)}


def _alarms(method, knob, lag, scores, labels, episodes, cal, cal_ctx) -> List[int]:
    st = _stream(scores, labels, episodes, cal, cal_ctx)
    return run_alarms(method, knob, st, lag).tolist()


def _compute(spec: Dict) -> Dict:
    P = prepare(spec["ds"], spec["arm"], spec["W"])
    m, kn, L = spec["method"], spec["knob"], spec["lag"]
    if spec["kind"] == "cv":
        held = [j for j in P.ds.folds[spec["fold"]] if P.train[j] is not None]
        cal = _pool(P.train, held)
        ctx = causal_local_context(cal).reshape(-1, 1)
        alarms = {}
        for j in held:
            sc = P.train[j]
            alarms[str(j)] = _alarms(m, kn, L, sc, np.zeros(sc.size, bool), [], cal, ctx)
        return {"alarms": alarms}
    cal = _pool(P.train)
    ctx = causal_local_context(cal).reshape(-1, 1)
    items = P.test if spec["kind"] == "test" else P.train_pos
    whole = {str(i): _alarms(m, kn, L, v.scores, v.labels, v.episodes, cal, ctx)
             for i, v in enumerate(items)}
    return {"whole": whole}


def _unit(args) -> Dict:
    cache_dir, key, spec = args
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())["result"]
        except (json.JSONDecodeError, KeyError):
            pass
    res = _compute(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp{os.getpid()}"
    tmp.write_text(json.dumps({"key": key, "spec": spec, "result": res}))
    os.replace(tmp, path)
    return res


def run_units(specs: Sequence[Dict], cache_dir: Path, desc: str, workers: int) -> Dict[str, Dict]:
    unique: Dict[str, Dict] = {}
    for s in specs:
        unique.setdefault(_key(s), s)
    keys = list(unique)
    print(f"  {desc}: {len(keys)} units", flush=True)
    outs = parallel_map(_unit, [(str(cache_dir), k, unique[k]) for k in keys], workers=workers, desc=desc)
    return dict(zip(keys, outs))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def lovo_counts(R: Dict, spec_fn, n_folds: int) -> Tuple[int, int, Dict[str, int]]:
    k = n = 0
    per: Dict[str, int] = {}
    for f in range(n_folds):
        for j, al in R[_key(spec_fn(f))]["alarms"].items():
            n += 1
            k += int(len(al) > 0)
            per[j] = len(al)
    return k, n, per


def pos_detection(P: Prepared, R_pos: Dict) -> Dict:
    """SENTINEL-E episode detection on the included training positives."""
    n_ep = n_det = 0
    for i, v in enumerate(P.train_pos):
        r = score_labelled_stream(R_pos["whole"][str(i)], v.labels, v.episodes, P.ds.grace, P.ds.fps)
        n_ep += r["n_episodes"]
        n_det += r["n_detected"]
    return {"n_episodes": int(n_ep), "n_detected": int(n_det),
            "detection": (n_det / n_ep) if n_ep else None}


def test_view(P: Prepared, R_test: Dict) -> Dict:
    """test_metrics over the included test population of one configuration."""
    ds_view = dataclasses.replace(P.ds, test=P.test)
    return test_metrics(ds_view, {"whole": R_test["whole"], "prefix": {}})


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def available(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    ok, skipped = [], []
    for n in names:
        if not n.startswith("mivia"):
            raise SystemExit(f"{n}: warm-up grid {W_GRID} s is not feasible for this corpus "
                             f"(25 fps, training videos of 120-1511 frames); exp_r4 is Mivia-only")
        d = DATA_ROOT / n
        if not (d / "backbone_auc.json").exists():
            print(f"  SKIPPED {n}: {d} is not complete (no backbone_auc.json)", flush=True)
            skipped.append(n)
            continue
        ok.append(n)
    return ok, skipped


def run(names: Sequence[str], grid: Sequence[int], cache_dir: Path, workers: int) -> Dict:
    t0 = time.time()
    names, skipped = available(names)
    print(f"Datasets: {names}", flush=True)
    R: Dict[str, Dict] = {}
    folds_n = {n: len(get_ds(n).folds) for n in names}

    # Phase A: SENTINEL-E LOVO / fold curves for both arms, every W and alpha.
    specs = [cv_spec(n, arm, W, "SENTINEL-E", a, L, f)
             for n in names for arm in ARMS for W in W_GRID for a in ALPHAS
             for L in grid for f in range(folds_n[n])]
    R.update(run_units(specs, cache_dir, "A: SENTINEL-E training-normal curves", workers))
    sel: Dict = {n: {arm: {} for arm in ARMS} for n in names}
    for n in names:
        for arm in ARMS:
            for W in W_GRID:
                sel[n][arm][W] = {}
                for a in ALPHAS:
                    curve = {}
                    for L in grid:
                        k, nn, _ = lovo_counts(R, lambda f, L=L, a=a, W=W, arm=arm:
                                               cv_spec(n, arm, W, "SENTINEL-E", a, L, f), folds_n[n])
                        curve[L] = (k, nn)
                    Lst, failed = select_lag(curve, a, grid)
                    k_at, n_at = curve[Lst]
                    sel[n][arm][W][a] = {
                        "lag": int(Lst), "selection_failed": bool(failed),
                        "lovo_k": int(k_at), "lovo_n": int(n_at),
                        "lovo_curve": {str(L): {"k": int(curve[L][0]), "n": int(curve[L][1])} for L in grid},
                    }
                    print(f"  {n:16s} {arm:4s} W={W:<3d} alpha={a:<5g} L*={Lst:<3d} "
                          f"LOVO {k_at}/{n_at}{'  NOT FEASIBLE' if failed else ''}", flush=True)

    # Phase A2: training-positive detection for the W rule (warm arm, SENTINEL-E).
    pos_specs = []
    for n in names:
        if not prepare(n, "warm", W_GRID[0]).info["train_positives_available"]:
            continue
        for W in W_GRID:
            for a in ALPHAS:
                pos_specs.append(run_spec(n, "warm", W, "pos", "SENTINEL-E", a, sel[n]["warm"][W][a]["lag"]))
    R.update(run_units(pos_specs, cache_dir, "A2: training-positive detection", workers))

    # Step 2-4 of the pre-registered rule.
    wsel: Dict[str, Dict[float, Optional[int]]] = {n: {} for n in names}
    wrule: Dict[str, Dict] = {n: {} for n in names}
    for n in names:
        has_pos = prepare(n, "warm", W_GRID[0]).info["train_positives_available"]
        for a in ALPHAS:
            feas = {W: not sel[n]["warm"][W][a]["selection_failed"] for W in W_GRID}
            det: Dict[int, Optional[float]] = {}
            per_w = {}
            for W in W_GRID:
                P = prepare(n, "warm", W)
                if has_pos:
                    sp = run_spec(n, "warm", W, "pos", "SENTINEL-E", a, sel[n]["warm"][W][a]["lag"])
                    pd = pos_detection(P, R[_key(sp)])
                else:
                    pd = {"n_episodes": 0, "n_detected": 0, "detection": None}
                det[W] = pd["detection"]
                per_w[str(W)] = {**pd, "feasible": bool(feas[W]), "lag": sel[n]["warm"][W][a]["lag"],
                                 "lovo_k": sel[n]["warm"][W][a]["lovo_k"],
                                 "lovo_n": sel[n]["warm"][W][a]["lovo_n"]}
            W_star, reason = choose_w(feas, det)
            wsel[n][a] = W_star
            wrule[n][str(a)] = {"selected_W": W_star, "reason": reason, "per_W": per_w}
            print(f"  W-rule {n:16s} alpha={a:<5g} -> W*={W_star}  ({reason})", flush=True)

    # Phase B: baseline LOVO at W*(alpha) and each arm's own L*; knobs selected as in exp_r3.
    bspecs = []
    for n in names:
        for a in ALPHAS:
            W = wsel[n][a]
            if W is None:
                continue
            for arm in ARMS:
                Lb = sel[n][arm][W][a]["lag"]
                for m in BASELINES:
                    for kn in SWEEPS[m]:
                        for f in range(folds_n[n]):
                            bspecs.append(cv_spec(n, arm, W, m, kn, Lb, f))
    R.update(run_units(bspecs, cache_dir, "B: baseline training-normal curves", workers))
    base_sel: Dict = {n: {} for n in names}
    knob_sel: Dict[Tuple[str, float, str, str], float] = {}
    for n in names:
        for a in ALPHAS:
            W = wsel[n][a]
            base_sel[n][str(a)] = {}
            if W is None:
                continue
            for arm in ARMS:
                Lb = sel[n][arm][W][a]["lag"]
                base_sel[n][str(a)][arm] = {}
                for m in BASELINES:
                    rows = []
                    for kn in SWEEPS[m]:
                        k, nn, _ = lovo_counts(R, lambda f, kn=kn, m=m, Lb=Lb, arm=arm, W=W:
                                               cv_spec(n, arm, W, m, kn, Lb, f), folds_n[n])
                        rows.append((k, nn))
                    idx, no_adm = select_knob(rows, a)
                    knob_sel[(n, a, arm, m)] = SWEEPS[m][idx]
                    base_sel[n][str(a)][arm][m] = {
                        "knob": float(SWEEPS[m][idx]), "lag": int(Lb), "no_admissible_knob": bool(no_adm),
                        "lovo": [{"knob": float(kn), "k": int(k), "n": int(nn)}
                                 for kn, (k, nn) in zip(SWEEPS[m], rows)],
                    }

    # Phase C: test evaluation at W*(alpha), and a W sensitivity for SENTINEL-E (not selected on training).
    tspecs = []
    for n in names:
        for a in ALPHAS:
            W = wsel[n][a]
            if W is not None:
                for arm in ARMS:
                    Lb = sel[n][arm][W][a]["lag"]
                    tspecs.append(run_spec(n, arm, W, "test", "SENTINEL-E", a, Lb))
                    for m in BASELINES:
                        tspecs.append(run_spec(n, arm, W, "test", m, knob_sel[(n, a, arm, m)], Lb))
            for arm in ARMS:
                for Wn in W_GRID:
                    tspecs.append(run_spec(n, arm, Wn, "test", "SENTINEL-E", a, sel[n][arm][Wn][a]["lag"]))
    R.update(run_units(tspecs, cache_dir, "C: test", workers))

    test_out: Dict = {n: {"at_selected_W": {}, "W_sensitivity": {}} for n in names}
    for n in names:
        for a in ALPHAS:
            W = wsel[n][a]
            if W is not None:
                block = {}
                for arm in ARMS:
                    Lb = sel[n][arm][W][a]["lag"]
                    P = prepare(n, arm, W)
                    entry = {"W": int(W), "lag": int(Lb), "info": P.info}
                    entry["SENTINEL-E"] = {"knob": a, **test_view(P, R[_key(run_spec(n, arm, W, "test", "SENTINEL-E", a, Lb))])}
                    for m in BASELINES:
                        kn = knob_sel[(n, a, arm, m)]
                        entry[m] = {"knob": float(kn), **test_view(P, R[_key(run_spec(n, arm, W, "test", m, kn, Lb))])}
                    block[arm] = entry
                test_out[n]["at_selected_W"][str(a)] = block
            sens = {}
            for arm in ARMS:
                for Wn in W_GRID:
                    P = prepare(n, arm, Wn)
                    Lb = sel[n][arm][Wn][a]["lag"]
                    sens[f"{arm}_W{Wn}"] = {"arm": arm, "W": int(Wn), "lag": int(Lb), "info": P.info,
                                            **test_view(P, R[_key(run_spec(n, arm, Wn, "test", "SENTINEL-E", a, Lb))])}
            test_out[n]["W_sensitivity"][str(a)] = sens

    meta = {}
    for n in names:
        ds = get_ds(n)
        meta[n] = {"label": LABELS[n], "family": ds.family, "fps": ds.fps, "grace_frames": ds.grace,
                   "fold_scheme": "mod10", "n_train_videos": len(ds.train), "n_folds": len(ds.folds),
                   "n_test_videos": len(ds.test),
                   "n_test_negative": sum(v.onset is None for v in ds.test),
                   "n_test_positive": sum(v.onset is not None for v in ds.test),
                   "train_positives_available": bool(prepare(n, "warm", W_GRID[0]).info["train_positives_available"]),
                   "min_null_frames": ds.min_null_frames}
    return {"meta": meta, "skipped": skipped, "datasets": names, "grid": list(grid),
            "selection": {n: {arm: {str(W): {str(a): sel[n][arm][W][a] for a in ALPHAS}
                                    for W in W_GRID} for arm in ARMS} for n in names},
            "w_rule": wrule, "selected_W": {n: {str(a): wsel[n][a] for a in ALPHAS} for n in names},
            "baselines": base_sel, "test": test_out, "runtime_s": time.time() - t0}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _ci(ci, nd: int = 3) -> str:
    return f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"


def _kn(p: Dict) -> str:
    return f"{p['k']}/{p['n']}"


def _det_cell(d: Optional[Dict]) -> str:
    if d is None or not d.get("n_episodes"):
        return "--"
    return f"{fmt(d['detection']['pfa'], 2)} {_ci(d['detection']['ci'], 2)} ({_kn(d['detection'])})"


def write_outputs(payload: Dict, pooled_test: Dict) -> None:
    names = payload["datasets"]
    test = payload["test"]
    pooled = pooled_test

    # Headline: pooled (exp_r3) vs post-warm-up raw (control) vs warm-up normalised, SENTINEL-E.
    rows = []
    for n in names:
        for a in ALPHAS:
            pe = pooled[n][str(a)]["SENTINEL-E"]
            rows.append([LABELS[n], f"{a:g}", "pooled, full video (exp\\_r3)", "--", pe["lag"],
                         f"{fmt(pe['null_primary']['pfa'], 3)} {_ci(pe['null_primary']['ci'])} ({_kn(pe['null_primary'])})",
                         f"{fmt(pe['fa_per_hour'], 2)} {_ci(pe['fa_per_hour_ci'], 2)}",
                         f"{fmt(pe['episodes']['pfa'], 2)} {_ci(pe['episodes']['ci'], 2)} ({_kn(pe['episodes'])})",
                         _det_cell(pe["by_event_type"].get("static dumping")),
                         _det_cell(pe["by_event_type"].get("dynamic dumping")),
                         fmt(pe["delay"]["median_s"], 1),
                         f"{pe['null_primary']['n']} / {pe['episodes']['n']}"])
            blk = test[n]["at_selected_W"].get(str(a))
            if blk is None:
                rows.append([LABELS[n], f"{a:g}", "no admissible W", "--", "--", "--", "--", "--", "--", "--", "--", "--"])
                continue
            for arm in ARMS:
                e = blk[arm]
                s = e["SENTINEL-E"]
                bt = s["by_event_type"]
                rows.append([LABELS[n], f"{a:g}", ARM_LABELS[arm], e["W"], e["lag"],
                             f"{fmt(s['null_primary']['pfa'], 3)} {_ci(s['null_primary']['ci'])} ({_kn(s['null_primary'])})",
                             f"{fmt(s['fa_per_hour'], 2)} {_ci(s['fa_per_hour_ci'], 2)}",
                             f"{fmt(s['episodes']['pfa'], 2)} {_ci(s['episodes']['ci'], 2)} ({_kn(s['episodes'])})",
                             _det_cell(bt.get("static dumping")), _det_cell(bt.get("dynamic dumping")),
                             fmt(s["delay"]["median_s"], 1),
                             f"{s['null_primary']['n']} / {s['episodes']['n']}"])
    write_latex_table(
        rows,
        ["dataset", r"$\alpha$", "source", "$W$ (s)", r"$L^\ast$", "test per-video FA [95\\% CI] (k/n)",
         "FA / hour [95\\% CI]", "episode detection [95\\% CI] (k/n)", "static detection [95\\% CI] (k/n)",
         "dynamic detection [95\\% CI] (k/n)", "median delay (s)", "null videos / episodes"],
        caption=("SENTINEL-E on Mivia-IWDD, pooled calibration (protocol v3, full videos, from "
                 "\\texttt{exp\\_r3\\_per\\_video}) against the same method after a per-camera warm-up "
                 "(\\,$W$ selected on training videos by the rule fixed before the test evaluation\\,) and against the "
                 "post-warm-up raw scores (control). Each test video is monitored alone with a fresh "
                 "detector; calibration on all usable training normal videos. Warm-up arms: negative test "
                 "videos with less than 2\\,s after warm-up and positives with onset before $W+2$\\,s are excluded."),
        label="tab:r4_warmup", name="tabr4_warmup", align="llllrrrrrrrr",
        notes=("Pooled rows use every test video (50 negative, 50 positive, per Mivia backbone). Warm-up rows use the "
               "post-warm-up population of the same $W$. Per-video FA is over negative test videos; episode detection "
               "over positive test videos (one episode each). Deployment assumption: the first $W$ s of each video "
               "are normal installation footage. $L^\\ast$ and baseline knobs were selected on training videos only."),
    )

    # Selection table (training data only).
    rows = []
    for n in names:
        for a in ALPHAS:
            rule = payload["w_rule"][n][str(a)]
            for W in (5, 10, 20):
                pw = rule["per_W"][str(W)]
                info = payload["test"][n]["W_sensitivity"][str(a)][f"warm_W{W}"]["info"]
                ex_n = info["train_excluded_short"]
                posn = (f"{pw['n_detected']}/{pw['n_episodes']}" if pw["n_episodes"] else "not evaluable")
                rows.append([LABELS[n], f"{a:g}", W, pw["lag"], f"{pw['lovo_k']}/{pw['lovo_n']}",
                             "yes" if pw["feasible"] else "no",
                             f"{info['train_positives_included']} ({info['train_positives_excluded']['onset_before_W_plus_2s']} excl.)"
                             if info["train_positives_available"] else "none scored",
                             posn, f"{ex_n}",
                             "selected" if rule["selected_W"] == W else ""])
    write_latex_table(
        rows,
        ["dataset", r"$\alpha$", "$W$ (s)", r"$L^\ast_W$", "training-normal LOVO FA (k/n)", "FA constraint met",
         "training positives used", "training-positive detection (k/n)", "normal videos too short",
         "selection"],
        caption=("Warm-up length selection on training videos only, by the rule fixed before the test evaluation: "
                 "feasible $W$ meets the per-video false-alarm bound at its own $L^\\ast_W(\\alpha)$ on training "
                 "normal videos; among feasible $W$ the one with the highest SENTINEL-E detection on training "
                 "positive videos is chosen, ties to the smallest $W$."),
        label="tab:r4_warmup_select", name="tabr4_warmup_select", align="llrrrlllrl",
        notes=("Training-normal LOVO (Mivia: 10 folds by video index), so denominators are the usable training "
               "videos. Detection on training positives is not evaluable for Mivia-IWDD PatchCore (no training-positive "
               "PatchCore scores), so the tie-break decides there; see the report."),
    )

    # Baselines at alpha = 0.1: pooled vs warm-up normalised.
    rows = []
    for n in names:
        blk = test[n]["at_selected_W"].get(str(HEADLINE))
        if blk is None:
            continue
        for m in ["SENTINEL-E"] + BASELINES:
            pe = pooled[n][str(HEADLINE)][m]
            we = blk["warm"][m]
            rows.append([LABELS[n], m, "pooled", _fmt_knob(pe["knob"]),
                         f"{fmt(pe['null_primary']['pfa'], 3)} {_ci(pe['null_primary']['ci'])}",
                         f"{fmt(pe['episodes']['pfa'], 2)} {_ci(pe['episodes']['ci'], 2)}"])
            rows.append([LABELS[n], m, f"warm-up, $W={blk['warm']['W']}$", _fmt_knob(we["knob"]),
                         f"{fmt(we['null_primary']['pfa'], 3)} {_ci(we['null_primary']['ci'])}",
                         f"{fmt(we['episodes']['pfa'], 2)} {_ci(we['episodes']['ci'], 2)}"])
    if rows:
        write_latex_table(
            rows,
            ["dataset", "method", "source", "knob", "test per-video FA [95\\% CI]", "episode detection [95\\% CI]"],
            caption=("Baselines at nominal $\\alpha=0.1$ on Mivia-IWDD, pooled (protocol v3) against warm-up "
                     "normalised scores. Each method is thinned at its own $L^\\ast$ for its source and takes "
                     "its knob from training videos only, by the exp\\_r3 rule."),
            label="tab:r4_warmup_baselines", name="tabr4_warmup_baselines", align="llllrr",
            notes="Population as in Table~\\ref{tab:r4_warmup}.",
        )

    # W sensitivity (not selected on training data).
    rows = []
    for n in names:
        for a in ALPHAS:
            for key in ("warm_W5", "warm_W10", "warm_W20", "raw_W5", "raw_W10", "raw_W20"):
                s = test[n]["W_sensitivity"][str(a)][key]
                info = s["info"]
                rows.append([LABELS[n], f"{a:g}", ARM_LABELS[s["arm"]], s["W"], s["lag"],
                             f"{info['test_negative_included']} / {info['test_positive_included']}",
                             f"{fmt(s['null_primary']['pfa'], 3)} {_ci(s['null_primary']['ci'])} ({_kn(s['null_primary'])})",
                             f"{fmt(s['episodes']['pfa'], 2)} {_ci(s['episodes']['ci'], 2)} ({_kn(s['episodes'])})"])
    write_latex_table(
        rows,
        ["dataset", r"$\alpha$", "source", "$W$ (s)", r"$L^\ast_W$", "test neg. / pos. included",
         "test per-video FA [95\\% CI] (k/n)", "episode detection [95\\% CI] (k/n)"],
        caption=("Sensitivity of SENTINEL-E to the warm-up length $W$ on the test videos, for both arms. "
                 "Every $W$ is shown and none is chosen from these numbers: the headline $W$ was fixed on training "
                 "data (Table~\\ref{tab:r4_warmup_select})."),
        label="tab:r4_warmup_sensitivity", name="tabr4_warmup_sensitivity", align="llllrrrr",
        notes="Each row uses its own $L^\\ast_W(\\alpha)$ selected on training videos for that $W$ and arm.",
    )


def main(argv: Optional[Sequence[str]] = None) -> Optional[Dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma-separated; default: mivia_flow, mivia_patchcore, and mivia_supervised if complete")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny lag grid; writes nothing under results/ (cache goes to the scratchpad)")
    ap.add_argument("--extract-train-positives", action="store_true",
                    help="score the 200 labelled-positive Mivia training videos (flow) and exit")
    a = ap.parse_args(argv)
    if a.extract_train_positives:
        extract_train_positives()
        return None
    if a.datasets:
        names = [x.strip() for x in a.datasets.split(",") if x.strip()]
    else:
        names = list(BASE_DATASETS)
        for opt in OPTIONAL_DATASETS:
            if (DATA_ROOT / opt / "backbone_auc.json").exists():
                names.append(opt)
    grid = SMOKE_LAG_GRID if a.smoke else LAG_GRID
    cache = SMOKE_DIR if a.smoke else CACHE_DIR
    print(f"Experiment R4 (warm-up normalisation): W={list(W_GRID)} s, eps={EPS}, smoke={a.smoke}, "
          f"lags={grid}, cache={cache}", flush=True)
    payload = run(names, grid, cache, a.workers)
    payload["protocol"] = {
        "warm_up_W_s": list(W_GRID), "eps": EPS, "alphas": ALPHAS, "headline_alpha": HEADLINE,
        "lag_grid": list(grid), "min_null_s": MIN_NULL_S, "onset_margin_s": MIN_NULL_S,
        "arms": ARM_LABELS, "fold_scheme": "mod10 (video index)",
        "w_rule": "pre-registered in the module docstring; tie-break to the smallest W",
        "deployment_assumption": "first W s of every video are normal installation footage",
        "sweeps": SWEEPS, "workers": a.workers,
    }
    if a.smoke:
        print(json.dumps({n: payload["selected_W"][n] for n in payload["datasets"]}, indent=1)[:2000])
        return payload
    pooled = load_json("exp_r3_per_video")
    payload["pooled_source"] = "results/exp_r3_per_video.json (unchanged)"
    save_json(payload, "exp_r4_warmup")
    write_outputs(payload, pooled["test"])
    print(f" wrote results/exp_r4_warmup.json and tables tabr4_warmup*.tex; runtime {payload['runtime_s']:.0f}s; "
          f"skipped={payload['skipped']}", flush=True)
    return payload


if __name__ == "__main__":
    main()

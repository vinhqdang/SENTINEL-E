"""Experiment R3 (protocol v3) --- one video is one monitored stream, real video.

The unit of replication is a single video. Every video is monitored on its own
with a freshly started detector: no splicing, no permutation, no frame reused
between calibration and monitoring.

* Operating points are chosen on TRAINING videos only, by leave-one-video-out
  (LOVO) for UCSD Ped2 and CUHK Avenue, and by 10 folds over video index
  (``i % 10``) for Mivia-IWDD (200 training videos make LOVO too costly to
  report with the same care). A held-out video's calibration set never contains
  that video.
* SENTINEL-E's betting lag ``L*(alpha)`` is the smallest grid lag whose LOVO
  per-video false-alarm proportion is at most ``alpha``. Each classical baseline
  is thinned at ``L*(alpha)`` and takes the first knob (in the order of
  :data:`exp_r2_delayfar.SWEEPS`, most aggressive first) whose LOVO proportion is
  at most ``alpha``.
* Test metrics use calibration on ALL training videos and are reported with exact
  Clopper--Pearson (proportions) and exact Poisson (false alarms per hour)
  intervals over independent videos.

Every unit (one method, one knob, one lag, one calibration fold, or one test
configuration) is written atomically to ``results/cache/exp_r3/`` as it finishes,
so an interrupted run resumes from the completed units.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from experiments.common import (
    FIGURES, REPO_ROOT, RESULTS, TABLES, fmt, parallel_map, save_json,
    setup_matplotlib, style_for, write_latex_table,
)
from experiments.real_data.exp_r2_delayfar import SWEEPS, _fmt_knob
from experiments.real_data.protocol_v2 import run_alarms, score_labelled_stream
from sentinel_e.datasets.real_vad import DATA_ROOT, causal_local_context, load_real_vad
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import Stream
from sentinel_e.temporal import acf, thin_indices

DATASETS = ("ped2", "ped2_patchcore", "avenue", "avenue_patchcore", "mivia_flow", "mivia_patchcore",
            "mivia_supervised")
LABELS = {
    "ped2": "UCSD Ped2 (flow)", "ped2_patchcore": "UCSD Ped2 (PatchCore)",
    "avenue": "CUHK Avenue (flow)", "avenue_patchcore": "CUHK Avenue (PatchCore)",
    "mivia_flow": "Mivia-IWDD (flow)", "mivia_patchcore": "Mivia-IWDD (PatchCore)",
    "mivia_supervised": "Mivia-IWDD (supervised)",
}
ALPHAS = [0.05, 0.1, 0.2]
HEADLINE = 0.1
LAG_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
SMOKE_LAG_GRID = [2, 10]
#: Minimum length of a pre-onset segment to count as a null trial (seconds).
MIN_NULL_S = 2.0
#: Floor used to draw zero false-alarm rates on the log-scale figure (per hour).
FA_FLOOR = 0.1
WORKERS = 2
BASELINES = [m for m in SWEEPS if m != "SENTINEL-E"]
CACHE_DIR = RESULTS / "cache" / "exp_r3"
SMOKE_DIR = Path("/tmp/sentinel_e_scratch/exp_r3_smoke")


# --------------------------------------------------------------------------- #
# Pure helpers (exercised by tests/test_exp_r3.py)
# --------------------------------------------------------------------------- #
def build_calibration(train: Sequence[np.ndarray], held: Sequence[int]) -> np.ndarray:
    """Concatenation of every training video whose index is not in ``held``."""
    held_set = set(int(h) for h in held)
    parts = [np.asarray(v, float) for j, v in enumerate(train) if j not in held_set]
    if not parts:
        raise ValueError("no calibration videos left after holding out the fold")
    return np.concatenate(parts)


def lovo_folds(n_videos: int, scheme: str) -> List[List[int]]:
    """Held-out index groups: leave-one-video-out, or ``i % 10`` folds."""
    if scheme == "lovo":
        return [[j] for j in range(n_videos)]
    if scheme == "mod10":
        return [[j for j in range(n_videos) if j % 10 == f] for f in range(10)]
    raise ValueError(f"unknown fold scheme {scheme!r}")


def select_lag(curve: Dict[int, Tuple[int, int]], alpha: float, grid: Sequence[int]) -> Tuple[int, bool]:
    """Smallest grid lag whose LOVO false-alarm proportion is at most ``alpha``.

    ``curve`` maps lag -> (k, n). Returns ``(L*, selection_failed)``; when no lag
    is admissible the largest lag is returned and the flag is set.
    """
    ordered = sorted(int(g) for g in grid)
    for lag in ordered:
        k, n = curve[lag]
        if n and k <= alpha * n + 1e-9:
            return lag, False
    return ordered[-1], True


def select_knob(pfas_in_sweep_order: Sequence[Tuple[int, int]], alpha: float) -> Tuple[int, bool]:
    """Index of the first knob (in sweep order, most aggressive first) with (k, n) admissible.

    Returns ``(index, no_admissible_knob)``; when none is admissible the most
    conservative knob (last index) is returned and the flag is set.
    """
    for i, (k, n) in enumerate(pfas_in_sweep_order):
        if n and k <= alpha * n + 1e-9:
            return i, False
    return len(pfas_in_sweep_order) - 1, True


def cp_interval(k: int, n: int) -> Tuple[float, float]:
    """Exact Clopper--Pearson 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def poisson_rate_interval(k: int, hours: float) -> Tuple[float, float]:
    """Exact Poisson 95% interval for a count ``k`` observed over ``hours``."""
    if hours <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.chi2.ppf(0.025, 2 * k) / 2.0)
    hi = float(stats.chi2.ppf(0.975, 2 * k + 2) / 2.0)
    return lo / hours, hi / hours


def prop(k: int, n: int) -> Dict:
    lo, hi = cp_interval(k, n)
    return {"k": int(k), "n": int(n), "pfa": (k / n) if n else None, "ci": [lo, hi]}


def _runs(labels: np.ndarray) -> List[Tuple[int, int]]:
    runs, in_run, start = [], False, 0
    for i, v in enumerate(labels):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(labels)))
    return runs


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
@dataclass
class TestVideo:
    name: str
    scores: np.ndarray
    labels: np.ndarray
    episodes: List[Tuple[int, int]]     # video-local [s, e)
    onset: Optional[int]                 # first labelled frame, None if no anomaly
    event_type: Optional[str]


@dataclass
class DS:
    name: str
    family: str                          # "vad" or "mivia"
    fps: float
    grace: int
    train: List[np.ndarray]
    train_names: List[str]
    test: List[TestVideo]
    folds: List[List[int]]
    min_null_frames: int
    mivia_episodes_ok: bool


_DS_CACHE: Dict[str, DS] = {}


def load_ds(name: str) -> DS:
    corpus = load_real_vad(name)
    meta = json.loads((DATA_ROOT / name / "test_meta.json").read_text())
    fps = float(corpus.fps)
    family = "mivia" if name.startswith("mivia") else "vad"
    train = [np.asarray(corpus.calibration_scores[lo:hi], float).copy()
             for _n, lo, hi in corpus.train_boundaries]
    train_names = [n for n, _lo, _hi in corpus.train_boundaries]
    tests, ok = [], True
    for (vn, lo, hi), m in zip(corpus.video_boundaries, meta):
        labels = np.asarray(corpus.test_labels[lo:hi], bool).copy()
        eps = _runs(labels)
        if family == "mivia":
            # One episode per positive video, running from onset to the end of the video.
            ok = ok and (len(eps) == 0 or (len(eps) == 1 and eps[0][1] == labels.size))
        onset = eps[0][0] if eps else None
        tests.append(TestVideo(vn, np.asarray(corpus.test_scores[lo:hi], float).copy(),
                               labels, eps, onset, m.get("event_type")))
    scheme = "mod10" if family == "mivia" else "lovo"
    return DS(
        name=name, family=family, fps=fps, grace=int(round(fps)), train=train,
        train_names=train_names, test=tests, folds=lovo_folds(len(train), scheme),
        min_null_frames=int(round(MIN_NULL_S * fps)), mivia_episodes_ok=bool(ok),
    )


def get_ds(name: str) -> DS:
    if name not in _DS_CACHE:
        _DS_CACHE[name] = load_ds(name)
    return _DS_CACHE[name]


def _stream(scores, labels, episodes, cal, cal_ctx) -> Stream:
    scores = np.asarray(scores, float)
    return Stream(
        scores=scores,
        context=causal_local_context(scores).reshape(-1, 1),
        labels=np.asarray(labels, bool),
        change_point=episodes[0][0] if episodes else None,
        calibration_scores=cal,
        calibration_context=cal_ctx,
        episodes=[(int(s), int(e)) for s, e in episodes],
    )


# --------------------------------------------------------------------------- #
# Units: each one is cached on disk, keyed by its spec
# --------------------------------------------------------------------------- #
def _key(spec: Dict) -> str:
    parts = [spec["ds"], spec["kind"]]
    if "method" in spec:
        parts.append(_slug(spec["method"]))
    if "knob" in spec:
        parts.append(f"k{spec['knob']:.6g}")
    parts.append(f"L{spec['lag']}")
    if "fold" in spec:
        parts.append(f"f{spec['fold']}")
    return "__".join(parts)


def _compute_cv(ds: DS, spec: Dict) -> Dict:
    """LOVO / fold null streams: each held-out training video monitored alone."""
    held = ds.folds[spec["fold"]]
    cal = build_calibration(ds.train, held)
    cal_ctx = causal_local_context(cal).reshape(-1, 1)
    alarms = {}
    for j in held:
        sc = ds.train[j]
        st = _stream(sc, np.zeros(sc.size, bool), [], cal, cal_ctx)
        alarms[str(j)] = run_alarms(spec["method"], spec["knob"], st, spec["lag"]).tolist()
    return {"alarms": alarms}


def _compute_test(ds: DS, spec: Dict) -> Dict:
    """Test videos monitored alone, calibrated on ALL training videos.

    ``whole``: every test video as one stream (detection, FA per hour, and the
    no-anomaly null trials). ``prefix``: the pre-onset segment of each positive
    test video whose segment lasts at least MIN_NULL_S seconds.
    """
    cal = np.concatenate(ds.train)
    cal_ctx = causal_local_context(cal).reshape(-1, 1)
    whole, prefix = {}, {}
    for i, v in enumerate(ds.test):
        st = _stream(v.scores, v.labels, v.episodes, cal, cal_ctx)
        whole[str(i)] = run_alarms(spec["method"], spec["knob"], st, spec["lag"]).tolist()
        if v.onset is not None and v.onset >= ds.min_null_frames:
            pre = v.scores[: v.onset]
            st_pre = _stream(pre, np.zeros(pre.size, bool), [], cal, cal_ctx)
            prefix[str(i)] = run_alarms(spec["method"], spec["knob"], st_pre, spec["lag"]).tolist()
    return {"whole": whole, "prefix": prefix}


def _compute_diag(ds: DS, spec: Dict) -> Dict:
    """Conformal p-values on the thinned grid for each held-out training video."""
    held = ds.folds[spec["fold"]]
    cal = build_calibration(ds.train, held)
    model = SentinelE(alpha=HEADLINE, calibration="pooled", lag=int(spec["lag"]), restart=True)
    model.fit(cal)
    out = {}
    for j in held:
        sc = ds.train[j]
        bet = thin_indices(sc.size, int(spec["lag"]))
        p, _active = model.p_values(sc[bet])
        out[str(j)] = [float(x) for x in p]
    return {"p": out}


def _unit(args) -> Dict:
    cache_dir, key, spec = args
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())["result"]
        except (json.JSONDecodeError, KeyError):
            pass  # a torn or foreign file is recomputed and overwritten
    ds = get_ds(spec["ds"])
    if spec["kind"] == "cv":
        res = _compute_cv(ds, spec)
    elif spec["kind"] == "test":
        res = _compute_test(ds, spec)
    elif spec["kind"] == "diag":
        res = _compute_diag(ds, spec)
    else:
        raise ValueError(spec["kind"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp{os.getpid()}"
    tmp.write_text(json.dumps({"key": key, "spec": spec, "result": res}))
    os.replace(tmp, path)
    return res


def run_units(specs: Sequence[Dict], cache_dir: Path, desc: str) -> Dict[str, Dict]:
    """Run every unit not yet cached, two worker processes at most. Returns key -> result."""
    unique: Dict[str, Dict] = {}
    for s in specs:
        unique.setdefault(_key(s), s)
    keys = list(unique)
    args = [(str(cache_dir), k, unique[k]) for k in keys]
    print(f"  {desc}: {len(keys)} units", flush=True)
    outs = parallel_map(_unit, args, workers=WORKERS, desc=desc)
    return dict(zip(keys, outs))


def cv_spec(ds: str, method: str, knob: float, lag: int, fold: int) -> Dict:
    return {"ds": ds, "kind": "cv", "method": method, "knob": float(knob), "lag": int(lag), "fold": int(fold)}


def test_spec(ds: str, method: str, knob: float, lag: int) -> Dict:
    return {"ds": ds, "kind": "test", "method": method, "knob": float(knob), "lag": int(lag)}


def diag_spec(ds: str, lag: int, fold: int) -> Dict:
    return {"ds": ds, "kind": "diag", "lag": int(lag), "fold": int(fold)}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def lovo_prop(ds: DS, R: Dict, method: str, knob: float, lag: int) -> Tuple[int, int, Dict[str, int]]:
    k = n = 0
    per_video: Dict[str, int] = {}
    for f in range(len(ds.folds)):
        for j, al in R[_key(cv_spec(ds.name, method, knob, lag, f))]["alarms"].items():
            n += 1
            k += int(len(al) > 0)
            per_video[j] = len(al)
    return k, n, per_video


def test_metrics(ds: DS, res: Dict) -> Dict:
    """Per-video null trials, FA per hour, episode detection and delays for one configuration."""
    whole, prefix = res["whole"], res["prefix"]
    fa = 0
    hours = 0.0
    n_ep = n_det = 0
    delays: List[int] = []
    by_type: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "det": 0, "delays": []})
    noanom: List[bool] = []
    pre: List[bool] = []
    for i, v in enumerate(ds.test):
        r = score_labelled_stream(whole[str(i)], v.labels, v.episodes, ds.grace, ds.fps)
        fa += r["n_false_alarms"]
        hours += r["normal_hours"]
        n_ep += r["n_episodes"]
        n_det += r["n_detected"]
        delays += r["delays"]
        if v.event_type is not None and v.event_type != "normal" and v.episodes:
            bt = by_type[v.event_type]
            bt["n"] += r["n_episodes"]
            bt["det"] += r["n_detected"]
            bt["delays"] += r["delays"]
        if v.onset is None:
            noanom.append(len(whole[str(i)]) > 0)
        elif str(i) in prefix:
            pre.append(len(prefix[str(i)]) > 0)
    # Mivia: primary null trials are the negative videos; ped2/avenue: pre-onset segments plus no-anomaly videos.
    primary = noanom if ds.family == "mivia" else noanom + pre
    kp, npr = sum(primary), len(primary)
    kn, nn = sum(noanom), len(noanom)
    kq, nq = sum(pre), len(pre)
    fa_lo, fa_hi = poisson_rate_interval(fa, hours)
    det_prop = prop(n_det, n_ep)

    def _delay(ds_list):
        if not ds_list:
            return {"median_s": None, "mean_s": None}
        return {"median_s": float(np.median(ds_list)) / ds.fps,
                "mean_s": float(np.mean(ds_list)) / ds.fps}

    types = {}
    for t, bt in sorted(by_type.items()):
        types[t] = {"n_episodes": bt["n"], "n_detected": bt["det"],
                    "detection": prop(bt["det"], bt["n"]), **_delay(bt["delays"])}
    return {
        "n_videos": len(ds.test),
        "null_primary": prop(kp, npr),
        "null_no_anomaly": prop(kn, nn),
        "null_pre_onset": prop(kq, nq),
        "false_alarms": int(fa),
        "normal_hours": float(hours),
        "fa_per_hour": (fa / hours) if hours > 0 else None,
        "fa_per_hour_ci": [fa_lo, fa_hi],
        "episodes": det_prop,
        "delay": _delay(delays),
        "by_event_type": types,
    }


def dependence_5a(ds: DS) -> Dict:
    raw = np.concatenate(ds.train)
    r = acf(raw, n_lags=200)
    below = np.flatnonzero(r[1:] < 0.05)
    within = {}
    for lag in (1, 5, 10, 25):
        vals = [acf(v, n_lags=25)[lag] for v in ds.train if v.size > 25]
        within[str(lag)] = {"mean": float(np.mean(vals)), "n_videos": len(vals)}
    return {
        "pooled_acf_length": int(below[0] + 1) if below.size else None,
        "pooled_acf_max_lag_checked": 200,
        "within_video_acf": within,
    }


def dependence_5b(ds: DS, R: Dict, lag_star: int) -> Dict:
    out = {}
    for label, L in (("L_star", lag_star), ("lag_1", 1)):
        ps: Dict[str, List[float]] = {}
        for f in range(len(ds.folds)):
            ps.update(R[_key(diag_spec(ds.name, L, f))]["p"])
        videos = [np.asarray(ps[str(j)], float) for j in range(len(ds.train))]
        xs = [v[:-1] for v in videos if v.size > 1]
        ys = [v[1:] for v in videos if v.size > 1]
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        allp = np.sort(np.concatenate(videos))
        m = allp.size
        F = np.searchsorted(allp, allp, side="right") / m
        mask = x <= 0.1
        out[label] = {
            "lag": int(L),
            "n_p_values": int(m),
            "n_pairs": int(x.size),
            "lag1_corr": float(np.corrcoef(x, y)[0, 1]),
            "sup_dev_pos": float(max(0.0, np.max(F - allp))),
            "cond_mean_pt_given_prev_le_0p1": float(np.mean(y[mask])) if mask.any() else None,
            "cond_count": int(mask.sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def available(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    ok, skipped = [], []
    for n in names:
        d = DATA_ROOT / n
        if not d.is_dir():
            print(f"  SKIPPED {n}: {d} does not exist", flush=True)
            skipped.append(n)
            continue
        try:
            get_ds(n)
        except FileNotFoundError as exc:
            print(f"  SKIPPED {n}: {exc}", flush=True)
            skipped.append(n)
            continue
        ok.append(n)
    return ok, skipped


def run(datasets: Sequence[str], grid: Sequence[int], cache_dir: Path) -> Dict:
    t0 = time.time()
    names, skipped = available(datasets)
    DSS = {n: get_ds(n) for n in names}
    R: Dict[str, Dict] = {}
    print(f"Datasets: {names}", flush=True)

    # Phase A: SENTINEL-E LOVO grid over (alpha, lag).
    specs = [cv_spec(n, "SENTINEL-E", a, L, f)
             for n in names for a in ALPHAS for L in grid for f in range(len(DSS[n].folds))]
    R.update(run_units(specs, cache_dir, "A: SENTINEL-E LOVO"))
    selection: Dict[str, Dict] = {}
    lag_star: Dict[Tuple[str, float], int] = {}
    for n in names:
        ds = DSS[n]
        selection[n] = {}
        for a in ALPHAS:
            curve = {}
            for L in grid:
                k, nn, _pv = lovo_prop(ds, R, "SENTINEL-E", a, L)
                curve[L] = (k, nn)
            Lst, failed = select_lag(curve, a, grid)
            lag_star[(n, a)] = Lst
            k_at, n_at = curve[Lst]
            selection[n][str(a)] = {
                "lag": int(Lst), "selection_failed": bool(failed),
                "lovo_k": int(k_at), "lovo_n": int(n_at),
                "lovo_curve": {str(L): {"k": int(curve[L][0]), "n": int(curve[L][1]),
                                         "pfa": curve[L][0] / curve[L][1]} for L in grid},
            }
            print(f"  {n:18s} alpha={a:<5g} L*={Lst:<3d} LOVO {k_at}/{n_at}"
                  f"{'  SELECTION FAILED' if failed else ''}", flush=True)

    # Phase B: baselines LOVO at L*(alpha), every knob, then the admissible knob.
    specs = [cv_spec(n, m, kn, lag_star[(n, a)], f)
             for n in names for a in ALPHAS for m in BASELINES for kn in SWEEPS[m]
             for f in range(len(DSS[n].folds))]
    R.update(run_units(specs, cache_dir, "B: baseline LOVO"))
    baseline_sel: Dict[str, Dict] = {}
    knob_sel: Dict[Tuple[str, float, str], float] = {}
    for n in names:
        ds = DSS[n]
        baseline_sel[n] = {}
        for a in ALPHAS:
            Lst = lag_star[(n, a)]
            baseline_sel[n][str(a)] = {}
            for m in BASELINES:
                rows = []
                for kn in SWEEPS[m]:
                    k, nn, _pv = lovo_prop(ds, R, m, kn, Lst)
                    rows.append((k, nn))
                idx, no_adm = select_knob(rows, a)
                knob = SWEEPS[m][idx]
                knob_sel[(n, a, m)] = knob
                baseline_sel[n][str(a)][m] = {
                    "lag": int(Lst), "selected_knob": float(knob),
                    "no_admissible_knob": bool(no_adm),
                    "lovo": [{"knob": float(kn), "k": int(k), "n": int(nn),
                              "pfa": (k / nn) if nn else None,
                              "admissible": bool(nn and k <= a * nn + 1e-9)}
                             for kn, (k, nn) in zip(SWEEPS[m], rows)],
                }

    # Phase C: test evaluation (calibration on all training videos).
    specs = []
    for n in names:
        for a in ALPHAS:
            specs.append(test_spec(n, "SENTINEL-E", a, lag_star[(n, a)]))
            for m in BASELINES:
                specs.append(test_spec(n, m, knob_sel[(n, a, m)], lag_star[(n, a)]))
        for L in grid:
            specs.append(test_spec(n, "SENTINEL-E", HEADLINE, L))
    R.update(run_units(specs, cache_dir, "C: test"))
    test_out: Dict[str, Dict] = {}
    for n in names:
        ds = DSS[n]
        test_out[n] = {}
        for a in ALPHAS:
            Lst = lag_star[(n, a)]
            entry = {}
            r = R[_key(test_spec(n, "SENTINEL-E", a, Lst))]
            entry["SENTINEL-E"] = {"knob": a, "lag": int(Lst), **test_metrics(ds, r)}
            for m in BASELINES:
                kn = knob_sel[(n, a, m)]
                r = R[_key(test_spec(n, m, kn, Lst))]
                entry[m] = {"knob": float(kn), "lag": int(Lst), **test_metrics(ds, r)}
            test_out[n][str(a)] = entry
        lag_out = {}
        for L in grid:
            r = R[_key(test_spec(n, "SENTINEL-E", HEADLINE, L))]
            lag_out[str(L)] = {"lag": int(L), "is_lag_star": bool(L == lag_star[(n, HEADLINE)]),
                               **test_metrics(ds, r)}
        test_out[n]["lag_sensitivity_alpha_0.1"] = lag_out

    # Phase D: p-value diagnostics on the LOVO streams at L*(0.1) and at lag 1.
    specs = [diag_spec(n, L, f) for n in names
             for L in sorted({lag_star[(n, HEADLINE)], 1}) for f in range(len(DSS[n].folds))]
    R.update(run_units(specs, cache_dir, "D: p-value diagnostics"))
    diagnostics: Dict[str, Dict] = {}
    for n in names:
        ds = DSS[n]
        diagnostics[n] = {"5a": dependence_5a(ds),
                          "5b": dependence_5b(ds, R, lag_star[(n, HEADLINE)])}

    meta = {}
    for n in names:
        ds = DSS[n]
        meta[n] = {
            "label": LABELS[n], "family": ds.family, "fps": ds.fps, "grace_frames": ds.grace,
            "fold_scheme": "mod10" if ds.family == "mivia" else "lovo",
            "n_train_videos": len(ds.train), "n_folds": len(ds.folds),
            "n_test_videos": len(ds.test),
            "n_test_negative": sum(v.onset is None for v in ds.test),
            "n_test_positive": sum(v.onset is not None for v in ds.test),
            "n_test_episodes": sum(len(v.episodes) for v in ds.test),
            "mivia_episode_structure_ok": ds.mivia_episodes_ok if ds.family == "mivia" else None,
            "min_null_frames": ds.min_null_frames,
        }
    return {
        "meta": meta, "skipped": skipped, "datasets": names, "grid": list(grid),
        "selection": selection, "baselines": baseline_sel, "test": test_out,
        "diagnostics": diagnostics, "runtime_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _pct(p: Optional[float], nd: int = 3) -> str:
    return fmt(p, nd)


def _ci(ci, nd: int = 3) -> str:
    lo, hi = ci
    return f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"


def _kn(p: Dict) -> str:
    return f"{p['k']}/{p['n']}"


def _delay_cell(delay: Dict) -> str:
    return fmt(delay["median_s"], 1)


def write_outputs(payload: Dict) -> None:
    meta, names, test = payload["meta"], payload["datasets"], payload["test"]
    video_note = "; ".join(f"{LABELS[n]}: {meta[n]['n_test_videos']} test videos" for n in names)
    train_note = "; ".join(f"{LABELS[n]}: {meta[n]['n_train_videos']} training videos" for n in names)

    # tabr3_selection
    rows = []
    for n in names:
        for a in ALPHAS:
            s = payload["selection"][n][str(a)]
            rows.append([LABELS[n], f"{a:g}", s["lag"], f"{s['lovo_k']}/{s['lovo_n']}",
                         "yes" if s["selection_failed"] else "no"])
    write_latex_table(
        rows, ["dataset", r"$\alpha$", r"$L^\ast$", "LOVO FA (k/n)", "no admissible lag"],
        caption=("SENTINEL-E betting lag $L^\\ast(\\alpha)$ selected on training videos only: the smallest grid lag "
                 "whose leave-one-video-out per-video false-alarm proportion is at most $\\alpha$. "
                 f"Denominators are training videos ({train_note})."),
        label="tab:r3_selection", name="tabr3_selection", align="llrrc",
        notes=("Grid $\\{1,2,3,5,7,10,15,20,30,50\\}$. A video is a false alarm if it raises at least one alarm "
               "when monitored alone. Mivia-IWDD uses 10 folds by video index, so its denominator is the 200 "
               "training videos, each held out once."),
    )

    # tabr3_test (alpha = 0.1 headline)
    rows = []
    for n in names:
        for m in ["SENTINEL-E"] + BASELINES:
            e = test[n][str(HEADLINE)][m]
            rows.append([LABELS[n], m, _fmt_knob(e["knob"]),
                         f"{_pct(e['null_primary']['pfa'], 3)} {_ci(e['null_primary']['ci'])} ({_kn(e['null_primary'])})",
                         f"{fmt(e['fa_per_hour'], 2)} {_ci(e['fa_per_hour_ci'], 2)}",
                         f"{_pct(e['episodes']['pfa'], 2)} {_ci(e['episodes']['ci'], 2)} ({_kn(e['episodes'])})",
                         _delay_cell(e["delay"])])
    write_latex_table(
        rows, ["dataset", "method", "knob", "test per-video FA [95\\% CI] (k/n)", "FA / hour [95\\% CI]",
               "episode detection [95\\% CI] (k/n)", "median delay (s)"],
        caption=(f"Test results at nominal $\\alpha=0.1$, each test video monitored alone, calibration on all "
                 f"training videos, and each method at $L^\\ast(0.1)$ from Table~\\ref{{tab:r3_selection}}. "
                 f"Test videos: {video_note}."),
        label="tab:r3_test", name="tabr3_test", align="llrrrrr",
        notes=("Per-video FA is over no-anomaly test videos and the pre-onset segments (at least 2\\,s) of anomalous "
               "test videos for UCSD Ped2 and CUHK Avenue, and over the negative test videos for Mivia-IWDD (pre-onset "
               "segments of positive Mivia videos are reported in the supporting data). Episodes are the labelled "
               "runs; for Mivia-IWDD, one per positive video. Median delay is over detected episodes. "
               "Baseline knobs are chosen on training videos (first admissible knob in sweep order; if no knob is admissible, the most conservative one)."),
    )

    # tabr3_test_alphas
    rows = []
    for n in names:
        for a in ALPHAS:
            e = test[n][str(a)]["SENTINEL-E"]
            rows.append([LABELS[n], f"{a:g}", payload["selection"][n][str(a)]["lag"],
                         f"{_pct(e['null_primary']['pfa'], 3)} {_ci(e['null_primary']['ci'])} ({_kn(e['null_primary'])})",
                         f"{fmt(e['fa_per_hour'], 2)} {_ci(e['fa_per_hour_ci'], 2)}",
                         f"{_pct(e['episodes']['pfa'], 2)} {_ci(e['episodes']['ci'], 2)} ({_kn(e['episodes'])})",
                         _delay_cell(e["delay"])])
    write_latex_table(
        rows, ["dataset", r"$\alpha$", "lag $L^\\ast$", "test per-video FA [95\\% CI] (k/n)", "FA / hour [95\\% CI]",
               "episode detection [95\\% CI] (k/n)", "median delay (s)"],
        caption=("SENTINEL-E alone at each nominal $\\alpha$, at $L^\\ast(\\alpha)$, on the test videos "
                 f"(each monitored alone; calibration on all training videos). Test videos: {video_note}."),
        label="tab:r3_test_alphas", name="tabr3_test_alphas", align="llrrrrrr",
        notes="Null-trial definitions as in the main-text test table.",
    )

    # tabr3_mivia_types
    rows = []
    for n in names:
        if not n.startswith("mivia"):
            continue
        for m in ["SENTINEL-E"] + BASELINES:
            e = test[n][str(HEADLINE)][m]
            for t in ("static dumping", "dynamic dumping"):
                d = e["by_event_type"].get(t)
                if d is None:
                    rows.append([LABELS[n], m, _fmt_knob(e["knob"]), t, "0/0", "--", "--", "--"])
                    continue
                rows.append([LABELS[n], m, _fmt_knob(e["knob"]), t, d["n_episodes"],
                             f"{_pct(d['detection']['pfa'], 2)} {_ci(d['detection']['ci'], 2)} ({_kn(d['detection'])})",
                             fmt(d["median_s"], 1), "--"])
            rows.append([LABELS[n], m, _fmt_knob(e["knob"]), "normal videos", e["null_no_anomaly"]["n"],
                         "--", "--",
                         f"{_pct(e['null_no_anomaly']['pfa'], 3)} {_ci(e['null_no_anomaly']['ci'])} ({_kn(e['null_no_anomaly'])})"])
    if rows:
        write_latex_table(
            rows, ["dataset", "method", "knob", "event type", "n", "detection [95\\% CI] (k/n)",
                   "median delay (s)", "normal-video FA [95\\% CI] (k/n)"],
            caption=("Mivia-IWDD test results at $\\alpha=0.1$ split by event type (static versus dynamic dumping), "
                     "and per-video false alarms on the negative test videos "
                     "(100 test videos: 50 negative, 25 static, 25 dynamic), for both backbones."),
            label="tab:r3_mivia_types", name="tabr3_mivia_types", align="llrlrrrr",
            notes=("Detection by event type counts each positive video's episode separately. Normal-video false "
                   "alarms are the primary null trials of the headline table."),
        )

    # tabr3_lags
    rows = []
    for n in names:
        lg = test[n]["lag_sensitivity_alpha_0.1"]
        for key in sorted(lg, key=lambda x: int(x)):
            e = lg[key]
            rows.append([LABELS[n], e["lag"], "$\\ast$" if e["is_lag_star"] else "",
                         f"{_pct(e['null_primary']['pfa'], 3)} {_ci(e['null_primary']['ci'])} ({_kn(e['null_primary'])})",
                         f"{fmt(e['fa_per_hour'], 2)} {_ci(e['fa_per_hour_ci'], 2)}",
                         f"{_pct(e['episodes']['pfa'], 2)} {_ci(e['episodes']['ci'], 2)} ({_kn(e['episodes'])})",
                         _delay_cell(e["delay"])])
    write_latex_table(
        rows, ["dataset", "lag $L$", r"$L^\ast$", "test per-video FA [95\\% CI] (k/n)", "FA / hour [95\\% CI]",
               "episode detection [95\\% CI] (k/n)", "median delay (s)"],
        caption=(f"Lag sensitivity of SENTINEL-E at $\\alpha=0.1$ on the test videos, each monitored alone. "
                 f"Test videos: {video_note}. This table is a sensitivity analysis: the headline lag is "
                 "the one selected on training videos."),
        label="tab:r3_lags", name="tabr3_lags", align="llcrrrr",
        notes="Calibration on all training videos at each lag. The starred row is $L^\\ast(0.1)$.",
    )

    # tabr3_dependence
    rows = []
    for n in names:
        d5a = payload["diagnostics"][n]["5a"]
        w = d5a["within_video_acf"]
        rows.append([LABELS[n], d5a["pooled_acf_length"] if d5a["pooled_acf_length"] is not None
                     else f">{d5a['pooled_acf_max_lag_checked']}",
                     fmt(w["1"]["mean"], 2), fmt(w["5"]["mean"], 2), fmt(w["10"]["mean"], 2),
                     fmt(w["25"]["mean"], 2)])
    rows_b = []
    for n in names:
        d5b = payload["diagnostics"][n]["5b"]
        for key in ("L_star", "lag_1"):
            e = d5b[key]
            grid = f"$L^\\ast(0.1)={e['lag']}$" if key == "L_star" else "$L=1$"
            rows_b.append([LABELS[n], grid, fmt(e["lag1_corr"], 3), fmt(e["sup_dev_pos"], 3),
                           fmt(e["cond_mean_pt_given_prev_le_0p1"], 3),
                           f"{e['cond_count']} / {e['n_p_values']}"])
    rows_all = [[r"\multicolumn{6}{l}{\textit{(a) Score dependence on the raw training scores}}"]]
    rows_all += rows
    rows_all += [[r"\multicolumn{6}{l}{\textit{(b) Conformal p-values on the LOVO / fold null streams}}"],
                 ["dataset", "grid", "lag-1 corr.", r"$\sup_u(F_n(u)-u)$", r"$E[p_t\mid p_{t-1}\le 0.1]$",
                  r"pairs with $p_{t-1}\le0.1$ / p-values"]]
    rows_all += rows_b
    write_latex_table(
        rows_all, ["dataset", "pooled ACF length", "ACF lag 1", "ACF lag 5", "ACF lag 10", "ACF lag 25"],
        caption=("Temporal dependence of the scores and the conformal p-values on real normal footage. "
                 "(a) pooled autocorrelation length of the concatenated training scores (first lag with "
                 "$|\\mathrm{ACF}|<0.05$), and within-video autocorrelation averaged over training videos. "
                 "(b) per-video lag-1 correlation of consecutive p-values on the thinned grid, the maximum "
                 "positive deviation of the pooled empirical p-value CDF, and the mean of $p_t$ given "
                 f"$p_{{t-1}}\\le0.1$. Training videos: {train_note}."),
        label="tab:r3_dependence", name="tabr3_dependence", align="lrrrrr",
        notes="Under valid, independent p-values the conditional mean is 0.5 and the CDF deviation is 0.",
    )

    # Figure
    plt = setup_matplotlib()
    cols = min(3, len(names))
    nrows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.7 * cols, 3.5 * nrows), squeeze=False)
    from matplotlib.ticker import NullFormatter
    # Spare grid cells (e.g. the ninth cell with seven datasets) hold the legend, so no
    # data area is covered and no panel is left blank.
    spare = list(axes.ravel()[len(names):])
    for ax, n in zip(axes.ravel(), names):
        for m in ["SENTINEL-E"] + BASELINES:
            e = test[n][str(HEADLINE)][m]
            x = max(e["fa_per_hour"] or 0.0, FA_FLOOR)
            y = e["episodes"]["pfa"]
            lo, hi = e["episodes"]["ci"]
            if y is None or not np.isfinite(y):
                continue
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor="#999999", lw=0.8, capsize=2)
            ax.plot([x], [y], label=m, markeredgecolor="white", markeredgewidth=0.4, **style_for(m))
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", labelsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"test FA per hour (zero drawn at {FA_FLOOR:g})")
        ax.set_ylabel("episode detection rate")
        nt = meta[n]["n_test_videos"]
        ax.set_title(f"{LABELS[n]} ($n$={nt} videos)")
    if spare:
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        spare[0].axis("off")
        spare[0].legend(handles, labels, loc="center", fontsize=8, frameon=False, title="method")
        for ax in spare[1:]:
            ax.set_visible(False)
    else:
        axes.ravel()[0].legend(loc="lower right", fontsize=6.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_r3_per_video.{ext}")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> Optional[Dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny lag grid; writes nothing under results/ (cache goes to the scratchpad)")
    a = ap.parse_args(argv)
    names = [x.strip() for x in a.datasets.split(",") if x.strip()]
    grid = SMOKE_LAG_GRID if a.smoke else LAG_GRID
    cache = SMOKE_DIR if a.smoke else CACHE_DIR
    print(f"Experiment R3 (protocol v3): per-video monitoring, real video  "
          f"(smoke={a.smoke}, lags={grid}, cache={cache})", flush=True)
    payload = run(names, grid, cache)
    payload["protocol"] = {
        "alphas": ALPHAS, "headline_alpha": HEADLINE, "lag_grid": list(grid),
        "min_null_s": MIN_NULL_S, "fa_floor_per_hour": FA_FLOOR, "workers": WORKERS,
        "calibration_test": "all training videos", "calibration_lovo": "all other training videos",
        "sweeps": SWEEPS,
    }
    if a.smoke:
        print(json.dumps({n: payload["selection"][n] for n in payload["datasets"]}, indent=1)[:2000])
        return payload
    save_json(payload, "exp_r3_per_video")
    write_outputs(payload)
    print(f" wrote results/exp_r3_per_video.json, tables tabr3_*, fig_r3_per_video; "
          f"runtime {payload['runtime_s']:.0f}s; skipped={payload['skipped']}", flush=True)
    return payload


if __name__ == "__main__":
    main()

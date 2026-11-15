"""Experiment R5 --- block betting versus thinning, real video, per-video protocol.

Motivation. On real footage the thinned conformal p-values of exp_r3 stay
serially dependent (lag-one correlation 0.76--0.99 at L*), so the conditional
super-uniformity that the e-process theorem assumes is violated. The block
variant (:mod:`sentinel_e.block_betting`) bets once per block of ``B``
consecutive per-frame p-values, merges each block by a combiner that is valid
under arbitrary within-block dependence, and optionally skips ``G`` frames
between blocks. The conditional super-uniformity across blocks remains an
assumption; this experiment measures how well it holds.

Protocol (pre-registered in this docstring before any test video is looked at).

* Datasets: the exp_r3 datasets (:data:`exp_r3_per_video.DATASETS`). The
  ``mivia_supervised`` dataset is added only if its ``backbone_auc.json`` exists.
* Calibration: pooled conformal calibration on the training videos, thinned at
  the lag ``L*(alpha)`` that exp_r3 selected on training videos only. Block mode
  therefore changes the betting only, not the calibration, so the comparison
  with thinned SENTINEL-E is like for like. The thinned method is run inside
  the same code, and its alarms are checked against exp_r3's test results.
* Configurations: the thinned path (``thin``) and 16 block configurations,
  ``B`` in {5, 10, 20, 30}, ``G`` in {0, B}, combiner in {bonferroni, mean2}.
* Diagnostics (part a): on the leave-one-video-out (LOVO) or 10-fold null
  streams of the training videos, at ``L*(0.1)``, for every configuration:
  lag-one correlation of consecutive block p-values, ``sup_u (F_n(u) - u)`` on
  the pooled empirical CDF, ``E[p_b | p_{b-1} <= 0.1]``, and the pair counts.
  Under valid conditionally super-uniform p-values the conditional mean is 0.5.
* Selection (part b), training videos only, separately for each dataset and
  alpha. A configuration is admissible if its LOVO per-video false-alarm
  proportion is at most alpha. Among admissible configurations, the selected one
  has the smallest ``B + G``; ties go to the training conditional mean closest
  to 0.5, and then to the combiner order (bonferroni first). If none is
  admissible, the configuration with the lowest LOVO proportion is taken and
  flagged.
  Deviation from the brief: the brief asked for the training positives' LOVO
  detection on Mivia-IWDD. The Mivia training videos carry no anomaly labels
  (``train_meta.json`` has no label field, and the dataset README says they are
  all labelled normal), so the rule above applies to every dataset. The number
  of labelled training positives is recorded in the output.
* Test (part c): calibration on all training videos, each test video monitored
  alone, at alpha in {0.05, 0.1, 0.2}. Per-video null trials, false alarms per
  hour, episode detection and median delay follow exp_r3's ``test_metrics``. The
  intervals are exact Clopper--Pearson (proportions) and exact Poisson (rates).

Every unit (one LOVO fold or one test configuration) is written atomically to
``results/cache/exp_r5/`` as it completes, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.common import RESULTS, TABLES, fmt, parallel_map, save_json, write_latex_table
from experiments.real_data.exp_r3_per_video import (
    ALPHAS, DATASETS, DS, HEADLINE, LABELS, available, build_calibration, cp_interval,
    get_ds, poisson_rate_interval, prop, test_metrics,
)
from sentinel_e.block_betting import BLOCK_COMBINERS, block_pvalues
from sentinel_e.datasets.real_vad import DATA_ROOT
from sentinel_e.pipeline import SentinelE
from sentinel_e.temporal import thin_indices

BLOCK_SIZES = (5, 10, 20, 30)
CACHE_DIR = RESULTS / "cache" / "exp_r5"
SMOKE_DIR = Path("/tmp/sentinel_e_scratch/exp_r5_smoke")
EXP_R3_JSON = "exp_r3_per_video"
#: Lag at which the part (a) diagnostics are taken; the same as exp_r3's headline diagnostic.
DIAG_ALPHA = HEADLINE
#: Thresholds of the conditional-mean diagnostic.
COND_THRESHOLD = 0.1
WORKERS = 1


def _cfg_table() -> Dict[str, Optional[Tuple[int, int, str]]]:
    cfgs: Dict[str, Optional[Tuple[int, int, str]]] = {"thin": None}
    for B in BLOCK_SIZES:
        for G in (0, B):
            for c in BLOCK_COMBINERS:
                cfgs[f"B{B}_G{G}_{c}"] = (B, G, c)
    return cfgs


CFGS = _cfg_table()
CFG_NAMES: List[str] = list(CFGS)
BLOCK_NAMES: List[str] = [n for n in CFG_NAMES if n != "thin"]


def _cfg_label(name: str) -> str:
    if name == "thin":
        return "thinned (exp_r3 path)"
    B, G, c = CFGS[name]
    return f"block $B$={B}, $G$={G}, {c}"


def _set_cfg(model: SentinelE, name: str) -> None:
    cfg = CFGS[name]
    if cfg is None:
        model.block_size, model.block_gap, model.block_combiner = None, 0, "bonferroni"
    else:
        B, G, c = cfg
        model.block_size, model.block_gap, model.block_combiner = B, G, c


def _alarms(model: SentinelE, scores: np.ndarray, name: str) -> List[int]:
    _set_cfg(model, name)
    res = model.run(np.asarray(scores, float))
    return [int(i) for i in np.flatnonzero(res.alarms)]


def _lag_star_from_exp_r3() -> Dict[Tuple[str, float], int]:
    payload = json.loads((RESULTS / f"{EXP_R3_JSON}.json").read_text())["payload"]
    out = {}
    for n, by_a in payload["selection"].items():
        for a, s in by_a.items():
            out[(n, float(a))] = int(s["lag"])
    return out


def _exp_r3_test(name: str) -> Dict:
    payload = json.loads((RESULTS / f"{EXP_R3_JSON}.json").read_text())["payload"]
    return payload["test"][name]


# --------------------------------------------------------------------------- #
# Units (cached on disk)
# --------------------------------------------------------------------------- #
def _key(spec: Dict) -> str:
    if spec["kind"] == "lovo":
        return f"lovo__{spec['ds']}__a{spec['alpha']:g}__L{spec['lag']}__f{spec['fold']}"
    return f"test__{spec['ds']}__a{spec['alpha']:g}__L{spec['lag']}__{spec['cfg']}"


def _compute_lovo(ds: DS, spec: Dict) -> Dict:
    """Frame-level p-values and alarms of every configuration on the held-out training videos of one fold."""
    held = ds.folds[spec["fold"]]
    cal = build_calibration(ds.train, held)
    model = SentinelE(alpha=spec["alpha"], calibration="pooled", lag=spec["lag"], restart=True).fit(cal)
    P: Dict[str, List[float]] = {}
    A: Dict[str, Dict[str, List[int]]] = {n: {} for n in CFG_NAMES}
    for j in held:
        sc = ds.train[j]
        p, _active = model.p_values(sc)
        P[str(j)] = [float(x) for x in p]
        for n in CFG_NAMES:
            A[n][str(j)] = _alarms(model, sc, n)
    return {"p": P, "alarms": A}


def _compute_test(ds: DS, spec: Dict) -> Dict:
    """One configuration on the test videos, calibration on all training videos."""
    cal = np.concatenate(ds.train)
    model = SentinelE(alpha=spec["alpha"], calibration="pooled", lag=spec["lag"], restart=True).fit(cal)
    whole: Dict[str, List[int]] = {}
    prefix: Dict[str, List[int]] = {}
    for i, v in enumerate(ds.test):
        whole[str(i)] = _alarms(model, v.scores, spec["cfg"])
        if v.onset is not None and v.onset >= ds.min_null_frames:
            prefix[str(i)] = _alarms(model, v.scores[: v.onset], spec["cfg"])
    return {"whole": whole, "prefix": prefix}


def _unit(args) -> Dict:
    cache_dir, key, spec = args
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            if doc.get("spec") == spec:
                return doc["result"]
        except (json.JSONDecodeError, KeyError):
            pass  # a torn file is recomputed and overwritten
    ds = get_ds(spec["ds"])
    res = _compute_lovo(ds, spec) if spec["kind"] == "lovo" else _compute_test(ds, spec)
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
# Statistics
# --------------------------------------------------------------------------- #
def _block_or_thin(p: np.ndarray, lag: int, name: str) -> np.ndarray:
    cfg = CFGS[name]
    if cfg is None:
        return p[thin_indices(p.size, lag)]
    B, G, c = cfg
    return block_pvalues(p, B, G, c)[0]


def diag_stats(videos: Sequence[np.ndarray]) -> Dict:
    """Lag-one correlation, pooled CDF deviation and the conditional mean for one set of p-value sequences."""
    xs = [v[:-1] for v in videos if v.size > 1]
    ys = [v[1:] for v in videos if v.size > 1]
    if not xs:
        return {"lag1_corr": None, "sup_dev_pos": None, "cond_mean": None, "cond_count": 0, "n_p": 0, "n_pairs": 0}
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    allp = np.sort(np.concatenate(videos))
    m = allp.size
    F = np.searchsorted(allp, allp, side="right") / m
    mask = x <= COND_THRESHOLD
    corr = float(np.corrcoef(x, y)[0, 1]) if (x.std() > 0 and y.std() > 0) else None
    return {
        "lag1_corr": corr,
        "sup_dev_pos": float(max(0.0, np.max(F - allp))),
        "cond_mean": float(np.mean(y[mask])) if mask.any() else None,
        "cond_count": int(mask.sum()),
        "n_p": int(m),
        "n_pairs": int(x.size),
    }


def _ordered_videos(P: Dict[str, List[float]], n_train: int) -> List[np.ndarray]:
    return [np.asarray(P[str(j)], float) for j in range(n_train)]


def select_config(lovo_k: Dict[str, Tuple[int, int]], cond: Dict[str, Optional[float]], alpha: float) -> Dict:
    """Training-only selection rule of the module docstring."""
    adm = {}
    for n in BLOCK_NAMES:
        k, nn = lovo_k[n]
        adm[n] = bool(nn and k <= alpha * nn + 1e-9)
    admissible = [n for n in BLOCK_NAMES if adm[n]]
    order = {c: i for i, c in enumerate(BLOCK_COMBINERS)}

    def key(n):
        B, G, c = CFGS[n]
        e = cond[n]
        dist = abs(e - 0.5) if e is not None else float("inf")
        return (B + G, dist, order[c])

    if admissible:
        chosen = min(admissible, key=key)
        flagged = False
    else:
        chosen = min(BLOCK_NAMES, key=lambda n: (lovo_k[n][0] / max(lovo_k[n][1], 1), key(n)))
        flagged = True
    return {"selected": chosen, "no_admissible_config": flagged, "admissible": adm}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _positives_in_train(name: str) -> int:
    """Training videos whose metadata marks an anomaly (0 when the metadata carries no labels)."""
    meta = json.loads((DATA_ROOT / name / "train_meta.json").read_text())
    return sum(1 for m in meta if m.get("label") or m.get("event_type") not in (None, "normal"))


def run(datasets: Sequence[str], cache_dir: Path, workers: int) -> Dict:
    t0 = time.time()
    names, skipped = available(datasets)
    DSS = {n: get_ds(n) for n in names}
    lag_star = _lag_star_from_exp_r3()

    # Part (a)/(b): LOVO / fold units at L*(alpha), every configuration.
    specs = [{"kind": "lovo", "ds": n, "alpha": a, "lag": lag_star[(n, a)], "fold": f}
             for n in names for a in ALPHAS for f in range(len(DSS[n].folds))]
    R = run_units(specs, cache_dir, "A: LOVO / fold block and thinned streams", workers)

    selection: Dict[str, Dict] = {}
    diagnostics: Dict[str, Dict] = {}
    for n in names:
        ds = DSS[n]
        selection[n] = {}
        for a in ALPHAS:
            L = lag_star[(n, a)]
            P: Dict[str, List[float]] = {}
            A: Dict[str, Dict[str, List[int]]] = {c: {} for c in CFG_NAMES}
            for f in range(len(ds.folds)):
                unit = R[_key({"kind": "lovo", "ds": n, "alpha": a, "lag": L, "fold": f})]
                P.update(unit["p"])
                for c in CFG_NAMES:
                    A[c].update(unit["alarms"][c])
            k_of = {}
            for c in CFG_NAMES:
                k = sum(1 for j in range(len(ds.train)) if len(A[c][str(j)]) > 0)
                k_of[c] = (k, len(ds.train))
            videos = _ordered_videos(P, len(ds.train))
            per_cfg = {}
            for c in CFG_NAMES:
                per_cfg[c] = diag_stats([_block_or_thin(v, L, c) for v in videos])
            cond = {c: per_cfg[c]["cond_mean"] for c in BLOCK_NAMES}
            sel = select_config({c: k_of[c] for c in BLOCK_NAMES}, cond, a)
            selection[n][str(a)] = {
                "lag_cal": int(L), "selected": sel["selected"],
                "no_admissible_config": sel["no_admissible_config"],
                "lovo": {c: {"k": int(k_of[c][0]), "n": int(k_of[c][1]),
                             "pfa": k_of[c][0] / k_of[c][1], "admissible": bool(sel["admissible"].get(c, False)) if c != "thin" else None,
                             "cond_mean_train": per_cfg[c]["cond_mean"]} for c in CFG_NAMES},
                "thin_lovo": {"k": int(k_of["thin"][0]), "n": int(k_of["thin"][1])},
            }
            print(f"  {n:18s} alpha={a:<5g} L_cal={L:<3d} selected={sel['selected']}"
                  f"{' (no admissible: flagged)' if sel['no_admissible_config'] else ''}", flush=True)
            if a == DIAG_ALPHA:
                diagnostics[n] = {"lag_cal": int(L), "per_config": per_cfg}
        diagnostics.setdefault(n, {})["training_labelled_positives"] = _positives_in_train(n)

    # Part (c): test units for the selected block configuration and the thinned method.
    specs = []
    for n in names:
        for a in ALPHAS:
            L = lag_star[(n, a)]
            for c in (selection[n][str(a)]["selected"], "thin"):
                specs.append({"kind": "test", "ds": n, "alpha": a, "lag": L, "cfg": c})
    R = run_units(specs, cache_dir, "B: test", workers)
    test_out: Dict[str, Dict] = {}
    checks: Dict[str, Dict] = {}
    for n in names:
        ds = DSS[n]
        test_out[n], checks[n] = {}, {}
        exp3 = _exp_r3_test(n)
        for a in ALPHAS:
            L = lag_star[(n, a)]
            entry = {}
            for c in (selection[n][str(a)]["selected"], "thin"):
                r = R[_key({"kind": "test", "ds": n, "alpha": a, "lag": L, "cfg": c})]
                entry[c] = test_metrics(ds, r)
            test_out[n][str(a)] = entry
            ref = exp3[str(a)]["SENTINEL-E"]
            mine = entry["thin"]
            checks[n][str(a)] = bool(
                mine["null_primary"]["k"] == ref["null_primary"]["k"]
                and mine["episodes"]["k"] == ref["episodes"]["k"]
                and mine["false_alarms"] == ref["false_alarms"]
            )
            if not checks[n][str(a)]:
                print(f"  WARNING {n} alpha={a}: thinned path differs from exp_r3 SENTINEL-E", flush=True)

    meta = {}
    for n in names:
        ds = DSS[n]
        meta[n] = {
            "label": LABELS[n], "family": ds.family, "fps": ds.fps,
            "n_train_videos": len(ds.train), "n_folds": len(ds.folds), "fold_scheme": "mod10" if ds.family == "mivia" else "lovo",
            "n_test_videos": len(ds.test), "n_test_negative": sum(v.onset is None for v in ds.test),
            "n_test_positive": sum(v.onset is not None for v in ds.test),
            "min_null_frames": ds.min_null_frames,
        }
    return {
        "meta": meta, "skipped": skipped, "datasets": names,
        "block_sizes": list(BLOCK_SIZES), "configurations": {k: (list(v) if v else None) for k, v in CFGS.items()},
        "lag_star_exp_r3": {f"{n}|{a:g}": lag_star[(n, a)] for n in names for a in ALPHAS},
        "selection": selection, "diagnostics": diagnostics, "test": test_out,
        "exp_r3_thin_crosscheck": checks, "runtime_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _ci(ci, nd: int = 3) -> str:
    lo, hi = ci
    return f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"


def write_outputs(payload: Dict) -> None:
    names = payload["datasets"]
    train_note = "; ".join(f"{LABELS[n]}: {payload['meta'][n]['n_train_videos']}" for n in names)
    video_note = "; ".join(f"{LABELS[n]}: {payload['meta'][n]['n_test_videos']}" for n in names)

    rows = []
    for n in names:
        d = payload["diagnostics"][n]
        for c in CFG_NAMES:
            s = d["per_config"][c]
            rows.append([LABELS[n], _cfg_label(c), fmt(s["lag1_corr"], 3), fmt(s["sup_dev_pos"], 3),
                         fmt(s["cond_mean"], 3), f"{s['cond_count']} / {s['n_p']}"])
    write_latex_table(
        rows, ["dataset", "betting grid", "lag-1 corr.", r"$\sup_u(F_n(u)-u)$",
               r"$E[p_b\mid p_{b-1}\le 0.1]$", r"pairs with $p_{b-1}\le0.1$ / p-values"],
        caption=("Conditional super-uniformity diagnostic of block p-values on the leave-one-video-out "
                 "(Ped2, Avenue) and 10-fold (Mivia-IWDD) null streams of the training videos, at "
                 f"$\\alpha={DIAG_ALPHA:g}$, calibration thinned at $L^\\ast({DIAG_ALPHA:g})$ as in exp_r3. "
                 "Each row is one betting grid: the thinned path, or block p-values for block size $B$, "
                 "gap $G$ and combiner. Under valid conditionally super-uniform p-values the conditional "
                 f"mean is 0.5. Training videos: {train_note}."),
        label="tab:r5_block_diag", name="tabr5_block_diag", align="llrrrr",
        notes=("Bonferroni: $\\min(1,B\\min_i p_i)$; mean2: $\\min(1,2\\bar p)$. Both are marginally "
               "super-uniform under arbitrary within-block dependence. The pooled CDF deviation is the maximum "
               "positive excess of the empirical CDF over the identity. Counts are the pairs whose "
               "earlier p-value is at most 0.1, over all p-values."),
    )

    rows = []
    for n in names:
        for a in ALPHAS:
            s = payload["selection"][n][str(a)]
            t = payload["test"][n][str(a)]
            for c in (s["selected"], "thin"):
                e = t[c]
                lv = s["lovo"][c]
                rows.append([LABELS[n], f"{a:g}", s["lag_cal"], _cfg_label(c),
                             f"{lv['k']}/{lv['n']}",
                             f"{fmt(e['null_primary']['pfa'], 3)} {_ci(e['null_primary']['ci'])} ({e['null_primary']['k']}/{e['null_primary']['n']})",
                             f"{fmt(e['fa_per_hour'], 2)} {_ci(e['fa_per_hour_ci'], 2)}",
                             f"{fmt(e['episodes']['pfa'], 2)} {_ci(e['episodes']['ci'], 2)} ({e['episodes']['k']}/{e['episodes']['n']})",
                             fmt(e["delay"]["median_s"], 1)])
    write_latex_table(
        rows, ["dataset", r"$\alpha$", r"$L^\ast$", "betting", "LOVO FA (k/n)",
               "test per-video FA [95\\% CI] (k/n)", "FA / hour [95\\% CI]",
               "episode detection [95\\% CI] (k/n)", "median delay (s)"],
        caption=("Block betting against thinned SENTINEL-E on the test videos, each monitored alone, at each "
                 "nominal $\\alpha$. The block configuration is selected on training videos only "
                 "(smallest $B+G$ with LOVO per-video false-alarm proportion at most $\\alpha$, ties broken by "
                 "the training conditional mean closest to 0.5). Calibration on all training videos, thinned at "
                 f"$L^\\ast(\\alpha)$ from exp_r3. Test videos: {video_note}."),
        label="tab:r5_block_test", name="tabr5_block_test", align="llrlrrrrr",
        notes=("Null-trial definitions follow exp_r3 (Ped2, Avenue: no-anomaly videos and pre-onset segments; "
               "Mivia-IWDD: negative videos). Thinned rows reproduce exp_r3's SENTINEL-E test results; the check "
               "is recorded in the supporting JSON. Exact Clopper--Pearson and Poisson intervals."),
    )


def main(argv: Optional[Sequence[str]] = None) -> Optional[Dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--smoke", action="store_true", help="one dataset, scratch cache, nothing written under results/")
    a = ap.parse_args(argv)
    names = [x.strip() for x in a.datasets.split(",") if x.strip()]
    cache = SMOKE_DIR if a.smoke else CACHE_DIR
    print(f"Experiment R5: block betting vs thinning (smoke={a.smoke}, cache={cache}, workers={a.workers})", flush=True)
    payload = run(names, cache, a.workers)
    payload["protocol"] = {
        "alphas": ALPHAS, "headline_alpha": HEADLINE, "diag_alpha": DIAG_ALPHA,
        "cond_threshold": COND_THRESHOLD, "calibration": "pooled, thinned at exp_r3 L*(alpha)",
        "selection": "smallest B+G with LOVO FA <= alpha; ties: |E-0.5|, then combiner order",
        "fold_scheme": "lovo (Ped2, Avenue), mod10 (Mivia-IWDD)",
    }
    if a.smoke:
        print(json.dumps({n: payload["selection"][n] for n in payload["datasets"]}, indent=1)[:3000])
        return payload
    save_json(payload, "exp_r5_block")
    write_outputs(payload)
    print(f" wrote results/exp_r5_block.json, tables tabr5_block_diag, tabr5_block_test; "
          f"runtime {payload['runtime_s']:.0f}s; skipped={payload['skipped']}", flush=True)
    return payload


if __name__ == "__main__":
    main()

"""Experiment R1 (protocol v2) --- false-alarm validity and lag selection, real video.

Replaces the v1 protocol, which had two flaws a reviewer identified:

* the betting-grid lag was chosen by sweeping lags on streams built from the
  *test* split and then evaluated on that same split (test-set leakage);
* 6,000-frame null streams were spliced from short runs sampled with
  replacement, so a 362-normal-frame test split was reused about 17 times.

The v2 protocol proceeds in four steps, all on real Farneback-flow scores:

a. **Lag selection on training data only.** Null streams are permutations of
   the official training videos in group ``sel`` (see
   :func:`sentinel_e.datasets.real_vad.train_video_groups`). The selected lag
   L* is the smallest grid lag whose null false-alarm rate is at most the
   headline level at that lag *and at every larger grid lag*.
b. **Null validity on held-out training videos.** Group ``null`` is used only
   here, at L*, for every alpha.
c. **Detection on the real labelled test split** at L*, every test video used
   exactly once per stream, real labels and onsets. Reference lags are reported
   alongside so a reader can see what a test-tuned lag would have shown.
d. **Lag quantities and conditional super-uniformity diagnostics** on the
   null group, at L* and at lag 1 for contrast.

No test frame or anomaly label is read in step (a).
"""

from __future__ import annotations

import argparse
import functools
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from experiments.common import (
    FIGURES, fmt, fmt_int, parallel_map, save_json, setup_matplotlib, write_latex_table,
)
from experiments.real_data.protocol_v2 import (
    FPS, GRACE_FRAMES, SEED, make_stream, rate, rng_for, run_alarms, score_labelled_stream,
    iid_rng, state, test_summary,
)
from sentinel_e.datasets.real_vad import causal_local_context, concat_whole_videos
from sentinel_e.pipeline import SentinelE
from sentinel_e.temporal import acf, estimate_decorrelation_lag, ljung_box, thin_indices

DATASETS = ("ped2", "avenue")
LABELS = {"ped2": "UCSD Ped2", "avenue": "CUHK Avenue"}
HEADLINE_ALPHA = 0.1
LAG_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
ALPHAS = [0.3, 0.2, 0.1, 0.05, 0.02]
REFERENCE_LAGS = [1, 3, 5, 7, 10]
REPS = 400
SEL_REPS = 200
TEST_REPS = 200
#: Kept for ``experiments/run_all.py``, which passes it as the second argument.
LAG_REPS = SEL_REPS
SEED = 20_260_101
# --------------------------------------------------------------------------- #
# Worker jobs (module-level so the process pool can pickle them)
# --------------------------------------------------------------------------- #
def _sel_job(args) -> bool:
    dataset, lag, rep = args
    s = state(dataset)
    scores, labels, eps = concat_whole_videos(s["groups"]["sel"], rng_for(dataset, "sel", rep))
    st = make_stream(scores, labels, eps, s["cal"])
    return bool(run_alarms("SENTINEL-E", HEADLINE_ALPHA, st, lag).size > 0)


def _null_job(args) -> bool:
    dataset, alpha, lag, rep = args
    s = state(dataset)
    scores, labels, eps = concat_whole_videos(s["groups"]["null"], rng_for(dataset, "null", rep))
    st = make_stream(scores, labels, eps, s["cal"])
    return bool(run_alarms("SENTINEL-E", alpha, st, lag).size > 0)


def _test_job(args) -> dict:
    dataset, alpha, lag, rep = args
    s = state(dataset)
    scores, labels, eps = concat_whole_videos(
        [v for v, _ in s["tv"]], rng_for(dataset, "test", rep), [l for _, l in s["tv"]],
    )
    st = make_stream(scores, labels, eps, s["cal"])
    idx = run_alarms("SENTINEL-E", alpha, st, lag)
    return score_labelled_stream(idx, labels, eps, GRACE_FRAMES, FPS)


def _pvalue_job(args) -> np.ndarray:
    """Conformal p-values on the thinned grid of one held-out null stream."""
    dataset, lag, rep = args
    s = state(dataset)
    scores, _, _ = concat_whole_videos(s["groups"]["null"], rng_for(dataset, "null", rep))
    model = SentinelE(alpha=HEADLINE_ALPHA, calibration="pooled", lag=lag, restart=True)
    model.fit(s["cal"], causal_local_context(s["cal"]).reshape(-1, 1))
    bet = thin_indices(scores.size, lag)
    p, active = model.p_values(scores[bet])
    return p[active]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def select_lag(pfa_by_lag: Sequence[float], lags: Sequence[int], level: float):
    """Smallest lag at which the PFA is <= ``level`` here and at every larger lag."""
    ok = [p <= level for p in pfa_by_lag]
    for i in range(len(lags)):
        if all(ok[i:]):
            return int(lags[i]), False
    return int(lags[-1]), True


def superuniformity(p_streams: Sequence[np.ndarray]) -> Dict:
    """Diagnostics of whether thinned p-values are (conditionally) i.i.d. uniform."""
    pooled = np.concatenate(p_streams)
    ks = stats.kstest(pooled, "uniform")
    # Super-uniformity: P(p <= u) <= u. The empirical CDF minus the identity
    # should therefore be at most ~0; a positive value means anti-conservative.
    xs = np.sort(pooled)
    max_excess = float(np.max(np.arange(1, xs.size + 1) / xs.size - xs))
    prev = np.concatenate([p[:-1] for p in p_streams])
    cur = np.concatenate([p[1:] for p in p_streams])
    corr = float(np.corrcoef(prev, cur)[0, 1]) if prev.size > 2 else float("nan")

    lb_p = []
    for p in p_streams:
        n_lags = min(10, p.size // 4)
        if n_lags >= 1:
            lb_p.append(ljung_box(p, n_lags=n_lags)[1])
    lb_p = np.asarray(lb_p)

    lo_m = prev <= 0.1
    hi_m = ~lo_m
    ks_lo = stats.kstest(cur[lo_m], "uniform")
    ks_hi = stats.kstest(cur[hi_m], "uniform")
    return {
        "n_pvalues": int(pooled.size),
        "n_streams": len(p_streams),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mean_p": float(pooled.mean()),
        "max_ecdf_minus_identity": max_excess,
        "lag1_corr": corr,
        "median_ljung_box_p": float(np.median(lb_p)) if lb_p.size else None,
        "frac_ljung_box_gt_0_05": float(np.mean(lb_p > 0.05)) if lb_p.size else None,
        "n_streams_ljung_box": int(lb_p.size),
        "cond_prev_le_0_1": {"n": int(lo_m.sum()), "ks_p": float(ks_lo.pvalue),
                             "ks_stat": float(ks_lo.statistic)},
        "cond_prev_gt_0_1": {"n": int(hi_m.sum()), "ks_p": float(ks_hi.pvalue),
                             "ks_stat": float(ks_hi.statistic)},
        "cond_mean_given_prev_le_0_1": float(cur[lo_m].mean()) if lo_m.any() else None,
    }


def iid_reference(dataset: str, lag: int, n_streams: int) -> Dict:
    """Same diagnostics on i.i.d. Gaussian scores, matched in calibration and stream size.

    Shows what the KS and correlation diagnostics look like when the
    conformal layer is the only thing in play. Uses the dataset's calibration
    and null-stream lengths and the same pooled calibrator.
    """
    s = state(dataset)
    n_cal = int(s["cal"].size)
    T = int(sum(v.size for v in s["groups"]["null"]))
    rng = iid_rng(dataset)
    cal = rng.normal(size=n_cal)
    model = SentinelE(alpha=HEADLINE_ALPHA, calibration="pooled", lag=lag, restart=True)
    model.fit(cal, None)
    bet_n = thin_indices(T, lag).size
    p_streams = []
    for _ in range(n_streams):
        p, _a = model.p_values(rng.normal(size=bet_n))
        p_streams.append(p)
    return {"lag": int(lag), **superuniformity(p_streams)}


def lag_quantities(cal: np.ndarray, selected_lag: int) -> Dict:
    kmax = int(min(500, cal.size // 3))
    r = acf(cal, n_lags=kmax)
    # |acf| < 0.05, matching sentinel_e.temporal.estimate_decorrelation_lag.
    below = np.flatnonzero(np.abs(r[1:]) < 0.05)
    acf_len = int(below[0] + 1) if below.size else f">{kmax}"
    return {
        "acf_decorrelation_length": acf_len,
        "acf_search_max": kmax,
        "estimated_lag": int(estimate_decorrelation_lag(cal)),
        "estimated_lag_max": 200,
        "selected_lag": int(selected_lag),
    }


# --------------------------------------------------------------------------- #
# Per-dataset protocol
# --------------------------------------------------------------------------- #
def run_dataset(dataset: str, reps: int, sel_reps: int, test_reps: int) -> Dict:
    s = state(dataset)
    corpus = s["corpus"]
    g = s["groups"]
    print(f" {dataset}: cal={sum(v.size for v in g['cal'])} frames "
          f"({len(g['cal'])} videos), sel={len(g['sel'])} videos, "
          f"null={len(g['null'])} videos, test={len(s['tv'])} videos", flush=True)

    # (a) lag selection on training data only --------------------------------
    sel_rows = []
    for lag in LAG_GRID:
        hits = parallel_map(_sel_job, [(dataset, lag, r) for r in range(sel_reps)],
                            desc=f"select {dataset} lag={lag}")
        k = int(sum(hits))
        row = {"lag": lag, **rate(k, sel_reps)}
        sel_rows.append(row)
        print(f"  select lag={lag:<3d} PFA={row['pfa']:.3f} "
              f"CI=[{row['ci'][0]:.3f},{row['ci'][1]:.3f}]", flush=True)
    lag_star, failed = select_lag([r["pfa"] for r in sel_rows], LAG_GRID, HEADLINE_ALPHA)
    print(f"  L* = {lag_star} (selection_failed={failed})", flush=True)

    # (b) null validity on held-out training videos ---------------------------
    null_horizon_s = float(sum(v.size for v in g["null"])) / FPS
    null_rows = []
    for a in ALPHAS:
        hits = parallel_map(_null_job, [(dataset, a, lag_star, r) for r in range(reps)],
                            desc=f"null {dataset} alpha={a}")
        row = {"alpha": a, **rate(int(sum(hits)), reps), "horizon_s": null_horizon_s}
        null_rows.append(row)
        print(f"  null alpha={a:<5} PFA={row['pfa']:.3f} "
              f"CI=[{row['ci'][0]:.3f},{row['ci'][1]:.3f}]", flush=True)

    # (c) detection on the real labelled test split --------------------------
    def _detect(lag: int, a: float) -> Dict:
        rows = parallel_map(_test_job, [(dataset, a, lag, r) for r in range(test_reps)],
                            desc=f"test {dataset} lag={lag} alpha={a}")
        return test_summary(rows)

    test_rows = []
    for a in ALPHAS:
        summ = _detect(lag_star, a)
        test_rows.append({"alpha": a, **summ})
        print(f"  test alpha={a:<5} det={summ['episode_detection_rate']:.3f} "
              f"FA/h={summ['false_alarms_per_hour']:.2f}", flush=True)
    ref_rows = []
    for lag in REFERENCE_LAGS:
        summ = _detect(lag, HEADLINE_ALPHA)
        ref_rows.append({"lag": lag, **summ})
        print(f"  reference lag={lag:<3d} det={summ['episode_detection_rate']:.3f} "
              f"FA/h={summ['false_alarms_per_hour']:.2f}", flush=True)

    # (d) lag quantities and super-uniformity diagnostics --------------------
    lq = lag_quantities(s["cal"], lag_star)
    diag = {}
    for name, lag in (("selected", lag_star), ("lag1", 1)):
        p_streams = parallel_map(_pvalue_job, [(dataset, lag, r) for r in range(reps)],
                                 desc=f"pvalues {dataset} lag={lag}")
        diag[name] = {"lag": lag, **superuniformity(p_streams)}
        print(f"  diag {name} (lag={lag}): KS p={diag[name]['ks_p']:.3g} "
              f"corr1={diag[name]['lag1_corr']:.3f} "
              f"medLB={diag[name]['median_ljung_box_p']:.3g}", flush=True)

    diag["iid_reference_selected"] = iid_reference(dataset, lag_star, reps)
    print(f"  iid reference at L*: KS p={diag['iid_reference_selected']['ks_p']:.3g} "
          f"mean p={diag['iid_reference_selected']['mean_p']:.3f}", flush=True)

    # Frame-reuse audit ------------------------------------------------------
    n_normal = int(np.sum(~corpus.test_labels))
    reuse = {
        "test_normal_frames": n_normal,
        "test_frames": int(corpus.test_labels.size),
        "test_episodes": len(corpus.episodes),
        "v1_null_stream_frames": 6000,
        "v1_reuse_factor": 6000.0 / n_normal if n_normal else float("nan"),
        # v2 concatenates each whole video once per stream (see
        # tests/test_protocol_v2.py), so no frame recurs within a stream.
        "v2_within_stream_reuse": 0,
    }

    return {
        "n_cal_frames": int(s["cal"].size),
        "n_cal_videos": len(g["cal"]),
        "n_sel_videos": len(g["sel"]),
        "n_null_videos": len(g["null"]),
        "n_test_videos": len(s["tv"]),
        "selection": {
            "level": HEADLINE_ALPHA,
            "rule": "smallest grid lag with PFA <= level at it and every larger grid lag",
            "rows": sel_rows,
            "selected_lag": lag_star,
            "selection_failed": failed,
        },
        "null_validity": null_rows,
        "test_detection": test_rows,
        "test_reference_lags": ref_rows,
        "lag_quantities": lq,
        "superuniformity": diag,
        "frame_reuse": reuse,
    }


# --------------------------------------------------------------------------- #
# Tables and figure
# --------------------------------------------------------------------------- #
def make_tables(payload: Dict) -> None:
    ds_items = payload["datasets"].items()

    rows = []
    for ds, d in ds_items:
        L = d["selection"]["selected_lag"]
        for r in d["selection"]["rows"]:
            mark = r"$\leftarrow L^\ast$" if r["lag"] == L else ""
            rows.append([LABELS.get(ds, ds), fmt_int(r["lag"]), fmt(r["pfa"], 3),
                         f"[{fmt(r['ci'][0], 3)}, {fmt(r['ci'][1], 3)}]", mark])
    write_latex_table(
        rows, ["dataset", "lag", "PFA", "95\\% CI", ""],
        caption=(
            "Lag selection on official training videos only. Each row is the "
            "fraction of permuted training-video streams (group \\emph{sel}) "
            f"with at least one alarm at $\\alpha={HEADLINE_ALPHA}$. The test "
            "split is not used. The selected lag is the smallest lag at which "
            "the rate is at most $\\alpha$ and stays so for every larger lag."
        ),
        label="tab:r1_lag_selection", name="tabr1_lag_selection", align="llrrl",
    )

    rows = []
    for ds, d in ds_items:
        det = {r["alpha"]: r for r in d["test_detection"]}
        for r in d["null_validity"]:
            t = det[r["alpha"]]
            rows.append([
                LABELS.get(ds, ds), fmt(r["alpha"], 2), fmt(r["pfa"], 3),
                f"[{fmt(r['ci'][0], 3)}, {fmt(r['ci'][1], 3)}]",
                fmt(r["horizon_s"], 1), fmt(t["episode_detection_rate"], 2),
                fmt(t["median_delay_s"], 1), fmt(t["false_alarms_per_hour"], 2),
            ])
    write_latex_table(
        rows,
        ["dataset", r"$\alpha$", "null PFA", "95\\% CI", "null horizon (s)",
         "test episode det.", "median delay (s)", "FA / hour"],
        caption=(
            "Validity and detection at the lag selected on training data. Null "
            "PFA is measured on held-out training videos (group \\emph{null}, "
            "never used for selection). Detection and false-alarm columns use "
            "the real labelled test split, every test video once per stream."
        ),
        label="tab:r1_validity", name="tabr1b_validity", align="llrrrrrr",
        notes=(
            r"The null horizon is the length in seconds of one held-out null "
            r"stream, so a PFA is a probability over that horizon only. "
            r"Delays are over detected episodes pooled across test streams."
        ),
    )

    rows = []
    for ds, d in ds_items:
        lq = d["lag_quantities"]
        rows.append([LABELS.get(ds, ds), fmt_int(lq["acf_decorrelation_length"]),
                     fmt_int(lq["estimated_lag"]), fmt_int(lq["selected_lag"])])
    write_latex_table(
        rows,
        ["dataset", "ACF decorrelation length", "estimated lag (cap 200)", "selected lag $L^\\ast$"],
        caption=(
            "Three lag quantities per dataset, all computed on the training "
            "calibration videos. The ACF length is the first lag where "
            "$|\\mathrm{acf}|<0.05$ (search up to $\\min(500,n/3)$). The "
            "estimated lag uses the library default, which caps at 200."
        ),
        label="tab:r1_lags", name="tabr1c_lags", align="lrrr",
    )

    rows = []
    for ds, d in ds_items:
        for key, tag in (("selected", ""), ("lag1", ""),
                         ("iid_reference_selected", " (i.i.d. ref.)")):
            u = d["superuniformity"][key]
            c1, c2 = u["cond_prev_le_0_1"], u["cond_prev_gt_0_1"]
            rows.append([
                LABELS.get(ds, ds) + tag, fmt_int(u["lag"]), fmt(u["ks_p"], 3),
                fmt(u["mean_p"], 3), fmt(u["max_ecdf_minus_identity"], 3),
                fmt(u["lag1_corr"], 3), fmt(u["median_ljung_box_p"], 3),
                fmt(u["frac_ljung_box_gt_0_05"], 2),
                f"{fmt(c1['ks_p'], 3)} ({fmt_int(c1['n'])})",
                f"{fmt(c2['ks_p'], 3)} ({fmt_int(c2['n'])})",
                fmt(u["cond_mean_given_prev_le_0_1"], 3),
            ])
    write_latex_table(
        rows,
        ["dataset", "lag", "KS $p$", "mean $p$", "$\\sup(\\hat F(u)-u)$",
         "lag-1 corr.", "median LB $p$", "frac. LB$>$0.05",
         "cond. KS $p$, $p_{t-1}\\le0.1$ (n)", "cond. KS $p$, $p_{t-1}>0.1$ (n)",
         "mean $p_t\\mid p_{t-1}\\le0.1$"],
        caption=(
            "Super-uniformity of conformal p-values on held-out training-video "
            "null streams, at the selected lag and at lag 1, with an i.i.d. "
            "Gaussian reference of matched size. Conformal p-values with the "
            "exact order-statistic correction are super-uniform, so they can "
            "sit above $U(0,1)$ even when valid; the sup statistic is the "
            "direct test of that. Conditional uniformity needs KS $p$ not "
            "small, lag-1 correlation near zero and conditional mean near 0.5."
        ),
        label="tab:r1_superuniform", name="tabr1d_superuniformity",
        align="lrrrrrrrrrr",
        notes=(
            r"KS $p$ tests pooled thinned p-values against $U(0,1)$; it rejects "
            r"any conservative (super-uniform) p-value distribution. The "
            r"Ljung--Box lag count is $\min(10, T/4)$ per stream. Conformal "
            r"p-values with a finite calibration set take a discrete set of "
            r"values, so KS $p$ values are approximate."
        ),
    )

    rows = []
    for ds, d in ds_items:
        for r in d["test_reference_lags"]:
            rows.append([LABELS.get(ds, ds), fmt_int(r["lag"]),
                         fmt(r["episode_detection_rate"], 2), fmt(r["median_delay_s"], 1),
                         fmt(r["false_alarms_per_hour"], 2)])
    write_latex_table(
        rows, ["dataset", "lag", "test episode det.", "median delay (s)", "FA / hour"],
        caption=(
            "Test-split detection at $\\alpha=0.1$ for reference lags, shown so "
            "a reader can see what a lag tuned on the test split would have "
            "reported. These lags are not used to select anything."
        ),
        label="tab:r1_reference_lags", name="tabr1e_reference_lags", align="llrrr",
    )


def make_figure(payload: Dict) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = {"ped2": "#0b3d91", "avenue": "#c1272d"}
    FLOOR = 2e-3

    ax = axes[0]
    for ds, d in payload["datasets"].items():
        rows = d["selection"]["rows"]
        lags = np.array([r["lag"] for r in rows], dtype=float)
        pfa = np.array([r["pfa"] for r in rows])
        y = np.maximum(pfa, FLOOR)
        lo_end = np.maximum(np.array([r["ci"][0] for r in rows]), FLOOR * 0.5)
        hi_end = np.maximum(np.array([r["ci"][1] for r in rows]), y)
        ax.errorbar(lags, y, yerr=[y - lo_end, hi_end - y],
                    color=colors.get(ds), marker="o", ms=3.5, lw=1.0, capsize=2,
                    label=LABELS.get(ds, ds))
        L = d["selection"]["selected_lag"]
        ax.axvline(L, color=colors.get(ds), lw=0.9, ls=":")
        ax.annotate(f"$L^\\ast$={L}", xy=(L, 0.02), color=colors.get(ds), fontsize=7,
                    rotation=90, va="bottom", ha="right")
    ax.axhline(HEADLINE_ALPHA, color="k", lw=0.9, ls="--", label=rf"$\alpha={HEADLINE_ALPHA}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.7, 1.2)
    ax.set_xlabel("betting-grid lag (frames)")
    ax.set_ylabel("PFA on training-selection streams")
    ax.set_title("(a) lag selection (training videos only)")
    ax.legend(loc="lower left", fontsize=6.5)

    ax = axes[1]
    for ds, d in payload["datasets"].items():
        rows = d["null_validity"]
        a = np.array([r["alpha"] for r in rows])
        pfa = np.array([r["pfa"] for r in rows])
        ci_hi = np.array([r["ci"][1] for r in rows])
        obs = pfa > 0
        if obs.any():
            ax.errorbar(a[obs], pfa[obs],
                        yerr=[pfa[obs] - np.array([r["ci"][0] for r in rows])[obs],
                              ci_hi[obs] - pfa[obs]],
                        color=colors.get(ds), marker="o", ms=3.5, lw=1.0, capsize=2,
                        label=LABELS.get(ds, ds))
        if (~obs).any():
            ax.plot(a[~obs], np.maximum(ci_hi[~obs], FLOOR), linestyle="none", marker="v",
                    markerfacecolor="white", markeredgecolor=colors.get(ds), ms=5)
    ax.plot(ALPHAS, ALPHAS, "k--", lw=1.0, label=r"nominal $\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.012, 0.4)
    ax.set_ylim(FLOOR * 0.7, 1.2)
    ax.set_xlabel(r"target level $\alpha$")
    ax.set_ylabel("null PFA on held-out training videos")
    ax.set_title("(b) validity on held-out videos at L*")
    ax.legend(loc="lower right", fontsize=6.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_r1_validity.{ext}")
    plt.close(fig)


def main(reps: int = REPS, sel_reps: int = SEL_REPS, test_reps: int = TEST_REPS,
         datasets: Sequence[str] = DATASETS):
    print("Experiment R1 (v2): lag selection on training videos, validity, real-test detection")
    payload_ds = {}
    for ds in datasets:
        print(f"dataset {ds}", flush=True)
        payload_ds[ds] = run_dataset(ds, reps, sel_reps, test_reps)
    payload = {
        "protocol": "v2",
        "headline_alpha": HEADLINE_ALPHA,
        "lag_grid": LAG_GRID,
        "alphas": ALPHAS,
        "reference_lags": REFERENCE_LAGS,
        "reps": reps,
        "sel_reps": sel_reps,
        "test_reps": test_reps,
        "grace_frames": GRACE_FRAMES,
        "fps": FPS,
        "seed": SEED,
        "datasets": payload_ds,
    }
    save_json(payload, "exp_r1_validity")
    make_figure(payload)
    make_tables(payload)
    print(" wrote results/exp_r1_validity.json, fig_r1_validity, tabr1* tables")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--sel-reps", type=int, default=SEL_REPS, dest="sel_reps")
    ap.add_argument("--test-reps", type=int, default=TEST_REPS, dest="test_reps")
    a = ap.parse_args()
    main(reps=a.reps, sel_reps=a.sel_reps, test_reps=a.test_reps,
         datasets=tuple(x.strip() for x in a.datasets.split(",") if x.strip()))

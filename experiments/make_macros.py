"""Emit LaTeX macros for every number quoted in the manuscript prose.

The paper must never contain a hand-typed result.  This script reads the JSON
payloads written by the experiments and defines one macro per quoted quantity,
so a re-run of the protocol updates the text automatically and a stale number is
impossible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.common import TABLES, load_json

FPS = 25.0


def _fmt(x, nd=2):
    if x is None:
        return "--"
    v = float(x)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{nd}f}"


def _pct(x, nd=1):
    return "--" if x is None or not np.isfinite(float(x)) else f"{100 * float(x):.{nd}f}"


def _int(x):
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{int(round(float(x))):,}".replace(",", "{,}")


# The illustrative calibration size and budget quoted in Section 4.1.  They are
# named here so the DKW/Beta comparison in the prose cannot drift away from the
# numbers the library actually produces -- the previous hand-typed pair was
# computed at two different settings and disagreed with both.
CAL_N = 2_000
CAL_DELTA = 1e-3


def _derived() -> Dict[str, str]:
    """Constants computed from the library itself, not from an experiment run."""
    from sentinel_e.conformal import beta_calibration_levels, dkw_inflation
    from sentinel_e.episodic import EpisodePrior

    m: Dict[str, str] = {}
    m["CalN"] = f"{CAL_N:,}".replace(",", "{,}")
    m["CalDelta"] = "10^{-3}"
    raw = 1.0 / (CAL_N + 1)
    dkw = raw + dkw_inflation(CAL_N, CAL_DELTA)
    beta0 = float(beta_calibration_levels(CAL_N, CAL_DELTA)[0])
    m["CalRawPmin"] = f"1/{CAL_N + 1}"
    m["DkwPmin"] = _fmt(dkw, 4)
    m["BetaPmin"] = _fmt(beta0, 4)
    m["BetaOverDkwSpeedup"] = _fmt(dkw / beta0, 1)

    prior = EpisodePrior()
    k, e, pi, _ = prior.grid()
    m["EpiGridPoints"] = str(int(k.size))
    m["EpiGridKappa"] = str(len(prior.kappa))
    m["EpiGridEta"] = str(len(prior.eta))
    m["EpiGridPi"] = str(len(prior.pi))
    m["EpiEtaMin"] = f"{min(prior.eta):g}"

    # Simulator configuration (defaults of sentinel_e.streams.StreamConfig).
    from sentinel_e.streams import StreamConfig
    sc = StreamConfig()
    m["SimCalN"] = _int(sc.n_calibration)
    m["SimDiurnalFrames"] = _int(sc.diurnal_period)
    m["SimDiurnalMin"] = _fmt(sc.diurnal_period / FPS / 60.0, 1)
    m["NetPropProb"] = _fmt(sc.propagation_prob, 2)
    m["NetPropLagSec"] = _fmt(sc.propagation_lag / FPS, 0)
    return m


def _at_alpha(rows, alpha):
    """The sweep row whose knob is the given nominal level, or None."""
    return next((r for r in rows if abs(r["knob"] - alpha) < 1e-12), None)


# Real-video macros from the obsolete exp_r1 / exp_r2 protocol. Any of these
# that cannot be computed from the current payloads is defined as \todo{obsolete}.
_OBSOLETE_REAL = [
    "RealReps", "RealLagReps",
    "RealPedTwoLag", "RealAvenueLag",
    "RealPedTwoPfaHeadline", "RealAvenuePfaHeadline",
    "RealPedTwoMissHeadline", "RealAvenueMissHeadline",
    "RealPedTwoCiHiHeadline", "RealAvenueCiHiHeadline",
    "RealPedTwoWorstAlpha", "RealAvenueWorstAlpha",
    "RealPedTwoWorstPfa", "RealAvenueWorstPfa",
    "RealDelayReps",
    "RealDelayPedTwoSec", "RealDelayPedTwoPfa", "RealDelayPedTwoMiss",
    "RealDelayPedTwoNBaselinesPass", "RealDelayPedTwoNBaselines",
    "RealDelayAvenueSec", "RealDelayAvenuePfa", "RealDelayAvenueMiss",
    "RealDelayAvenueNBaselinesPass", "RealDelayAvenueNBaselines",
]

# Real-data protocol v3 (experiments/real_data/exp_r3_per_video.py). One tag per
# dataset-backbone pair; macros are \RThree<tag><Quantity>.
_RTHREE_TAGS = {
    "ped2": "PedTwoFlow", "ped2_patchcore": "PedTwoPC",
    "avenue": "AvenueFlow", "avenue_patchcore": "AvenuePC",
    "mivia_flow": "MiviaFlow", "mivia_patchcore": "MiviaPC",
}


def _collect_real_v3(m: Dict[str, str]) -> None:
    """Macros for the per-video protocol at the headline alpha, plus dependence diagnostics."""
    try:
        r3 = load_json("exp_r3_per_video")
    except FileNotFoundError:
        return
    head = str(r3["protocol"]["headline_alpha"])
    repo = Path(TABLES).parents[1]
    for ds, tag in _RTHREE_TAGS.items():
        try:
            auc = json.loads((repo / "data" / "real_vad" / ds / "backbone_auc.json").read_text())["auc"]
            m[f"RThree{tag}Auc"] = _fmt(auc, 3)
        except (OSError, KeyError, ValueError):
            pass
    dep_corr, dep_sup, dep_cond, dep_cnt, dep_corr1, dep_sup1 = [], [], [], [], [], []
    wacf1, wacf25 = [], []
    for ds, tag in _RTHREE_TAGS.items():
        try:
            sel = r3["selection"][ds][head]
            t = r3["test"][ds][head]["SENTINEL-E"]
            m[f"RThree{tag}Lag"] = str(sel["lag"])
            m[f"RThree{tag}LovoK"] = str(sel["lovo_k"])
            m[f"RThree{tag}LovoN"] = str(sel["lovo_n"])
            m[f"RThree{tag}NVideos"] = str(t["n_videos"])
            nl = t["null_primary"]
            m[f"RThree{tag}FAk"] = str(nl["k"])
            m[f"RThree{tag}FAn"] = str(nl["n"])
            m[f"RThree{tag}FAp"] = _fmt(nl["pfa"], 3)
            m[f"RThree{tag}FALo"] = _fmt(nl["ci"][0], 3)
            m[f"RThree{tag}FAHi"] = _fmt(nl["ci"][1], 3)
            m[f"RThree{tag}FAhrP"] = _fmt(t["fa_per_hour"], 2)
            m[f"RThree{tag}FAhrLo"] = _fmt(t["fa_per_hour_ci"][0], 2)
            m[f"RThree{tag}FAhrHi"] = _fmt(t["fa_per_hour_ci"][1], 2)
            ep = t["episodes"]
            m[f"RThree{tag}Detk"] = str(ep["k"])
            m[f"RThree{tag}Detn"] = str(ep["n"])
            m[f"RThree{tag}DetP"] = _fmt(ep["pfa"], 2)
            m[f"RThree{tag}DetLo"] = _fmt(ep["ci"][0], 2)
            m[f"RThree{tag}DetHi"] = _fmt(ep["ci"][1], 2)
            m[f"RThree{tag}DelayMed"] = _fmt(t["delay"]["median_s"], 1)
            for key, prefix in (("static dumping", "Static"), ("dynamic dumping", "Dyn")):
                bt = t["by_event_type"].get(key)
                if bt is None:
                    continue
                d = bt["detection"]
                m[f"RThree{tag}{prefix}Detk"] = str(d["k"])
                m[f"RThree{tag}{prefix}Detn"] = str(d["n"])
                m[f"RThree{tag}{prefix}DetLo"] = _fmt(d["ci"][0], 2)
                m[f"RThree{tag}{prefix}DetHi"] = _fmt(d["ci"][1], 2)
            diag = r3["diagnostics"][ds]
            L = diag["5b"]["L_star"]
            one = diag["5b"]["lag_1"]
            m[f"RThree{tag}DepCorr"] = _fmt(L["lag1_corr"], 3)
            m[f"RThree{tag}DepSup"] = _fmt(L["sup_dev_pos"], 3)
            m[f"RThree{tag}DepCond"] = _fmt(L["cond_mean_pt_given_prev_le_0p1"], 3)
            m[f"RThree{tag}DepCondCount"] = str(L["cond_count"])
            m[f"RThree{tag}DepCorrOne"] = _fmt(one["lag1_corr"], 3)
            m[f"RThree{tag}DepSupOne"] = _fmt(one["sup_dev_pos"], 3)
            w = diag["5a"]["within_video_acf"]
            m[f"RThree{tag}WithinLagOne"] = _fmt(w["1"]["mean"], 2)
            m[f"RThree{tag}WithinLagTwentyFive"] = _fmt(w["25"]["mean"], 2)
            dep_corr.append(L["lag1_corr"]); dep_sup.append(L["sup_dev_pos"])
            dep_cond.append(L["cond_mean_pt_given_prev_le_0p1"]); dep_cnt.append(L["cond_count"])
            dep_corr1.append(one["lag1_corr"]); dep_sup1.append(one["sup_dev_pos"])
            wacf1.append(w["1"]["mean"]); wacf25.append(w["25"]["mean"])
        except (KeyError, TypeError, ValueError):
            continue
    # Baseline configurations (6 baselines x datasets) with no admissible knob on training videos.
    n_base = n_noknob = 0
    for ds in _RTHREE_TAGS:
        try:
            base = r3["baselines"][ds][head]
        except KeyError:
            continue
        for method, v in base.items():
            n_base += 1
            n_noknob += int(bool(v.get("no_admissible_knob")))
    m["RThreeBaselineConfigs"] = str(n_base)
    m["RThreeBaselineNoKnob"] = str(n_noknob)
    if dep_corr:
        m["RThreeDepCorrMin"] = _fmt(min(dep_corr), 3)
        m["RThreeDepCorrMax"] = _fmt(max(dep_corr), 3)
        m["RThreeDepSupMax"] = _fmt(max(dep_sup), 3)
        m["RThreeDepCondMin"] = _fmt(min(dep_cond), 3)
        m["RThreeDepCondMax"] = _fmt(max(dep_cond), 3)
        m["RThreeDepCountMin"] = str(min(dep_cnt))
        m["RThreeDepCountMax"] = str(max(dep_cnt))
        m["RThreeDepCorrOneMax"] = _fmt(max(dep_corr1), 3)
        m["RThreeDepSupOneMax"] = _fmt(max(dep_sup1), 3)
        m["RThreeWithinLagOneMin"] = _fmt(min(wacf1), 2)
        m["RThreeWithinLagOneMax"] = _fmt(max(wacf1), 2)
        m["RThreeWithinLagTwentyFiveMin"] = _fmt(min(wacf25), 2)
        m["RThreeWithinLagTwentyFiveMax"] = _fmt(max(wacf25), 2)


def collect() -> Dict[str, str]:
    m: Dict[str, str] = {}
    m.update(_derived())

    # ---- Experiment 1 ---------------------------------------------------- #
    try:
        e1 = load_json("exp1_validity")
        m["ExpOneReps"] = str(e1["reps"])
        m["ExpOneHorizonReps"] = str(e1.get("horizon_reps", e1["reps"]))
        a = e1["alpha_sweep"]
        at01 = next(r for r in a if abs(r["alpha"] - 0.01) < 1e-12)
        m["ValidityPfaSentinel"] = _fmt(at01["sentinel_pfa"], 3)
        m["ValidityPfaPvalue"] = _fmt(at01["pvalue_pfa"], 3)
        # nd=1: at 0.998 the abstract's headline claim, rounding to 0 decimals
        # would silently print "100%", overstating an already stark number.
        m["ValidityPfaPvaluePct"] = _pct(at01["pvalue_pfa"], 1)
        m["ValidityPfaOracleLoose"] = _fmt(a[0]["oracle_pfa"], 3)
        m["ValidityAlphaLoose"] = _fmt(a[0]["alpha"], 2)
        m["ValidityMaxSentinelPfa"] = _fmt(max(r["sentinel_pfa"] for r in a), 3)
        m["ValidityPvalueArl"] = _int(at01["pvalue_arl0"])
        m["ValidityPvalueArlMin"] = _fmt(at01["pvalue_arl0"] / (25.0 * 60.0), 1)
        m["ValidityTightestAlpha"] = _fmt(min(r["alpha"] for r in a), 3)
        tightest = min(a, key=lambda r: r["alpha"])
        m["ValidityTightestPfa"] = _fmt(tightest["sentinel_pfa"], 3)
        m["ValidityTightestCiHi"] = _fmt(tightest["sentinel_ci"][1], 3)
        m["ValidityWorstCiHi"] = _fmt(max(r["sentinel_ci"][1] for r in a), 3)
        h = e1["horizon_sweep"]
        last = h[-1]
        m["HorizonMaxHours"] = _fmt(last["hours"], 1)
        m["HorizonMaxFrames"] = _int(last["T"])
        m["HorizonSentinelPfa"] = _fmt(last["sentinel_pfa"], 3)
        m["HorizonPvaluePfa"] = _fmt(last["pvalue_pfa"], 3)
        m["HorizonPvaluePfaPct"] = _pct(last["pvalue_pfa"], 0)
        m["HorizonFixedPfa"] = _fmt(last["fixed_pfa"], 3)
        m["HorizonFixedPfaPct"] = _pct(last["fixed_pfa"], 0)
        first = h[0]
        m["HorizonMinHours"] = _fmt(first["hours"], 1)
        m["HorizonFixedPfaShort"] = _fmt(first["fixed_pfa"], 3)
    except FileNotFoundError:
        pass

    # ---- Experiment 2 ---------------------------------------------------- #
    try:
        e2 = load_json("exp2_delay_far")
        res = e2["results"]
        m["ExpTwoReps"] = str(e2["reps"])

        def best_at(method, cap=0.05):
            pts = [r for r in res[method]
                   if r["add"] is not None and np.isfinite(r["add"])
                   and r["pfa"] <= cap and r["miss_rate"] < 0.9]
            return min(pts, key=lambda r: r["add_censored"]) if pts else None

        # Headline SENTINEL-E row = the common nominal level used in tab3
        # (both exact e-processes are shown at alpha = 0.01 there).
        se = _at_alpha(res["SENTINEL-E"], 0.01)
        if se:
            m["DelaySentinelSec"] = _fmt(se["add_censored"] / FPS, 1)
            m["DelaySentinelCondSec"] = _fmt(se["add"] / FPS, 1)
            m["DelaySentinelPfa"] = _fmt(se["pfa"], 3)
            m["DelaySentinelMiss"] = _fmt(se["miss_rate"], 3)
            m["DelaySentinelCi"] = _fmt(1.96 * (se["add_censored_se"] or 0) / FPS, 1)
        # Equal-nominal-level comparison of the two exact e-processes.
        for a, tag in ((0.5, "Half"), (0.2, "Fifth"), (0.01, "Hundredth")):
            for name, vt in (("SENTINEL-E", "SE"), ("Changepoint mixture", "Cp")):
                r = _at_alpha(res[name], a)
                if r is None:
                    continue
                m[f"Eq{vt}{tag}DelaySec"] = _fmt(r["add_censored"] / FPS, 1)
                m[f"Eq{vt}{tag}CiSec"] = _fmt(
                    1.96 * (r["add_censored_se"] or 0) / FPS, 1)
                m[f"Eq{vt}{tag}Miss"] = _fmt(r["miss_rate"], 3)
                m[f"Eq{vt}{tag}Pfa"] = _fmt(r["pfa"], 3)
        admissible, inadmissible = [], []
        for name in res:
            if name == "SENTINEL-E":
                continue
            b = best_at(name)
            (admissible if b else inadmissible).append(name)
            if b:
                key = name.replace("-", "").replace(" ", "").replace("$", "")
                key = "".join(c for c in key if c.isalpha())
                m[f"Delay{key}Sec"] = _fmt(b["add_censored"] / FPS, 1)
                m[f"Delay{key}Pfa"] = _fmt(b["pfa"], 3)
                if se and se["add_censored"] > 0:
                    m[f"Speedup{key}"] = _fmt(
                        b["add_censored"] / se["add_censored"], 1)
        m["DelayNumInadmissible"] = str(len(inadmissible))
        m["DelayInadmissibleList"] = ", ".join(inadmissible) if inadmissible else "none"
    except FileNotFoundError:
        pass

    # ---- Experiment 3 ---------------------------------------------------- #
    try:
        e3 = load_json("exp3_shift")
        m["ExpThreeReps"] = str(e3["reps"])
        act = e3["activity"]
        keys = sorted(act, key=float)
        worst = keys[-1]
        m["ShiftMaxRatio"] = _fmt(float(worst), 1)
        for v, tag in (("pooled", "Pooled"), ("Mondrian", "Mondrian"),
                       ("residual", "Residual"), ("residual + LR wts", "ResidualLR")):
            r = next(x for x in act[worst] if x["variant"] == v)
            m[f"Shift{tag}Pfa"] = _fmt(r["pfa"], 3)
            m[f"Shift{tag}Delay"] = _fmt(
                None if r.get("add_censored") is None else r["add_censored"] / FPS, 1)
            m[f"Shift{tag}Miss"] = _fmt(r["miss_rate"], 2)
        # The level at which the unweighted residual construction is *clearly*
        # invalid, judged from the Wilson interval rather than the point
        # estimate: at 200 reps a point-estimate comparison cannot distinguish
        # e.g. PFA 0.02 [0.0078, 0.0503] from nominal, and reporting that as a
        # breaking point overstates what the design can show.
        alpha_nom = 0.01
        broke = None
        for k in keys:
            r = next(x for x in act[k] if x["variant"] == "residual")
            if r["pfa_ci"][0] > alpha_nom + 1e-12:
                broke = k
                break
        m["ShiftResidualBreaks"] = _fmt(float(broke), 1) if broke else "--"
        if broke:
            r = next(x for x in act[broke] if x["variant"] == "residual")
            w = next(x for x in act[broke] if x["variant"] == "residual + LR wts")
            m["ShiftBreakResidualPfa"] = _fmt(r["pfa"], 3)
            m["ShiftBreakResidualPfaLo"] = _fmt(r["pfa_ci"][0], 3)
            m["ShiftBreakResidualPfaHi"] = _fmt(r["pfa_ci"][1], 3)
            m["ShiftBreakResidualMiss"] = _fmt(r["miss_rate"], 2)
            m["ShiftBreakWeightedPfa"] = _fmt(w["pfa"], 3)
            m["ShiftBreakWeightedMiss"] = _fmt(w["miss_rate"], 2)
        # Largest shift at which the interval's upper end still clears nominal,
        # i.e. the construction is *clearly* safe, not merely not-yet-broken.
        ok = [k for k in keys
              if next(x for x in act[k] if x["variant"] == "residual")["pfa_ci"][1]
              <= alpha_nom + 1e-12]
        m["ShiftResidualSafeUpTo"] = _fmt(float(max(ok, key=float)), 1) if ok else "--"
        # Shifts where neither interval end resolves against nominal: the rate
        # may or may not be inflated and the design cannot tell at this reps count.
        unresolved = [k for k in keys if k not in ok and k != broke]
        m["ShiftResidualUnresolvedRatios"] = ", ".join(
            f"$\\times${_fmt(float(k), 1)}" for k in sorted(unresolved, key=float)
        ) if unresolved else "none"
        cov = e3["coverage"]
        for reg, tag in (("representative", "Rep"), ("full_cycle_benign", "NoRain"),
                         ("daytime_only", "Daytime")):
            r = next(x for x in cov[reg] if x["variant"] == "residual")
            m[f"Cover{tag}Pfa"] = _fmt(r["pfa"], 3)
            m[f"Cover{tag}Miss"] = _fmt(r["miss_rate"], 2)
            m[f"Cover{tag}Abstain"] = _pct(r["abstention"], 0)
    except FileNotFoundError:
        pass

    # ---- Experiment 4 ---------------------------------------------------- #
    try:
        e4 = load_json("exp4_network")
        rows = {r["controller"]: r for r in e4["rows"]}
        m["NetCameras"] = str(e4["n_cameras"])
        m["NetFleets"] = str(e4["n_eval"])
        m["NetFdrLevel"] = _fmt(e4["fdr_level"], 2)
        m["NetCreditWindowSec"] = _fmt(e4.get("credit_window", 0) / FPS, 0)
        for k, tag in (("none", "NoGraph"), ("heuristic", "Heuristic"),
                       ("gnn", "Gnn"), ("gnn_windowed", "GnnWindowed")):
            r = rows.get(k)
            if not r:
                continue
            m[f"Net{tag}DelaySec"] = _fmt(r["add"] / FPS, 1)
            m[f"Net{tag}CensDelaySec"] = _fmt(
                None if r.get("add_censored") is None else r["add_censored"] / FPS, 1)
            m[f"Net{tag}Miss"] = _fmt(r["miss_rate"], 2)
            m[f"Net{tag}Power"] = _fmt(r["power"], 3)
            m[f"Net{tag}Fdr"] = _fmt(r["fdr"], 3)
            m[f"Net{tag}NullPfa"] = _fmt(r["null_fleet_alarm_rate"], 3)
    except FileNotFoundError:
        pass

    # ---- Experiment 5 ---------------------------------------------------- #
    try:
        e5 = load_json("exp5_compute")
        bb = e5["backbone"].get("seconds_per_frame")
        e2e = e5["end_to_end"]
        m["CostBackboneMs"] = _fmt(bb * 1e3, 1) if bb else "--"
        m["CostBackboneParams"] = _fmt(e5["backbone"]["parameters"] / 1e6, 1)
        m["CostSentinelUs"] = _fmt(e2e["microseconds_per_frame"], 2)
        m["CostSentinelPct"] = (
            f"{100 * e2e['microseconds_per_frame'] * 1e-6 / bb:.4f}" if bb else "--"
        )
        m["CostRealtimeFactor"] = _int(e2e["realtime_factor"])
        m["CostLag"] = str(e2e["decorrelation_lag"])
        m["CostFitSec"] = _fmt(e2e["calibration_fit_seconds"], 2)
        det = e5["detector"]
        r32 = next(d for d in det if d["n_grid"] == 32)
        m["CostStepUs"] = _fmt(r32["sec_per_step_streaming"] * 1e6, 1)
        if e5.get("fleet"):
            f32 = min(e5["fleet"], key=lambda d: abs(d["n_cameras"] - 32))
            m["CostGnnPerCameraUs"] = _fmt(f32["sec_per_camera_step"] * 1e6, 1)
    except FileNotFoundError:
        pass

    # ---- Experiment 6 ---------------------------------------------------- #
    try:
        e6 = load_json("exp6_ablation")
        rows = {r["name"]: r for r in e6["rows"]}
        m["AblationAlpha"] = _fmt(e6["alpha"], 3)
        pairs = [
            ("SENTINEL-E (full)", "AblFull"),
            ("changepoint mixture ($\\eta = 0$)", "AblChangepoint"),
            ("enable dilation ($\\pi$ grid)", "AblDilation"),
            ("bet on every frame", "AblEveryFrame"),
            ("raw calibration (not thinned)", "AblRawCal"),
            ("pooled conformal", "AblPooled"),
            ("DKW inflation", "AblDkw"),
            ("no correction ($\\delta=0$)", "AblNoCc"),
            ("fixed start (changepoint core)", "AblPointPrior"),
            ("linear betting", "AblLinear"),
        ]
        for name, tag in pairs:
            r = rows.get(name)
            if not r:
                continue
            m[f"{tag}Pfa"] = _fmt(r["pfa"], 3)
            m[f"{tag}Delay"] = _fmt(
                None if r.get("add_censored") is None else r["add_censored"] / FPS, 1)
            m[f"{tag}Miss"] = _fmt(r["miss_rate"], 2)
        m["AblLag"] = str(rows["SENTINEL-E (full)"]["lag"])
        full, dkw = rows.get("SENTINEL-E (full)"), rows.get("DKW inflation")
        if (full and dkw and full.get("add_censored")
                and np.isfinite(full["add_censored"])):
            m["AblDkwSpeedup"] = _fmt(
                dkw["add_censored"] / full["add_censored"], 1)
        # Wilson-interval verdict, matching exp6_ablation._verdict: a point
        # estimate against alpha has no power to resolve most of these rows
        # at 200 replicates (see tab8_ablation's own caption).
        alpha = e6["alpha"]
        n_bad = sum(1 for r in e6["rows"] if r["pfa_ci"][0] > alpha + 1e-9)
        n_unresolved = sum(
            1 for r in e6["rows"]
            if not (r["pfa_ci"][1] <= alpha + 1e-9 or r["pfa_ci"][0] > alpha + 1e-9)
        )
        m["AblNumInvalid"] = str(n_bad)
        m["AblNumUnresolved"] = str(n_unresolved)
        m["AblNumVariants"] = str(len(e6["rows"]))
        m["AblReps"] = str(e6["reps"])
    except FileNotFoundError:
        pass

    # ---- Experiment 8 ---------------------------------------------------- #
    try:
        e8 = load_json("exp8_episodes")
        m["EpiReps"] = str(e8["reps"])
        counts = e8["episode_counts"]
        m["EpiMaxCount"] = str(counts[-1])
        # The count at which the episodic mixture helps most, by relative
        # reduction in censored delay -- reported rather than cherry-picked.
        best_k, best_gain = None, 0.0
        for k in counts:
            e = next(x for x in e8["by_episodes"][str(k)] if x["variant"] == "episodic")
            c = next(x for x in e8["by_episodes"][str(k)]
                     if x["variant"] == "changepoint")
            if e["add_censored"] and c["add_censored"]:
                g = 1.0 - e["add_censored"] / c["add_censored"]
                if g > best_gain:
                    best_k, best_gain = k, g
        if best_k is not None:
            m["EpiBestCount"] = str(best_k)
            m["EpiBestGainPct"] = _fmt(100 * best_gain, 0)
            e = next(x for x in e8["by_episodes"][str(best_k)]
                     if x["variant"] == "episodic")
            c = next(x for x in e8["by_episodes"][str(best_k)]
                     if x["variant"] == "changepoint")
            m["EpiBestEpiMiss"] = _fmt(e["miss_rate"], 3)
            m["EpiBestCpMiss"] = _fmt(c["miss_rate"], 3)
            m["EpiBestEpiDelay"] = _fmt(e["add_censored"] / FPS, 0)
            m["EpiBestCpDelay"] = _fmt(c["add_censored"] / FPS, 0)
            if e["miss_rate"] > 0:
                m["EpiBestMissRatio"] = _fmt(c["miss_rate"] / e["miss_rate"], 1)
        m["EpiOneCount"] = str(counts[0])
        mid_k = counts[len(counts) // 2] if len(counts) > 2 else counts[-1]
        m["EpiMidCount"] = str(mid_k)
        for k, tag in ((counts[0], "One"), (mid_k, "Mid"), (counts[-1], "Many")):
            for v, vt in (("episodic", "Epi"), ("changepoint", "Cp")):
                r = next(x for x in e8["by_episodes"][str(k)] if x["variant"] == v)
                m[f"Epi{tag}{vt}Miss"] = _fmt(r["miss_rate"], 3)
                m[f"Epi{tag}{vt}Delay"] = _fmt(
                    None if r["add_censored"] is None else r["add_censored"] / FPS, 0)
                m[f"Epi{tag}{vt}Pfa"] = _fmt(r["pfa"], 3)
        pis = e8["intermittency"]
        m["EpiMinPi"] = _fmt(pis[-1], 2)
        for v, vt in (("episodic", "Epi"), ("changepoint", "Cp")):
            r = next(x for x in e8["by_intermittency"][str(pis[-1])]
                     if x["variant"] == v)
            m[f"EpiHardPi{vt}Miss"] = _fmt(r["miss_rate"], 3)
            m[f"EpiHardPi{vt}Delay"] = _fmt(
                None if r["add_censored"] is None else r["add_censored"] / FPS, 1)
    except FileNotFoundError:
        pass

    # ---- Experiment 11 (single-stream worked example) --------------------- #
    try:
        e11 = load_json("exp11_worked_example")
        m["WorkedT"] = _fmt(e11["T"] / FPS, 0)
        m["WorkedNEpisodes"] = str(e11["n_episodes_in_window"])
        eps = e11["episodes"]
        m["WorkedEp"] = ", ".join(
            f"[{a/FPS:.0f}, {b/FPS:.0f}]\\,s" for a, b in eps
        )
        if e11["first_alarm"] is not None and e11["change_point"] is not None:
            delay = (e11["first_alarm"] - e11["change_point"]) / FPS
            m["WorkedDelaySec"] = _fmt(delay, 1)
            m["WorkedClimbSec"] = _fmt(delay, 0)
    except FileNotFoundError:
        pass

    # ---- Experiment 9 (paired episodic vs. changepoint) ------------------- #
    try:
        e9 = load_json("exp9_paired")
        m["ExpNineReps"] = str(e9["reps"])
        counts9 = e9.get("episode_counts") or sorted(
            (int(k) for k in e9["by_episodes"]))
        for k, tag in ((counts9[0], "One"), (counts9[1], "Two"),
                       (counts9[len(counts9) // 2], "Mid"), (counts9[-1], "Many")):
            d = e9["by_episodes"][str(k)]["paired"]["epi_vs_cp24"]["delay"]
            mi = e9["by_episodes"][str(k)]["paired"]["epi_vs_cp24"]["miss"]
            m[f"EpiPairedDiff{tag}"] = _fmt(d["mean_diff"] / FPS, 1)
            m[f"EpiPairedCI{tag}"] = (
                f"[{d['ci95'][0] / FPS:+.1f}, {d['ci95'][1] / FPS:+.1f}]"
            )
            m[f"EpiPairedPboot{tag}"] = _fmt(d["p_boot"], 3)
            # b01: changepoint-only detections; b10: episodic-only. The macro
            # always names whichever count favours the episodic mixture, i.e.
            # the count run in the paper's prose for that row (see sec:episodes).
            m[f"EpiPairedDiscordant{tag}"] = str(max(mi["b01"], mi["b10"]))
            m[f"EpiPairedPMcnemar{tag}"] = _fmt(mi["p_exact"], 3)
    except FileNotFoundError:
        pass

    # ---- Experiment 10 (per-episode mechanism check) ----------------------- #
    try:
        e10 = load_json("exp10_per_episode")
        m["ExpTenReps"] = str(e10["reps"])
        m["EpiEventLenSec"] = _fmt(e10.get("event_length", 800) / FPS, 0)
        counts10 = e10["episode_counts"]
        for k, tag in ((counts10[0], "One"), (counts10[len(counts10) // 2], "Mid"),
                       (counts10[-1], "Many")):
            row = e10["by_episodes"][str(k)]
            for v, vt in (("episodic", "Epi"), ("changepoint", "Cp")):
                r = row[v]
                m[f"PerEpi{tag}{vt}Miss"] = _fmt(r["episode_miss_rate"], 3)
                m[f"PerEpi{tag}{vt}StreamMiss"] = _fmt(r["stream_miss_rate"], 3)
    except FileNotFoundError:
        pass

    # ---- Real-video corpora (dataset facts, not an experiment run) -------- #
    try:
        from sentinel_e.datasets.real_vad import (
            clips_from_runs, load_real_vad,
        )
        from sentinel_e.temporal import acf

        labels = {"ped2": "PedTwo", "avenue": "Avenue"}
        for name, tag in labels.items():
            c = load_real_vad(name)
            m[f"Real{tag}Auc"] = _fmt(c.backbone_auc, 3)
            m[f"Real{tag}NTrainVideos"] = str(len(c.train_boundaries))
            m[f"Real{tag}NTestVideos"] = str(len(c.video_boundaries))
            m[f"Real{tag}NEpisodes"] = str(len(c.episodes))
            durs = [e - s for s, e in c.episodes]
            m[f"Real{tag}EpiMedian"] = _int(np.median(durs))
            m[f"Real{tag}EpiMin"] = _int(min(durs))
            m[f"Real{tag}EpiMax"] = _int(max(durs))
            r = acf(c.calibration_scores, n_lags=min(500, c.calibration_scores.size // 3))
            below = np.flatnonzero(r < 0.05)
            m[f"Real{tag}FullLag"] = _int(below[0] + 1) if below.size else f">{len(r)}"
            neg_runs, lab_arr, _ids = clips_from_runs(c, include_calibration_as_negative=False)
            neg = np.concatenate([neg_runs[i] for i in np.flatnonzero(lab_arr == 0)])
            p95 = np.percentile(c.calibration_scores, 95)
            m[f"Real{tag}ShiftFrac"] = _pct((neg > p95).mean(), 1)
    except FileNotFoundError:
        pass

    # ---- Experiment R1 (real-video validity) ------------------------------ #
    # The exp_r1 payload is from an obsolete protocol. Keys it lacks are
    # skipped; unfilled names fall back to \todo{obsolete} in collect().
    try:
        er1 = load_json("exp_r1_validity")
        try:
            m["RealReps"] = str(er1["reps"])
        except KeyError:
            pass
        try:
            m["RealLagReps"] = str(er1["lag_reps"])
        except KeyError:
            pass
        labels = {"ped2": "PedTwo", "avenue": "Avenue"}
        for name, tag in labels.items():
            try:
                m[f"Real{tag}Lag"] = str(er1["dataset_lag"][name])
                rows = er1["alpha_sweep"][name]
                at01 = next(r for r in rows if abs(r["alpha"] - 0.1) < 1e-12)
                m[f"Real{tag}PfaHeadline"] = _fmt(at01["pfa"], 3)
                m[f"Real{tag}MissHeadline"] = _fmt(at01["miss_rate"], 2)
                m[f"Real{tag}CiHiHeadline"] = _fmt(at01["pfa_ci"][1], 3)
                worst = max(rows, key=lambda r: r["pfa"])
                m[f"Real{tag}WorstAlpha"] = _fmt(worst["alpha"], 2)
                m[f"Real{tag}WorstPfa"] = _fmt(worst["pfa"], 3)
            except (KeyError, StopIteration, TypeError):
                pass
    except FileNotFoundError:
        pass

    # ---- Experiment R2 (real-video delay-vs-FAR against baselines) -------- #
    # The exp_r2 payload is from the same obsolete protocol as exp_r1.
    try:
        er2 = load_json("exp_r2_delayfar")
        try:
            m["RealDelayReps"] = str(er2["reps"])
        except KeyError:
            pass
        labels = {"ped2": "PedTwo", "avenue": "Avenue"}
        for name, tag in labels.items():
            try:
                res = er2["results"][name]
                se = min(
                    (r for r in res["SENTINEL-E"]
                     if r["add"] is not None and np.isfinite(r["add"])
                     and r["pfa"] <= 0.1 and r["miss_rate"] < 0.9),
                    key=lambda r: (r["pfa"], r["add_censored"]), default=None,
                )
                if se:
                    m[f"RealDelay{tag}Sec"] = _fmt(se["add_censored"] / FPS, 1)
                    m[f"RealDelay{tag}Pfa"] = _fmt(se["pfa"], 3)
                    m[f"RealDelay{tag}Miss"] = _fmt(se["miss_rate"], 3)
                n_pass = 0
                for method, rs in res.items():
                    if method == "SENTINEL-E":
                        continue
                    usable = [r for r in rs
                              if r["add_censored"] is not None
                              and np.isfinite(r["add_censored"]) and r["miss_rate"] < 0.9]
                    if usable and min(r["pfa"] for r in usable) <= 0.1:
                        n_pass += 1
                m[f"RealDelay{tag}NBaselinesPass"] = str(n_pass)
                m[f"RealDelay{tag}NBaselines"] = str(len(res) - 1)
            except (KeyError, TypeError, ValueError):
                pass
    except FileNotFoundError:
        pass

    # ---- Experiment R0 (qualitative real-frame example) ------------------- #
    try:
        er0 = load_json("exp_r0_qualitative")
        labels = {"ped2": "PedTwo", "avenue": "Avenue"}
        for name, tag in labels.items():
            m[f"QualExample{tag}Normal"] = _fmt(er0["scores"][name]["normal"], 3)
            m[f"QualExample{tag}Anomalous"] = _fmt(er0["scores"][name]["anomalous"], 3)
    except FileNotFoundError:
        pass

    # ---- Experiment 7 ---------------------------------------------------- #
    try:
        e7 = load_json("exp7_diagnostics")
        m["DiagStreams"] = str(e7["n_streams"])
        m["DiagLagMean"] = _fmt(e7["lag_mean"], 1)
        m["DiagLagMin"] = str(e7["lag_min"])
        m["DiagLagMax"] = str(e7["lag_max"])
        m["DiagAcfRaw"] = _fmt(e7["acf1_raw"], 3)
        m["DiagAcfThin"] = _fmt(e7["acf1_thinned"], 3)
        lb_raw = e7["ljung_box_raw_median"]
        m["DiagLjungRaw"] = "$<10^{-6}$" if lb_raw < 1e-6 else _fmt(lb_raw, 4)
        m["DiagLjungThin"] = _fmt(e7["ljung_box_thinned_median"], 3)
        m["DiagPmin"] = _fmt(e7["p_min"], 4)
    except FileNotFoundError:
        pass

    # ---- Experiment 12 (episodic vs changepoint across simulator corners) -- #
    try:
        e12 = load_json("exp12_corner_sweep")
        rows12 = {r["name"]: r for r in e12["rows"]}
        m["CornerReps"] = str(e12["reps"])
        m["CornerFloorMiss"] = _fmt(
            rows12["L=200, gap=1500, pi=0.2, 4 ep."]["epi"]["miss_rate"], 2)
        for name, tag in (
            ("L=1600, gap=1500, pi=0.2, 4 ep.", "CornerShortGap"),
            ("L=1600, gap=12000, pi=0.2, 4 ep.", "CornerLongGap"),
            ("L=1600, gap=1500, pi=0.7, 4 ep.", "CornerDense"),
            ("single episode, L=800, pi=0.35", "CornerSingle"),
            ("permanent change, pi=1.0 (L=70000)", "CornerPermanent"),
        ):
            r = rows12[name]
            d, ci = r["delay_diff_s"], r["delay_diff_ci95_s"]
            m[f"{tag}Diff"] = _fmt(d, 1)
            m[f"{tag}AbsDiff"] = _fmt(abs(d), 1)
            m[f"{tag}Ci"] = f"[{d_sign(ci[0])}, {d_sign(ci[1])}]"
            m[f"{tag}EpiOnly"] = str(r["miss_epi_only_detect"])
            m[f"{tag}CpOnly"] = str(r["miss_cp24_only_detect"])
            m[f"{tag}P"] = _fmt(r["miss_p_mcnemar"], 3)
    except (FileNotFoundError, KeyError, StopIteration):
        pass

    # ---- Experiment 13 (week-long horizon, real 24 h diurnal cycle) ------- #
    try:
        e13 = load_json("exp13_horizon_week")
        rows13 = e13["rows"]
        one_h = min(rows13, key=lambda r: r["hours"])
        six_h = next(r for r in rows13 if abs(r["hours"] - 6.0) < 1e-9)
        m["HorizonWeekReps"] = str(e13["reps_sentinel"])
        m["HorizonWeekBaselineReps"] = str(e13["reps_baseline"])
        m["HorizonWeekMaxHours"] = _fmt(max(r["hours"] for r in rows13), 0)
        m["HorizonWeekMaxMFrames"] = _fmt(max(r["frames"] for r in rows13) / 1e6, 2)
        m["HorizonWeekSentinelPfaMax"] = _fmt(
            max(r["sentinel"]["pfa"] for r in rows13), 3)
        m["HorizonWeekSentinelCiHi"] = _fmt(
            max(r["sentinel"]["pfa_ci"][1] for r in rows13), 3)
        m["HorizonWeekPvalueOneHour"] = _fmt(one_h["pvalue"]["pfa"], 3)
        m["HorizonWeekFixedOneHour"] = _fmt(one_h["fixed"]["pfa"], 3)
        m["HorizonWeekFixedSixHour"] = _fmt(six_h["fixed"]["pfa"], 3)
        m["HorizonWeekPvalueSixHour"] = _fmt(six_h["pvalue"]["pfa"], 3)
        m["HorizonWeekCalN"] = _int(e13["n_calibration"])
        m["HorizonWeekCalDefaultPct"] = _fmt(
            100 * e13["default_calibration_cycle_fraction"], 1)
    except (FileNotFoundError, KeyError, StopIteration):
        pass

    # ---- Experiment 3b (activity shift, unweighted vs weighted, gate) ----- #
    try:
        e3b = load_json("exp3b_shift_default")
        for key, tag in (("1.0", "ShiftDefOne"), ("1.5", "ShiftDefOneHalf"),
                         ("2.0", "ShiftDefTwo"), ("3.0", "ShiftDefThree")):
            cell = e3b["cells"][key]
            r, w = cell["residual"], cell["residual + LR wts"]
            m[f"{tag}ResPfa"] = _fmt(r["pfa"], 3)
            m[f"{tag}ResCiLo"] = _fmt(r["pfa_ci"][0], 3)
            m[f"{tag}ResCiHi"] = _fmt(r["pfa_ci"][1], 3)
            m[f"{tag}ResVerdict"] = r["verdict"]
            m[f"{tag}ResMiss"] = _fmt(r["miss_rate"], 3)
            m[f"{tag}WtPfa"] = _fmt(w["pfa"], 3)
            m[f"{tag}WtMiss"] = _fmt(w["miss_rate"], 3)
        dec = e3b["decision"]
        m["ShiftDefMissDelta"] = _fmt(dec["condition_b_miss_delta_at_1.0"], 3)
        m["ShiftDefMissDeltaMax"] = _fmt(dec["condition_b_threshold"], 2)
        m["ShiftDefRecommendWeighted"] = (
            "recommend" if dec["recommend_weighted_default"] else "keep unweighted")
        m["ShiftDefGateMinFrac"] = _fmt(min(
            g["frac_streams_triggered"] for g in e3b["gate"].values()), 2)
        m["ShiftDefGateMinBlocks"] = _fmt(min(
            g["mean_frac_blocks_triggered"] for g in e3b["gate"].values()), 2)
        m["ShiftDefNullStreams"] = str(e3b["reps_null"])
        m["ShiftDefSignalStreams"] = str(e3b["reps_signal"])
    except (FileNotFoundError, KeyError, StopIteration):
        pass

    # ---- Cost derivation (exp5 end-to-end and component benchmarks) ------- #
    try:
        e5 = load_json("exp5_compute")
        e2e = e5["end_to_end"]
        conf = max(e5["conformal"], key=lambda d: d["n_calibration"])
        det32 = next(d for d in e5["detector"] if d["n_grid"] == 32)
        lag = e2e["decorrelation_lag"]
        conf_us = conf["sec_per_frame"] * 1e6
        step_us = det32["sec_per_step_streaming"] * 1e6
        vec_us = det32["sec_per_step_vectorised"] * 1e6
        m["CostConformalUs"] = _fmt(conf_us, 2)
        m["CostConformalN"] = _int(conf["n_calibration"])
        m["CostStepVecUs"] = _fmt(vec_us, 1)
        m["CostAmortStreamUs"] = _fmt(conf_us + step_us / lag, 1)
        m["CostAmortVecUs"] = _fmt(conf_us + vec_us / lag, 2)
        m["CostFrames"] = _int(e2e["frames"])
        m["CostBetSteps"] = _int(e2e["betting_steps"])
        m["CostMonitorSec"] = _fmt(e2e["monitor_seconds"], 3)
        m["CostBetEndToEndUs"] = _fmt(
            1e6 * e2e["monitor_seconds"] / e2e["betting_steps"], 1)
    except (FileNotFoundError, KeyError, StopIteration):
        pass

    # ---- Fleet (exp4) and diagnostic (exp7) configuration counts ---------- #
    try:
        e4 = load_json("exp4_network")
        m["NetEvalMinutes"] = _fmt(e4["T_eval"] / FPS / 60.0, 0)
        m["NetTrainFleets"] = str(e4["n_train"])
    except (FileNotFoundError, KeyError):
        pass
    try:
        e7 = load_json("exp7_diagnostics")
        m["DiagPN"] = _int(e7["coverage"]["0.02"]["n"])
        m["DiagLjungFrac"] = _fmt(e7["ljung_box_thinned_frac_above_005"], 3)
    except (FileNotFoundError, KeyError):
        pass

    _collect_real_v3(m)

    for name in _OBSOLETE_REAL:
        m.setdefault(name, "\\todo{obsolete}")

    return m


def d_sign(x: float) -> str:
    """A signed one-decimal delay, e.g. +38.5 or -55.7."""
    return f"{float(x):+.1f}"


# LaTeX control sequences the manuscript uses that are not result macros.
_NOT_MACROS = {
    "PassOptionsToPackage", "T", "Beta", "E", "Einf", "Pinf", "Fcal", "Gfilt", "ADD",
    "PFA", "ARL", "Cref", "Crefname",
    # Standard LaTeX control sequences that match the CamelCase pattern.
    "Bigl", "Bigr", "Big", "Bigg", "Biggl", "Biggr", "Biggm", "Bigm",
    "LaTeX", "TeX", "Roman", "Alph", "Huge", "Large", "Small",
    "IfFileExists", "Cref", "Crefformat",
    # algpseudocode control sequences.
    "Comment", "EndFor", "EndIf", "Require", "State", "For", "If", "Ensure",
    "While", "EndWhile", "Return", "Function", "EndFunction", "Procedure",
    "EndProcedure",
}


def _referenced_names(tex: Path) -> set:
    """CamelCase control sequences appearing in the manuscript."""
    import re
    if not tex.exists():
        return set()
    body = tex.read_text()
    names = set(re.findall(r"\\([A-Z][A-Za-z]{3,})", body))
    return {n for n in names if n not in _NOT_MACROS}


def main():
    macros = collect()
    tex = Path(TABLES).parents[1] / "paper" / "sentinel_e.tex"
    missing = sorted(_referenced_names(tex) - set(macros))

    lines = [
        "% Auto-generated by experiments/make_macros.py -- do not edit.",
        "% Every number quoted in the manuscript prose is defined here and comes",
        "% directly from the JSON payloads written by the experiment scripts.",
        "",
        "% Visible placeholder for a value that is still open or obsolete.",
        "\\providecommand{\\todo}[1]{\\textbf{[TODO: #1]}}",
        "",
    ]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    if missing:
        lines += [
            "",
            "% Placeholders for experiments that have not been run yet.  These make",
            "% the manuscript compile from a partial results directory; a '??' in the",
            "% PDF means the corresponding experiment output is missing.",
        ]
        for k in missing:
            lines.append(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}}}")

    path = Path(TABLES) / "macros.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(macros)} defined, {len(missing)} placeholders")
    if missing:
        print("  missing:", ", ".join(missing))


if __name__ == "__main__":
    main()

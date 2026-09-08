"""Emit LaTeX macros for every number quoted in the manuscript prose.

The paper must never contain a hand-typed result.  This script reads the JSON
payloads written by the experiments and defines one macro per quoted quantity,
so a re-run of the protocol updates the text automatically and a stale number is
impossible.
"""

from __future__ import annotations

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


def collect() -> Dict[str, str]:
    m: Dict[str, str] = {}

    # ---- Experiment 1 ---------------------------------------------------- #
    try:
        e1 = load_json("exp1_validity")
        m["ExpOneReps"] = str(e1["reps"])
        m["ExpOneHorizonReps"] = str(e1.get("horizon_reps", e1["reps"]))
        a = e1["alpha_sweep"]
        at01 = next(r for r in a if abs(r["alpha"] - 0.01) < 1e-12)
        m["ValidityPfaSentinel"] = _fmt(at01["sentinel_pfa"], 3)
        m["ValidityPfaPvalue"] = _fmt(at01["pvalue_pfa"], 3)
        m["ValidityPfaPvaluePct"] = _pct(at01["pvalue_pfa"], 0)
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

        se = best_at("SENTINEL-E")
        if se:
            m["DelaySentinelSec"] = _fmt(se["add_censored"] / FPS, 1)
            m["DelaySentinelCondSec"] = _fmt(se["add"] / FPS, 1)
            m["DelaySentinelPfa"] = _fmt(se["pfa"], 3)
            m["DelaySentinelMiss"] = _fmt(se["miss_rate"], 2)
            m["DelaySentinelCi"] = _fmt(1.96 * (se["add_censored_se"] or 0) / FPS, 1)
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
        for k, tag in (("none", "NoGraph"), ("heuristic", "Heuristic"), ("gnn", "Gnn")):
            r = rows.get(k)
            if not r:
                continue
            m[f"Net{tag}DelaySec"] = _fmt(r["add"] / FPS, 1)
            m[f"Net{tag}Power"] = _fmt(r["power"], 3)
            m[f"Net{tag}Fdr"] = _fmt(r["fdr"], 3)
            m[f"Net{tag}NullPfa"] = _fmt(r["null_fleet_alarm_rate"], 3)
        if "none" in rows and "gnn" in rows:
            red = 100 * (1 - rows["gnn"]["add"] / rows["none"]["add"])
            m["NetGnnDelayReduction"] = _fmt(red, 1)
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
        n_bad = sum(1 for r in e6["rows"] if r["pfa"] > e6["alpha"] + 1e-9)
        m["AblNumInvalid"] = str(n_bad)
        m["AblNumVariants"] = str(len(e6["rows"]))
    except FileNotFoundError:
        pass

    # ---- Experiment 8 ---------------------------------------------------- #
    try:
        e8 = load_json("exp8_episodes")
        m["EpiReps"] = str(e8["reps"])
        counts = e8["episode_counts"]
        m["EpiMaxCount"] = str(counts[-1])
        for k, tag in ((counts[0], "One"), (counts[-1], "Many")):
            for v, vt in (("episodic", "Epi"), ("changepoint", "Cp")):
                r = next(x for x in e8["by_episodes"][str(k)] if x["variant"] == v)
                m[f"Epi{tag}{vt}Miss"] = _fmt(r["miss_rate"], 3)
                m[f"Epi{tag}{vt}Delay"] = _fmt(
                    None if r["add_censored"] is None else r["add_censored"] / FPS, 1)
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

    return m


# LaTeX control sequences the manuscript uses that are not result macros.
_NOT_MACROS = {
    "PassOptionsToPackage", "T", "Beta", "E", "Einf", "Pinf", "Fcal", "ADD",
    "PFA", "ARL", "Cref", "Crefname",
    # Standard LaTeX control sequences that match the CamelCase pattern.
    "Bigl", "Bigr", "Big", "Bigg", "Biggl", "Biggr", "Biggm", "Bigm",
    "LaTeX", "TeX", "Roman", "Alph", "Huge", "Large", "Small",
    "IfFileExists", "Cref", "Crefformat",
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

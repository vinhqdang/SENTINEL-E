"""Experiment 12 --- does the episodic advantage hold beyond one simulator setting?

Experiment 9 compared the episodic e-process (``epi``, shipped 6x4 grid) with a
changepoint mixture on a 24-point kappa grid (``cp24``) at a single simulator
configuration.  A reviewer asked two things: does the paired advantage hold
across the event-shape parameters the simulator exposes, and when is the simpler
changepoint mixture the better choice?

This script reuses the exp9 arms and paired helpers unchanged.  Every
configuration runs both arms on the same seeds (null and post-change streams
are paired by seed), at alpha = 0.01, T = 90,000 frames, change point at 20,000,
representative calibration.  Configurations:

* 8 corners: event_length in {200, 1600} x episode_gap in {1500, 12000} x
  event_intermittency pi in {0.2, 0.7}, all with 4 episodes;
* a single episode: n_episodes = 1, event_length = 800, pi = 0.35;
* a permanent change: n_episodes = 1, event_length = T - change point (the
  event runs to the horizon), pi = 1.0 (the simulator draws
  ``rng.random < pi``, so pi = 1 marks every frame active).

Verdict: "equivalent (|diff| < 1 s)" when the point estimate of the censored
delay difference (cp24 minus epi, positive = episodic faster) is smaller than
the 1 s practical margin in absolute value.  Otherwise "episodic better" when the
95% bootstrap CI lies above zero, "changepoint better" when it lies below zero,
and "no resolved difference" when the CI contains zero.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from experiments.common import (
    FIGURES,
    fmt,
    save_json,
    setup_matplotlib,
    write_latex_table,
)
from experiments.exp9_paired import (
    ARMS,
    _censored,
    _mcnemar,
    _missed,
    _paired_delay,
)
from experiments.runner import evaluate
from sentinel_e.metrics import wilson_interval

REPS = 200
T = 90_000
CHANGE_POINT = 20_000
FPS = 25.0
ALPHA = 0.01
CAL_REGIME = "representative"
ARM_NAMES = ("cp24", "epi")
WORKERS = 3


def build_configs() -> List[Dict]:
    cfgs: List[Dict] = []
    for L in (200, 1600):
        for gap in (1500, 12000):
            for pi in (0.2, 0.7):
                cfgs.append({
                    "name": f"L={L}, gap={gap}, pi={pi}, 4 ep.",
                    "group": "corner",
                    "cfg": dict(T=T, n_episodes=4, event_length=L, episode_gap=gap,
                                event_intermittency=pi,
                                calibration_regime=CAL_REGIME),
                })
    cfgs.append({
        "name": "single episode, L=800, pi=0.35",
        "group": "single",
        "cfg": dict(T=T, n_episodes=1, event_length=800, episode_gap=6_000,
                    event_intermittency=0.35, calibration_regime=CAL_REGIME),
    })
    cfgs.append({
        "name": f"permanent change, pi=1.0 (L={T - CHANGE_POINT})",
        "group": "permanent",
        "cfg": dict(T=T, n_episodes=1, event_length=T - CHANGE_POINT,
                    episode_gap=6_000, event_intermittency=1.0,
                    calibration_regime=CAL_REGIME),
    })
    return cfgs


MARGIN_S = 1.0   # practical equivalence margin on the delay difference, seconds


def _verdict(diff_s: float, ci_s) -> str:
    """Verdict on the cp24-minus-epi delay difference (seconds, > 0 = episodic faster)."""
    if abs(diff_s) < MARGIN_S:
        return f"equivalent (|diff| < {MARGIN_S:g} s)"
    lo, hi = ci_s
    if lo > 0:
        return "episodic better"
    if hi < 0:
        return "changepoint better"
    return "no resolved difference"


def run_config(entry: Dict, reps: int) -> Dict:
    arms = {}
    for name in ARM_NAMES:
        out, null, sig = evaluate("SENTINEL-E", ALPHA, entry["cfg"], reps,
                                  CHANGE_POINT, options=ARMS[name], workers=WORKERS)
        delay = _censored(sig, T, CHANGE_POINT)
        # The per-stream vector must reproduce the harness aggregate (see exp9).
        assert abs(delay.mean() - out.add_censored) < 1e-6, (name, delay.mean(),
                                                            out.add_censored)
        arms[name] = {
            "pfa": out.pfa, "pfa_ci": list(out.pfa_ci),
            "miss_rate": out.miss_rate,
            "censored_delay_s": float(delay.mean()) / FPS,
            "delay": delay, "miss": _missed(sig),
        }
    d = _paired_delay(arms["epi"]["delay"], arms["cp24"]["delay"], seed=7)
    m = _mcnemar(arms["epi"]["miss"], arms["cp24"]["miss"])
    ci_s = [d["ci95"][0] / FPS, d["ci95"][1] / FPS]
    row = {
        "name": entry["name"], "group": entry["group"], "cfg": entry["cfg"],
        "cp24": {k: v for k, v in arms["cp24"].items() if k not in ("delay", "miss")},
        "epi": {k: v for k, v in arms["epi"].items() if k not in ("delay", "miss")},
        "delay_diff_s": d["mean_diff"] / FPS,        # cp24 - epi; > 0 = episodic faster
        "delay_diff_ci95_s": ci_s,
        "delay_diff_se_s": d["se"] / FPS,
        "delay_p_boot": d["p_boot"],
        "miss_epi_only_detect": m["b01"],            # epi detects, cp24 misses
        "miss_cp24_only_detect": m["b10"],           # cp24 detects, epi misses
        "miss_p_mcnemar": m["p_exact"],
        "verdict": _verdict(d["mean_diff"] / FPS, ci_s),
    }
    print(f"  {entry['name']:36s} cp24 PFA={arms['cp24']['pfa']:.3f} "
          f"epi PFA={arms['epi']['pfa']:.3f} | diff={row['delay_diff_s']:+.2f}s "
          f"[{ci_s[0]:+.2f}, {ci_s[1]:+.2f}] | miss {m['b01']}/{m['b10']} "
          f"p={m['p_exact']:.3f} | {row['verdict']}", flush=True)
    return row


def run(reps: int = REPS) -> Dict:
    rows = [run_config(e, reps) for e in build_configs()]
    return {"reps": reps, "T": T, "change_point": CHANGE_POINT, "alpha": ALPHA,
            "fps": FPS, "calibration_regime": CAL_REGIME,
            "arms": {"cp24": "changepoint mixture, 24-point kappa grid (eta=0)",
                     "epi": "episodic, shipped 6x4 grid (kappa x eta)"},
            "rows": rows}


def make_table(payload: Dict) -> None:
    reps = payload["reps"]
    body = []
    for r in payload["rows"]:
        c, e = r["cp24"], r["epi"]
        if c["miss_rate"] >= 1.0 and e["miss_rate"] >= 1.0:
            r["verdict"] = "both miss every episode"
        else:
            r["verdict"] = _verdict(r["delay_diff_s"], r["delay_diff_ci95_s"])
        lo, hi = r["delay_diff_ci95_s"]
        body.append([
            r["name"],
            f"{fmt(c['pfa'], 3)} [{fmt(c['pfa_ci'][0], 3)}, {fmt(c['pfa_ci'][1], 3)}]",
            f"{fmt(e['pfa'], 3)} [{fmt(e['pfa_ci'][0], 3)}, {fmt(e['pfa_ci'][1], 3)}]",
            fmt(c["miss_rate"], 2), fmt(e["miss_rate"], 2),
            fmt(c["censored_delay_s"], 1), fmt(e["censored_delay_s"], 1),
            f"{r['delay_diff_s']:+.1f} [{lo:+.1f}, {hi:+.1f}]",
            f"{r['miss_epi_only_detect']}/{r['miss_cp24_only_detect']}",
            fmt(r["miss_p_mcnemar"], 3),
            r["verdict"],
        ])
    write_latex_table(
        body,
        ["configuration", "PFA cp24 [95\\% CI]", "PFA epi [95\\% CI]",
         "miss cp24", "miss epi", "delay cp24 (s)", "delay epi (s)",
         "diff.\\ (s) [95\\% CI]", "miss disc.\\ epi/cp24", "$p$ McNemar",
         "verdict"],
        caption=(
            r"Paired episodic-versus-changepoint comparison across simulator "
            r"configurations at $\alpha=0.01$, "
            f"{reps} seeded pairs per configuration. Delay is censored (misses "
            r"charged the remaining horizon); the difference is cp24 minus epi, so "
            r"a positive value favours the episodic arm. Miss discordance counts "
            r"streams that only the episodic arm (epi) or only the changepoint arm "
            r"(cp24) detect. Verdicts treat a difference smaller than 1\,s in "
            r"absolute value as equivalent, whatever its CI."
        ),
        label="tab:corner_sweep",
        name="tab_corner_sweep",
        align="lrrrrrrrrrrl",
    )


def _short_name(r: Dict) -> str:
    """Compact tick label: L = event length, g = gap, pi = intermittency."""
    c = r["cfg"]
    if r["group"] == "corner":
        return f"L{c['event_length']} g{c['episode_gap']} pi{c['event_intermittency']:g}"
    if r["group"] == "single":
        return "single episode"
    return "permanent change"


def make_figure(payload: Dict) -> None:
    plt = setup_matplotlib()
    rows = payload["rows"]
    x = np.arange(len(rows))
    d = np.array([r["delay_diff_s"] for r in rows])
    lo = np.array([r["delay_diff_ci95_s"][0] for r in rows])
    hi = np.array([r["delay_diff_ci95_s"][1] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.axhline(0.0, color="k", lw=1.0)
    colors = ["#0b3d91" if r["verdict"] == "episodic better" else
              ("#f9a825" if r["verdict"] == "changepoint better" else "#777777")
              for r in rows]  # equivalent and unresolved rows are grey
    for xi, di, l, h, c in zip(x, d, lo, hi, colors):
        ax.errorbar(xi, di, yerr=[[di - l], [h - di]], fmt="o", color=c,
                    capsize=3, ms=5)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(r) for r in rows], rotation=40,
                       ha="right", fontsize=6.5)
    ax.set_ylabel("cp24 minus epi censored delay (s)\n(> 0: episodic faster)")
    ax.set_title("Paired delay difference with 95% bootstrap CI")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_corner_sweep.{ext}")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run with 4 streams per arm; writes nothing")
    args = ap.parse_args(argv)
    if args.smoke:
        run(reps=4)
        print("smoke test ok")
        return
    print(f"Experiment 12: corner sweep ({args.reps} paired streams per arm)")
    payload = run(reps=args.reps)
    save_json(payload, "exp12_corner_sweep")
    make_table(payload)
    make_figure(payload)
    print(" wrote results/exp12_corner_sweep.json, tab_corner_sweep, fig_corner_sweep")


if __name__ == "__main__":
    main()

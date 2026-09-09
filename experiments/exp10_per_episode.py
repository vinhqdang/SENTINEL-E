"""Experiment 10 --- the per-episode mechanism claim, measured rather than asserted.

Every other experiment in this paper scores a stream as detected once *any*
alarm follows the *first* onset (``sentinel_e.metrics.summarize_runs``). With
several episodes per camera that is a stream-level statistic: a run that sleeps
through episodes 1 through k-1 and only fires on episode k is scored a clean
detection, which is invisible to a reader and hides exactly the mechanism the
episodic e-process claims credit for --- recovering after a missed episode
rather than carrying a burned wealth process into the next one (see
``sec:episodes`` and ``sentinel_e.metrics.per_episode_outcomes``).

This experiment runs the same two arms as Experiment 9 (episodic vs. the
changepoint mixture on a matched 24-point kappa grid, so the comparison is not
confounded by grid size) and scores every episode's own detection window
independently, aggregating to a genuine per-episode miss rate rather than a
per-stream one.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from experiments.common import fmt, parallel_map, save_json, write_latex_table
from experiments.exp9_paired import ETA, KAPPA24, KAPPA6
from sentinel_e.episodic import EpisodePrior
from sentinel_e.metrics import per_episode_outcomes
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator

REPS = 200
T = 90_000
CHANGE_POINT = 20_000
FPS = 25.0
ALPHA = 0.01
EPISODE_COUNTS = [1, 2, 4, 8]

ARMS = {
    "episodic": EpisodePrior(kappa=KAPPA6, eta=ETA),
    "changepoint": EpisodePrior(kappa=KAPPA24, eta=(0.0,)),
}


def _one_run(args) -> Dict:
    variant, seed, n_episodes = args
    cfg = StreamConfig(T=T, n_episodes=n_episodes, event_length=800,
                       episode_gap=6_000, event_intermittency=0.35)
    stream = StreamSimulator(cfg, seed=seed).simulate_camera(change_point=CHANGE_POINT)
    prior = ARMS[variant]
    model = SentinelE(alpha=ALPHA, calibration="residual", weighted=False,
                      detector="episodic", episode_prior=prior)
    res = model.run_stream(stream)
    per_ep = per_episode_outcomes(res.alarms, stream.episodes, horizon=T)
    stream_missed = not np.any(res.alarms[CHANGE_POINT:])
    return {
        "variant": variant, "n_episodes": n_episodes,
        "per_episode": [
            {"delay": o.delay, "missed": o.missed,
             "window_len": o.window_end - o.start}
            for o in per_ep
        ],
        "stream_missed": bool(stream_missed),
    }


def run(reps: int = REPS, counts=EPISODE_COUNTS, workers=None) -> Dict:
    jobs = [
        (variant, 20_000 + rep, n_episodes)
        for n_episodes in counts
        for variant in ARMS
        for rep in range(reps)
    ]
    print(f"Experiment 10: {len(jobs)} streams, per-episode mechanism check")
    results = parallel_map(_one_run, jobs, workers, desc="per-episode runs")

    by_key: Dict[str, List[Dict]] = {}
    for r in results:
        by_key.setdefault((r["n_episodes"], r["variant"]), []).append(r)

    payload: Dict = {"reps": reps, "T": T, "change_point": CHANGE_POINT,
                     "alpha": ALPHA, "episode_counts": list(counts),
                     "event_length": 800, "by_episodes": {}}
    for n_episodes in counts:
        row = {}
        for variant in ARMS:
            runs = by_key[(n_episodes, variant)]
            episodes = [o for r in runs for o in r["per_episode"]]
            n = len(episodes)
            misses = sum(o["missed"] for o in episodes)
            delays = [o["delay"] for o in episodes if o["delay"] is not None]
            # A missed episode is charged its own window's remainder, exactly
            # as summarize_runs charges the whole-horizon remainder for a
            # stream-level miss -- the per-episode analogue of censored delay.
            censored = [
                o["delay"] if o["delay"] is not None else o["window_len"]
                for o in episodes
            ]
            agg = {
                "n_episodes_scored": n,
                "episode_miss_rate": misses / n if n else float("nan"),
                "episode_add": float(np.mean(delays)) if delays else float("nan"),
                "episode_add_se": (
                    float(np.std(delays, ddof=1) / np.sqrt(len(delays)))
                    if len(delays) > 1 else float("nan")
                ),
                "episode_add_censored": float(np.mean(censored)) if censored else float("nan"),
                "episode_add_censored_se": (
                    float(np.std(censored, ddof=1) / np.sqrt(len(censored)))
                    if len(censored) > 1 else float("nan")
                ),
            }
            stream_miss = float(np.mean([r["stream_missed"] for r in runs]))
            row[variant] = {**agg, "stream_miss_rate": stream_miss}
            print(f"  n_episodes={n_episodes:2d} {variant:12s} "
                  f"episode_miss={agg['episode_miss_rate']:.3f} "
                  f"episode_add_c={agg['episode_add_censored']/FPS:7.1f}s "
                  f"(stream_miss={stream_miss:.3f})", flush=True)
        payload["by_episodes"][str(n_episodes)] = row
    save_json(payload, "exp10_per_episode")
    print(" wrote results/exp10_per_episode.json")
    return payload


def make_table(payload: Dict) -> None:
    rows = []
    for k in payload["episode_counts"]:
        row = payload["by_episodes"][str(k)]
        for i, variant in enumerate(("episodic", "changepoint")):
            r = row[variant]
            rows.append([
                str(k) if i == 0 else "", variant,
                fmt(r["stream_miss_rate"], 3), fmt(r["episode_miss_rate"], 3),
                fmt(None if r["episode_add_censored"] is None
                    else r["episode_add_censored"] / FPS, 1),
            ])
    write_latex_table(
        rows,
        ["episodes", "mixture", "stream miss", "per-episode miss",
         "per-episode cens.\\ delay (s)"],
        caption=(
            f"Stream-level miss rate (any alarm after the first onset) against "
            r"the per-episode miss rate (each episode's own detection window "
            f"scored independently, {payload['reps']} streams per row). The two "
            r"columns diverge sharply as episodes accumulate: a stream can look "
            r"perfectly detected while most of its individual episodes are "
            r"missed, which is exactly what the stream-level statistic used "
            r"elsewhere in this paper cannot show."
        ),
        label="tab:per_episode", name="tab14_per_episode", align="rlrrr",
    )


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else REPS
    payload = run(reps=reps)
    make_table(payload)
    print(" wrote results/tables/tab14_per_episode.tex")

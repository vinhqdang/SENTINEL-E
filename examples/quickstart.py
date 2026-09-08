"""A ten-minute tour of SENTINEL-E on one simulated camera.

    python examples/quickstart.py

Shows the three things a deployment actually needs: fit the conformal layer on
confirmed no-dumping frames, monitor a stream, and read off when the detector
alarmed and how confident it was that an episode was in progress.
"""

from __future__ import annotations

import numpy as np

from sentinel_e.episodic import EpisodicEDetector
from sentinel_e.pipeline import SentinelE
from sentinel_e.streams import StreamConfig, StreamSimulator

FPS = 25.0


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. A camera: ~1 hour of video with four dumping episodes.
    # ------------------------------------------------------------------ #
    cfg = StreamConfig(
        T=90_000,                       # frames (~60 min at 25 fps)
        n_episodes=4,
        event_length=800,               # ~32 s per episode
        event_intermittency=0.35,       # only a third of frames show the act
        calibration_regime="representative",
    )
    stream = StreamSimulator(cfg, seed=7).simulate_camera(change_point=20_000)
    print(f"stream: {stream.T:,} frames, {stream.n_episodes} episodes at "
          f"{[e[0] for e in stream.episodes]}")
    print(f"calibration: {stream.calibration_scores.size:,} confirmed "
          f"no-dumping frames\n")

    # ------------------------------------------------------------------ #
    # 2. Fit and monitor.  alpha is a *time-uniform* level: the probability
    #    that this camera ever raises a false alarm, however long it runs.
    # ------------------------------------------------------------------ #
    model = SentinelE(alpha=0.01)
    model.fit(stream.calibration_scores, stream.calibration_context)
    print(f"decorrelation lag       : {model.decorrelation_lag} frames "
          f"({FPS / model.decorrelation_lag:.1f} bets per second)")
    print(f"effective calibration n : {model.n_calibration_effective:,} "
          f"(raw count divided by the lag)")

    result = model.run(stream.scores, stream.context)
    fired = np.flatnonzero(result.alarms)
    if fired.size:
        first_onset = stream.episodes[0][0]
        delay = fired[0] - first_onset
        print(f"first alarm             : frame {fired[0]:,} "
              f"({delay / FPS:+.1f} s relative to the first onset)")
    else:
        print("first alarm             : none")
    print(f"abstention rate         : {result.abstention_rate:.1%} of frames "
          f"outside the calibrated context region\n")

    # ------------------------------------------------------------------ #
    # 3. The same stream on a quiet camera: the guarantee in action.
    # ------------------------------------------------------------------ #
    quiet = StreamSimulator(cfg, seed=8).simulate_camera(change_point=None)
    quiet_result = SentinelE(alpha=0.01).run_stream(quiet)
    print(f"null stream alarms      : {int(quiet_result.alarms.sum())} "
          f"(alpha = 0.01 bounds the probability of any alarm, at any frame)")

    # ------------------------------------------------------------------ #
    # 4. The episode posterior, which is what the camera graph passes around.
    # ------------------------------------------------------------------ #
    p = result.p_values[result.bet_index]
    post = EpisodicEDetector(alpha=1e-12).posteriors(p)     # threshold unreachable
    inside = np.zeros(result.bet_index.size, dtype=bool)
    for a, b in stream.episodes:
        inside |= (result.bet_index >= a) & (result.bet_index < b)
    print(f"episode posterior       : {post[inside].mean():.2f} inside episodes, "
          f"{post[~inside].mean():.2f} outside")


if __name__ == "__main__":
    main()

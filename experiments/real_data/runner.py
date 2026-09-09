"""Monte-Carlo workers for the real-data experiments.

Builds every stream from :func:`sentinel_e.datasets.real_vad.build_real_stream`
instead of the simulator, then hands it to
:func:`experiments.runner.run_spec_on_stream` -- the exact same method
dispatch (SENTINEL-E, every classical baseline, the decorrelation-lag
thinning convention) used throughout the rest of this paper. Nothing about
how a method is run changes; only where the per-frame scores come from.

Every stream is built with ironclad calibration/deployment separation
(:func:`build_real_stream`): the calibration set is always the real dataset's
official training split, and the background/spliced-event pool is always
drawn from labelled runs inside the real test split. A single fixed dataset
(``'ped2'`` or ``'avenue'``) supplies both parts unless ``shift_from`` names a
different one, in which case calibration is drawn from ``shift_from`` and the
deployment stream from ``dataset`` -- the real cross-scene shift used in
``exp_r4_shift.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.common import parallel_map
from experiments.runner import Spec, Trace, _to_alarm_array, run_spec_on_stream
from sentinel_e.datasets.real_vad import (
    RealVADCorpus,
    build_real_stream,
    causal_local_context,
    load_real_vad,
)
from sentinel_e.metrics import RunOutcome, summarize_runs

_CORPUS_CACHE: Dict[str, RealVADCorpus] = {}


def _corpus(name: str) -> RealVADCorpus:
    if name not in _CORPUS_CACHE:
        _CORPUS_CACHE[name] = load_real_vad(name)
    return _CORPUS_CACHE[name]


@dataclass(frozen=True)
class RealSpec:
    method: str
    knob: float
    seed: int
    change_point_flag: bool          # True: build a signal stream; False: null
    dataset: str
    n_frames: int
    n_calibration: int
    n_events: int
    change_point: int                # fixed onset, shared across the ensemble
    options: Tuple = ()
    shift_from: Optional[str] = None  # calibrate on THIS dataset's official
                                       # split instead (cross-scene shift)
    calibration_regime: str = "official"  # 'official' or 'representative'
    min_event_length: int = 0

    @property
    def opt(self) -> Dict:
        return dict(self.options)


def run_real_spec(spec: RealSpec) -> Trace:
    corpus = _corpus(spec.dataset)
    rng = np.random.default_rng(spec.seed)
    common = dict(
        n_frames=spec.n_frames, rng=rng,
        change_point=spec.change_point if spec.change_point_flag else None,
    )

    if spec.shift_from:
        # Cross-scene shift: calibration is the *other* real dataset's
        # official training split; deployment is this dataset's real test
        # stream. The representative-calibration fix does not apply here by
        # construction -- there is no "held-out normal test frame" of a
        # scene the detector has never seen.
        cal_corpus = _corpus(spec.shift_from)
        cal_rng = np.random.default_rng(spec.seed + 3_000_000)
        pool = cal_corpus.calibration_scores
        idx0 = int(cal_rng.integers(0, max(1, pool.size - spec.n_calibration)))
        cal = pool[idx0: idx0 + spec.n_calibration]
        st = build_real_stream(
            corpus, n_calibration=0, n_events=spec.n_events if spec.change_point_flag else 0,
            require_normal=not spec.change_point_flag,
            min_event_length=spec.min_event_length, **common,
        )
        st.calibration_scores = cal
        st.calibration_context = causal_local_context(cal).reshape(-1, 1)
    else:
        st = build_real_stream(
            corpus, n_calibration=spec.n_calibration,
            n_events=spec.n_events if spec.change_point_flag else 0,
            require_normal=not spec.change_point_flag,
            calibration_regime=spec.calibration_regime,
            min_event_length=spec.min_event_length, **common,
        )

    pfa_spec = Spec(
        method=spec.method, knob=spec.knob, seed=spec.seed,
        change_point=st.change_point, cfg={}, options=spec.options,
    )
    return run_spec_on_stream(st, pfa_spec)


def evaluate_real(
    method: str,
    knob: float,
    dataset: str,
    reps: int,
    n_frames: int,
    n_calibration: int,
    change_point: int,
    n_events: int = 1,
    options: Optional[Dict] = None,
    seed0: int = 10_000,
    workers: Optional[int] = None,
    shift_from: Optional[str] = None,
    calibration_regime: str = "official",
    min_event_length: int = 0,
) -> Tuple[RunOutcome, List[Trace], List[Trace]]:
    """Real-data analogue of :func:`experiments.runner.evaluate`.

    ``change_point`` is fixed across the whole replicate ensemble (matching
    the simulator's own convention), so ``summarize_runs`` can be given a
    single onset for every signal trace.
    """
    opts = tuple(sorted((options or {}).items()))
    null_specs = [
        RealSpec(method, knob, seed0 + i, False, dataset, n_frames, n_calibration,
                 0, change_point, opts, shift_from, calibration_regime, min_event_length)
        for i in range(reps)
    ]
    sig_specs = [
        RealSpec(method, knob, seed0 + 500_000 + i, True, dataset, n_frames,
                 n_calibration, n_events, change_point, opts, shift_from,
                 calibration_regime, min_event_length)
        for i in range(reps)
    ]
    null = parallel_map(run_real_spec, null_specs, workers)
    sig = parallel_map(run_real_spec, sig_specs, workers)
    outcome = summarize_runs(
        [_to_alarm_array(t) for t in null],
        [_to_alarm_array(t) for t in sig],
        [change_point] * len(sig),
    )
    return outcome, null, sig

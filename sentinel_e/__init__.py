"""SENTINEL-E: Sequential E-process NeTwork for INtermittent Litter/dumping Events.

Anytime-valid sequential detection for streaming illegal-dumping surveillance.

The package converts any frozen per-frame dumping detector into a network of
anytime-valid sequential detectors with time-uniform false-alarm control:

  Layer 1 (:mod:`sentinel_e.conformal`)
      Raw detector scores -> (weighted, calibration-conditional) conformal
      p-values against a per-camera set of confirmed no-dumping frames.
  Layer 2 (:mod:`sentinel_e.edetector`)
      p-values -> a betting wealth process that is a non-negative
      supermartingale under the no-dumping null, thresholded at ``1/alpha``.
  Layer 3 (:mod:`sentinel_e.gnn`, :mod:`sentinel_e.ebh`)
      A camera graph supplies a *predictable* spatial prior that steers the
      bets, and e-BH combines fleet evidence with FDR control under arbitrary
      dependence.
"""

__version__ = "1.0.0"

from sentinel_e.conformal import (
    ConformalCalibrator,
    WeightedConformalCalibrator,
    MondrianConformalCalibrator,
    QuantileTaxonomy,
    dkw_inflation,
    weighted_dkw_inflation,
)
from sentinel_e.temporal import acf, estimate_decorrelation_lag, ljung_box, thin_indices
from sentinel_e.betting import (
    BettingFunction,
    PowerBet,
    LinearBet,
    MixtureBet,
    AdaptiveLinearBet,
)
from sentinel_e.edetector import EDetector, EDetectorState, ChangepointPrior
from sentinel_e.episodic import EpisodicEDetector, EpisodePrior, EpisodicState
from sentinel_e.pipeline import SentinelE, FleetSentinelE
from sentinel_e.ebh import ebh, ebh_rejections, global_e_merge
from sentinel_e.graph import CameraGraph
from sentinel_e.metrics import (
    detection_delay,
    run_length,
    summarize_runs,
    delay_far_curve,
    fdr_power,
)

__all__ = [
    "__version__",
    "ConformalCalibrator",
    "WeightedConformalCalibrator",
    "MondrianConformalCalibrator",
    "QuantileTaxonomy",
    "dkw_inflation",
    "weighted_dkw_inflation",
    "acf",
    "estimate_decorrelation_lag",
    "ljung_box",
    "thin_indices",
    "BettingFunction",
    "PowerBet",
    "LinearBet",
    "MixtureBet",
    "AdaptiveLinearBet",
    "EDetector",
    "EpisodicEDetector",
    "EpisodePrior",
    "EpisodicState",
    "SentinelE",
    "FleetSentinelE",
    "EDetectorState",
    "ChangepointPrior",
    "ebh",
    "ebh_rejections",
    "global_e_merge",
    "CameraGraph",
    "detection_delay",
    "run_length",
    "summarize_runs",
    "delay_far_curve",
    "fdr_power",
]

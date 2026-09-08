"""Dataset adapters for SENTINEL-E.

SENTINEL-E never touches pixels: it consumes the scalar output of a frozen
per-frame detector.  Every corpus is therefore reduced to the same interface,
:class:`~sentinel_e.datasets.base.ScoredCorpus` --- a list of per-clip (or
per-image) detector-score arrays plus binary labels --- from which
:func:`~sentinel_e.streams.build_stream_from_clips` builds long monitoring
streams.

Supported sources and their roles in the evaluation protocol:

==============  ===============================  ==================================
Corpus          Modality                          Role
==============  ===============================  ==================================
Mivia-IWDD      fixed-camera dumping video        primary streaming benchmark
TACO            litter images in the wild         backbone pre-training
ZeroWaste       cluttered waste imagery           backbone pre-training, robustness
UAVVaste        UAV litter imagery                UAV deployment scenario
AerialWaste     aerial/satellite dumpsite imagery wide-area deployment scenario
MARIDA / MADOS  Sentinel-2 marine debris          cross-domain generalisation
==============  ===============================  ==================================

None of these corpora redistribute per-frame detector scores, and several
require a signed request form, so the adapters read a small, documented
intermediate file that the user produces once by running their own backbone
(see :func:`sentinel_e.datasets.base.write_scored_corpus`).  When the
intermediate file is absent the loader raises a clear error rather than silently
substituting synthetic data; :mod:`sentinel_e.streams` is the explicit,
clearly-labelled simulation path.
"""

from sentinel_e.datasets.base import (
    ScoredCorpus,
    load_scored_corpus,
    write_scored_corpus,
    corpus_from_frame_table,
)
from sentinel_e.datasets.adapters import (
    ADAPTERS,
    DatasetSpec,
    describe_datasets,
    load_dataset,
)

__all__ = [
    "ScoredCorpus",
    "load_scored_corpus",
    "write_scored_corpus",
    "corpus_from_frame_table",
    "ADAPTERS",
    "DatasetSpec",
    "describe_datasets",
    "load_dataset",
]

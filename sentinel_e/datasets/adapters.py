"""Per-dataset adapters and provenance notes.

Each :class:`DatasetSpec` records where a corpus comes from, how it is obtained,
what role it plays in the SENTINEL-E evaluation and what the frozen backbone is
expected to emit.  ``load_dataset`` then reads the ``.npz`` intermediate that a
one-off backbone pass produces.

Design decision: **no adapter ever downloads or fabricates data.**  Mivia-IWDD
and AerialWaste are distributed under request forms, TACO's images live on
Flickr, and none of the corpora ship detector scores, so an adapter that
silently produced numbers would make the evaluation unreproducible in the worst
possible way.  A missing intermediate raises :class:`FileNotFoundError` with the
exact recipe for producing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from sentinel_e.datasets.base import ScoredCorpus, load_scored_corpus

__all__ = ["DatasetSpec", "ADAPTERS", "load_dataset", "describe_datasets"]

DEFAULT_ROOT = Path("data/scored")


@dataclass(frozen=True)
class DatasetSpec:
    """Provenance and protocol metadata for one corpus."""

    key: str
    name: str
    modality: str
    role: str
    access: str
    url: str
    backbone_output: str
    notes: str = ""
    filename: str = field(default="")

    def path(self, root: Path | str = DEFAULT_ROOT) -> Path:
        fname = self.filename or f"{self.key}.npz"
        return Path(root) / fname

    def recipe(self, root: Path | str = DEFAULT_ROOT) -> str:
        return (
            f"{self.name}: obtain from {self.url} ({self.access}). Run the frozen "
            f"per-frame detector to obtain {self.backbone_output}, then call "
            f"sentinel_e.datasets.write_scored_corpus(corpus, '{self.path(root)}')."
        )


ADAPTERS: Dict[str, DatasetSpec] = {
    "mivia_iwdd": DatasetSpec(
        key="mivia_iwdd",
        name="Mivia Illegal Waste Dumping Detection (Mivia-IWDD)",
        modality="fixed-camera surveillance video, clip-level dumping labels",
        role=(
            "primary streaming benchmark: negative clips are concatenated into "
            "long monitoring streams and positive clips are spliced in at random "
            "onsets to measure detection delay at a controlled false-alarm rate"
        ),
        access="request form to the MIVIA Lab, University of Salerno",
        url="https://mivia.unisa.it/datasets/",
        backbone_output="one dumping score per frame per clip",
        notes=(
            "The reference detector released with the corpus is the natural "
            "backbone; SENTINEL-E freezes it and only consumes its scores."
        ),
    ),
    "taco": DatasetSpec(
        key="taco",
        name="TACO: Trash Annotations in Context",
        modality="litter images in the wild with instance masks",
        role="backbone pre-training and cross-domain robustness check",
        access="open; images hosted on Flickr and fetched by the official script",
        url="http://tacodataset.org/",
        backbone_output="one waste-presence score per image (clips of length 1)",
    ),
    "zerowaste": DatasetSpec(
        key="zerowaste",
        name="ZeroWaste",
        modality="cluttered waste-sorting imagery and video frames",
        role="backbone pre-training; heavy-clutter robustness check",
        access="open download",
        url="http://ai.bu.edu/zerowaste/",
        backbone_output="one waste-presence score per frame",
    ),
    "uavvaste": DatasetSpec(
        key="uavvaste",
        name="UAVVaste",
        modality="UAV imagery of urban and natural litter",
        role="UAV deployment scenario: moving platform, wide field of view",
        access="open; annotations on GitHub, images fetched by the official script",
        url="https://github.com/PUTvision/UAVVaste",
        backbone_output="one litter score per frame of a simulated flight line",
        notes=(
            "Frames along a flight path are ordered by acquisition to form a "
            "pseudo-stream; each overflight is one 'camera' in the fleet."
        ),
    ),
    "aerialwaste": DatasetSpec(
        key="aerialwaste",
        name="AerialWaste",
        modality="aerial and satellite imagery of candidate illegal dumpsites",
        role="wide-area monitoring scenario with revisit-based streams",
        access="request form / registered download",
        url="https://aerialwaste.org/",
        backbone_output="one dumpsite score per tile per revisit",
        notes="Tiles are the nodes of the graph and revisits index the stream.",
    ),
    "marida": DatasetSpec(
        key="marida",
        name="MARIDA: Marine Debris Archive",
        modality="Sentinel-2 multispectral scenes with marine-debris labels",
        role="cross-domain generalisation to marine pollution surveillance",
        access="open (Zenodo)",
        url="https://zenodo.org/records/5151941",
        backbone_output="one debris score per pixel-patch per revisit",
    ),
    "mados": DatasetSpec(
        key="mados",
        name="MADOS: Marine Debris and Oil Spill",
        modality="Sentinel-2 scenes with debris and oil-spill annotations",
        role="cross-domain generalisation; multi-class pollution surveillance",
        access="open (Zenodo)",
        url="https://zenodo.org/records/10664073",
        backbone_output="one pollution score per patch per revisit",
    ),
}


def load_dataset(key: str, root: Path | str = DEFAULT_ROOT) -> ScoredCorpus:
    """Load the scored intermediate for ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not a known corpus.
    FileNotFoundError
        If the intermediate has not been produced yet; the message contains the
        exact recipe.
    """
    if key not in ADAPTERS:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(ADAPTERS)}")
    spec = ADAPTERS[key]
    path = spec.path(root)
    if not path.exists():
        raise FileNotFoundError(spec.recipe(root))
    corpus = load_scored_corpus(path)
    corpus.name = spec.name
    return corpus


def describe_datasets(root: Path | str = DEFAULT_ROOT) -> List[Dict[str, object]]:
    """Report, for every known corpus, whether its scored intermediate exists."""
    rows: List[Dict[str, object]] = []
    for spec in ADAPTERS.values():
        path = spec.path(root)
        rows.append(
            {
                "key": spec.key,
                "name": spec.name,
                "role": spec.role,
                "access": spec.access,
                "url": spec.url,
                "available": path.exists(),
                "path": str(path),
            }
        )
    return rows

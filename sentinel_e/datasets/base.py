"""The common corpus representation and its on-disk format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "ScoredCorpus",
    "load_scored_corpus",
    "write_scored_corpus",
    "corpus_from_frame_table",
]


@dataclass
class ScoredCorpus:
    """Per-clip detector scores and clip-level labels.

    Attributes
    ----------
    clip_scores : list of 1-D float arrays
        Frame-by-frame detector scores for each clip.  A still-image corpus is
        represented as clips of length one.
    clip_labels : ndarray of int
        ``0`` = confirmed negative (no dumping / no waste), ``1`` = positive.
    clip_ids : list of str
    context : list of 2-D float arrays or None
        Optional per-frame context features (illumination, weather flag, ...)
        used by the weighted conformal layer.
    name : str
    fps : float
    """

    clip_scores: List[np.ndarray]
    clip_labels: np.ndarray
    clip_ids: List[str]
    context: Optional[List[np.ndarray]] = None
    name: str = "corpus"
    fps: float = 25.0

    def __post_init__(self) -> None:
        self.clip_labels = np.asarray(self.clip_labels, dtype=int).ravel()
        if len(self.clip_scores) != self.clip_labels.size:
            raise ValueError("clip_scores and clip_labels must align")
        if len(self.clip_ids) != len(self.clip_scores):
            raise ValueError("clip_ids and clip_scores must align")
        if self.context is not None and len(self.context) != len(self.clip_scores):
            raise ValueError("context and clip_scores must align")

    # -- summaries --------------------------------------------------------- #
    @property
    def n_clips(self) -> int:
        return len(self.clip_scores)

    @property
    def n_frames(self) -> int:
        return int(sum(len(s) for s in self.clip_scores))

    @property
    def n_positive(self) -> int:
        return int((self.clip_labels == 1).sum())

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "clips": self.n_clips,
            "frames": self.n_frames,
            "positive_clips": self.n_positive,
            "fps": self.fps,
        }

    def split(self, frac: float = 0.5, seed: int = 0) -> tuple:
        """Split clips into two disjoint corpora (calibration / evaluation)."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_clips)
        cut = int(round(frac * self.n_clips))
        return self._subset(idx[:cut]), self._subset(idx[cut:])

    def _subset(self, idx: Sequence[int]) -> "ScoredCorpus":
        idx = list(int(i) for i in idx)
        return ScoredCorpus(
            clip_scores=[self.clip_scores[i] for i in idx],
            clip_labels=self.clip_labels[idx],
            clip_ids=[self.clip_ids[i] for i in idx],
            context=None if self.context is None else [self.context[i] for i in idx],
            name=self.name,
            fps=self.fps,
        )


def write_scored_corpus(corpus: ScoredCorpus, path: str | Path) -> Path:
    """Serialise a corpus to a single ``.npz`` file.

    Clips are stored concatenated with an offset index so that a corpus of tens
    of thousands of clips loads in one read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(s) for s in corpus.clip_scores], dtype=np.int64)
    flat = np.concatenate(corpus.clip_scores) if corpus.n_clips else np.zeros(0)
    payload = {
        "scores": flat.astype(np.float32),
        "lengths": lengths,
        "labels": corpus.clip_labels.astype(np.int64),
        "ids": np.array(corpus.clip_ids, dtype=object),
        "name": np.array(corpus.name),
        "fps": np.array(corpus.fps),
    }
    if corpus.context is not None:
        payload["context"] = np.vstack(corpus.context).astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_scored_corpus(path: str | Path) -> ScoredCorpus:
    """Read a corpus written by :func:`write_scored_corpus`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"scored corpus not found at {path}. Produce it once by running your "
            "frozen detector over the corpus and calling write_scored_corpus(); "
            "see sentinel_e/datasets/adapters.py for the per-dataset recipe."
        )
    with np.load(path, allow_pickle=True) as z:
        lengths = z["lengths"]
        offs = np.concatenate([[0], np.cumsum(lengths)])
        scores = z["scores"]
        clips = [scores[offs[i]:offs[i + 1]].astype(float) for i in range(len(lengths))]
        context = None
        if "context" in z.files:
            ctx = z["context"]
            context = [ctx[offs[i]:offs[i + 1]].astype(float) for i in range(len(lengths))]
        return ScoredCorpus(
            clip_scores=clips,
            clip_labels=z["labels"],
            clip_ids=[str(x) for x in z["ids"]],
            context=context,
            name=str(z["name"]),
            fps=float(z["fps"]),
        )


def corpus_from_frame_table(
    frame_ids: Sequence[str],
    clip_ids: Sequence[str],
    scores: Sequence[float],
    clip_labels: Dict[str, int],
    context: Optional[np.ndarray] = None,
    name: str = "corpus",
    fps: float = 25.0,
) -> ScoredCorpus:
    """Build a corpus from a flat, frame-level table.

    This is the convenient entry point when a backbone is run with a simple
    ``for frame in video`` loop that appends ``(frame_id, clip_id, score)``
    rows.  Frames are grouped by ``clip_id`` preserving their order of
    appearance, which for a sequential pass over a video is temporal order.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if not (len(frame_ids) == len(clip_ids) == scores.size):
        raise ValueError("frame_ids, clip_ids and scores must have equal length")
    ctx = None if context is None else np.atleast_2d(np.asarray(context, dtype=float))
    if ctx is not None and len(ctx) != scores.size:
        raise ValueError("context rows must match number of frames")

    order: List[str] = []
    buckets: Dict[str, List[int]] = {}
    for i, c in enumerate(clip_ids):
        if c not in buckets:
            buckets[c] = []
            order.append(c)
        buckets[c].append(i)

    missing = [c for c in order if c not in clip_labels]
    if missing:
        raise KeyError(f"no label supplied for clips: {missing[:5]}")

    return ScoredCorpus(
        clip_scores=[scores[buckets[c]] for c in order],
        clip_labels=np.array([clip_labels[c] for c in order], dtype=int),
        clip_ids=order,
        context=None if ctx is None else [ctx[buckets[c]] for c in order],
        name=name,
        fps=fps,
    )

"""Per-unit checkpoint cache for the long-running experiments.

Each experiment is split into units (one Monte-Carlo sweep cell, one
``evaluate`` call, ...).  When a unit finishes its result is written to
``results/cache/<script>/<unit_key>.json``; on the next run the unit is loaded
instead of recomputed.  Writes are atomic (temp file + ``os.replace``), so a
container restart can never leave a half-written unit behind.

A cache file stores the unit's parameters next to its result.  A cached unit is
used only if its stored parameters equal the parameters the current run would
use, so changing a replicate count or a configuration silently recomputes
instead of mixing results.

Seeds are not handled here: every unit's seeds are fixed functions of the
unit's own arguments (see :func:`experiments.runner.evaluate`), so a unit gives
the same numbers whenever and in whatever order it is computed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

from experiments.common import RESULTS


def safe_name(text: object) -> str:
    """Filesystem-safe fragment of a unit key (``"p-value threshold"`` -> ``p_value_threshold``)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def _np_default(obj):
    if hasattr(obj, "item"):          # numpy scalars and 0-d arrays
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _normalise(obj):
    """Round-trip through JSON so fresh and cached results have identical types."""
    return json.loads(json.dumps(obj, default=_np_default))


class UnitCache:
    """Checkpoint store for one experiment script.

    ``enabled=False`` (``--no-cache``) neither reads nor writes the cache.
    ``smoke=True`` uses ``<script>_smoke`` so a smoke run never touches the real
    cache.
    """

    def __init__(self, script: str, enabled: bool = True, smoke: bool = False):
        name = f"{script}_smoke" if smoke else script
        self.dir: Path = RESULTS / "cache" / name
        self.enabled = enabled
        self.hits = 0
        self.computed = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _load(self, key: str, params) -> Optional[Dict]:
        try:
            doc = json.loads(self._path(key).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict) or doc.get("params") != params:
            return None
        return doc.get("result")

    def _store(self, key: str, params, result) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        doc = {"unit": key, "params": params, "result": result}
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(doc, fh, default=_np_default)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get_or_compute(self, key: str, params: Dict, fn: Callable[[], Dict]) -> Dict:
        """Return the cached result for ``key`` or compute, store and return it."""
        params = _normalise(params)
        if self.enabled:
            cached = self._load(key, params)
            if cached is not None:
                self.hits += 1
                print(f"    [cache] {key}", flush=True)
                return cached
        result = _normalise(fn())
        self.computed += 1
        if self.enabled:
            self._store(key, params, result)
        return result

"""Shared harness for the SENTINEL-E experiments.

Every experiment follows the same shape: sweep a method's tuning knob, run many
independent Monte-Carlo streams under the null and under a change, and report
the *realised* operating point rather than the nominal one.  Streams are
independent, so the work is embarrassingly parallel and is farmed out to a
process pool.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
for _d in (RESULTS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_WORKERS = max(1, min(os.cpu_count() or 1, 8))


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(payload, name: str) -> Path:
    """Write a result payload to ``results/<name>.json`` with provenance."""
    path = RESULTS / f"{name}.json"
    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "payload": _jsonable(payload),
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def load_json(name: str):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text())["payload"]


# --------------------------------------------------------------------------- #
# Parallel Monte Carlo
# --------------------------------------------------------------------------- #
def parallel_map(
    fn: Callable,
    args: Sequence,
    workers: Optional[int] = None,
    desc: str = "",
) -> List:
    """Map ``fn`` over ``args`` in a process pool, preserving order."""
    workers = DEFAULT_WORKERS if workers is None else int(workers)
    args = list(args)
    if workers <= 1 or len(args) <= 1:
        return [fn(a) for a in args]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(fn, args))
    if desc:
        print(f"    {desc}: {len(args)} runs in {time.time() - t0:.1f}s "
              f"on {workers} workers", flush=True)
    return out


# --------------------------------------------------------------------------- #
# LaTeX table emission
# --------------------------------------------------------------------------- #
def write_latex_table(
    rows: Sequence[Sequence[object]],
    header: Sequence[str],
    caption: str,
    label: str,
    name: str,
    align: Optional[str] = None,
    notes: str = "",
) -> Path:
    """Emit a booktabs table to ``results/tables/<name>.tex``."""
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(str(h) for h in header) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join("" if v is None else str(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\begin{minipage}{\linewidth}\vspace{2pt}\footnotesize " + notes + r"\end{minipage}")
    lines.append(r"\end{table}")
    path = TABLES / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def fmt(x, nd: int = 3, dash: str = "--") -> str:
    """Format a float for a table, rendering non-finite values as an em dash."""
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return dash
    return f"{v:.{nd}f}"


def fmt_int(x, dash: str = "--") -> str:
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return dash
    return f"{int(round(v)):,}"


# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
def setup_matplotlib():
    """Consistent, print-ready styling for every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )
    return plt


#: Stable colour and marker assignment so a method looks the same in every figure.
METHOD_STYLE: Dict[str, Dict[str, object]] = {
    "SENTINEL-E":            {"color": "#0b3d91", "marker": "o", "zorder": 5},
    "SENTINEL-E (no graph)": {"color": "#3f7fd1", "marker": "s", "zorder": 4},
    "SENTINEL-E (graph)":    {"color": "#0b3d91", "marker": "o", "zorder": 5},
    "SENTINEL-E (heuristic graph)": {"color": "#7ab3f5", "marker": "^", "zorder": 4},
    "Fixed threshold":       {"color": "#c1272d", "marker": "v", "zorder": 2},
    "p-value threshold":     {"color": "#e07b39", "marker": "<", "zorder": 2},
    "CUSUM":                 {"color": "#2e7d32", "marker": "D", "zorder": 3},
    "Shiryaev--Roberts":     {"color": "#7b1fa2", "marker": "^", "zorder": 3},
    "Parametric e-detector": {"color": "#795548", "marker": "P", "zorder": 3},
}


def style_for(name: str) -> Dict[str, object]:
    return dict(METHOD_STYLE.get(name, {"color": "#555555", "marker": "x", "zorder": 2}))

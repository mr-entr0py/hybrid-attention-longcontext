"""Figure style and provenance for the protocol plots (protocol §12).

* **k** (number of F_g layers, ordinal) is coloured with **cividis** (perceptually uniform, CVD-safe);
  :func:`k_color` maps k in [0, 12] to a fixed colour so a config never changes colour between figures.
* **Families** use **Okabe–Ito**: retrieval blue, multi-hop vermillion, aggregation bluish green, each
  also with its own marker so colour never carries identity alone.
* **Every seed is drawn** (:func:`plot_seeds`): thin per-seed markers/lines under the mean.
* Length axes are **log2** (:func:`log2_length_axis`); the extrapolation region is shaded
  (:func:`shade_extrapolation`).
* :func:`savefig` writes vector PDF, a 200-dpi PNG and ``<name>.inputs.json`` listing the SHA-256 of
  every input file (results JSONL and the registry), the code SHA-256 and the CI type in the caption.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402

OKABE_ITO: dict[str, str] = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
FAMILY_COLOR: dict[str, str] = {"retrieval": OKABE_ITO["blue"], "multihop": OKABE_ITO["vermillion"],
                                "aggregation": OKABE_ITO["bluish_green"]}
FAMILY_MARKER: dict[str, str] = {"retrieval": "o", "multihop": "s", "aggregation": "^"}
FAMILY_LABEL: dict[str, str] = {"retrieval": "Retrieval", "multihop": "Multi-hop", "aggregation": "Aggregation"}
INK, INK_MUTED, GRID = "#1a1a1a", "#5c5c5c", "#e3e3e3"
_CIVIDIS = colormaps["cividis"]


def k_color(k: int | float, k_max: int = 12) -> tuple[float, float, float, float]:
    """Fixed cividis colour for k. Uses the 0.05–0.80 range so the light end stays visible on white."""
    x = np.log1p(float(k)) / np.log1p(k_max)  # k is spread 0,1,2,3,6,12: log spacing separates small k
    return _CIVIDIS(0.05 + 0.75 * x)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def new_figure(ncols: int = 1, nrows: int = 1, width: float = 3.4, height: float = 2.6, **kw: Any):
    apply_style()
    return plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False, **kw)


def log2_length_axis(ax, lengths: Sequence[int], axis: str = "x") -> None:
    """log2 axis with ticks at the evaluated lengths, labelled 1K, 2K, ..."""
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    set_scale("log", base=2)
    ticks = sorted(set(int(t) for t in lengths))
    labels = [f"{t // 1024}K" if t >= 1024 else str(t) for t in ticks]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, labels)
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)("Context length T (log2)")


def shade_extrapolation(ax, l_ext: int | None, t_max: int = 32768) -> None:
    """Shade T > L_ext (only if L_ext = 16K was chosen, §13.3 FEAS fallback)."""
    if l_ext is not None and l_ext < t_max:
        ax.axvspan(l_ext * 1.0001, t_max * 1.2, color="#f2f2f2", zorder=0, lw=0)
        ax.text(l_ext * 1.05, 0.98, "extrapolation", transform=ax.get_xaxis_transform(), va="top",
                fontsize=7, color=INK_MUTED)


def plot_seeds(ax, x: Sequence[float], per_seed: Mapping[Any, Sequence[float]] | np.ndarray, *, color,
               marker: str = "o", label: str | None = None, connect: bool = True, jitter: float = 0.0,
               mean: bool = True) -> None:
    """Draw every seed (thin, translucent) and the seed mean (solid). ``per_seed``: seed -> y values."""
    rows = list(per_seed.values()) if isinstance(per_seed, Mapping) else list(np.atleast_2d(per_seed))
    x = np.asarray(x, float)
    for i, y in enumerate(rows):
        xx = x * (1 + jitter * (i - (len(rows) - 1) / 2)) if jitter else x
        ax.plot(xx, y, ls="-" if connect else "none", lw=0.8, marker=marker, ms=3, color=color, alpha=0.45,
                zorder=2)
    if mean and rows:
        m = np.nanmean(np.vstack([np.asarray(r, float) for r in rows]), axis=0)
        ax.plot(x, m, ls="-" if connect else "none", lw=2.0, marker=marker, ms=6, color=color, label=label,
                zorder=3, markeredgecolor="white", markeredgewidth=0.8)


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def savefig(fig, name: str, inputs: Iterable[str | Path], *, out_dir: str | Path = "plots/out",
            caption: str = "", ci_type: str = "", extra: Mapping[str, Any] | None = None,
            code_file: str | Path | None = None) -> dict[str, Any]:
    """Write ``<out_dir>/<name>.pdf`` (vector), ``.png`` and ``.inputs.json`` (input SHA-256s).

    The JSON records the sorted input files with their SHA-256, an aggregate hash over them, the
    generating script's SHA-256, the CI type stated in the caption and any ``extra`` metadata.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted({str(p) for p in inputs})
    hashes = {p: _sha(p) for p in files if Path(p).is_file()}
    agg = hashlib.sha256("".join(f"{p}:{h}\n" for p, h in sorted(hashes.items())).encode()).hexdigest()
    meta = dict(name=name, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_inputs=len(hashes),
                inputs_sha256=agg, inputs=hashes, caption=caption, ci_type=ci_type,
                code_sha256=_sha(code_file) if code_file and Path(code_file).is_file() else None,
                **(dict(extra) if extra else {}))
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    (out / f"{name}.inputs.json").write_text(json.dumps(meta, indent=1))
    plt.close(fig)
    return meta


def placeholder(ax, msg: str) -> None:
    """Empty-state panel used when a figure's inputs do not exist yet."""
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=INK_MUTED, fontsize=8, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

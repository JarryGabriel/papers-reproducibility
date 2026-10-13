"""Shared publication-quality matplotlib style for the paper figures.

Single source of truth so every figure in the paper looks consistent. The
rcParams follow the "Professional Publication Style" of the matplotlib skill
(sans-serif, thin spines, inward ticks, dpi 300 on save), with graceful font
fallback (Arial -> DejaVu Sans) so it renders the same headless.

Usage::
    import paper_style as PS
    PS.use()                       # apply rcParams (call once at import time)
    fig, ax = PS.figure()          # constrained-layout figure
    PS.save(fig, "name", channel)  # 300-dpi save + close

Palettes:
    PS.ALGO[algo]      front-end colours (bilateral / savgol / varthresh)
    PS.CHANNEL[ch]     per-channel accent colour
    PS.type_colors(n)  qualitative ramp for the n typecodes
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display, deterministic rendering
import matplotlib.pyplot as plt
from opensky26_plateau import paths as P

# ----------------------------------------------------------------------------
# Palettes — qualitative for front-ends/channels (tab10-derived, colourblind ok)
# ----------------------------------------------------------------------------
# Front-ends: the three candidate detectors. Stable colours across every figure.
ALGO = {
    "bilateral": "#0173B2",   # blue   (the operator that wins resolved channels)
    "savgol":    "#DE8F05",   # orange (fidelity-leader that over-segments)
    "varthresh": "#029E73",   # green  (the gaming baseline)
    "pelt":      "#949494",   # grey   (external off-the-shelf baseline)
}
ALGO_LABEL = {"bilateral": "bilateral", "savgol": "Savitzky-Golay",
              "varthresh": "variance-threshold", "pelt": "PELT (external)"}

# Per-channel accent (used for single-channel detection plots).
CHANNEL = {
    "vz":    "#0173B2",
    "alt":   "#CC78BC",
    "gamma": "#029E73",
    "cas":   "#DE8F05",
    "mach":  "#D55E00",
}

# Selected-operator band/line (committed plateaus on detection plots).
BAND = "#ffd8a8"
LINE = "#e8800a"
RAW = "0.45"        # raw signal grey
EXCL = "0.85"       # excluded-zone grey span


def use() -> None:
    """Apply the publication rcParams. Idempotent; call once."""
    plt.rcParams.update({
        # Figure
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        # Font (graceful fallback so headless == local)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        # Axes
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(
            color=["#0173B2", "#DE8F05", "#029E73", "#CC78BC",
                   "#D55E00", "#949494"]),
        # Grid
        "grid.color": "0.8",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        # Ticks (inward, publication look)
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        # Legend
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
    })


def figure(nrows: int = 1, ncols: int = 1, figsize=None, **kw):
    """Constrained-layout figure with the OO interface. Returns (fig, ax) for a
    single panel, (fig, 1-D array) for a single row/column, and (fig, 2-D array)
    for a true grid --- matching matplotlib's default squeeze behaviour so
    callers can unpack a row of axes directly."""
    if figsize is None:
        figsize = (4.2 * ncols, 3.4 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


def type_colors(n: int):
    """Qualitative colour ramp for n typecodes (tab10/tab20, distinct)."""
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return [cmap(i % cmap.N) for i in range(n)]


def save(fig, name: str, channel: str | None = None) -> str:
    """Save at 300 dpi to the channel's figures/ dir; close the figure (no memory
    leak) and return the path. The LaTeX png/ copy is handled by the caller."""
    out = (P.figures_dir(channel) if channel else P.REPO / "figures")
    out.mkdir(parents=True, exist_ok=True)
    main = out / f"{name}.png"
    fig.savefig(main)
    plt.close(fig)
    return str(main)

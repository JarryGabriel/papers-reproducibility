"""The figures for what an operating point costs across airframe populations.

Three panels, one per question the manifest committed to:

  fig_dispersion    — where each fold's frozen point lands, per channel, against
                      the budget it must respect. The spread IS the measurement.
  fig_deficit       — what re-selecting on each fold alone would have bought.
  fig_margin_ratio  — the dispersion-over-margin ratio, against the ordering the
                      archived round reported, so a reader sees it fail to
                      reproduce rather than reading that it did.

Reproduce:
    uv run --active python scripts/04_figures.py
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = EXPERIMENT_DIR / "outputs"
FIGURES = EXPERIMENT_DIR / "figures"

CHANNELS = ("alt", "gamma", "vz", "mach", "cas")
LABEL = {
    "alt": "altitude",
    "gamma": "flight-path\nangle",
    "vz": "vertical\nspeed",
    "mach": "Mach",
    "cas": "CAS/IAS",
}
#: One colour per fold, kept identical across all three figures so a reader can
#: follow a single population of airframes from panel to panel. Five hues that
#: stay apart in greyscale as well as in colour — an earlier version put two
#: near-identical navies on folds 0 and 4 and they could not be told apart.
FOLD_COLOURS = ("#0b2545", "#1b6ca8", "#4bb3a5", "#e8a33d", "#b5443f")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def fig_dispersion(held: pl.DataFrame) -> None:
    """Budget use per fold, per channel — the spread against the constraint.

    Plotted as a share of budget rather than in physical units: the five
    channels have five incommensurable units (metres, radians, ft/min, knots,
    dimensionless), and what matters is how close each sits to the line it may
    not cross.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for x, channel in enumerate(CHANNELS):
        rows = held.filter(pl.col("channel") == channel).sort("fold")
        used = rows["budget_used_pct"].to_list()
        # A single vertical rule spanning the fold values: the range is the
        # measurement, and the earlier bracketed caps only drew attention to
        # whichever point happened to be extreme. Centred on the dot cluster —
        # the dots are offset by (fold - 2) * 0.12, so their centre is x, and
        # drawing the rule anywhere else makes it read as a sixth series.
        ax.plot([x, x], [min(used), max(used)], color="#d8d8d8", lw=1.4, zorder=1)
        for fold, value in enumerate(used):
            ax.scatter(
                x + (fold - 2) * 0.12,
                value,
                s=46,
                color=FOLD_COLOURS[fold],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=f"fold {fold}" if x == 0 else None,
            )

    ax.axhline(100.0, color="#c1121f", lw=1.2, ls="--", zorder=1)
    ax.text(4.45, 101.5, "budget", color="#c1121f", fontsize=8, ha="right")
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([LABEL[c] for c in CHANNELS])
    ax.set_ylabel("worst segment's error,\nas a share of budget (%)")
    ax.set_ylim(50, 106)
    ax.set_title(
        "Where each fold's frozen point lands against its budget", fontsize=10, loc="left"
    )
    # Below the axes: inside the plot it sat on Mach's cluster, which is the
    # lowest and the most interesting.
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.45))
    fig.savefig(FIGURES / "fig_dispersion.png")
    plt.close(fig)


def fig_deficit(refit: pl.DataFrame) -> None:
    """What re-selecting on each fold alone would have bought, in coverage pts.

    A log x-axis: the deficits span two orders of magnitude, and on a linear
    axis the four small channels collapse onto zero and stop being readable.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    order = list(reversed(CHANNELS))
    for y, channel in enumerate(order):
        rows = refit.filter(pl.col("channel") == channel).sort("fold")
        values = [v or 0.0 for v in rows["deficit_pts"].to_list()]
        same = rows["same_setting"].to_list()
        # The dots straddle y by (fold - 2) * 0.12, so the band must be centred
        # on y and tall enough to hold all five, or it reads as an offset row.
        ax.axhspan(y - 0.3, y + 0.3, color="#f4f4f4", zorder=0)
        for fold, (value, is_same) in enumerate(zip(values, same)):
            ax.scatter(
                value,
                y + (fold - 2) * 0.12,
                s=46,
                color=FOLD_COLOURS[fold],
                marker="D" if is_same else "o",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=f"fold {fold}" if y == 0 else None,
            )

    # Linear, not log: a third of the deficits are exactly zero, and a log axis
    # has no place for zero — the previous version pinned them to 0.01, which
    # put a fabricated position on the one value that matters most here.
    ax.set_xlim(-0.45, 14.0)
    ax.axvline(0.0, color="#bbb", lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.15)
    ax.set_xlabel("coverage re-selection would have bought (points)")
    ax.set_title(
        "The cost of freezing, per fold\n"
        "◆ = re-selection returned the very same setting (deficit 0)",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.grid(axis="x", color="#eee", zorder=0)
    fig.savefig(FIGURES / "fig_deficit.png")
    plt.close(fig)


def margin_ratios(held: pl.DataFrame) -> dict[str, float | None]:
    """Reproduce the emitter's rounded ratios from public per-fold measurements."""
    ratios = {}
    for channel in CHANNELS:
        rows = held.filter(pl.col("channel") == channel)
        errors = rows["worst_segment_error"].to_list()
        margin = rows["budget"][0] - median(errors)
        ratios[channel] = round((max(errors) - min(errors)) / margin, 4) if margin > 0 else None
    return ratios


def fig_margin_ratio(held: pl.DataFrame) -> None:
    """Dispersion over margin, this round against the archived one.

    The archived round reported this ratio ordering its channels' failures
    without exception. Showing both makes the non-reproduction the figure's
    subject rather than a sentence a reader has to take on trust.
    """
    archived = {"vz": 2.32, "cas": 1.30, "mach": 0.61, "gamma": 0.19}
    here = margin_ratios(held)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.36
    labelled = False
    for x, channel in enumerate(CHANNELS):
        ratio = here[channel]
        ax.bar(
            x - width / 2,
            ratio,
            width,
            color="#1b4965",
            label="this round" if x == 0 else None,
            zorder=3,
        )
        ax.text(x - width / 2, ratio + 0.04, f"{ratio:.2f}", ha="center", fontsize=7.5)
        if channel in archived:
            # Label the first archived bar that actually exists — altitude has
            # none (it breached), so keying the legend to x == 0 silently lost
            # the entry and left the pale bars unexplained.
            ax.bar(
                x + width / 2,
                archived[channel],
                width,
                color="#9ecae1",
                label=None if labelled else "archived round",
                zorder=3,
            )
            labelled = True
            ax.text(
                x + width / 2,
                archived[channel] + 0.04,
                f"{archived[channel]:.2f}",
                ha="center",
                fontsize=7.5,
            )
        else:
            # Altitude had negative margin in the archived round: the ratio was
            # not large there, it was undefined, and a missing bar says so more
            # honestly than a tall one would.
            ax.text(
                x + width / 2,
                0.06,
                "breached\n(no margin)",
                ha="center",
                fontsize=7,
                color="#c1121f",
                rotation=90,
                va="bottom",
            )

    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([LABEL[c] for c in CHANNELS])
    ax.set_ylabel("error spread ÷ margin on budget")
    ax.set_title(
        "The ratio the archived round found ordering every failure —\nand what it does here",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#eee", zorder=0)
    fig.savefig(FIGURES / "fig_margin_ratio.png")
    plt.close(fig)


def main() -> int:
    held = pl.read_csv(OUTPUTS / "heldout.csv")
    refit = pl.read_csv(OUTPUTS / "reselected.csv")

    FIGURES.mkdir(exist_ok=True)
    fig_dispersion(held)
    fig_deficit(refit)
    fig_margin_ratio(held)
    for name in ("fig_dispersion", "fig_deficit", "fig_margin_ratio"):
        print(f"wrote {FIGURES / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

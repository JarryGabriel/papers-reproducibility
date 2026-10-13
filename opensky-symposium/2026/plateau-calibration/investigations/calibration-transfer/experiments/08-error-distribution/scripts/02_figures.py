"""Two figures: the distribution the maximum was standing for, and the breaches.

  fig_error_distribution — the per-segment error as a share of budget, per
                           channel and per date, with the median and the maximum
                           on the same axis. The gap between them IS the finding.
  fig_breach_share       — how few segments breach, and how little coverage they
                           carry, against what the maximum implied.

Reproduce:
    uv run --active python scripts/02_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

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
CALIBRATION = "2025-11-01"
DATES = ("2025-02-17", "2025-05-21", "2025-08-29", "2025-11-01")
SHORT = {"2025-02-17": "Feb", "2025-05-21": "May",
         "2025-08-29": "Aug", "2025-11-01": "Nov (calib.)"}
DATE_COLOURS = {"2025-02-17": "#0b2545", "2025-05-21": "#1b6ca8",
                "2025-08-29": "#4bb3a5", "2025-11-01": "#b0b0b0"}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


#: The manuscript's text block, measured from the venue class: 404 pt, which is
#: 5.59 in. A figure generated wider than this is scaled DOWN by LaTeX and its
#: fonts shrink with it, so it is generated at its final width instead.
TEXT_IN = 5.59

def fig_error_distribution(seg: pl.DataFrame) -> None:
    """Median, p95 and maximum on one axis, per channel and per date.

    Log scale on the y axis: the medians sit near a tenth of budget and the
    maxima above it, and on a linear axis the medians collapse into the floor —
    which is exactly the thing a reader must see has not moved.
    """
    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.9))

    for x, channel in enumerate(CHANNELS):
        rows = seg.filter(pl.col("channel") == channel)
        for k, date in enumerate(DATES):
            here = rows.filter(pl.col("date") == date)["error_over_budget"]
            offset = (k - 1.5) * 0.15
            median = float(here.median())
            p95 = float(here.quantile(0.95))
            top = float(here.max())
            # A stem from the median to the maximum: the maximum is what the
            # earlier experiments recorded, the median is what they discarded.
            ax.plot([x + offset] * 2, [median, top], color="#d8d8d8", lw=1.2,
                    zorder=1)
            ax.scatter(x + offset, top, s=26, marker="_",
                       color=DATE_COLOURS[date], zorder=3)
            ax.scatter(x + offset, p95, s=14, marker="_", color="#999",
                       zorder=2)
            ax.scatter(x + offset, median, s=42, color=DATE_COLOURS[date],
                       edgecolor="white", linewidth=0.5, zorder=4,
                       label=SHORT[date] if x == 0 else None)

    ax.axhline(1.0, color="#c1121f", lw=1.2, ls="--", zorder=2)
    ax.text(4.45, 1.06, "budget", color="#c1121f", fontsize=8, ha="right")
    ax.set_yscale("log")
    # Ticks stop at 2.0 but the axis must not: CAS/IAS reaches 6.3x budget on
    # one segment of 13 969 in May, and clipping the stem would hide exactly
    # the spread this figure exists to show.
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([LABEL[c] for c in CHANNELS])
    ax.set_ylabel("per-segment error,\nas a share of budget (log)")
    # The narration the title carried is the caption's job; what stays is the
    # key, which names the marks and belongs on the drawing.
    ax.set_title("● median   ─ p95 (grey) and maximum (coloured)",
                 fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, -0.40))
    ax.grid(axis="y", color="#f4f4f4", zorder=0)
    fig.savefig(FIGURES / "fig_error_distribution.png")
    plt.close(fig)


def fig_breach_share(seg: pl.DataFrame) -> None:
    """How few segments breach, and how little coverage they carry.

    Two panels because they are two different quantities: a count of segments,
    and the share of the committed trajectory those segments hold. The second is
    weighted by finite samples, never by span.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    for x, channel in enumerate(CHANNELS):
        rows = seg.filter(pl.col("channel") == channel)
        for k, date in enumerate(DATES):
            here = rows.filter(pl.col("date") == date)
            over = here.filter(pl.col("error_over_budget") > 1.0)
            offset = (k - 1.5) * 0.15
            share = over.height / here.height * 100.0
            coverage = (over["n_samples"].sum() / here["n_samples"].sum()
                        * 100.0) if here.height else 0.0
            left.scatter(x + offset, share, s=44, color=DATE_COLOURS[date],
                         edgecolor="white", linewidth=0.5, zorder=3,
                         label=SHORT[date] if x == 0 else None)
            right.scatter(x + offset, coverage, s=44, color=DATE_COLOURS[date],
                          edgecolor="white", linewidth=0.5, zorder=3)

    for ax, title, ylabel in (
        (left, "Segments in breach", "share of committed segments (%)"),
        (right, "What they carry", "share of committed coverage (%)"),
    ):
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_xticklabels([LABEL[c] for c in CHANNELS], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(-0.008, None)
        ax.grid(axis="y", color="#f4f4f4", zorder=0)
        ax.axhline(0.0, color="#ccc", lw=0.8, zorder=1)

    total = seg.filter(pl.col("error_over_budget") > 1.0).height
    fig.suptitle(f"{total} of {seg.height} committed segments exceed their "
                 f"budget ({total / seg.height * 100:.3f} %)",
                 fontsize=10, x=0.005, ha="left")
    # subplots_adjust rather than tight_layout: tight_layout re-fits the axes
    # around the legend and drags it back over the tick labels.
    fig.subplots_adjust(bottom=0.30, top=0.84, wspace=0.28)
    left.legend(frameon=False, fontsize=8, ncol=4,
                loc="upper center", bbox_to_anchor=(1.15, -0.24))
    fig.savefig(FIGURES / "fig_breach_share.png")
    plt.close(fig)


def main() -> int:
    seg = pl.read_csv(OUTPUTS / "segments.csv")
    FIGURES.mkdir(exist_ok=True)
    fig_error_distribution(seg)
    fig_breach_share(seg)
    for name in ("fig_error_distribution", "fig_breach_share"):
        print(f"wrote {FIGURES / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

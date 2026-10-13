"""Two figures: how the difference is distributed, and who disagrees.

  fig_difference    — the signed difference between the two observers, in feet
                      and in units of the 25 ft step, over all paired plateaux.
  fig_disagreement  — what distinguishes the plateaux beyond one flight level,
                      described rather than explained.

Reproduce:
    uv run --active python scripts/02_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = EXPERIMENT_DIR / "outputs"
FIGURES = EXPERIMENT_DIR / "figures"

STEP_FT = 25.0
FLIGHT_LEVEL_FT = 100.0
AGREE = "#1b6ca8"
DISAGREE = "#b5443f"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


#: The manuscript's text block, measured from the venue class: 404 pt, which is
#: 5.59 in. A figure generated wider than this is scaled DOWN by LaTeX and its
#: fonts shrink with it, so it is generated at its final width instead.
TEXT_IN = 5.59

def fig_difference(paired: pl.DataFrame) -> None:
    """The signed difference, zoomed to where the mass is and stated where not.

    Clipped at +/- 300 ft with the overflow counted in the annotation rather
    than hidden: a histogram spanning the full range would be one bar at zero
    and a flat line, which shows nothing.
    """
    values = paired["difference_ft"].to_numpy()
    limit = 300.0
    inside = values[np.abs(values) <= limit]
    beyond = int((np.abs(values) > limit).sum())

    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.8))
    ax.hist(
        inside,
        bins=np.arange(-limit, limit + STEP_FT, STEP_FT),
        color=AGREE,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for edge in (-FLIGHT_LEVEL_FT, FLIGHT_LEVEL_FT):
        ax.axvline(edge, color=DISAGREE, lw=1.2, ls="--", zorder=4)
    ax.axvline(0.0, color="#333", lw=0.9, zorder=4)

    ax.set_xlabel("plateau value − broadcast selected altitude (ft)")
    ax.set_ylabel("plateaux")
    # The median and the within-one-level share belong to the prose, not here:
    # a number printed in both reads as two results. What DOES stay is the
    # count clipped off the axis — without it the histogram misrepresents its
    # own population — and it sits inside the axes as an annotation rather than
    # in the title, so a reader meets it as a property of the drawing.
    ax.annotate(
        f"{beyond} plateaux beyond\n±{limit:.0f} ft not drawn",
        # Bottom left and wrapped onto two lines: the zero line and the two
        # dashed edges run the full height of the axes and the central bars
        # reach the top, so a single long line meets the drawing at any height
        # it is placed. Two short lines fit in the empty left tail.
        xy=(0.02, 0.30),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        color="#555",
    )
    ax.grid(axis="y", color="#f4f4f4", zorder=0)

    # A second axis in units of the broadcast step, so a reader sees that most
    # of the mass sits inside one step — below what the data can resolve.
    top = ax.secondary_xaxis("top", functions=(lambda x: x / STEP_FT, lambda x: x * STEP_FT))
    top.set_xlabel("in units of the 25 ft quantisation step", fontsize=8)

    fig.savefig(FIGURES / "fig_difference.png")
    plt.close(fig)


def fig_disagreement(paired: pl.DataFrame) -> None:
    """What distinguishes the plateaux beyond one flight level.

    Four panels, one per declared descriptor. Described, not explained: nothing
    here attributes a cause, and the manifest says so.
    """
    agree = paired.filter(pl.col("within_flight_level"))
    disagree = paired.filter(~pl.col("within_flight_level"))

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0))

    # 1. where in the flight
    axes[0].hist(
        [agree["position_in_flight"].to_numpy(), disagree["position_in_flight"].to_numpy()],
        bins=np.linspace(0, 1, 11),
        density=True,
        color=[AGREE, DISAGREE],
        zorder=3,
    )
    axes[0].set_xlabel("position in flight")
    axes[0].set_ylabel("density")
    axes[0].set_title("Where", fontsize=9, loc="left")

    # 2. how long
    for frame, colour in ((agree, AGREE), (disagree, DISAGREE)):
        axes[1].hist(
            np.log10(frame["duration_s"].to_numpy()),
            bins=20,
            density=True,
            histtype="step",
            lw=1.6,
            color=colour,
            zorder=3,
        )
    axes[1].set_xlabel("duration (log₁₀ s)")
    axes[1].set_title("How long", fontsize=9, loc="left")

    # 3. at what altitude
    for frame, colour in ((agree, AGREE), (disagree, DISAGREE)):
        axes[2].hist(
            frame["plateau_alt_ft"].to_numpy() / 1000.0,
            bins=20,
            density=True,
            histtype="step",
            lw=1.6,
            color=colour,
            zorder=3,
        )
    axes[2].set_xlabel("plateau altitude (kft)")
    axes[2].set_title("How high", fontsize=9, loc="left")

    # 4. does the register move under it
    shares = [
        float(agree["mcp_changes_within"].mean()) * 100.0,
        float(disagree["mcp_changes_within"].mean()) * 100.0,
    ]
    axes[3].bar([0, 1], shares, color=[AGREE, DISAGREE], width=0.6, zorder=3)
    for x, share in zip((0, 1), shares):
        axes[3].text(x, share + 1.5, f"{share:.0f} %", ha="center", fontsize=8)
    axes[3].set_xticks([0, 1])
    axes[3].set_xticklabels(
        [f"agree\nn={agree.height}", f"disagree\nn={disagree.height}"], fontsize=8
    )
    axes[3].set_ylabel("register changes\nmid-plateau (%)")
    axes[3].set_ylim(0, max(shares) * 1.3)
    axes[3].set_title("Does it move", fontsize=9, loc="left")

    for ax in axes:
        ax.grid(axis="y", color="#f4f4f4", zorder=0)

    fig.suptitle(
        "The plateaux beyond one flight level, described — "
        f"{disagree.height} of {paired.height} "
        "(blue: agreeing · red: disagreeing)",
        fontsize=10,
        x=0.005,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "fig_disagreement.png")
    plt.close(fig)


def main() -> int:
    frame = pl.read_csv(OUTPUTS / "paired_plateaux.csv")
    paired = frame.filter(pl.col("mcp_broadcast_ft").is_not_null())

    FIGURES.mkdir(exist_ok=True)
    fig_difference(paired)
    fig_disagreement(paired)
    for name in ("fig_difference", "fig_disagreement"):
        print(f"wrote {FIGURES / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

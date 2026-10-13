"""The figures for what the retained points cost across days.

Three panels:

  fig_by_date       — coverage against budget use, per channel and per date. The
                      point of the figure is that one barely moves while the
                      other leaves the page: the failure is invisible in the
                      quantity the literature reports.
  fig_deficit       — what re-calibrating on each day alone would have bought,
                      with its sign.
  fig_budget_vs_deficit — the two against each other. This is the mechanism: a
                      point over its budget loses coverage to come back inside
                      it, a point within budget gains from re-calibration.

Reproduce:
    uv run --active python scripts/03_figures.py
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
#: The calibration date last and in grey — it is the reference the others are
#: read against, not a fourth observation.
DATE_COLOURS = {"2025-02-17": "#0b2545", "2025-05-21": "#1b6ca8",
                "2025-08-29": "#4bb3a5", "2025-11-01": "#b0b0b0"}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def fig_by_date(carried: pl.DataFrame) -> None:
    """Coverage and budget use side by side — the experiment's whole point.

    Two panels sharing an x axis so a reader compares the SAME channel across
    them: coverage hardly moves between days, budget use leaves the page. A
    detector judged on coverage alone looks healthy on all four dates.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    for x, channel in enumerate(CHANNELS):
        rows = carried.filter(pl.col("channel") == channel)
        for date in DATES:
            row = rows.filter(pl.col("date") == date)
            offset = (DATES.index(date) - 1.5) * 0.15
            left.scatter(x + offset, row["coverage_pct"][0], s=44,
                         color=DATE_COLOURS[date], edgecolor="white",
                         linewidth=0.5, zorder=3,
                         label=SHORT[date] if x == 0 else None)
            right.scatter(x + offset, row["budget_used_pct"][0], s=44,
                          color=DATE_COLOURS[date], edgecolor="white",
                          linewidth=0.5, zorder=3)

    for ax, title in ((left, "Coverage: barely moves"),
                      (right, "Error against budget: leaves the page")):
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_xticklabels([LABEL[c] for c in CHANNELS], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color="#f0f0f0", zorder=0)

    # Log scale on coverage: the five channels span 4 % to 62 %, and on a linear
    # axis the flight-path angle and vertical speed collapse onto the floor —
    # the reader cannot see that they barely move either, which is the claim.
    left.set_yscale("log")
    left.set_yticks([4, 6, 10, 20, 40, 60])
    left.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Silence the minor ticks too, or the log locator writes a stray 3x10^1
    # between the labelled decades.
    left.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    left.set_ylabel("committed coverage (%, log scale)")
    right.set_ylabel("worst segment's error,\nas a share of budget (%)")
    right.axhline(100.0, color="#c1121f", lw=1.2, ls="--", zorder=2)
    right.text(4.4, 106, "budget", color="#c1121f", fontsize=8, ha="right")
    left.legend(frameon=False, fontsize=8, ncol=4,
                loc="lower center", bbox_to_anchor=(1.1, -0.42))
    fig.savefig(FIGURES / "fig_by_date.png")
    plt.close(fig)


def fig_deficit(refit: pl.DataFrame) -> None:
    """What re-calibrating on each day alone would have bought, with its sign.

    Signed, and the zero line drawn: negative is the direction this experiment
    found, and an unsigned magnitude would have hidden it.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    order = list(reversed(CHANNELS))
    for y, channel in enumerate(order):
        rows = refit.filter(pl.col("channel") == channel).sort("date")
        ax.axhspan(y - 0.3, y + 0.3, color="#f4f4f4", zorder=0)
        for date, value in zip(rows["date"].to_list(),
                               rows["deficit_pts"].to_list()):
            ax.scatter(value or 0.0, y + (DATES.index(date) - 1) * 0.15, s=46,
                       color=DATE_COLOURS[date], edgecolor="white",
                       linewidth=0.5, zorder=3,
                       label=SHORT[date] if y == 0 else None)

    ax.axvline(0.0, color="#c1121f", lw=1.2, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("coverage re-calibrating on that day alone would have bought "
                  "(points)")
    ax.set_title("Re-calibrating per day mostly COSTS coverage\n"
                 "left of the red line: the frozen point already covers more",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.46))
    ax.grid(axis="x", color="#f0f0f0", zorder=0)
    fig.savefig(FIGURES / "fig_deficit.png")
    plt.close(fig)


def fig_budget_vs_deficit(carried: pl.DataFrame, refit: pl.DataFrame) -> None:
    """The mechanism: budget use against what re-calibration buys.

    One point per channel-date. Over budget, the rule must give coverage back to
    return inside it; within budget, it has room and re-calibration pays. This
    is the figure that turns a list of numbers into an explanation.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    marker = {"alt": "o", "gamma": "s", "vz": "^", "mach": "D", "cas": "v"}

    for channel in CHANNELS:
        drows = refit.filter(pl.col("channel") == channel)
        for date, deficit in zip(drows["date"].to_list(),
                                 drows["deficit_pts"].to_list()):
            used = carried.filter(
                (pl.col("channel") == channel) & (pl.col("date") == date)
            )["budget_used_pct"][0]
            ax.scatter(used, deficit or 0.0, s=52, marker=marker[channel],
                       color=DATE_COLOURS[date], edgecolor="white",
                       linewidth=0.6, zorder=3)

    ax.axvline(100.0, color="#c1121f", lw=1.2, ls="--", zorder=1)
    ax.axhline(0.0, color="#999", lw=0.9, zorder=1)
    ax.text(101, ax.get_ylim()[1] * 0.86, "over budget →", color="#c1121f",
            fontsize=8)
    ax.set_xlabel("budget the frozen point uses (%)")
    ax.set_ylabel("coverage re-calibration\nwould have bought (points)")
    ax.set_title("Where the frozen point is over budget, re-calibrating buys "
                 "nothing\n— it gives coverage back to return inside it",
                 fontsize=10, loc="left")

    handles = [plt.Line2D([], [], marker=marker[c], color="#555", ls="",
                          label=LABEL[c].replace("\n", " "), markersize=6)
               for c in CHANNELS]
    handles += [plt.Line2D([], [], marker="o", color=DATE_COLOURS[d], ls="",
                           label=SHORT[d], markersize=6) for d in DATES[:3]]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, -0.38))
    ax.grid(color="#f4f4f4", zorder=0)
    fig.savefig(FIGURES / "fig_budget_vs_deficit.png")
    plt.close(fig)


def main() -> int:
    carried = pl.read_csv(OUTPUTS / "by_date.csv")
    refit = pl.read_csv(OUTPUTS / "recalibrated.csv")

    FIGURES.mkdir(exist_ok=True)
    fig_by_date(carried)
    fig_deficit(refit)
    fig_budget_vs_deficit(carried, refit)
    for name in ("fig_by_date", "fig_deficit", "fig_budget_vs_deficit"):
        print(f"wrote {FIGURES / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

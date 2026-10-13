"""What each date's own derivation gives, as a share of the November budget.

One panel. Every channel is plotted as a ratio to the budget experiment 02
locked, so five incommensurable units (metres, radians, ft/min, knots,
dimensionless) sit on one axis and the question — does it move? — is read as
distance from 1.

Reproduce:
    uv run --active python scripts/03_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "outputs"
FIGURES = EXP / "figures"

CHANNELS = ("alt", "gamma", "vz", "mach", "cas")
LABEL = {
    "alt": "altitude",
    "gamma": "flight-path\nangle",
    "vz": "vertical\nspeed",
    "mach": "Mach",
    "cas": "CAS/IAS",
}
SHORT = {"2025-02-17": "Feb", "2025-05-21": "May",
         "2025-08-29": "Aug", "2025-11-01": "Nov (locked)"}
DATE_COLOURS = {"2025-02-17": "#0b2545", "2025-05-21": "#1b6ca8",
                "2025-08-29": "#4bb3a5", "2025-11-01": "#b0b0b0"}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def main() -> int:
    budgets = pl.read_csv(OUT / "budgets_by_date.csv")
    dates = sorted(budgets["date"].unique().to_list())

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for x, channel in enumerate(CHANNELS):
        rows = budgets.filter(pl.col("channel") == channel).sort("date")
        ratios = rows["ratio_to_locked"].to_list()
        # A rule spanning the four values: where it is invisible, the budget
        # does not move, and that is the finding for three channels of five.
        ax.plot([x, x], [min(ratios), max(ratios)], color="#d8d8d8", lw=1.4,
                zorder=1)
        for date, ratio in zip(rows["date"].to_list(), ratios):
            ax.scatter(x + (dates.index(date) - 1.5) * 0.13, ratio, s=48,
                       color=DATE_COLOURS[date], edgecolor="white",
                       linewidth=0.5, zorder=3,
                       label=SHORT[date] if x == 0 else None)

    ax.axhline(1.0, color="#c1121f", lw=1.2, ls="--", zorder=2)
    ax.text(4.45, 1.008, "November budget", color="#c1121f", fontsize=8,
            ha="right")
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([LABEL[c] for c in CHANNELS])
    ax.set_ylabel("budget derived on that date,\nas a share of the locked one")
    ax.set_title("The quantised channels give the same budget on every date;\n"
                 "only the two reconstructed from ERA5 move",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.grid(axis="y", color="#f4f4f4", zorder=0)

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_budgets_by_date.png")
    plt.close(fig)
    print(f"wrote {FIGURES / 'fig_budgets_by_date.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

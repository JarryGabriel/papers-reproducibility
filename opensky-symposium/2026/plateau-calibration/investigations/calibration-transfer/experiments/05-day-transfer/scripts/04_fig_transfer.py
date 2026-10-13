"""fig_transfer — the two transport axes in one object.

The paper's what-freezing-costs subsection draws on three experiments, and the
assertion they share is single: carried to airframes it never saw and to days it
never read, the frozen operating point holds its committed coverage, while the
worst committed segment's error leaves the budget entirely on the day axis. Two
panels rather than two figures, because a reader asked to hold the comparison
across two floats will not make it.

The airframe axis is experiment 04's, read here as a pinned input; the date axis
is this experiment's own.

Reproduce:
    uv run --active python scripts/04_fig_transfer.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = EXPERIMENT_DIR / "outputs"
INPUTS = EXPERIMENT_DIR / "inputs"
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

#: Airframe folds in one warm colour, dates in the cool scale the other figures
#: of this experiment already use, so the two axes stay distinguishable at a
#: glance without a second legend.
FOLD_COLOUR = "#d1793a"
DATE_COLOUR = "#1b6ca8"
CALIB_COLOUR = "#b0b0b0"

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

def _jitter(n: int, width: float = 0.16) -> np.ndarray:
    """Spread n points across a band so overlapping values stay countable."""
    if n == 1:
        return np.zeros(1)
    return np.linspace(-width, width, n)


def main() -> None:
    folds = pl.read_csv(INPUTS / "heldout.csv")
    dates = pl.read_csv(OUTPUTS / "by_date.csv")

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(TEXT_IN, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.12},
    )

    for index, channel in enumerate(CHANNELS):
        fold_rows = folds.filter(pl.col("channel") == channel)
        date_rows = dates.filter(pl.col("channel") == channel)
        held = date_rows.filter(pl.col("date") != CALIBRATION)
        calib = date_rows.filter(pl.col("date") == CALIBRATION)

        # --- top panel: committed coverage, both axes -----------------------
        cov_f = fold_rows["coverage_pct"].to_numpy()
        top.scatter(index - 0.17 + _jitter(len(cov_f), 0.09), cov_f,
                    s=22, color=FOLD_COLOUR, zorder=3,
                    label="airframe folds" if index == 0 else None)
        cov_d = held["coverage_pct"].to_numpy()
        top.scatter(index + 0.17 + _jitter(len(cov_d), 0.09), cov_d,
                    s=22, color=DATE_COLOUR, zorder=3,
                    label="held-out dates" if index == 0 else None)
        cov_c = calib["coverage_pct"].to_numpy()
        top.scatter(np.full(len(cov_c), index + 0.17), cov_c,
                    s=46, marker="_", linewidths=1.8, color="#333333", zorder=4,
                    label="calibration date" if index == 0 else None)

        # --- bottom panel: worst-segment error over budget ------------------
        err_f = fold_rows["budget_used_pct"].to_numpy() / 100.0
        bottom.scatter(index - 0.17 + _jitter(len(err_f), 0.09), err_f,
                       s=22, color=FOLD_COLOUR, zorder=3)
        err_d = held["budget_used_pct"].to_numpy() / 100.0
        bottom.scatter(index + 0.17 + _jitter(len(err_d), 0.09), err_d,
                       s=22, color=DATE_COLOUR, zorder=3)
        err_c = calib["budget_used_pct"].to_numpy() / 100.0
        bottom.scatter(np.full(len(err_c), index + 0.17), err_c,
                       s=46, marker="_", linewidths=1.8, color="#333333", zorder=4)

    top.set_ylabel("committed coverage (%)")
    top.set_ylim(bottom=0)
    top.grid(axis="y", alpha=0.25, linewidth=0.6)
    # Above the panel rather than inside it: the narrating title is gone, so
    # this is the free strip, and anchored in the axes the key sat on the
    # altitude points it exists to explain.
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
               ncols=3, fontsize=8, borderaxespad=0.2)

    bottom.axhline(1.0, color="#c1272d", linestyle="--", linewidth=1.1, zorder=2)
    bottom.text(-0.55, 1.02, "budget $E$", color="#c1272d", fontsize=8,
                ha="left", va="bottom")
    bottom.set_yscale("log")
    # Plain multiples of the budget rather than powers of ten: the reader is
    # asked how many times over budget a point sits, not its order of magnitude.
    bottom.yaxis.set_major_locator(FixedLocator([0.6, 0.8, 1.0, 1.5, 2.0, 2.5]))
    bottom.yaxis.set_minor_locator(NullLocator())
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    bottom.yaxis.set_major_formatter(formatter)
    bottom.set_ylabel("worst segment error\n/ budget (log)")
    bottom.grid(axis="y", alpha=0.25, linewidth=0.6)

    bottom.set_xticks(range(len(CHANNELS)))
    bottom.set_xticklabels([LABEL[c] for c in CHANNELS])
    bottom.set_xlim(-0.6, 4.6)

    fig.savefig(FIGURES / "fig_transfer.png")
    plt.close(fig)
    print(f"wrote {FIGURES / 'fig_transfer.png'}")

    # Echo what the figure asserts, so the caption is written from the render.
    for channel in CHANNELS:
        d = dates.filter(pl.col("channel") == channel)
        f = folds.filter(pl.col("channel") == channel)
        held = d.filter(pl.col("date") != CALIBRATION)
        print(
            f"  {channel:6s} coverage folds "
            f"{f['coverage_pct'].min():.1f}-{f['coverage_pct'].max():.1f}  "
            f"dates {d['coverage_pct'].min():.1f}-{d['coverage_pct'].max():.1f}  "
            f"| budget use held-out {held['budget_used_pct'].min():.1f}-"
            f"{held['budget_used_pct'].max():.1f} %  "
            f"breaching {held.filter(pl.col('budget_used_pct') > 100).height}/3"
        )


if __name__ == "__main__":
    main()

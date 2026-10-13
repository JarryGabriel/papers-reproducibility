"""Two figures, both read from outputs/ so they re-render from the evidence alone.

  * fig_reconstruction — the ERA5 error against the register, and the measurement
    that refutes the wind explanation.
"""
from __future__ import annotations

# Resolve this checkout by its marker, regardless of investigation nesting.
import sys as _sys
from pathlib import Path as _Path

_checkout = next((p for p in _Path(__file__).resolve().parents
                  if (p / "src/opensky26_plateau/project_paths.py").is_file()), None)
if _checkout is None:
    raise FileNotFoundError(f"No project source tree above {__file__}")
_sys.path.insert(0, str(_checkout / "src"))
from opensky26_plateau.project_paths import data_dir, experiment_dir, pipeline_config, project_root


import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

EXP = Path(__file__).resolve().parent.parent
PAPER = project_root(__file__)
OUT = EXP / "outputs"
FIG = EXP / "figures"


try:
    from opensky26_plateau import paper_style  # noqa: F401
except Exception:  # pragma: no cover - styling is cosmetic
    pass

QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
LABEL = {"alt": "altitude", "vz": "vertical speed", "cas": "CAS/IAS", "mach": "Mach"}
UNIT = {"alt": "ft", "vz": "ft/min", "cas": "kt", "mach": "Mach"}


def fig_reconstruction() -> None:
    """The reconstruction error, its size against the budget, and the wind test.

    Three panels: how much of each channel is reconstructed, the error's
    quantiles against the budget round 2 used, and the error binned by wind —
    which is the measurement that settles the wind explanation.
    """
    recon = pl.read_csv(OUT / "reconstruction.csv")
    wind = pl.read_csv(OUT / "wind_dependence.csv")
    budgets = pl.read_csv(OUT / "budgets.csv")

    channels = recon["channel"].to_list()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # Panels 1 and 2: one channel each, in its OWN physical unit. An earlier
    # version put both on one axis normalised by their own budgets, which made
    # the two curves coincide exactly — the normalisation had removed the only
    # thing that distinguished them, and the figure read as a plotting bug.
    for ax, channel in zip(axes[:2], channels):
        row = recon.filter(pl.col("channel") == channel).to_dicts()[0]
        budget_row = budgets.filter(pl.col("channel") == channel).to_dicts()[0]
        values = [float(row[f"p{q}"]) for q in QUANTILES[:-1]]
        ax.plot(QUANTILES[:-1], values, marker="o", ms=4.5, lw=1.8,
                color="#1f4e79", zorder=3)

        derived = float(budget_row["budget"])
        ax.axhline(derived, color="#c0392b", lw=1.6, ls="--", zorder=2)
        # Label ON the line, at the left edge, so nothing floats over the data.
        ax.annotate(f"budget  {derived:.4g} {UNIT[channel]}", xy=(1, derived),
                    xytext=(3, 4), textcoords="offset points", fontsize=8,
                    color="#c0392b", va="bottom")
        # Where the curve crosses the budget is the share of samples the
        # reconstruction cannot resolve within it, which is the number the
        # panel exists to show.
        crossing = float(np.interp(derived, values, QUANTILES[:-1]))
        ax.annotate(f"{100 - crossing:.0f} % of samples exceed it",
                    xy=(crossing, derived), xytext=(6, -14),
                    textcoords="offset points", fontsize=8, color="#c0392b",
                    va="top")

        ax.set_title(
            f"{LABEL[channel]} — the reanalysis error against the budget",
            fontsize=9,
        )
        ax.set_xlabel("quantile of |reconstruction error|")
        ax.set_ylabel(f"|ERA5 − register|  ({UNIT[channel]})")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.annotate(
            f"{float(row['reconstructed_pct']):.1f} % of the channel is "
            f"reconstructed\n{int(row['n_paired']):,} paired samples",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8, color="#555555",
        )

    # 3. the wind test — the panel that refutes the motivating hypothesis. Both
    # channels belong here because what is compared is a SHAPE (flat or not),
    # not a level, so normalising is legitimate.
    ax = axes[2]
    for colour, marker, channel in zip(("#1f4e79", "#c0392b"), ("s", "^"), channels):
        subset = wind.filter((pl.col("channel") == channel)
                             & (pl.col("binned_on") == "wind_kt"))
        if not subset.height:
            continue
        # The median wind of each bin, not the midpoint of its edges: the last
        # bin spans 80-200 kt, whose midpoint of 140 is a speed nothing in it
        # flies at, and plotting there draws a rise the data does not carry.
        centres = subset["bin_centre"].to_list()
        derived = float(budgets.filter(pl.col("channel") == channel)["budget"][0])
        medians = [m / derived for m in subset["median_abs_error"].to_list()]
        r = float(subset["pearson_r_full_range"][0])
        ax.plot(centres, medians, marker=marker, ms=5, lw=1.8, color=colour,
                ls="-" if channel == channels[0] else "--",
                label=f"{LABEL[channel]}   r = {r:+.3f}")
    ax.set_xlabel("wind speed (kt), at each bin's own median")
    ax.set_ylabel("median |error| / derived budget")
    # From zero, so the eye judges the variation against the quantity itself
    # rather than against a cropped window that magnifies it.
    ax.set_ylim(0, 1.0)
    ax.set_title("An eightfold rise in wind moves the error by 7 %", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Seven samples in ten on the speed channels are a reanalysis product, "
        "and the budget is what its error imposes",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig_reconstruction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_reconstruction()
    print(f"wrote {FIG / 'fig_reconstruction.png'}")


if __name__ == "__main__":
    main()

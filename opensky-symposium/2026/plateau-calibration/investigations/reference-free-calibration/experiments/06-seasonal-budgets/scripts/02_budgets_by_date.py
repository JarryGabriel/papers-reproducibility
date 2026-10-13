"""The budget experiment 02's own convention would produce from each date alone.

The point is the SPREAD of those four values, not any one of them. If it is
small, the November budget was not mis-estimated and experiment 05's breaches
are about the signal; if it is large, one day was a thin basis for the quantity.

The conventions are experiment 02's as it locked them, applied per date:

  * quantised channels (altitude, vertical speed) -> the FULL step. 02 first
    derived half a step and its dated correction block records why that was
    untenable: a budget finer than the sensor's resolution is unsatisfiable by
    any real plateau. `budgets_full_step.json`, what every downstream experiment
    consumes, carries the full step, so that is what is comparable here.
  * reconstructed channels (CAS/IAS, Mach) -> the robust sigma of the ERA5
    reconstruction error.
  * the derived channel (flight-path angle) -> the median propagated budget.

Reproduce:
    uv run --active python scripts/02_budgets_by_date.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "outputs"

FT_TO_M = 0.3048

#: What experiment 02 locked, derived on 2025-11-01 alone. Not an input to
#: anything computed here — it is the value each date's own derivation is
#: reported against, since the question is whether that one day was
#: representative.
LOCKED = {
    "alt": 7.62,
    "vz": 64.0,
    "gamma": 0.0014288,
    "cas": 2.572742,
    "mach": 0.0066725,
}
UNIT = {"alt": "m", "vz": "ft/min", "gamma": "rad", "cas": "kt", "mach": "-"}


def main() -> int:
    quantisation = pl.read_csv(OUT / "quantisation_by_date.csv")
    reconstruction = pl.read_csv(OUT / "reconstruction_by_date.csv")
    propagation = pl.read_csv(OUT / "propagation_by_date.csv")

    dates = sorted(quantisation["date"].unique().to_list())
    rows: list[dict] = []

    for date in dates:
        quant = quantisation.filter(pl.col("date") == date)
        recon = reconstruction.filter(pl.col("date") == date)
        prop = propagation.filter(pl.col("date") == date)

        def step_of(channel: str, frame: pl.DataFrame = quant) -> float:
            row = frame.filter(pl.col("channel") == channel)
            if not bool(row["quantised"][0]):
                raise SystemExit(
                    f"{date} {channel}: no quantisation step found "
                    f"(share {float(row['share_multiple_of_step'][0]):.3f}). "
                    f"SOURCES.md reads this channel's budget from its step, so "
                    f"stopping rather than substituting another statistic."
                )
            return float(row["step"][0])

        def sigma_of(channel: str, frame: pl.DataFrame = recon) -> float:
            return float(
                frame.filter(pl.col("channel") == channel)["robust_sigma"][0])

        values = {
            # Full step, in metres for altitude and in ft/min for vz — the units
            # every downstream experiment scores in.
            "alt": step_of("alt") * FT_TO_M,
            "vz": step_of("vz"),
            # p50 of the propagated budget, doubled like the quantised channels:
            # gamma propagates FROM the vertical-speed step, so 02's full-step
            # correction applies to it too (FULL_STEP = alt, vz, gamma there).
            "gamma": float(prop["p50"][0]) * 2.0,
            "cas": sigma_of("cas"),
            "mach": sigma_of("mach"),
        }

        for channel, value in values.items():
            rows.append({
                "date": date,
                "channel": channel,
                "budget": value,
                "unit": UNIT[channel],
                "locked_budget": LOCKED[channel],
                "ratio_to_locked": value / LOCKED[channel],
            })

    frame = pl.DataFrame(rows)
    frame.write_csv(OUT / "budgets_by_date.csv")

    print(f"{'channel':8s} " + " ".join(f"{d[5:]:>10s}" for d in dates)
          + "   spread   max/min")
    for channel in ("alt", "gamma", "vz", "mach", "cas"):
        here = frame.filter(pl.col("channel") == channel).sort("date")
        values = here["budget"].to_list()
        print(f"{channel:8s} " + " ".join(f"{v:10.5g}" for v in values)
              + f"  {max(values) - min(values):9.4g}"
              + f"  {max(values) / min(values):8.3f}")
    print(f"\nwrote {OUT / 'budgets_by_date.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

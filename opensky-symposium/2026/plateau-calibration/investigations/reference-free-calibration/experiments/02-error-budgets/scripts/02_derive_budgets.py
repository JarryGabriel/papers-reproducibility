"""Assemble the five budgets from the four measurements.

Which statistic each channel reads was fixed in inputs/SOURCES.md and committed
before anything ran, so nothing here is a choice made in view of the numbers:

  * quantised channels (altitude, vertical speed) -> step / 2
  * reconstructed channels (CAS/IAS, Mach)        -> robust sigma of the ERA5
                                                     error against the register
  * the derived channel (flight-path angle)       -> median of the propagated
                                                     budget

The ratio to what round 2 used is reported beside each, because where the two
disagree the disagreement IS the result.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "outputs"

FT_TO_M = 0.3048

#: What round 2 used, for the comparison. Not an input to anything here.
ROUND_2 = {
    "alt": 2.0,
    "vz": 50.0,
    "gamma": 0.0008726646259971648,
    "cas": 1.0,
    "mach": 0.005,
}
UNIT = {"alt": "m", "vz": "ft/min", "gamma": "rad", "cas": "kt", "mach": "-"}


def main() -> None:
    quantisation = pl.read_csv(OUT / "quantisation.csv")
    reconstruction = pl.read_csv(OUT / "reconstruction.csv")
    propagation = pl.read_csv(OUT / "propagation.csv")

    def step_of(channel: str) -> float:
        row = quantisation.filter(pl.col("channel") == channel)
        if not bool(row["quantised"][0]):
            raise SystemExit(
                f"{channel}: no quantisation step found "
                f"(share {float(row['share_multiple_of_step'][0]):.3f}), and "
                f"SOURCES.md reads this channel's budget from its step. The "
                f"measurement contradicts the declared rule; stopping rather "
                f"than substituting another statistic."
            )
        return float(row["step"][0])

    def sigma_of(channel: str) -> float:
        return float(reconstruction.filter(pl.col("channel") == channel)["robust_sigma"][0])

    alt_step_ft = step_of("alt")
    vz_step = step_of("vz")

    budgets = {
        # THE ONE CHANNEL WITH TWO UNITS. The step is a property of the
        # encoding and the standard names it in feet: 25 ft. The detector
        # scores altitude in metres, because `densify_front._signal` converts
        # (and its flat tolerance of 30.48 is 100 ft wearing a metric label).
        # So the budget is carried in metres, where it is used, and reported in
        # feet, where it is understood.
        "alt": {
            "value": alt_step_ft / 2.0 * FT_TO_M,
            "value_native": alt_step_ft / 2.0,
            "native_unit": "ft",
            "basis": "quantisation step / 2",
            "measured": f"step = {alt_step_ft:g} ft "
                        f"({float(quantisation.filter(pl.col('channel') == 'alt')['share_multiple_of_step'][0]) * 100:.1f} % of gaps are multiples)",
        },
        "vz": {
            "value": vz_step / 2.0,
            "basis": "quantisation step / 2",
            "measured": f"step = {vz_step:g} ft/min "
                        f"({float(quantisation.filter(pl.col('channel') == 'vz')['share_multiple_of_step'][0]) * 100:.1f} % of gaps are multiples)",
        },
        "cas": {
            "value": sigma_of("cas"),
            "basis": "robust sigma of the ERA5 reconstruction error",
            "measured": f"{float(reconstruction.filter(pl.col('channel') == 'cas')['reconstructed_pct'][0]):.1f} % "
                        f"of the channel is reconstructed; "
                        f"{int(reconstruction.filter(pl.col('channel') == 'cas')['n_paired'][0]):,} paired samples",
        },
        "mach": {
            "value": sigma_of("mach"),
            "basis": "robust sigma of the ERA5 reconstruction error",
            "measured": f"{float(reconstruction.filter(pl.col('channel') == 'mach')['reconstructed_pct'][0]):.1f} % "
                        f"of the channel is reconstructed; "
                        f"{int(reconstruction.filter(pl.col('channel') == 'mach')['n_paired'][0]):,} paired samples",
        },
        "gamma": {
            "value": float(propagation["p50"][0]),
            "basis": "median of the propagated budget",
            "measured": f"dvz = {float(propagation['d_vz_ftmin'][0]):g} ft/min, "
                        f"dV = {float(propagation['d_speed_kt'][0]):.3g} kt, "
                        f"the vz term dominating by {float(propagation['term_ratio'][0]):.0f}x",
        },
    }

    rows = []
    for channel in ("alt", "vz", "gamma", "cas", "mach"):
        entry = budgets[channel]
        rows.append({
            "channel": channel,
            "unit": UNIT[channel],
            "budget": entry["value"],
            # Only altitude differs; every other channel is scored in the unit
            # it was measured in, so the two columns coincide.
            "budget_native": entry.get("value_native", entry["value"]),
            "native_unit": entry.get("native_unit", UNIT[channel]),
            "basis": entry["basis"],
            "measured": entry["measured"],
            "round_2_budget": ROUND_2[channel],
            "ratio_to_round_2": entry["value"] / ROUND_2[channel],
        })
    table = pl.DataFrame(rows)
    table.write_csv(OUT / "budgets.csv")
    (OUT / "budgets.json").write_text(
        json.dumps({c: budgets[c]["value"] for c in ROUND_2}, indent=1) + "\n"
    )

    print(f"{'channel':8s} {'budget':>13s} {'round 2':>13s} {'ratio':>7s}  basis")
    for row in rows:
        native = (f"{row['budget_native']:.6g} {row['native_unit']}"
                  if row["native_unit"] != row["unit"] else "")
        shown = f"{row['budget']:.6g} {row['unit']}"
        print(f"{row['channel']:8s} {shown:>13s} "
              f"{f'{row['round_2_budget']:.6g} {row['unit']}':>13s} "
              f"{row['ratio_to_round_2']:7.2f}  {row['basis']}"
              + (f"   [= {native}]" if native else ""))


if __name__ == "__main__":
    main()

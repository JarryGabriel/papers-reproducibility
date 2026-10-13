"""Apply the full-step convention to the quantised channels.

`outputs/budgets.json` is locked by digest and is not touched: it records what
this folder derived, halving convention included. This writes a second file
carrying the corrected convention, which is what downstream experiments consume.

WHY THE CORRECTION. The halving reads: a plateau occupying one quantisation step
has a mean absolute deviation of about half a step. That describes a plateau
alternating cleanly between two adjacent steps; a real one wanders over three or
four and exceeds half a step.

Measured on flight 4076cc_BAW664_s0, a 202-minute cruise at FL350 that is
geometrically one plateau: the grid holds points committing it as a SINGLE
segment covering 100 % of it, with a MAE of 4.61 m. The half-step budget of
3.81 m rejects that plateau by 21 %, and the best point that does respect it cuts
the same cruise into 54 segments. The half step was not constraining error — it
was rejecting real plateaux and admitting only fragmentation, since cutting a
segment is the reliable way to drive a mean absolute deviation down.

APPLIED UNIFORMLY. Only altitude carries that measurement. Vertical speed takes
the full step by the same rule rather than by its own evidence, and the
flight-path angle follows mechanically, being computed from vz. Treating one
quantised channel differently from the other would trade a declared convention
for a per-channel judgement.

The reconstructed channels are untouched: their budgets read the robust sigma of
the ERA5 error, which is a dispersion and not a step, so there is nothing to
double.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "outputs"

#: The channels whose budget is read from a quantisation step, and therefore the
#: only ones the convention touches. gamma is here because it propagates from
#: vz, not because it carries a step of its own.
FULL_STEP = ("alt", "vz", "gamma")


def main() -> None:
    derived = json.loads((OUT / "budgets.json").read_text())
    corrected = {
        channel: value * 2.0 if channel in FULL_STEP else value
        for channel, value in derived.items()
    }
    (OUT / "budgets_full_step.json").write_text(
        json.dumps(corrected, indent=1) + "\n"
    )

    table = pl.read_csv(OUT / "budgets.csv")
    table = table.with_columns(
        pl.when(pl.col("channel").is_in(FULL_STEP))
        .then(pl.col("budget") * 2.0)
        .otherwise(pl.col("budget"))
        .alias("budget_full_step"),
        pl.when(pl.col("channel").is_in(FULL_STEP))
        .then(pl.lit("full quantisation step"))
        .otherwise(pl.lit("unchanged — no step, reads the ERA5 sigma"))
        .alias("convention"),
    )
    table.write_csv(OUT / "budgets_full_step.csv")

    print(f"{'channel':8s} {'derived':>13s} {'corrected':>13s}  convention")
    for row in table.iter_rows(named=True):
        print(f"{row['channel']:8s} {row['budget']:13.6g} "
              f"{row['budget_full_step']:13.6g}  {row['convention']}")


if __name__ == "__main__":
    main()

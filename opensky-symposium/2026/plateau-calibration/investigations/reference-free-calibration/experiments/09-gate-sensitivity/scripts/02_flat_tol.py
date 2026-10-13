"""Does the range tolerance decide which candidate the rule retains?

`flat_tol` is the widest span of the channel a run may cover and still be
offered as a segment — 30.48 m on altitude, 0.002 rad on the flight-path angle,
100 ft/min on vertical speed, 0.05 on Mach and 20 kt on CAS/IAS. Like `min_len`
it is applied inside the detector, before the selection rule sees anything, and
like it, it is fixed per channel and never swept.

It is tested in multiples rather than offsets because the five channels are in
five different units, and a fixed step would be a large perturbation on one
channel and a negligible one on another. The multiples are those the freeze
declares: 0.5, 0.75, 1.0, 1.5, 2.0.

Reproduce:
    <venv>/bin/python scripts/02_flat_tol.py            # the cohort
    <venv>/bin/python scripts/02_flat_tol.py --smoke 40 # a probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gate as G

#: Multiples of each channel's configured value, declared in freeze/.
MULTIPLES = (0.5, 0.75, 1.0, 1.5, 2.0)


#: Each channel's configured value, captured before the first perturbation so a
#: multiple is always taken of the configured value and never of a value a
#: previous setting already moved.
_BASE: dict[str, float] = {}


def apply_setting(channel: str, multiple) -> None:
    G.CFG[channel]["flat"] = _BASE[channel] * float(multiple)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0,
                        help="score this many flights into outputs/smoke/")
    args = parser.parse_args()

    grid = G.grids()
    budgets = G.budgets()
    frames = []
    for channel in G.CHANNELS:
        base = float(G.CFG[channel]["flat"])
        _BASE[channel] = base
        budget = budgets[channel]
        candidates = G.leading_candidates(channel, grid, budget)
        print(f"{channel}: base flat_tol={base:g}, testing "
              f"{[f'{m:g}x' for m in MULTIPLES]}, {len(candidates)} candidates",
              flush=True)
        try:
            scored = G.score_cohort(channel, candidates, list(MULTIPLES),
                                    apply_setting, n_flights=args.smoke)
        finally:
            G.CFG[channel]["flat"] = base
        scored = scored.with_columns(
            pl.lit("flat_tol").alias("gate"), pl.lit(1.0).alias("base_setting")
        )
        frames.append(scored)

    out = pl.concat(frames)
    target = (G.OUTPUTS / "smoke" if args.smoke else G.OUTPUTS) / "flat_tol_ranking.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(target)
    print(f"\nwrote {target}  ({out.height} rows)\n")

    for channel in G.CHANNELS:
        rows = out.filter(pl.col("channel") == channel)
        verdict = G.verdict_per_channel(rows, budgets[channel], 1.0)
        print(f"{channel:6s} {verdict['outcome']:14s} incumbent {verdict['incumbent']}")
        for move in verdict["moved"]:
            print(f"         flat_tol={move['setting']:g}x retains {move['to']} "
                  f"(+{move['gap_pts']:.3f} pts)")
        for setting in verdict["incumbent_over_budget"]:
            print(f"         flat_tol={setting:g} the incumbent breaches its budget")
        for setting in verdict["no_admissible"]:
            print(f"         flat_tol={setting:g} no candidate is admissible")
        for move in verdict["indeterminate"]:
            print(f"         flat_tol={move['setting']:g}x ties with {move['to']} "
                  f"({move['gap_pts']:.3f} pts, inside the band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

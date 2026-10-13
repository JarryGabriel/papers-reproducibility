"""Does the minimum run length decide which candidate the rule retains?

`min_len` is the fewest consecutive samples a run must hold to be offered as a
segment at all. It is applied inside the detector, before the selection rule
sees anything, and it is fixed per channel and never swept — 8 samples on
altitude and vertical speed, 10 on the flight-path angle, 15 on Mach and 5 on
CAS/IAS, against a 4-second sampling interval.

The range is the one the freeze declares and is not chosen here: {2, base-3,
base, base+3, base+6, base+10}, clipped below at 2 samples.

Reproduce:
    <venv>/bin/python scripts/01_min_len.py            # the cohort
    <venv>/bin/python scripts/01_min_len.py --smoke 40 # a probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gate as G

#: Offsets from each channel's configured value, declared in freeze/.
OFFSETS = (-3, 0, 3, 6, 10)
FLOOR = 2


def values_for(channel: str) -> list[int]:
    base = G.CFG[channel]["min_len"]
    return sorted({FLOOR} | {max(FLOOR, base + d) for d in OFFSETS})


def apply_setting(channel: str, value) -> None:
    G.CFG[channel]["min_len"] = int(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0,
                        help="score this many flights into outputs/smoke/")
    args = parser.parse_args()

    grid = G.grids()
    budgets = G.budgets()
    frames = []
    for channel in G.CHANNELS:
        base = G.CFG[channel]["min_len"]
        budget = budgets[channel]
        candidates = G.leading_candidates(channel, grid, budget)
        settings = values_for(channel)
        print(f"{channel}: base min_len={base}, testing {settings}, "
              f"{len(candidates)} candidates", flush=True)
        try:
            scored = G.score_cohort(channel, candidates, settings, apply_setting,
                                    n_flights=args.smoke)
        finally:
            # Restore, or the next channel's cascade priors run on a gate this
            # step moved.
            G.CFG[channel]["min_len"] = base
        scored = scored.with_columns(
            pl.lit("min_len").alias("gate"), pl.lit(base).alias("base_setting")
        )
        frames.append(scored)

    out = pl.concat(frames)
    target = (G.OUTPUTS / "smoke" if args.smoke else G.OUTPUTS) / "min_len_ranking.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(target)
    print(f"\nwrote {target}  ({out.height} rows)\n")

    for channel in G.CHANNELS:
        rows = out.filter(pl.col("channel") == channel)
        base = rows["base_setting"][0]
        verdict = G.verdict_per_channel(rows, budgets[channel], base)
        print(f"{channel:6s} {verdict['outcome']:14s} incumbent {verdict['incumbent']}")
        for move in verdict["moved"]:
            print(f"         min_len={move['setting']} retains {move['to']} "
                  f"(+{move['gap_pts']:.3f} pts)")
        for setting in verdict["incumbent_over_budget"]:
            print(f"         min_len={setting:g} the incumbent breaches its budget")
        for setting in verdict["no_admissible"]:
            print(f"         min_len={setting:g} no candidate is admissible")
        for move in verdict["indeterminate"]:
            print(f"         min_len={move['setting']} ties with {move['to']} "
                  f"({move['gap_pts']:.3f} pts, inside the band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

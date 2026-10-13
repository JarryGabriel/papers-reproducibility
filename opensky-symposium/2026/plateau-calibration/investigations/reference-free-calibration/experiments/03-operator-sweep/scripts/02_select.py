"""Apply the selection rule to the frontier.

The rule, from inputs/SOURCES.md and fixed before the grid was walked:

    among the settings whose worst segment error respects the budget, retain the
    one committing the largest share of samples; ties within 0.05 coverage
    points go to the lower worst-segment error, then to the fewer segments.

A channel with no admissible candidate is reported EMPTY, with the smallest
worst-segment error the grid reached and the factor by which it misses. The
budget is not widened to produce an answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
BUDGETS = EXPERIMENT_DIR / "inputs" / "budgets_full_step.json"
OUTPUTS = EXPERIMENT_DIR / "outputs"

#: Coverage within this of the maximum is a tie. An argmax on hundredths of a
#: point is choosing on sampling noise; declared in SOURCES.md before the run.
TIE_PTS = 0.05


def main() -> int:
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8"))
    sweep = pl.read_csv(OUTPUTS / "sweep.csv")

    rows: list[dict] = []
    for channel, budget in budgets.items():
        scored = sweep.filter(pl.col("channel") == channel)
        if not scored.height:
            print(f"{channel}: nothing scored", file=sys.stderr)
            raise SystemExit(1)

        admissible = scored.filter(pl.col("worst_segment_error") <= budget)
        if not admissible.height:
            # Reported, not rescued. The grid's best attempt and its shortfall
            # are what a reader needs to judge whether the budget or the grid is
            # the binding constraint.
            best = scored.sort("worst_segment_error").head(1).to_dicts()[0]
            rows.append({
                "channel": channel, "admissible": False,
                "algo": None, "butter": None, "knobs": None,
                "coverage_pct": None, "worst_segment_error": None,
                "budget": budget, "budget_used_pct": None,
                "worst_drift": None, "drift_over_budget": None,
                "segments_per_flight": None, "median_duration_s": None,
                "n_admissible": 0,
                "closest_worst_error": best["worst_segment_error"],
                "closest_factor": best["worst_segment_error"] / budget,
            })
            print(f"{channel:6s} EMPTY — closest reaches "
                  f"{best['worst_segment_error']:.5g} "
                  f"({best['worst_segment_error'] / budget:.1f}x the budget)")
            continue

        ceiling = admissible["coverage_pct"].max()
        tied = admissible.filter(pl.col("coverage_pct") >= ceiling - TIE_PTS)
        chosen = tied.sort(
            ["worst_segment_error", "segments_per_flight"]
        ).head(1).to_dicts()[0]

        rows.append({
            "channel": channel, "admissible": True,
            "algo": chosen["algo"], "butter": chosen["butter"],
            "knobs": chosen["knobs"],
            "coverage_pct": chosen["coverage_pct"],
            "worst_segment_error": chosen["worst_segment_error"],
            "budget": budget,
            "budget_used_pct": chosen["worst_segment_error"] / budget * 100.0,
            "worst_drift": chosen["worst_drift"],
            "drift_over_budget": chosen["worst_drift"] / budget,
            "segments_per_flight": chosen["segments_per_flight"],
            "median_duration_s": chosen["median_duration_s"],
            "n_admissible": admissible.height,
            "closest_worst_error": None,
            "closest_factor": None,
        })
        print(f"{channel:6s} {chosen['coverage_pct']:6.2f}%  "
              f"budget used {chosen['worst_segment_error'] / budget * 100:5.1f}%  "
              f"{chosen['segments_per_flight']:5.1f} seg/flt  "
              f"{chosen['median_duration_s']:5.0f} s  "
              f"drift {chosen['worst_drift'] / budget:4.1f}x  "
              f"{chosen['algo']} b={chosen['butter']} {chosen['knobs']}"
              + (f"  [tie among {tied.height}]" if tied.height > 1 else ""))

    pl.DataFrame(rows).write_csv(OUTPUTS / "retained.csv")
    print(f"wrote {OUTPUTS / 'retained.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

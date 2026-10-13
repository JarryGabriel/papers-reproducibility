"""What constraining the drift as well as the deviation would cost.

The selection rule binds the mean absolute deviation of each plateau and leaves
its SHAPE free. On a linear ramp the drift is about four times the deviation, so
a segment may decline steadily from one end to the other and still be admitted —
and the manifest's limit says so. This step measures what that costs in practice
rather than leaving the bound theoretical: the drift actually observed at each
retained point, and the coverage the same rule would retain if the drift were
constrained too.

Nothing is re-selected and no rule is changed. The frontier already carries
`worst_drift` for every one of its 27 414 candidates, so this is a re-read of
`sweep.csv` under a different filter — the whole reason the sweep writes the
front rather than only its winner.

Reproduce:
    uv run --active python scripts/04_drift_cost.py
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
INPUTS = EXPERIMENT_DIR / "inputs"
OUTPUTS = EXPERIMENT_DIR / "outputs"

CHANNELS = ("alt", "gamma", "vz", "mach", "cas")
#: The sweep's own tie-break, so the counterfactual rule is the same rule.
TIE_PTS = 0.05
#: Drift multiples of the budget to test. 1.0 is as strict as the deviation
#: constraint; 2.0 is the loosest that still forbids a segment from drifting
#: through its whole budget twice over.
MULTIPLES = (1.0, 2.0)


def retain(frame: pl.DataFrame, column: str, budget: float) -> dict | None:
    """The rule of `02_select.py`, with the constrained statistic swapped."""
    admissible = frame.filter(pl.col(column) <= budget)
    if not admissible.height:
        return None
    ceiling = admissible["coverage_pct"].max()
    tied = admissible.filter(
        pl.col("coverage_pct") >= ceiling - TIE_PTS
    ).sort([column, "segments_per_flight"])
    return tied.row(0, named=True)


def main() -> int:
    budgets = json.loads((INPUTS / "budgets_full_step.json").read_text())
    sweep = pl.read_csv(OUTPUTS / "sweep.csv")

    rows: list[dict] = []
    for channel in CHANNELS:
        frame = sweep.filter(pl.col("channel") == channel)
        budget = budgets[channel]

        base = retain(frame, "worst_segment_error", budget)
        if base is None:
            continue
        rows.append({
            "channel": channel,
            "rule": "deviation<=E",
            "budget": budget,
            "coverage_pct": base["coverage_pct"],
            "worst_segment_error": base["worst_segment_error"],
            "worst_drift": base["worst_drift"],
            # The number the limit quotes: how far past its budget the retained
            # point's worst segment drifts while respecting the deviation.
            "drift_over_budget": base["worst_drift"] / budget,
            "segments_per_flight": base["segments_per_flight"],
            "cost_vs_base_pts": 0.0,
            "knobs": base["knobs"],
        })

        for multiple in MULTIPLES:
            other = retain(frame, "worst_drift", budget * multiple)
            if other is None:
                rows.append({
                    "channel": channel, "rule": f"drift<={multiple:g}E",
                    "budget": budget * multiple, "coverage_pct": None,
                    "worst_segment_error": None, "worst_drift": None,
                    "drift_over_budget": None, "segments_per_flight": None,
                    "cost_vs_base_pts": None, "knobs": None,
                })
                continue
            rows.append({
                "channel": channel,
                "rule": f"drift<={multiple:g}E",
                "budget": budget * multiple,
                "coverage_pct": other["coverage_pct"],
                "worst_segment_error": other["worst_segment_error"],
                "worst_drift": other["worst_drift"],
                "drift_over_budget": other["worst_drift"] / budget,
                "segments_per_flight": other["segments_per_flight"],
                "cost_vs_base_pts": base["coverage_pct"] - other["coverage_pct"],
                "knobs": other["knobs"],
            })

    frame = pl.DataFrame(rows)
    OUTPUTS.mkdir(exist_ok=True)
    frame.write_csv(OUTPUTS / "drift_cost.csv")

    print(f"{'channel':7s} {'rule':14s} {'coverage':>9s} {'drift/E':>8s} {'cost':>7s}")
    for row in frame.iter_rows(named=True):
        if row["coverage_pct"] is None:
            print(f"{row['channel']:7s} {row['rule']:14s}   no admissible candidate")
            continue
        print(f"{row['channel']:7s} {row['rule']:14s} {row['coverage_pct']:9.2f} "
              f"{row['drift_over_budget']:8.2f} {row['cost_vs_base_pts']:7.2f}")
    print(f"\nwrote {OUTPUTS / 'drift_cost.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

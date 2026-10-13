"""What the frozen points achieve on each fold's airframes.

For every channel and every fold, at the setting experiment 03 retained on the
full cohort: committed coverage, the worst committed segment's own error, the
segment count and the median duration. The spread of those across folds is the
dispersion the manifest commits to reporting.

The cascade runs exactly as the sweep ran it — sequential, each channel seeing
only what its priors left, with the priors at their frozen points. Measuring a
downstream channel against the full trajectory would measure something the
detector never does.

Reproduce:
    uv run --active python scripts/02_heldout.py
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


import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PAPER_ROOT = project_root(__file__)
REPO_ROOT = project_root(__file__)
COHORT = data_dir() / 'flights.delta'
INPUTS = EXPERIMENT_DIR / "inputs"
OUTPUTS = EXPERIMENT_DIR / "outputs"


from opensky26_plateau import transfer_scoring as T  # noqa: E402

DATE = "2025-11-01"
DT_S = 4.0
VERTICAL = ("alt", "gamma", "vz")
SPEED = ("mach", "cas")


def frozen_points() -> dict[str, tuple[str, dict]]:
    """The retained settings, resolved back to the grid's own precision.

    `retained.csv` writes its knobs with `%g` — six significant digits — so a
    detector run on the stored string is not the detector that was scored.
    `resolve_frozen` maps each knob back to the grid value it can only have come
    from, and refuses when the match is ambiguous.
    """
    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    out: dict[str, tuple[str, dict]] = {}
    for row in pl.read_csv(INPUTS / "retained.csv").iter_rows(named=True):
        if not row["admissible"]:
            continue
        knobs = {
            k: float(v) for k, v in (p.split("=") for p in row["knobs"].split(","))
        }
        knobs["butter"] = int(row["butter"])
        resolved = T.resolve_frozen(knobs, row["channel"], row["algo"], grids)
        resolved["butter"] = int(row["butter"])
        out[row["channel"]] = (row["algo"], resolved)
    return out


def score_fold(channel: str, algo: str, knobs: dict, flights: list[pl.DataFrame],
               signals: list[dict], exclusion: list[np.ndarray],
               mach_config: dict) -> tuple[dict, list[np.ndarray]]:
    """One channel on one fold, and the exclusion it hands down."""
    raws = [s[channel] for s in signals]
    total = sum(int(np.isfinite(r).sum()) for r in raws)

    worst_error = worst_drift = 0.0
    committed = n_segments = n_flights_with = 0
    lengths: list[int] = []
    handed: list[np.ndarray] = []

    for raw, flight, excl in zip(raws, flights, exclusion):
        here = excl.copy()
        if channel == "cas":
            here = here | T.mach_mask(flight, mach_config)
        view = T.filtered(raw, knobs["butter"])
        claimed = np.zeros(raw.size, dtype=bool)
        seen = False
        for segment in T.detect(algo, channel, view, knobs, here):
            i, j = segment["start_idx"], segment["end_idx"]
            piece = raw[i:j + 1]
            finite = np.isfinite(piece)
            if not finite.any():
                continue
            values = piece[finite]
            worst_error = max(worst_error, float(np.abs(values - values.mean()).mean()))
            third = max(2, values.size // 3)
            worst_drift = max(worst_drift, float(
                abs(np.median(values[-third:]) - np.median(values[:third]))))
            committed += int(finite.sum())
            n_segments += 1
            lengths.append(j - i + 1)
            claimed[i:j + 1] = True
            seen = True
        if seen:
            n_flights_with += 1
        handed.append(excl | claimed)

    row = {
        "channel": channel,
        "coverage_pct": committed / total * 100.0 if total else 0.0,
        "worst_segment_error": worst_error,
        "worst_drift": worst_drift,
        "n_segments": n_segments,
        "segments_per_flight": n_segments / max(n_flights_with, 1),
        "median_duration_s": float(np.median(lengths)) * DT_S if lengths else 0.0,
        "n_flights": len(flights),
    }
    return row, handed


def main() -> int:
    points = frozen_points()
    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    budgets = json.loads((INPUTS / "budgets_full_step.json").read_text())
    folds = pl.read_csv(INPUTS / "folds.csv")

    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    ).filter(pl.col("_date") == DATE)
    mach_config = T.mach_config(INPUTS / "retained.csv", grids)

    rows: list[dict] = []
    for fold in sorted(folds["fold"].unique().to_list()):
        icaos = folds.filter(pl.col("fold") == fold)["raw_icao24"].to_list()
        here = frame.filter(pl.col("raw_icao24").is_in(icaos))
        ids = here["meta_flight_id"].unique(maintain_order=True).to_list()
        flights = [
            here.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
            for fid in ids
        ]
        signals = [T.signals(f) for f in flights]
        print(f"fold {fold}: {len(flights)} flights, {len(icaos)} airframes")

        for order in (VERTICAL, SPEED):
            exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
            for channel in order:
                algo, knobs = points[channel]
                row, exclusion = score_fold(channel, algo, knobs, flights,
                                            signals, exclusion, mach_config)
                row["fold"] = fold
                row["budget"] = budgets[channel]
                row["budget_used_pct"] = (
                    row["worst_segment_error"] / budgets[channel] * 100.0
                )
                rows.append(row)
                print(f"  {channel:6s} {row['coverage_pct']:6.2f} %  "
                      f"budget used {row['budget_used_pct']:6.1f} %  "
                      f"{row['segments_per_flight']:5.2f} seg/flt")

    OUTPUTS.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTPUTS / "heldout.csv")
    print(f"wrote {OUTPUTS / 'heldout.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

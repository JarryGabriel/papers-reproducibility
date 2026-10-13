"""What the frozen points achieve on each of the four dates.

Per channel and per date, at the setting experiment 03 retained on the November
cohort: committed coverage, the worst committed segment's own error, the segment
count and the median duration. The spread of those across dates is the
dispersion the manifest commits to reporting, and the share of budget each date
uses is what says whether the point still holds where it was not chosen.

The cascade runs exactly as the sweep ran it — sequential, each channel seeing
only the residual its priors leave, with the priors at their frozen points.

Reproduce:
    uv run --active python scripts/01_carry.py
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
import os
import sys
from pathlib import Path

#: Same cap as the sweep, and for the same reason — set before numpy/polars are
#: imported, since both size their thread pools at import time. This script is
#: short, but it loads the same Delta table and there is no reason for it to
#: take every core either.
_THREADS = os.environ.get("FDM_THREADS", "5")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "POLARS_MAX_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PAPER_ROOT = project_root(__file__)
REPO_ROOT = project_root(__file__)
COHORT = data_dir() / 'flights.delta'
INPUTS = EXPERIMENT_DIR / "inputs"
OUTPUTS = EXPERIMENT_DIR / "outputs"


from opensky26_plateau import transfer_scoring as T  # noqa: E402

#: All four, calibration last so the log reads outward from the unseen dates.
DATES = ("2025-02-17", "2025-05-21", "2025-08-29", "2025-11-01")
CALIBRATION = "2025-11-01"
DT_S = 4.0
VERTICAL = ("alt", "gamma", "vz")
SPEED = ("mach", "cas")


def frozen_points() -> dict[str, tuple[str, dict]]:
    """The retained settings, resolved back to the grid's own precision.

    `retained.csv` writes knobs with `%g` — six significant digits — so running
    the detector on the stored string is running a slightly different detector
    from the one that was scored. `resolve_frozen` maps each knob back to the
    grid value it can only have come from.
    """
    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    out: dict[str, tuple[str, dict]] = {}
    for row in pl.read_csv(INPUTS / "retained.csv").iter_rows(named=True):
        if not row["admissible"]:
            continue
        knobs = {k: float(v) for k, v in (p.split("=")
                                          for p in row["knobs"].split(","))}
        knobs["butter"] = int(row["butter"])
        resolved = T.resolve_frozen(knobs, row["channel"], row["algo"], grids)
        resolved["butter"] = int(row["butter"])
        out[row["channel"]] = (row["algo"], resolved)
    return out


def score_channel(channel: str, algo: str, knobs: dict,
                  flights: list[pl.DataFrame], signals: list[dict],
                  exclusion: list[np.ndarray],
                  mach_config: dict) -> tuple[dict, list[np.ndarray]]:
    """One channel on one date, and the exclusion it hands down."""
    raws = [s[channel] for s in signals]
    total = sum(int(np.isfinite(r).sum()) for r in raws)

    worst_error = worst_drift = 0.0
    committed = n_segments = n_with = 0
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
            worst_error = max(worst_error,
                              float(np.abs(values - values.mean()).mean()))
            third = max(2, values.size // 3)
            worst_drift = max(worst_drift, float(
                abs(np.median(values[-third:]) - np.median(values[:third]))))
            committed += int(finite.sum())
            n_segments += 1
            lengths.append(j - i + 1)
            claimed[i:j + 1] = True
            seen = True
        if seen:
            n_with += 1
        handed.append(excl | claimed)

    row = {
        "channel": channel,
        "coverage_pct": committed / total * 100.0 if total else 0.0,
        "worst_segment_error": worst_error,
        "worst_drift": worst_drift,
        "n_segments": n_segments,
        "segments_per_flight": n_segments / max(n_with, 1),
        "median_duration_s": float(np.median(lengths)) * DT_S if lengths else 0.0,
    }
    return row, handed


def main() -> int:
    points = frozen_points()
    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    budgets = json.loads((INPUTS / "budgets_full_step.json").read_text())
    mach_config = T.mach_config(INPUTS / "retained.csv", grids)

    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    )

    rows: list[dict] = []
    for date in DATES:
        here = frame.filter(pl.col("_date") == date)
        ids = here["meta_flight_id"].unique(maintain_order=True).to_list()
        flights = [here.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
                   for fid in ids]
        signals = [T.signals(f) for f in flights]
        tag = "calibration" if date == CALIBRATION else "held out"
        print(f"{date} ({tag}): {len(flights)} flights, "
              f"{here['raw_icao24'].n_unique()} airframes")

        for order in (VERTICAL, SPEED):
            exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
            for channel in order:
                algo, knobs = points[channel]
                row, exclusion = score_channel(channel, algo, knobs, flights,
                                               signals, exclusion, mach_config)
                row["date"] = date
                row["in_sample"] = date == CALIBRATION
                row["budget"] = budgets[channel]
                row["budget_used_pct"] = (
                    row["worst_segment_error"] / budgets[channel] * 100.0)
                row["n_flights"] = len(flights)
                rows.append(row)
                breach = " BREACH" if row["budget_used_pct"] > 100.0 else ""
                print(f"  {channel:6s} {row['coverage_pct']:6.2f} %  "
                      f"budget used {row['budget_used_pct']:6.1f} %  "
                      f"{row['segments_per_flight']:5.2f} seg/flt{breach}")

    OUTPUTS.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTPUTS / "by_date.csv")
    print(f"wrote {OUTPUTS / 'by_date.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

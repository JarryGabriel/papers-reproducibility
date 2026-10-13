"""One row per committed segment — the distribution the maximum was hiding.

The scoring is experiment 05's `score_channel`, unchanged in what it detects and
unchanged in the cascade it runs, with one difference: where that function keeps
`worst_error = max(worst_error, ...)` and drops the rest, this one emits every
segment. Nothing is re-selected and no rule is changed; only what is recorded.

Reproduce:
    uv run --active python scripts/01_segments.py
    uv run --active python scripts/01_segments.py --smoke 20
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


import argparse
import json
import os
import sys
from pathlib import Path

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

DATES = ("2025-02-17", "2025-05-21", "2025-08-29", "2025-11-01")
CALIBRATION = "2025-11-01"
DT_S = 4.0
#: The cascade order the sweep ran, and the reason the two orders are separate:
#: each channel sees only the residual its priors leave.
VERTICAL = ("alt", "gamma", "vz")
SPEED = ("mach", "cas")


def frozen_points() -> dict[str, tuple[str, dict]]:
    """The retained settings, resolved back to the grid's own precision."""
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


def segments_of(channel: str, algo: str, knobs: dict, date: str,
                flights: list[pl.DataFrame], signals: list[dict],
                exclusion: list[np.ndarray], budget: float,
                mach_config: dict) -> tuple[list[dict], list[np.ndarray]]:
    """Every committed segment of one channel on one date, and the residual.

    The detection, the cascade and the error are experiment 05's. What differs
    is that each segment becomes a row instead of being folded into a maximum.
    """
    raws = [s[channel] for s in signals]
    total = sum(int(np.isfinite(r).sum()) for r in raws)
    rows: list[dict] = []
    handed: list[np.ndarray] = []

    for raw, flight, excl in zip(raws, flights, exclusion):
        here = excl.copy()
        if channel == "cas":
            here = here | T.mach_mask(flight, mach_config)
        view = T.filtered(raw, knobs["butter"])
        claimed = np.zeros(raw.size, dtype=bool)

        for segment in T.detect(algo, channel, view, knobs, here):
            i, j = segment["start_idx"], segment["end_idx"]
            piece = raw[i:j + 1]
            finite = np.isfinite(piece)
            if not finite.any():
                continue
            values = piece[finite]
            mean = float(values.mean())
            error = float(np.abs(values - mean).mean())
            # The SECOND statistic 03 and 05 constrained, and they kept only its
            # maximum too. Recording it per segment here costs three lines and
            # closes the same hole permanently — otherwise the criticism this
            # experiment answers applies verbatim to drift the moment someone
            # asks. Definition is 05's `score_channel`, unchanged.
            third = max(2, values.size // 3)
            drift = float(abs(np.median(values[-third:])
                              - np.median(values[:third])))
            rows.append({
                "date": date,
                "in_sample": date == CALIBRATION,
                "channel": channel,
                "flight_id": flight["meta_flight_id"][0],
                "icao24": flight["raw_icao24"][0],
                "start_idx": i,
                "end_idx": j,
                "n_samples": int(finite.sum()),
                # SPAN, not covered time: a segment can hold interior holes, so
                # `span_s` and `n_samples * DT_S` differ on about 4 % of rows
                # (mostly Mach). Anything weighting a segment by how much of the
                # trajectory it carries must use `n_samples`, never this.
                "span_s": (j - i + 1) * DT_S,
                "covered_s": int(finite.sum()) * DT_S,
                "mean_value": mean,
                "error": error,
                "drift": drift,
                # Five channels carry five incommensurable units; the ratio is
                # what makes a distribution comparable across them.
                "error_over_budget": error / budget,
                "channel_total_samples": total,
            })
            claimed[i:j + 1] = True

        handed.append(excl | claimed)

    return rows, handed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="restrict to N flights per date, into outputs/smoke/")
    args = parser.parse_args()

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
        if args.smoke:
            ids = ids[:args.smoke]
        flights = [here.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
                   for fid in ids]
        signals = [T.signals(f) for f in flights]
        tag = "calibration" if date == CALIBRATION else "held out"
        print(f"{date} ({tag}): {len(flights)} flights")

        for order in (VERTICAL, SPEED):
            exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
            for channel in order:
                algo, knobs = points[channel]
                got, exclusion = segments_of(
                    channel, algo, knobs, date, flights, signals, exclusion,
                    budgets[channel], mach_config)
                rows += got
                if got:
                    err = np.array([r["error_over_budget"] for r in got])
                    breach = int((err > 1.0).sum())
                    print(f"  {channel:6s} {len(got):6d} segments  "
                          f"breach {breach:5d} ({breach / len(got) * 100:5.2f} %)  "
                          f"median {np.median(err):.3f}  max {err.max():.3f}"
                          " of budget")

    out = OUTPUTS / "smoke" if args.smoke else OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out / "segments.csv")
    print(f"\nwrote {out / 'segments.csv'} ({len(rows)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

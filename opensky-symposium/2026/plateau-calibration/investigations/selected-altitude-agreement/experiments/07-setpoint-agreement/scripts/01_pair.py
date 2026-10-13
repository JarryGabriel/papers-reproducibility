"""Pair each committed altitude plateau with the selected altitude broadcast.

One row per plateau the frozen detector commits to, whether or not a broadcast
value exists over its extent — an unpaired plateau is a row with a null
difference, never a dropped row. The share that cannot be compared bounds
everything else this experiment reports, so it has to be visible in the output
rather than inferred from what is missing.

The pairing rule is the one SOURCES.md fixed before the distribution was seen:
the median of `bds_mcp_alt_sel_ft` over the plateau's own extent.

Reproduce:
    uv run --active python scripts/01_pair.py
    uv run --active python scripts/01_pair.py --smoke 20
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

DATE = "2025-11-01"
DT_S = 4.0
FT_TO_M = 0.3048
#: Chosen agreement bracket of one flight level, not the BDS 4,0 encoding
#: resolution (16 ft). A reporting bracket, not a pass mark.
FLIGHT_LEVEL_FT = 100.0
#: The altitude channel's quantisation step, from experiment 02.
STEP_FT = 25.0


def frozen_altitude_point() -> tuple[str, dict]:
    """The altitude setting experiment 03 retained, resolved back to the grid.

    `retained.csv` writes knobs with `%g`, so running the detector on the stored
    string would run a slightly different detector from the one that was scored.
    """
    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    for row in pl.read_csv(INPUTS / "retained.csv").iter_rows(named=True):
        if row["channel"] != "alt" or not row["admissible"]:
            continue
        knobs = {k: float(v) for k, v in (p.split("=")
                                          for p in row["knobs"].split(","))}
        knobs["butter"] = int(row["butter"])
        resolved = T.resolve_frozen(knobs, "alt", row["algo"], grids)
        resolved["butter"] = int(row["butter"])
        return row["algo"], resolved
    raise SystemExit("no admissible altitude point in retained.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="restrict to N flights, writing to outputs/smoke/")
    args = parser.parse_args()

    algo, knobs = frozen_altitude_point()
    print(f"frozen altitude point: {algo}, butter={knobs['butter']:g}, "
          + ", ".join(f"{k}={v:g}" for k, v in sorted(knobs.items())
                      if k != "butter"))

    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    ).filter(pl.col("_date") == DATE)

    ids = frame["meta_flight_id"].unique(maintain_order=True).to_list()
    if args.smoke:
        ids = ids[:args.smoke]
    print(f"{len(ids)} flights")

    rows: list[dict] = []
    for n, fid in enumerate(ids):
        flight = frame.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
        signals = T.signals(flight)
        raw = signals["alt"]
        view = T.filtered(raw, knobs["butter"])
        exclusion = np.zeros(raw.size, dtype=bool)

        mcp = flight["bds_mcp_alt_sel_ft"].to_numpy().astype(float)
        fms = flight["bds_fms_alt_sel_ft"].to_numpy().astype(float)

        # `T.signals` writes NaN into the frozen head and tail rather than
        # shortening the array, so that indices stay aligned with the frame.
        # Normalising a position against `raw.size` therefore divides by an
        # extent the plateau can never reach: measured on the smoke cohort, 34
        # of 40 flights are trimmed and one loses 35 % of its rows. The finite
        # extent is what a position in the flight means.
        finite_idx = np.flatnonzero(np.isfinite(raw))
        if finite_idx.size == 0:
            # A flight whose altitude is NaN throughout produces no segment
            # anyway; skipping here keeps the extent well defined below.
            continue
        first, last = int(finite_idx[0]), int(finite_idx[-1])
        span = max(last - first, 1)

        for k, segment in enumerate(T.detect(algo, "alt", view, knobs, exclusion)):
            i, j = segment["start_idx"], segment["end_idx"]
            piece = raw[i:j + 1]
            finite = np.isfinite(piece)
            if not finite.any():
                continue
            # The plateau's announced value, in feet: the signal is scored in
            # metres and the register is broadcast in feet.
            value_ft = float(piece[finite].mean()) / FT_TO_M

            here = mcp[i:j + 1]
            present = np.isfinite(here)
            n_present = int(present.sum())
            broadcast = float(np.median(here[present])) if n_present else None
            distinct = int(np.unique(here[present]).size) if n_present else 0

            fms_here = fms[i:j + 1]
            fms_present = int(np.isfinite(fms_here).sum())

            difference = value_ft - broadcast if broadcast is not None else None
            rows.append({
                "flight_id": fid,
                "icao24": flight["raw_icao24"][0],
                "segment_index": k,
                "start_idx": i,
                "end_idx": j,
                "duration_s": (j - i + 1) * DT_S,
                "plateau_alt_ft": value_ft,
                "n_samples": int(finite.sum()),
                # Null when no register exists over this plateau: an UNPAIRED
                # plateau is a row, never a deletion.
                "mcp_broadcast_ft": broadcast,
                "mcp_samples_present": n_present,
                "mcp_distinct_values": distinct,
                "mcp_changes_within": distinct > 1,
                "fms_samples_present": fms_present,
                "difference_ft": difference,
                "difference_steps": (difference / STEP_FT
                                     if difference is not None else None),
                # Closed bound: |difference| <= 100 ft counts as within.
                "within_flight_level": (abs(difference) <= FLIGHT_LEVEL_FT
                                        if difference is not None else None),
                # Midpoint of the plateau along the FINITE extent, so measurement
                # 4 can describe the disagreeing set. A plateau spanning the
                # whole flight reads 0.5 — this is a centre of mass, not a start.
                "position_in_flight": ((i + j) / 2.0 - first) / span,
                # The stratum the smoke run revealed and SOURCES.md warned about
                # in the abstract: an aircraft on the ground with the MCP still
                # holding the level dialled on approach. Declared as a COLUMN,
                # never as a filter — the manifest forbids dropping plateaux for
                # disagreeing, and 1 000 ft is fixed here, before the full
                # distribution is seen, rather than chosen to suit it.
                "below_1000ft": value_ft < 1000.0,
            })

        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(ids)} flights, {len(rows)} plateaux")

    out = OUTPUTS / "smoke" if args.smoke else OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    frame_out = pl.DataFrame(rows)
    frame_out.write_csv(out / "paired_plateaux.csv")

    paired = frame_out.filter(pl.col("mcp_broadcast_ft").is_not_null())
    print(f"\n{frame_out.height} plateaux, {paired.height} paired "
          f"({paired.height / max(frame_out.height, 1) * 100:.1f} %)")
    if paired.height:
        d = paired["difference_ft"]
        print(f"  signed difference: median {d.median():+.1f} ft, "
              f"p5 {d.quantile(0.05):+.1f}, p95 {d.quantile(0.95):+.1f}")
        print(f"  within one flight level: "
              f"{paired['within_flight_level'].sum()} "
              f"({paired['within_flight_level'].mean() * 100:.1f} %)")
    print(f"wrote {out / 'paired_plateaux.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

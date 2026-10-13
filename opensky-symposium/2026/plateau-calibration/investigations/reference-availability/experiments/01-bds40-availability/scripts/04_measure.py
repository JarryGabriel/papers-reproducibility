"""Measure BDS 4,0 availability over the cohort.

Step ``measure`` of experiment 01-bds40-availability.

Writes the four aggregates the figures read, so a figure re-renders from
versioned evidence rather than from the 40 M-row Delta table:

* ``coverage_by_date.csv``       — does availability depend on the day
* ``coverage_by_typecode.csv``   — and on the aircraft type
* ``coverage_by_airframe.csv``   — and on the individual airframe
* ``signal_availability.csv``    — what the stream carries per signal

The cross-decoder census that settled the question is in
``05_comm_b_census.py``: it reads the raw Mode-S messages with an independent
decoder, which is what forecloses "your decoder is broken" as an explanation.

Reproduce:
    uv run python scripts/04_measure.py
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


from pathlib import Path
from typing import Annotated

import cyclopts

__all__ = ["app", "main"]

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
# parents: [0] experiments, [1] the paper, [2] papers/, [3] the repo root.
_REPO_ROOT = project_root(__file__)

DEFAULT_TABLE = data_dir() / 'flights.delta'
DEFAULT_OUT = _EXPERIMENT_ROOT / "outputs"

#: One row per signal in the availability table. The last entry has no column
#: on purpose: no Mode-S register carries a selected SPEED, and that absence is
#: the measurement — it is what the reconstruction has to supply.
SIGNALS: list[tuple[str, str]] = [
    ("raw_alt_ft", "altitude (ADS-B)"),
    ("raw_vz_ftmin", "vertical speed (ADS-B)"),
    ("bds_mach", "Mach (BDS 6,0)"),
    ("bds_ias_kt", "CAS/IAS (BDS 6,0)"),
    ("bds_mcp_alt_sel_ft", "selected altitude MCP (BDS 4,0)"),
    ("bds_fms_alt_sel_ft", "selected altitude FMS (BDS 4,0)"),
    ("era_temp_K", "ERA5 temperature"),
]

app = cyclopts.App(name="measure", help="Measure BDS 4,0 availability over the cohort.")


@app.default
def main(
    *,
    table: Annotated[Path, cyclopts.Parameter(help="Delta table.")] = DEFAULT_TABLE,
    out: Annotated[Path, cyclopts.Parameter(help="Measurement destination.")] = DEFAULT_OUT,
) -> None:
    """Write the four coverage aggregates."""
    import polars as pl

    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_delta(str(table)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("date")
    )
    print(f"cohort: {df.height:,} rows, {df['meta_flight_id'].n_unique():,} flights")

    pct = (100 * pl.col("mcp") / pl.col("rows")).round(2)

    (
        df.group_by("date")
        .agg(
            pl.len().alias("rows"),
            pl.col("meta_flight_id").n_unique().alias("flights"),
            pl.col("raw_icao24").n_unique().alias("airframes"),
            pl.col("bds_mcp_alt_sel_ft").is_not_null().sum().alias("mcp"),
            pl.col("bds_fms_alt_sel_ft").is_not_null().sum().alias("fms"),
        )
        .sort("date")
        .with_columns(
            pct.alias("mcp_pct"),
            (100 * pl.col("fms") / pl.col("rows")).round(2).alias("fms_pct"),
        )
        .write_csv(out / "coverage_by_date.csv")
    )

    (
        df.group_by("meta_aircraft_type")
        .agg(
            pl.len().alias("rows"),
            pl.col("meta_flight_id").n_unique().alias("flights"),
            pl.col("raw_icao24").n_unique().alias("airframes"),
            pl.col("bds_mcp_alt_sel_ft").is_not_null().sum().alias("mcp"),
        )
        .sort("meta_aircraft_type")
        .with_columns(pct.alias("mcp_pct"))
        .write_csv(out / "coverage_by_typecode.csv")
    )

    (
        df.group_by("raw_icao24")
        .agg(
            pl.col("meta_aircraft_type").first().alias("typecode"),
            pl.len().alias("rows"),
            pl.col("bds_mcp_alt_sel_ft").is_not_null().sum().alias("mcp"),
        )
        .with_columns(pct.alias("mcp_pct"))
        .sort("mcp_pct", descending=True)
        .write_csv(out / "coverage_by_airframe.csv")
    )

    rows: list[dict[str, object]] = [
        {
            "column": col,
            "label": label,
            "non_null": int(df[col].is_not_null().sum()),
            "pct": round(100 * df[col].is_not_null().sum() / df.height, 2),
        }
        for col, label in SIGNALS
    ]
    rows.append(
        {"column": "", "label": "selected speed (no register exists)", "non_null": 0, "pct": 0.0}
    )
    pl.DataFrame(rows).write_csv(out / "signal_availability.csv")

    for name in sorted(p.name for p in out.glob("*.csv")):
        print(f"  {name}")


if __name__ == "__main__":
    app()

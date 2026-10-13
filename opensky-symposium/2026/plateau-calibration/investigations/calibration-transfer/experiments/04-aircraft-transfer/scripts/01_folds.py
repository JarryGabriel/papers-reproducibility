"""Assign every airframe to one of five folds, once and for all.

The split is written to inputs/ and committed rather than recomputed at each
run. A selection regenerated per run can drift between executions, and every
number this experiment reports rests on the split being identical every time.

Two properties, both load-bearing:

  * **grouped by airframe** — every `raw_icao24` lands in exactly one fold, so
    the subsets are disjoint, but all contributed to global calibration. Repeated timestamps of one aircraft
    are not independent units; a split over samples would put the same airframe
    on both sides and measure memorisation rather than transfer.

  * **stratified by typecode** — each fold gets its share of every type, so a
    fold cannot come out badly merely for having drawn the turboprops.

The assignment is a deterministic function of the address and a versioned salt,
so there is no random state to share and no seed to trust:

    rank = position of SHA256(salt + ':' + raw_icao24) within its typecode
    fold = rank mod 5

Changing the split means changing the salt and saying so, never reshuffling in
place.

Reproduce:
    uv run --active python scripts/01_folds.py
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


import hashlib
from pathlib import Path

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = project_root(__file__)
COHORT = data_dir() / 'flights.delta'

DATE = "2025-11-01"
N_FOLDS = 5
#: Versioned. A different split is a different salt, declared as such.
SALT = "04-aircraft-transfer/v1"


def main() -> int:
    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    ).filter(pl.col("_date") == DATE)

    # One row per airframe, with its typecode and how much it contributes. The
    # counts are carried so the analysis can say whether a fold's result rests
    # on a handful of flights.
    airframes = (
        frame.group_by("raw_icao24")
        .agg(
            pl.col("meta_aircraft_type").first().alias("typecode"),
            pl.col("meta_flight_id").n_unique().alias("n_flights"),
            pl.len().alias("n_samples"),
        )
        .sort("raw_icao24")
    )

    rows: list[dict] = []
    for typecode in sorted(airframes["typecode"].unique().to_list()):
        here = airframes.filter(pl.col("typecode") == typecode)
        # Rank within the type by the hash, then deal round-robin. Ranking
        # rather than taking the hash modulo keeps the folds balanced even when
        # a type has few airframes.
        digests = [
            (hashlib.sha256(f"{SALT}:{icao}".encode()).hexdigest(), icao)
            for icao in here["raw_icao24"].to_list()
        ]
        for rank, (_, icao) in enumerate(sorted(digests)):
            row = here.filter(pl.col("raw_icao24") == icao).to_dicts()[0]
            rows.append({
                "raw_icao24": icao,
                "typecode": typecode,
                "fold": rank % N_FOLDS,
                "n_flights": int(row["n_flights"]),
                "n_samples": int(row["n_samples"]),
            })

    table = pl.DataFrame(rows).sort(["fold", "typecode", "raw_icao24"])
    table.write_csv(EXPERIMENT_DIR / "inputs" / "folds.csv")

    print(f"{table.height} airframes over "
          f"{table['typecode'].n_unique()} typecodes, salt {SALT!r}")
    summary = (table.group_by("fold")
               .agg(pl.len().alias("airframes"),
                    pl.col("typecode").n_unique().alias("types"),
                    pl.col("n_flights").sum().alias("flights"))
               .sort("fold"))
    print(summary.to_pandas().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

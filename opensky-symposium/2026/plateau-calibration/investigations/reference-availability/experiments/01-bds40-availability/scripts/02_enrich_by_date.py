"""Enrich the cohort one date at a time, pruning ERA5 between dates.

Step ``enrich`` of experiment 01-bds40-availability.

``fastmeteo`` fills its local zarr store hour by hour from the earliest to the
latest timestamp it is handed (``Grid.sync_local``), with no notion of which
days are actually wanted. Handing it the whole four-date table therefore asks
for every hour between February and November — 258 days of global hourly fields
at roughly 4 GB each. That filled the disk at the fifth day of February on the
first attempt.

Bounding each call to one date makes the download proportional to the data. And
because the interpolated values land in the Delta table, the local ERA5 store is
only ever an intermediate: it can be pruned once a date is enriched, so the peak
footprint is one day (~4 GB) rather than the sum of four.

``write_columns`` merges rather than replaces, so the dates accumulate correctly
across calls.

Reproduce:
    uv run python scripts/02_enrich_by_date.py            # all four dates
    uv run python scripts/02_enrich_by_date.py --keep-cache
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
from opensky26_plateau.project_paths import data_dir, experiment_dir, pipeline_config, project_root, call_pipeline, resolved_pipeline_config


import json
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import cyclopts

__all__ = ["app", "main", "prune_era5"]

DATES = [date(2025, 2, 17), date(2025, 5, 21), date(2025, 8, 29), date(2025, 11, 1)]

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_PAPER_ROOT = project_root(__file__)
_REPO_ROOT = project_root(__file__)

DEFAULT_CONFIG = pipeline_config()
DEFAULT_CACHE = data_dir() / 'era5_cache'
DEFAULT_LOG = _EXPERIMENT_ROOT / "outputs" / "enrich_log.json"

#: The variables carrying the per-hour chunks. The coordinate arrays
#: (latitude, longitude, level) are tiny and shared, and must survive a prune:
#: fastmeteo refuses a store whose declared features are missing.
_CHUNKED = ("temperature", "u_component_of_wind", "v_component_of_wind")

app = cyclopts.App(name="enrich-by-date", help="Enrich one date at a time, pruning ERA5 between.")


def prune_era5(cache: Path) -> int:
    """Drop every hourly chunk from *cache*, keeping the store's structure.

    Returns the number of chunks removed. The coordinate arrays and the
    ``.zarray`` metadata stay: fastmeteo reopens the store and appends, and it
    rejects a store whose declared features are absent. Shapes are reset to zero
    along the time axis so the metadata matches what is left on disk.
    """
    removed = 0
    for name in (*_CHUNKED, "time"):
        arr = cache / name
        if not arr.is_dir():
            continue
        for chunk in arr.iterdir():
            if chunk.name.startswith("."):
                continue
            chunk.unlink()
            removed += 1
        meta_path = arr / ".zarray"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["shape"][0] = 0
            meta_path.write_text(json.dumps(meta, indent=4))
    consolidated = cache / ".zmetadata"
    if consolidated.exists():
        consolidated.unlink()
    return removed


def _size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=False)
    return int(out.stdout.split()[0]) / 1024 / 1024 if out.stdout.strip() else 0.0


@app.default
def main(
    *,
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config.")] = DEFAULT_CONFIG,
    cache: Annotated[Path, cyclopts.Parameter(help="ERA5 local store.")] = DEFAULT_CACHE,
    log_path: Annotated[
        Path, cyclopts.Parameter(name="--log", help="Where to record what ran.")
    ] = DEFAULT_LOG,
    keep_cache: Annotated[
        bool,
        cyclopts.Parameter(name="--keep-cache", help="Do not prune ERA5 between dates."),
    ] = False,
) -> None:
    """Run ``enrich`` once per date, pruning the ERA5 store between each."""
    from node_fdm_pipeline.commands.data import enrich

    record: list[dict[str, object]] = []
    for day in DATES:
        before = _size_gb(cache)
        call_pipeline(enrich,
            config=config,
            start_date=day.isoformat(),
            end_date=(day + timedelta(days=1)).isoformat(),
        )
        after = _size_gb(cache)

        entry: dict[str, object] = {
            "date": day.isoformat(),
            "era5_gb_before": round(before, 2),
            "era5_gb_after": round(after, 2),
            "era5_gb_downloaded": round(after - before, 2),
        }
        if not keep_cache:
            entry["chunks_pruned"] = prune_era5(cache)
            entry["era5_gb_pruned_to"] = round(_size_gb(cache), 2)
        record.append(entry)
        print(f"  {day}: {entry}", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"log: {log_path}")


if __name__ == "__main__":
    app()

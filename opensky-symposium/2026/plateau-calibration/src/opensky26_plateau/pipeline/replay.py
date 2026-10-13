"""Rebuild the input channels from a supplied aircraft list and dated flight plan.

No aircraft selection, detector calibration or historical outputs are regenerated.
The destination must be new: decoding a shared cache could include other dates
and aircraft. Run journals describe this execution, not historical executions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def replay(*, plan: Path, aircraft: Path, config: Path, destination: Path,
           smoke: int = 0, dry_run: bool = False, commands=None):
    """Prepare a new, isolated data store; dry-run only validates supplied inputs.

    ERA5 calls cover individual plan dates. Caches are retained (not deleted),
    so allow roughly 4-5 GB per date in addition to the trajectory data.
    """
    if smoke < 0:
        raise ValueError("smoke must be non-negative")
    rows, fleet = _rows(plan), _rows(aircraft)
    if not rows or not fleet:
        raise ValueError("The plan and aircraft inputs must not be empty")
    aircraft_ids = {row["icao24"] for row in fleet}
    if len(aircraft_ids) != len(fleet):
        raise ValueError("Duplicate aircraft in supplied list")
    if any(row["icao24"] not in aircraft_ids for row in rows):
        raise ValueError("Plan contains unknown aircraft")
    keys = [(row["icao24"], row["day"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate aircraft/day in plan")
    if smoke:
        chosen = set(sorted({row["icao24"] for row in rows})[:smoke])
        rows = [row for row in rows if row["icao24"] in chosen]
    selected = {row["icao24"] for row in rows}
    fleet = [row for row in fleet if row["icao24"] in selected]
    dates = sorted({datetime.fromisoformat(row["day"]).date() for row in rows})
    destination = destination.expanduser().resolve()
    settings = yaml.safe_load(config.read_text())
    paths = settings.setdefault("paths", {})
    paths.update(data_dir=str(destination), delta_table="flights.delta",
                 era5_cache_dir="era5_cache", preprocess_dir="preprocessed_parquet")
    record = {
        "status": "planned", "dates": [day.isoformat() for day in dates],
        "airframes": len(selected), "aircraft_days": len(rows), "smoke": smoke,
        "destination": str(destination), "completed": [],
        "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in [("plan", plan), ("aircraft", aircraft),
                                             ("config", config)]},
        "resolved_config": settings,
    }
    if destination.exists():
        raise FileExistsError(f"Use a new isolated destination: {destination}")
    if dry_run:
        return record
    if commands is None:
        from node_fdm_pipeline.commands import data as commands
    versions = {}
    for name in ("node-fdm-pipeline", "node-fdm-data", "traffic", "rs1090", "fastmeteo"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    record.update(versions=versions, started_at=datetime.now(UTC).isoformat())
    destination.mkdir(parents=True)
    for name, entries in [("aircraft_db.csv", fleet), ("flight_plan.csv", rows)]:
        with (destination / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    active_config = destination / "config.yaml"
    active_config.write_text(yaml.safe_dump(settings, sort_keys=False))
    journal = destination / "replay.json"

    def save():
        journal.write_text(json.dumps(record, indent=2) + "\n")

    def step(name, **kwargs):
        record.update(status="running", current_step=name, current_arguments=kwargs)
        save()
        getattr(commands, name)(config=active_config, **kwargs)
        record["completed"].append({"step": name, "arguments": kwargs,
                                    "finished_at": datetime.now(UTC).isoformat()})
        save()

    # Path arguments are strings in the journal and Paths at the API boundary.
    def download():
        record.update(status="running", current_step="download")
        save()
        commands.download(config=active_config, flight_plan=destination / "flight_plan.csv")
        record["completed"].append({"step": "download_and_decode",
                                    "finished_at": datetime.now(UTC).isoformat()})
        save()

    try:
        download()  # node-fdm download automatically chains decode.
        for name in ("identify", "preprocess", "flag"):
            step(name)
        for day in dates:
            # Separate stores prevent a cache implementation from filling the
            # months between already cached dates. Nothing is deleted.
            paths["era5_cache_dir"] = f"era5_cache/{day.isoformat()}"
            active_config.write_text(yaml.safe_dump(settings, sort_keys=False))
            step("enrich", start_date=day.isoformat(),
                 end_date=(day + timedelta(days=1)).isoformat())
        step("clean_speeds")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    record.update(status="complete", finished_at=datetime.now(UTC).isoformat())
    save()
    return record


def main():
    from opensky26_plateau.project_paths import experiment_dir, pipeline_config

    experiment = experiment_dir("01-bds40-availability")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=experiment / "inputs/flight_plan.csv")
    parser.add_argument("--aircraft", type=Path, default=experiment / "inputs/aircraft_db.csv")
    parser.add_argument("--config", type=Path, default=pipeline_config())
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path)
    args = vars(parser.parse_args())
    log = args.pop("log") or args["destination"] / "preparation_log.json"
    result = replay(**args)
    if not args["dry_run"]:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

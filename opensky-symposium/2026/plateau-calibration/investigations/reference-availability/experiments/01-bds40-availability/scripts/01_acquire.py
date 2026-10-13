"""Acquire the cohort declared in ``inputs/flight_plan.csv``.

Step ``acquire`` of experiment 01-bds40-availability.

The plan is an INPUT, not a product: it is written once, declared in
``inputs/SOURCES.md`` and consumed here. Regenerating it per run would let the
airframe selection drift between executions, and the selection is what makes the
four dates comparable — the same 200 airframes on each, so day and fleet
composition stop being confounded, which is what archived experiment 11 failed
to separate.

``--smoke N`` restricts the run to N airframes across every date. A full pass
costs 800 aircraft-day fetches plus three days of ERA5, so the chain is proved on
a couple of airframes first. The smoke run writes to a separate data directory: a
partial cohort must never be mistaken for the real one downstream, since the
cache keys on ``(date, icao24)`` and a later full run would silently top it up
rather than replace it.

The raw cache is NOT re-fetched by default. It holds the acquisition, which is
immutable: the same day fetched again returns the same messages. What moved was
the decode -- the November cohort was decoded before the EHS fixes and carried
zero BDS 4,0 values, while the same cached bytes decoded with the current code
yield 1 140 662. Re-running ``decode`` is what corrects a cohort; re-downloading
is not, and it costs 200 Trino queries to prove it.

Reproduce:
    uv run python scripts/01_acquire.py --smoke 2     # prove the chain
    uv run python scripts/01_acquire.py               # the full cohort
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
import shutil
from pathlib import Path
from typing import Annotated

import cyclopts

__all__ = ["app", "main"]

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_PAPER_ROOT = project_root(__file__)
_REPO_ROOT = project_root(__file__)

DEFAULT_PLAN = _EXPERIMENT_ROOT / "inputs" / "flight_plan.csv"
DEFAULT_CONFIG = pipeline_config()
DEFAULT_LOG = _EXPERIMENT_ROOT / "outputs" / "acquisition_log.json"

app = cyclopts.App(name="acquire", help="Acquire the four-date cohort from the frozen plan.")


def _smoke_plan(plan: Path, n: int, dest: Path) -> Path:
    """Write a plan restricted to *n* airframes, kept across every date.

    Airframes are taken in sorted order rather than sampled: a smoke test must
    select the same aircraft on every run, or a failure cannot be reproduced.
    One airframe per typecode is preferred while there are typecodes left, so a
    small N still exercises more than one fleet.
    """
    import polars as pl

    frame = pl.read_csv(plan, infer_schema=False)
    chosen = (
        frame.select("icao24", "typecode")
        .unique()
        .sort("typecode", "icao24")
        .group_by("typecode", maintain_order=True)
        .head(1)
        .head(n)["icao24"]
        .to_list()
    )
    out = frame.filter(pl.col("icao24").is_in(chosen)).sort("day", "icao24")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dest)
    return dest


def _smoke_config(config: Path, dest_dir: Path) -> Path:
    """Copy *config* with ``paths.data_dir`` redirected under *dest_dir*.

    A smoke run must not write into the real cohort's directory: a partial
    download that later looks like a complete one is exactly the kind of silent
    corruption this experiment exists to rule out.
    """
    import yaml

    cfg = yaml.safe_load(config.read_text())
    cfg.setdefault("paths", {})["data_dir"] = str(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # The airframe list lives beside the data directory, and download reads it.
    src_db = data_dir() / 'aircraft_db.csv'
    if src_db.exists():
        shutil.copy2(src_db, dest_dir / "aircraft_db.csv")
    return out


@app.default
def main(
    *,
    plan: Annotated[Path, cyclopts.Parameter(help="The frozen flight plan.")] = DEFAULT_PLAN,
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config.")] = DEFAULT_CONFIG,
    log_path: Annotated[
        Path, cyclopts.Parameter(name="--log", help="Acquisition log destination.")
    ] = DEFAULT_LOG,
    smoke: Annotated[
        int,
        cyclopts.Parameter(help="Restrict to N airframes and write to a scratch directory."),
    ] = 0,
    force_refresh: Annotated[
        bool,
        cyclopts.Parameter(
            name="--force-refresh",
            help="Re-fetch every (date, icao24), ignoring the raw cache.",
        ),
    ] = False,
    dry_run: Annotated[bool, cyclopts.Parameter(help="Validate without any I/O.")] = False,
) -> None:
    """Download and decode the cohort named by the plan."""
    import polars as pl
    from node_fdm_pipeline.commands.data import _read_flight_plan, download

    if not plan.exists():
        raise SystemExit(f"{plan}: the flight plan is an input and must exist")

    active_plan, active_config = plan, config
    scratch: Path | None = None
    if smoke:
        scratch = _EXPERIMENT_ROOT / "outputs" / f"smoke-{smoke}"
        active_plan = _smoke_plan(plan, smoke, scratch / "flight_plan.csv")
        active_config = _smoke_config(config, scratch)

    resolved = _read_flight_plan(active_plan)
    frame = pl.read_csv(active_plan, infer_schema=False)

    record = {
        "plan": str(active_plan),
        "config": str(active_config),
        "smoke": smoke or None,
        "data_dir": str(scratch) if scratch else None,
        "force_refresh": force_refresh,
        "dates": sorted(resolved),
        "airframes": frame["icao24"].n_unique(),
        "aircraft_days": sum(len(v) for v in resolved.values()),
        "typecodes": sorted(frame["typecode"].unique().to_list()),
    }

    print(f"plan: {active_plan}")
    print(
        f"  {record['airframes']} airframes x {len(resolved)} dates "
        f"= {record['aircraft_days']} aircraft-days"
    )
    for day in sorted(resolved):
        print(f"  {day}: {len(resolved[day])} aircraft")
    if smoke:
        print(f"  SMOKE RUN — writing to {scratch}")

    # The log is provenance — which airframes, which dates, resolved from which
    # plan — and none of that depends on the download having run. Write it
    # before, so a dry run still records what it would have fetched and the
    # step's declared output exists either way.
    record["downloaded"] = not dry_run
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"log: {log_path}")

    if dry_run:
        print("--dry-run: stopping before download")
        return

    call_pipeline(download, config=active_config, override_data=not bool(smoke),
                  flight_plan=active_plan, force_refresh=force_refresh)


if __name__ == "__main__":
    app()

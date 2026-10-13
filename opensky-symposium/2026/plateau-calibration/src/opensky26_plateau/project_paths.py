"""Project locations independent of checkout depth and working directory.

OPENSKY26_DATA_DIR names the directory containing flights.delta, raw/extended,
era5_cache and test_flights.parquet. Relative overrides resolve from the project,
never from an ancestor repository. OPENSKY26_PIPELINE_CONFIG selects an explicit
node-fdm pipeline configuration; the external pipeline must be installed.
"""
from __future__ import annotations

import os
import tomllib
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def _is_project_root(root: Path) -> bool:
    if (root / "project.yaml").is_file():
        return True
    try:
        public = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, ValueError):
        return False
    return public.get("project", {}).get("name") == "plateau-calibration"


def project_root(start: str | Path | None = None) -> Path:
    """Find this project's marker or standard Python package metadata."""
    override = os.environ.get("OPENSKY26_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not _is_project_root(root):
            raise FileNotFoundError(f"No project marker at OPENSKY26_PROJECT_ROOT={root}")
        return root
    anchor = Path(start or __file__).resolve()
    for candidate in (anchor, *anchor.parents):
        if _is_project_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"No project marker above {anchor}; set OPENSKY26_PROJECT_ROOT for an installed wheel"
    )


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    if not value:
        return default
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def data_dir() -> Path:
    """Return the explicit data location without creating or downloading data."""
    return _configured_path("OPENSKY26_DATA_DIR", project_root() / "data")


def pipeline_config() -> Path:
    """Return the configured acquisition pipeline YAML location."""
    return _configured_path(
        "OPENSKY26_PIPELINE_CONFIG",
        project_root() / "src" / "opensky26_plateau" / "pipeline" / "config.yaml",
    )


def experiment_dir(experiment_id: str, *, root: Path | None = None) -> Path:
    """Resolve a global experiment ID, rejecting absent or ambiguous matches."""
    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError(f"Expected a global experiment ID, got {experiment_id!r}")
    base = Path(root) if root is not None else project_root()
    matches = sorted(
        path for path in base.glob("investigations/*/experiments/*")
        if path.is_dir() and path.name == experiment_id
    )
    if not matches:
        raise FileNotFoundError(f"Experiment {experiment_id!r} missing under {base}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous experiment ID {experiment_id!r}: {matches}")
    return matches[0]


@contextmanager
def resolved_pipeline_config(config: Path, *, override_data: bool = True):
    """Pass absolute data paths to the external pipeline without editing its YAML.

    The environment override wins; otherwise a configured relative data directory
    is interpreted against project.yaml. Smoke configurations already contain an
    absolute scratch path and can opt out of the environment override.
    """
    import yaml

    settings = yaml.safe_load(Path(config).read_text())
    paths = settings.setdefault("paths", {})
    configured = Path(paths.get("data_dir", "data")).expanduser()
    if os.environ.get("OPENSKY26_DATA_DIR") and override_data:
        configured = data_dir()
    elif not configured.is_absolute():
        configured = project_root() / configured
    paths["data_dir"] = str(configured.resolve())
    with TemporaryDirectory(prefix="opensky26-config-") as temporary:
        target = Path(temporary) / "config.yaml"
        target.write_text(yaml.safe_dump(settings, sort_keys=False))
        yield target


def call_pipeline(command, *, config: Path, override_data: bool = True, **kwargs):
    """Invoke a pipeline command with a normalized, temporary configuration."""
    with resolved_pipeline_config(config, override_data=override_data) as active_config:
        return command(config=active_config, **kwargs)

"""Per-channel path resolution for Paper 0-A.

Each channel (vz, mach, cas, ...) is a self-contained folder under
``channels/<channel>/`` with ``figures/``, ``results/`` and ``ANALYSIS.md``.
All scripts take a ``--channel`` argument and resolve their I/O through here,
so the same code serves every channel.
"""

from __future__ import annotations

from opensky26_plateau.project_paths import data_dir, experiment_dir, pipeline_config, project_root


from pathlib import Path

# src/opensky26_plateau/paths.py -> repository root is three levels up.
REPO = project_root(__file__)
DATA = data_dir() / 'test_flights.parquet'

# Display metadata per channel: (pretty name, tolerance/value unit).
CHANNEL_META = {
    "vz": ("vertical speed", "ft/min"),
    "alt": ("altitude", "m"),
    "gamma": ("flight-path angle", "rad"),
    "mach": ("Mach", ""),
    "cas": ("CAS", "kt"),
}


def channel_name(channel: str) -> str:
    return CHANNEL_META.get(channel, (channel, ""))[0]


def channel_unit(channel: str) -> str:
    return CHANNEL_META.get(channel, (channel, ""))[1]


def channel_dir(channel: str) -> Path:
    return REPO / "results" / channel


def figures_dir(channel: str) -> Path:
    d = channel_dir(channel) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(channel: str) -> Path:
    d = channel_dir(channel) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pareto_parquet(channel: str) -> Path:
    return results_dir(channel) / "pareto.parquet"


def cache_parquet(channel: str) -> Path:
    return results_dir(channel) / "_detect_cache.parquet"

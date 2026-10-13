"""Census the raw Mode-S stream with an INDEPENDENT decoder.

Step ``census`` of experiment 01-bds40-availability.

This is the measurement that settles the question. The paper's first
contribution rests on 0 finite values out of 2 805 025 on both BDS 4,0
selected-altitude registers; counting those registers again with the same
decoder that produced the zero would prove nothing. So the raw messages are
decoded with ``pyModeS``, which shares no code with the ``rs1090`` path
node-fdm uses.

Three things come out of it, and together they close the falsifier:

* how much of the stream is Comm-B at all (is there anything to decode)
* what share of it identifies as BDS 4,0 (is the register broadcast)
* whether the aircraft themselves declare supporting 4,0, in their BDS 1,7
  capability report (do they claim to send it)

Reproduce:
    uv run --with pyModeS python scripts/05_comm_b_census.py
    uv run --with pyModeS python scripts/05_comm_b_census.py --sample 200000
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
from collections import Counter
from pathlib import Path
from typing import Annotated

import cyclopts

__all__ = ["app", "main"]

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
# parents: [0] experiments, [1] the paper, [2] papers/, [3] the repo root.
_REPO_ROOT = project_root(__file__)

DEFAULT_RAW = data_dir() / 'raw' / 'extended'
DEFAULT_OUT = _EXPERIMENT_ROOT / "outputs" / "comm_b_census.json"

#: Downlink formats carrying a Comm-B reply. DF4/DF5 are surveillance replies
#: with no Comm-B payload, so they are counted but never decoded for a register.
COMM_B_DF: frozenset[int] = frozenset({20, 21})

app = cyclopts.App(name="comm-b-census", help="Census the raw Mode-S stream with pyModeS.")


@app.default
def main(
    *,
    raw: Annotated[Path, cyclopts.Parameter(help="Raw `extended` cache.")] = DEFAULT_RAW,
    out: Annotated[Path, cyclopts.Parameter(help="Census destination.")] = DEFAULT_OUT,
    sample: Annotated[
        int,
        cyclopts.Parameter(help="Stop after roughly this many raw messages (0 = all)."),
    ] = 400_000,
    seed: Annotated[int, cyclopts.Parameter(help="Seed for the file shuffle.")] = 0,
) -> None:
    """Decode the raw Comm-B stream and count what it carries."""
    import random

    import polars as pl
    import pyModeS as pms  # noqa: N813 - the package spells itself that way

    files = sorted(raw.glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"{raw}: no raw messages — run scripts/01_acquire.py first")
    # Shuffle under a fixed seed so a sample spans airframes rather than taking
    # the first few whole, and so the same sample comes back on a re-run.
    random.Random(seed).shuffle(files)

    messages: list[str] = []
    for path in files:
        frame = pl.read_parquet(path)
        if frame.height:
            messages += frame["rawmsg"].to_list()
        if sample and len(messages) >= sample:
            break

    df_counts = Counter(int(m[0:2], 16) >> 3 for m in messages)
    comm_b = [m for m in messages if (int(m[0:2], 16) >> 3) in COMM_B_DF]

    registers: Counter[str] = Counter()
    capability: Counter[str] = Counter()
    bds40_with_mcp = 0
    bds40_with_fms = 0
    for msg in comm_b:
        try:
            decoded = pms.decode(msg)
        except Exception:  # an undecodable frame is a measurement, not a failure
            registers["undecodable"] += 1
            continue
        bds = decoded.get("bds")
        registers[str(bds)] += 1
        if bds == "1,7":
            for supported in decoded.get("supported_bds", []):
                capability[supported] += 1
        elif bds == "4,0":
            bds40_with_mcp += decoded.get("selected_altitude_mcp") is not None
            bds40_with_fms += decoded.get("selected_altitude_fms") is not None

    n_commb = len(comm_b)
    census = {
        "raw_messages": len(messages),
        "files_read": len(files),
        "seed": seed,
        "decoder": f"pyModeS {getattr(pms, '__version__', 'unknown')}",
        "downlink_formats": dict(sorted(df_counts.items())),
        "comm_b_messages": n_commb,
        "comm_b_pct": round(100 * n_commb / len(messages), 2) if messages else 0.0,
        "registers": {
            k: {"messages": v, "pct": round(100 * v / n_commb, 2) if n_commb else 0.0}
            for k, v in registers.most_common()
        },
        "bds40": {
            "messages": registers.get("4,0", 0),
            "with_selected_altitude_mcp": bds40_with_mcp,
            "with_selected_altitude_fms": bds40_with_fms,
        },
        "capability_bds17": dict(capability.most_common()),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(census, indent=2) + "\n")

    print(f"raw messages   {census['raw_messages']:,}")
    print(f"Comm-B         {n_commb:,} ({census['comm_b_pct']}%)")
    for reg, stats in list(census["registers"].items())[:6]:
        print(f"  BDS {reg:<12} {stats['messages']:>8,}  {stats['pct']:>5.2f}%")
    print(f"BDS 4,0 with a selected altitude: {bds40_with_mcp:,}")
    print(f"aircraft declaring 4,0 support:   {capability.get('4,0', 0):,} reports")
    print(f"census: {out}")


if __name__ == "__main__":
    app()

"""Cohort description figures for the paper's data section.

Step ``figures`` of experiment 01-bds40-availability.

Five panels, each answering a question a reader of the data section asks:

1. what the cohort IS — flights and samples per typecode, per date
2. does availability depend on the DAY — BDS 4,0 coverage across four dates
3. is coverage UNIFORM — its distribution over the 200 airframes
4. what the CHANNELS carry — per-signal availability, the reconstruction premise
5. what a set-point LOOKS like — one flight, altitude against its commanded value

Panel 5 is the one that carries the finding: a reader who has been told the
register is empty needs to see a commanded altitude next to the trajectory that
follows it.

Reproduce:
    uv run python scripts/03_cohort_figures.py
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


import sys
from pathlib import Path
from typing import Annotated

import cyclopts

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_PAPER_ROOT = project_root(__file__)
_REPO_ROOT = project_root(__file__)


DEFAULT_TABLE = data_dir() / 'flights.delta'
DEFAULT_OUT = _EXPERIMENT_ROOT / "figures"
DEFAULT_MEASURES = _EXPERIMENT_ROOT / "outputs"

DATE_LABEL = {
    "2025-02-17": "17 Feb",
    "2025-05-21": "21 May",
    "2025-08-29": "29 Aug",
    "2025-11-01": "1 Nov",
}

app = cyclopts.App(name="cohort-figures", help="Render the cohort description figures.")


def _load(table: Path):
    import polars as pl

    df = pl.read_delta(str(table))
    return df.with_columns(pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("date"))


def fig_composition(measures: Path, out: Path) -> str:
    """Flights and samples per typecode — what the cohort is made of."""
    import matplotlib.pyplot as plt
    from opensky26_plateau import paper_style as PS
    import polars as pl

    agg = (
        pl.read_csv(measures / "coverage_by_typecode.csv")
        .rename({"rows": "samples"})
        .sort("flights", descending=True)
    )
    types = agg["meta_aircraft_type"].to_list()
    fig, ax = PS.figure(figsize=(6.4, 3.4))
    ax.bar(types, agg["flights"].to_list(), color=PS.type_colors(len(types)))
    ax.set_ylabel("flights  (label: 4 s samples)")
    ax.set_xlabel("ICAO type-code")
    ax.set_title("Cohort composition: 200 airframes, 20 per type-code", loc="left")
    # The airframe count is 20 for every type-code by construction and already in
    # the title; the sample count is what actually varies between them.
    for i, (f, s) in enumerate(zip(agg["flights"], agg["samples"], strict=True)):
        ax.text(i, f, f"{s / 1000:.0f}k", ha="center", va="bottom", fontsize=7, color="0.35")
    ax.margins(y=0.15)
    fig.savefig(out / "fig_cohort_composition.png")
    plt.close(fig)
    return "fig_cohort_composition.png"


def fig_by_date(measures: Path, out: Path) -> str:
    """BDS 4,0 coverage per date — does availability depend on the day."""
    import matplotlib.pyplot as plt
    from opensky26_plateau import paper_style as PS
    import polars as pl

    agg = pl.read_csv(measures / "coverage_by_date.csv").sort("date")
    labels = [DATE_LABEL.get(d, d) for d in agg["date"].to_list()]
    x = range(len(labels))
    fig, ax = PS.figure(figsize=(5.2, 3.4))
    ax.bar([i - 0.19 for i in x], agg["mcp_pct"].to_list(), width=0.38,
           label="MCP/FCU selected altitude", color=PS.ALGO["bilateral"])
    ax.bar([i + 0.19 for i in x], agg["fms_pct"].to_list(), width=0.38,
           label="FMS selected altitude", color=PS.ALGO["savgol"])
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("samples carrying a set-point (%)")
    ax.set_title("BDS 4,0 availability across four dates spread over the year", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.margins(y=0.2)
    fig.savefig(out / "fig_bds40_by_date.png")
    plt.close(fig)
    return "fig_bds40_by_date.png"


def fig_per_airframe(measures: Path, out: Path) -> str:
    """Coverage distribution over airframes — the aggregate hides concentration."""
    import matplotlib.pyplot as plt
    from opensky26_plateau import paper_style as PS
    import polars as pl

    per = pl.read_csv(measures / "coverage_by_airframe.csv").sort(
        "mcp_pct", descending=True
    )
    pct = per["mcp_pct"].to_list()
    silent = sum(1 for p in pct if p == 0)
    fig, ax = PS.figure(figsize=(6.4, 3.4))
    ax.bar(range(len(pct)), pct, width=1.0,
           color=[PS.RAW if p == 0 else PS.ALGO["bilateral"] for p in pct])
    ax.set_xlabel("airframe, ranked by coverage")
    ax.set_ylabel("samples carrying a set-point (%)")
    ax.set_title("Coverage is a property of the airframe, not the fleet", loc="left")
    ax.axvspan(len(pct) - silent, len(pct), color=PS.EXCL, zorder=0)
    ax.annotate(f"{silent} airframes never broadcast it",
                xy=(len(pct) - silent / 2, 2), xytext=(len(pct) * 0.42, 30),
                fontsize=7, color="0.35",
                arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 0.8,
                            "connectionstyle": "arc3,rad=-0.2"})
    ax.set_xlim(0, len(pct))
    fig.savefig(out / "fig_coverage_per_airframe.png")
    plt.close(fig)
    return "fig_coverage_per_airframe.png"


def fig_channels(measures: Path, out: Path) -> str:
    """Per-signal availability — why reconstruction is still required."""
    import matplotlib.pyplot as plt
    from opensky26_plateau import paper_style as PS
    import polars as pl

    color_for = {
        "raw_alt_ft": "0.55",
        "raw_vz_ftmin": "0.55",
        "bds_mach": PS.CHANNEL["mach"],
        "bds_ias_kt": PS.CHANNEL["cas"],
        "bds_mcp_alt_sel_ft": PS.ALGO["bilateral"],
        "bds_fms_alt_sel_ft": PS.ALGO["savgol"],
    }
    frame = pl.read_csv(measures / "signal_availability.csv").filter(
        pl.col("label") != "ERA5 temperature"
    )
    names = frame["label"].to_list()
    vals = frame["pct"].to_list()
    colors = [color_for.get(c or "", "0.85") for c in frame["column"].fill_null("")]
    fig, ax = PS.figure(figsize=(5.6, 3.4))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("samples with a finite value (%)")
    ax.set_title("What the stream carries, and what it does not", loc="left")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1, i, "no register" if v == 0 else f"{v:.0f}%",
                va="center", fontsize=7, color="0.35")
    ax.set_xlim(0, 108)
    fig.savefig(out / "fig_channel_availability.png")
    plt.close(fig)
    return "fig_channel_availability.png"


def _frozen_ends(cols: list[list], min_run: int = 15) -> tuple[int, int]:
    """Return the first and last index that is not part of a frozen end.

    The tail of a flight often carries a stretch where EVERY channel repeats
    its previous value: on the example flight, from 254.7 min onward, altitude
    holds 125 ft, ground speed holds 95 kt and track holds
    95.41186915231704 -- fifteen identical decimals for eight minutes. An
    aircraft taxiing does not reproduce a heading to the fifteenth decimal;
    a decoder repeating its last frame does. The position keeps moving
    underneath, which is why per-field validity flags do not catch it: each
    channel on its own looks plausible, and only their SIMULTANEOUS constancy
    gives it away.

    One channel alone must never trigger this. A flat altitude is a level-off,
    which is precisely what the paper exists to detect; a flat Mach is a held
    Mach. It takes every channel freezing together, for at least *min_run*
    samples (a minute on the 4 s grid), to mean the signal stopped rather than
    the aircraft.

    Only the ends are trimmed. An interior frozen stretch is a dropout worth
    seeing, and cutting it would hide the anomaly a diagnostic plot exists for.
    """
    n = len(cols[0])

    def frozen_at(i: int) -> bool:
        # A sample continues a freeze when every channel that HAS a value
        # equals its predecessor. A null does not break the run: an absent
        # value is not the signal resuming, and a flight whose tail decays
        # into nulls (this one ends on 107 of them) would otherwise reset the
        # counter and keep the whole frozen stretch. A sample where every
        # channel is null is not evidence of movement either, so it continues
        # the run rather than ending it.
        seen = False
        for col in cols:
            a, b = col[i - 1], col[i]
            if a is None or b is None:
                continue
            seen = True
            if a != b:
                return False
        return True or seen

    first = 0
    run = 0
    for i in range(1, n):
        if frozen_at(i):
            run += 1
        else:
            if run >= min_run and first == 0:
                first = i - 1
            break
    if run >= min_run and first == 0:
        first = min(run, n - 1)

    # Walk in from the last sample that carries any value at all: a trailing
    # run of pure nulls is already absent from the plot and says nothing about
    # where the signal died.
    tail = n - 1
    while tail > 0 and all(col[tail] is None for col in cols):
        tail -= 1

    last = tail
    run = 0
    for i in range(tail, 0, -1):
        if frozen_at(i):
            run += 1
        else:
            if run >= min_run:
                last = i
            break

    return first, last


def _mask_invalid(values: list, cols: list[list]) -> list:
    """Null the samples belonging to a frozen run at either end of the flight."""
    first, last = _frozen_ends(cols)
    return [v if first <= i <= last else None for i, v in enumerate(values)]


def _despike(values: list, jump: float = 1500.0, max_run: int = 6) -> list:
    """Null short excursions that leave a level and return to it.

    A decode artefact shows as a run of samples that departs from the
    surrounding value by more than *jump* and comes back to it, over at most
    *max_run* samples. Real flight does not do that: an aircraft cannot climb
    1 500 ft and return within 24 s, and a set-point that genuinely changes
    stays changed.

    The threshold is deliberately below a flight level (2 000 ft): the artefact
    this removes from the example flight is a single sample reading 36 000 ft
    between two 34 000 ft neighbours — a false-positive register identification,
    the known Mode-S failure mode this experiment's limits name. It is worth
    stating plainly: the register exists, and it is not error-free.

    Nulling rather than dropping keeps the x-axis honest — the samples stay
    absent from the line instead of being bridged by interpolation.
    """
    out = list(values)
    n = len(out)
    i = 1
    while i < n - 1:
        before = out[i - 1]
        if before is None or out[i] is None or abs(out[i] - before) <= jump:
            i += 1
            continue
        # A departure: find where it comes back to the level it left.
        end = i
        while end < n - 1 and end - i < max_run:
            after = out[end + 1]
            if after is not None and abs(after - before) <= jump:
                for k in range(i, end + 1):
                    out[k] = None
                break
            end += 1
        i = end + 1
    return out


def fig_example(df, out: Path) -> str:
    """One flight: altitude against the altitude that was commanded."""
    import matplotlib.pyplot as plt
    from opensky26_plateau import paper_style as PS
    import polars as pl

    per = (
        df.group_by("meta_flight_id")
        .agg(
            pl.col("bds_mcp_alt_sel_ft").is_not_null().sum().alias("n"),
            pl.col("bds_mcp_alt_sel_ft").n_unique().alias("levels"),
        )
        .filter(pl.col("levels") >= 4)
        .sort("n", descending=True)
    )
    if per.is_empty():
        return ""
    fid = per["meta_flight_id"][0]
    g = df.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
    t = [(x - g["raw_timestamp"][0]).total_seconds() / 60 for x in g["raw_timestamp"]]

    # Trim the frozen runs at each end — see _frozen_ends. Nulled, never
    # dropped, and the x-axis stays raw_timestamp: filtering rows would
    # silently shorten the flight and hide exactly the gaps and anomalies a
    # diagnostic plot exists to show. A nulled sample leaves a visible break in
    # the line; a dropped one is invisible.
    channels = [
        g["raw_alt_ft"].to_list(),
        g["raw_gs_kt"].to_list(),
        g["raw_track_deg"].to_list(),
    ]
    flown = _mask_invalid(_despike(g["raw_alt_ft"].to_list()), channels)
    cmd = _mask_invalid(_despike(g["bds_mcp_alt_sel_ft"].to_list()), channels)

    fig, ax = PS.figure(figsize=(6.4, 3.4))
    ax.plot(t, flown, color=PS.RAW, lw=1.0, label="altitude flown")
    ax.plot(t, cmd, color=PS.LINE, lw=1.4,
            drawstyle="steps-post", label="altitude commanded (BDS 4,0)")
    ax.set_xlabel("time since first sample (min)")
    ax.set_ylabel("altitude (ft)")
    ax.set_title("A commanded altitude, and the trajectory that follows it", loc="left")
    # Upper-centre: the descent runs into the lower-right corner, and a legend
    # there covers exactly the step-downs a reader needs to check.
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncols=2)
    ax.margins(y=0.12)
    fig.savefig(out / "fig_setpoint_example.png")
    plt.close(fig)
    return "fig_setpoint_example.png"


@app.default
def main(
    *,
    measures: Annotated[
        Path, cyclopts.Parameter(help="Measurements written by 02_measure.py.")
    ] = DEFAULT_MEASURES,
    table: Annotated[
        Path, cyclopts.Parameter(help="Delta table (example flight only).")
    ] = DEFAULT_TABLE,
    out: Annotated[Path, cyclopts.Parameter(help="Figure destination.")] = DEFAULT_OUT,
) -> None:
    """Render every cohort figure."""
    from opensky26_plateau import paper_style as PS

    PS.use()
    out.mkdir(parents=True, exist_ok=True)
    # The four aggregate figures read outputs/, so they re-render from the
    # versioned evidence alone. Only the example flight needs the cohort itself:
    # it draws one trajectory, which no aggregate can carry.
    for fn in (fig_composition, fig_by_date, fig_per_airframe, fig_channels):
        print(f"  {fn(measures, out)}")

    df = _load(table)
    print(f"cohort: {df.height:,} rows, {df['meta_flight_id'].n_unique():,} flights")
    name = fig_example(df, out)
    if name:
        print(f"  {name}")


if __name__ == "__main__":
    app()

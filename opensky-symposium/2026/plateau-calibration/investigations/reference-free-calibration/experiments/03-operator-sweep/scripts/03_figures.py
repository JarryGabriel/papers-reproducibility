"""Two figures: the frontier the rule chose from, and what it chose.

The frontier says whether a choice was consequential or cosmetic. The cascade
says what the chosen points actually commit to — which is not the same question,
and the numbers cannot answer it. Every selection rule tried during this work
produced settings that satisfied their constraint by cutting a cruise into
fragments; the counts made that checkable and only a picture made it obvious.

The cascade needs the cohort, and the contract asks figures to read `outputs/`
so they re-render from versioned evidence alone. So a first step writes the
committed segments of one flight to `outputs/example_segments.csv`, and the
figure reads that.
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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PAPER_ROOT = project_root(__file__)
REPO_ROOT = project_root(__file__)
OUTPUTS = EXPERIMENT_DIR / "outputs"
FIGURES = EXPERIMENT_DIR / "figures"
COHORT = data_dir() / 'flights.delta'
GRIDS = EXPERIMENT_DIR / "inputs" / "grids.json"
BUDGETS = EXPERIMENT_DIR / "inputs" / "budgets_full_step.json"


from opensky26_plateau import transfer_scoring as T  # noqa: E402

DATE = "2025-11-01"
DT_S = 4.0
M_TO_FT = 1.0 / 0.3048
VERTICAL = ("alt", "gamma", "vz")
SPEED = ("mach", "cas")

LABEL = {
    "alt": "altitude",
    "gamma": "flight-path angle",
    "vz": "vertical speed",
    "cas": "CAS/IAS",
    "mach": "Mach",
}
UNIT = {"alt": "m", "gamma": "rad", "vz": "ft/min", "cas": "kt", "mach": "—"}
#: What each channel's error is DISPLAYED in, with the factor from the unit it
#: is scored in. Altitude is scored in metres because `signals` converts, but
#: nothing in the domain reads an altitude tolerance in metres — 25 ft is the
#: quantisation step and the number a reader recognises.
DISPLAY = {
    "alt": ("ft", M_TO_FT),
    "gamma": ("deg", 180.0 / np.pi),
    "vz": ("ft/min", 1.0),
    "cas": ("kt", 1.0),
    "mach": ("×10⁻³", 1e3),
}
#: One colour per detector family, shared by every panel and named once in a
#: figure-level legend — which family wins where is a result of the sweep.
FAMILY = {"bilateral": "#2b7bba", "savgol": "#e8a33d", "varthresh": "#59a14f", "pelt": "#b07aa1"}
#: One colour per CHANNEL, not per cascade position. Altitude and Mach shared a
#: green and vertical speed shared a blue with CAS/IAS, which reads as a claim
#: that the pairs are the same kind of thing; they are the first and second
#: stages of two different cascades. What matters is that no channel sits next
#: to a neighbour it can be confused with: the flight-path angle takes the
#: orange and Mach the violet, which separates each of them from the green and
#: the blue they are read against.
LINE = {"alt": "#1a7f37", "gamma": "#b45309", "vz": "#2563eb", "mach": "#7c3aed", "cas": "#be185d"}
BAND = {"alt": "#a8e6b4", "gamma": "#f0c99a", "vz": "#c8d8ff", "mach": "#cbb2f7", "cas": "#f5bdd3"}
RAW = "0.45"

#: The manuscript's text block, measured from the venue class: 404 pt, which is
#: 5.59 in. A figure generated wider than this is scaled DOWN by LaTeX, and its
#: fonts shrink with it — a 9 pt label inside a 13 in figure reaches the page at
#: under 4 pt, against 10 pt body text. Generating at the final width keeps the
#: labels at the size they are set to.
TEXT_IN = 5.59
#: A float may run to about two thirds of the text height before it pushes the
#: prose off its own page.
MAX_H_IN = 6.4


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            # Set against the manuscript's own 10 pt body: figures are now
            # generated at the text width, so what is set here is what prints.
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _points() -> dict[str, tuple[str, dict]]:
    """The retained points, resolved back to the grid's own precision.

    `retained.csv` writes its knobs with `%g`, six significant digits, so a
    detector run on the stored string is not quite the detector that was
    scored. `resolve_frozen` maps each knob back to the grid value it can only
    have come from.
    """
    grids = json.loads(GRIDS.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, dict]] = {}
    for row in pl.read_csv(OUTPUTS / "retained.csv").iter_rows(named=True):
        if not row["admissible"]:
            continue
        knobs = {k: float(v) for k, v in (p.split("=") for p in row["knobs"].split(","))}
        knobs["butter"] = int(row["butter"])
        resolved = T.resolve_frozen(knobs, row["channel"], row["algo"], grids)
        resolved["butter"] = int(row["butter"])
        out[row["channel"]] = (row["algo"], resolved)
    return out


#: Two flights, named rather than picked by a rule -- a figure whose subject
#: shifts when the cohort changes is not the same figure from one run to the
#: next. Both are complete flights, climb through descent, so every stage of
#: both cascades has something to arbitrate.
#:
#: The first is what the detector does on a clean signal: 116 minutes, three
#: altitude plateaux covering 73 % of the trace and 68 % on Mach. The second
#: commits 45 altitude plateaux over a cruise of similar length, its altitude
#: wandering two quantisation steps throughout. One example alone would either
#: flatter the detector or misrepresent it; two say what the spread is, which is
#: what the extended abstract's reviewers asked for.
#: A third flight, added for the SPEED cascade specifically. The two above are
#: chosen on their altitude behaviour, and on both of them Mach holds most of
#: the cruise — so CAS/IAS is left with almost nothing and the panel shows a
#: cascade that never has to share. DAL328 broadcasts Mach and CAS on 100 % of
#: its samples and the two split the trajectory: Mach commits 49.7 % of its
#: channel, CAS/IAS 38.2 % of what is left to it. That is what the speed
#: cascade looks like when both channels are actually present.
EXAMPLES = {"clean": "4bc883_PGT12RF_s0", "noisy": "4076cc_BAW664_s0", "speed": "a65454_DAL328_s3"}


def write_example_segments() -> str:
    """Run the cascade on one long flight and write what it commits.

    Sequentially, exactly as the sweep did: each channel sees only the residual
    its priors leave, and what is handed down is each segment's ENVELOPE rather
    than its finite samples alone.
    """
    points = _points()
    frame = (
        pl.read_delta(str(COHORT))
        .with_columns(pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date"))
        .filter(pl.col("_date") == DATE)
    )

    for kind, flight_id in EXAMPLES.items():
        _write_one(frame, points, kind, flight_id)
    return ", ".join(EXAMPLES.values())


def _mach_config() -> dict:
    """The Mach point the CAS mask must be built from — read, never assumed.

    `retained.csv` is the artefact; a constant copied out of it drifts, and the
    one this replaced had drifted for a month, building the CAS mask from a
    July setting 32x coarser than the point the sweep retained.
    """
    return T.mach_config(OUTPUTS / "retained.csv", json.loads(GRIDS.read_text(encoding="utf-8")))


def _write_one(frame: pl.DataFrame, points: dict, kind: str, flight_id: str) -> None:
    """Segments and traces for one example flight, under suffix *kind*."""
    flight = frame.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    signals = T.signals(flight)
    size = signals["alt"].size

    rows: list[dict] = []
    for order in (VERTICAL, SPEED):
        exclusion = np.zeros(size, dtype=bool)
        for channel in order:
            if channel not in points:
                continue
            algo, knobs = points[channel]
            view = T.filtered(signals[channel], knobs["butter"])
            here = exclusion.copy()
            if channel == "cas":
                here = here | T.mach_mask(flight, _mach_config())
            claimed = np.zeros(size, dtype=bool)
            for segment in T.detect(algo, channel, view, knobs, here):
                i, j = segment["start_idx"], segment["end_idx"]
                piece = view[i : j + 1]
                finite = np.isfinite(piece)
                if not finite.any():
                    continue
                claimed[i : j + 1] = True
                rows.append(
                    {
                        "cascade": "vertical" if order is VERTICAL else "speed",
                        "channel": channel,
                        "start_idx": int(i),
                        "end_idx": int(j),
                        "value": float(piece[finite].mean()),
                    }
                )
            exclusion |= claimed

    # All five series come from `signals`, which is what the detector saw — not
    # from the Delta columns. `_frozen_ends` cuts a trajectory's dead ends to
    # NaN inside it: a decoder replaying its last frame shows as every channel
    # repeating its previous value at once, and on one flight that was eight
    # minutes of taxi drawn as level flight. Plotting the raw column would put
    # that tail on the figure beside segments the detector never had the chance
    # to commit there, which reads as the detector missing something.
    trace = pl.DataFrame(
        {
            "sample": np.arange(size),
            "minutes": np.arange(size) * DT_S / 60.0,
            "alt_ft": signals["alt"] * M_TO_FT,
            "vz_ftmin": signals["vz"],
            "gamma_rad": signals["gamma"],
            "mach": signals["mach"],
            "cas_kt": signals["cas"],
        }
    )
    trace.write_csv(OUTPUTS / f"example_trace_{kind}.csv")
    pl.DataFrame(rows).write_csv(OUTPUTS / f"example_segments_{kind}.csv")
    print(f"{kind} example {flight_id}: {len(rows)} committed segments")


def fig_frontier() -> None:
    """Coverage against worst-segment error, per channel, with the budget.

    The point is not where the retained setting sits but what the frontier looks
    like around it: a steep frontier means the choice bought something, a flat
    one means the rule separated equivalents.
    """
    sweep = pl.read_csv(OUTPUTS / "sweep.csv")
    retained = pl.read_csv(OUTPUTS / "retained.csv")
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8"))

    # Cascade order, not alphabetical: altitude > gamma > vz on the top row,
    # Mach > CAS on the bottom. The figure then reads as the two orderings the
    # paper argues for rather than as a list of five channels.
    channels = [
        c
        for c in ("alt", "gamma", "vz", "mach", "cas")
        if c in sweep["channel"].unique().to_list()
    ]
    # Two rows of three rather than one row of five: on a page, five panels side
    # by side are too narrow to read the shape of a frontier, which is the whole
    # subject. The sixth cell carries the legend.
    fig, grid = plt.subplots(2, 3, figsize=(13.5, 7.4))
    axes = grid.ravel()
    for ax, channel in zip(axes, channels):
        scored = sweep.filter(pl.col("channel") == channel)
        budget = budgets[channel]
        error = scored["worst_segment_error"].to_numpy() / budget
        coverage = scored["coverage_pct"].to_numpy()
        algos = scored["algo"].to_numpy()

        # Coverage on x, error on y — the orientation the paper's own front
        # figures use, so a reader moving between them does not have to re-learn
        # the axes. It also asks the question the right way round: what does
        # this much coverage cost in error?
        over = error > 1.0
        # Lighter and smaller than the admissible points: at print width these
        # panels are about 5 cm across, and rejected candidates outnumber the
        # ones the rule chooses among. Drawn at equal weight they set the
        # visual scale and the useful region reads as a smudge in one corner.
        ax.scatter(
            coverage[over], error[over], s=2, c="#e8e8e8", linewidths=0, zorder=1, rasterized=True
        )
        for family, colour in FAMILY.items():
            here = (~over) & (algos == family)
            if here.any():
                ax.scatter(
                    coverage[here],
                    error[here],
                    s=6,
                    c=colour,
                    linewidths=0,
                    alpha=0.85,
                    zorder=2,
                    rasterized=True,
                )
        # The lower envelope: the least error reached at or below each level of
        # coverage. NOT a convex hull — closing the contour above would draw a
        # boundary through candidates nobody would pick. What the eye needs is
        # the floor, since the rule only ever chooses on it.
        order = np.argsort(coverage)
        envelope = np.minimum.accumulate(error[order][::-1])[::-1]
        ax.plot(coverage[order], envelope, color="#333333", lw=1.1, alpha=0.65, zorder=3)

        ax.axhline(1.0, color="#c0392b", lw=1.2, ls="--", zorder=4)

        row = retained.filter(pl.col("channel") == channel).to_dicts()[0]
        if row["admissible"]:
            ax.scatter(
                [row["coverage_pct"]],
                [row["worst_segment_error"] / budget],
                marker="*",
                s=190,
                c="black",
                zorder=5,
            )
        # Bounded just above the budget. Everything the rule can choose lies at
        # or below 1.0, so the panel's whole job is that band; showing candidates
        # out to 2.5x the budget spends three fifths of the height on points no
        # rule would take and squeezes the decision into the bottom two fifths.
        # The envelope past the budget is what Section 5.3 quotes from the
        # ceiling figures, not from this panel.
        ax.set_ylim(0, 1.25)
        ax.set_xlim(left=0)
        ax.set_title(
            f"{LABEL[channel]} ({UNIT[channel]}) — {row['coverage_pct']:.1f} %"
            if row["admissible"]
            else f"{LABEL[channel]} — empty",
            fontsize=10,
        )
        ax.set_xlabel("committed coverage (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("worst segment error / budget")
    axes[3].set_ylabel("worst segment error / budget")

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=6, color=colour, label=name)
        for name, colour in FAMILY.items()
    ] + [
        plt.Line2D([], [], marker="o", ls="", ms=6, color="#d9d9d9", label="over budget"),
        plt.Line2D(
            [], [], color="#333333", lw=1.1, alpha=0.65, label="least error at that coverage"
        ),
        plt.Line2D([], [], ls="--", color="#c0392b", lw=1.2, label="budget E"),
        plt.Line2D([], [], marker="*", ls="", ms=13, color="black", label="retained point"),
    ]
    # The sixth cell holds the legend rather than a panel: five channels do not
    # fill a 2x3 grid, and a legend there costs no page space.
    spare = axes[len(channels)]
    spare.axis("off")
    spare.legend(handles=handles, loc="center", frameon=False, fontsize=10, labelspacing=1.1)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_frontier.png")
    plt.close(fig)


def _cascade_panels(
    order: tuple[str, ...],
    panels: tuple,
    title: str,
    path: Path,
    figsize: tuple[float, float],
    kind: str = "noisy",
) -> None:
    """One cascade, one panel per stage.

    Composition carried over from the archived plotters, and each part of it
    earns its place: minutes on the x axis so a reader sees a flight rather than
    an index; a band for every stage on every panel, so any vertical slice says
    which channel owns that stretch; and the committed line drawn ONLY on its
    own panel — a gamma plateau is a value of gamma, and drawing it over the
    altitude trace would assert something never measured.
    """
    trace = pl.read_csv(OUTPUTS / f"example_trace_{kind}.csv")
    segments = pl.read_csv(OUTPUTS / f"example_segments_{kind}.csv")
    minutes = trace["minutes"].to_numpy()
    size = trace.height

    def mask_of(channel: str) -> np.ndarray:
        out = np.zeros(size, dtype=bool)
        for row in segments.filter(pl.col("channel") == channel).iter_rows(named=True):
            out[row["start_idx"] : row["end_idx"] + 1] = True
        return out

    def committed_of(channel: str, scale: float) -> np.ndarray:
        out = np.full(size, np.nan)
        for row in segments.filter(pl.col("channel") == channel).iter_rows(named=True):
            out[row["start_idx"] : row["end_idx"] + 1] = row["value"] * scale
        return out

    masks = {channel: mask_of(channel) for channel in order}
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    _draw_cascade(axes, order, panels, title, trace, segments, minutes, masks, committed_of)
    fig.savefig(path)
    plt.close(fig)


def _draw_cascade(
    axes,
    order,
    panels,
    title,
    trace,
    segments,
    minutes,
    masks,
    committed_of,
    panel_legends: bool = True,
) -> None:
    """Draw one cascade into axes somebody else owns.

    Split out of :func:`_cascade_panels` so the two cascades can share one
    figure: they are the same flight, so stacking them repeats an x axis and
    costs a page the venue does not have.
    """
    for ax, (channel, column, scale, ylabel, rawlabel, legend_at) in zip(axes, panels):
        # The first stage owns most of a cruise flight, so at equal alpha its
        # band swamps the ones the figure exists to show.
        for name in order:
            alpha = 0.3 if name == order[0] else 0.5
            for start, end in _runs(masks[name]):
                ax.axvspan(minutes[start], minutes[end], color=BAND[name], alpha=alpha, zorder=0)
        ax.plot(minutes, trace[column].to_numpy(), color=RAW, lw=0.7, label=rawlabel)
        ax.plot(
            minutes,
            committed_of(channel, scale),
            color=LINE[channel],
            lw=2.4,
            solid_capstyle="butt",
            label=f"{LABEL[channel]} held ({len(_runs(masks[channel]))})",
        )
        if channel in ("gamma", "vz"):
            ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_ylabel(ylabel)
        if panel_legends:
            ax.legend(loc=legend_at, framealpha=0.85)
    axes[-1].set_xlabel("Time [min]")
    axes[0].set_title(title, fontsize=10)


def _pareto(coverage: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Indices of the candidates nothing dominates: more coverage AND less error.

    Walking by decreasing coverage and keeping each new running minimum of the
    error gives exactly that set, in one pass and already in plotting order.
    """
    order = np.argsort(-coverage)
    keep, best = [], np.inf
    for index in order:
        if error[index] < best:
            keep.append(index)
            best = error[index]
    return np.array(keep[::-1], dtype=int)


def fig_front_zoom() -> None:
    """The attainable front, close up, in each channel's own unit.

    `fig_frontier` shows the whole scored population on one normalised scale,
    which answers "what did the grid contain". This answers the other question —
    what does a given coverage cost in error, in a unit an engineer argues with.
    Altitude appears in FEET: it is scored in metres because `signals` converts,
    but 25 ft is the quantisation step and the number the domain recognises,
    while "7.62 m" is an artefact of the conversion.

    Three layers, and each earns its place. The front itself, coloured by the
    family that reached it. The admissible candidates, which say how much room
    the rule had. And a light neighbourhood, cropped so the axes stay on the
    region the rule chooses in — showing every candidate would put the front
    inside a cloud and stretch the scale over ground nothing useful occupies.
    """
    sweep = pl.read_csv(OUTPUTS / "sweep.csv")
    retained = pl.read_csv(OUTPUTS / "retained.csv")
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8"))

    channels = [
        c
        for c in ("alt", "gamma", "vz", "mach", "cas")
        if c in sweep["channel"].unique().to_list()
    ]
    # The legend used to occupy the sixth cell of a 2x3 grid, so five panels
    # shared five sixths of the width. Below the panels as a horizontal strip it
    # costs one narrow band of height and gives every panel the full column.
    # The legend used to occupy the sixth cell of a 2x3 grid, so five panels
    # shared five sixths of the width. As a strip below them every panel gets
    # the full column; the strip needs its own reserved band, since a legend
    # anchored outside the axes is not laid out with them.
    # Two columns, not three. Five panels across 5.6 in of text width leaves
    # 1.8 in each, which is too little for an axis pair plus a budget
    # annotation; at two columns each panel is 2.7 in and the front is legible.
    # The sixth cell carries the legend, so no separate band is needed.
    fig = plt.figure(figsize=(TEXT_IN, 4.3))
    grid = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.40, left=0.07, right=0.985)
    axes = np.array([fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)])
    legend_band = axes[-1]
    for ax, channel in zip(axes, channels):
        scored = sweep.filter(pl.col("channel") == channel)
        unit, factor = DISPLAY[channel]
        budget = budgets[channel] * factor
        error = scored["worst_segment_error"].to_numpy() * factor
        coverage = scored["coverage_pct"].to_numpy()
        algos = scored["algo"].to_numpy()
        front = _pareto(coverage, error)

        # The neighbourhood: within 60 % of the front's own error at that level
        # of coverage. Enough to show the front is a boundary of something, not
        # so much that it sets the scale.
        reference = np.interp(coverage, coverage[front], error[front])
        near = (error <= reference * 1.6) & (error <= budget * 2.0)
        ax.scatter(
            coverage[near], error[near], s=5, c="#dddddd", linewidths=0, zorder=1, rasterized=True
        )

        admissible = error <= budget
        ax.scatter(
            coverage[admissible],
            error[admissible],
            s=7,
            c="#9ecae1",
            linewidths=0,
            alpha=0.8,
            zorder=2,
            rasterized=True,
        )

        ax.plot(coverage[front], error[front], color="#666666", lw=1.1, zorder=3)
        for family, colour in FAMILY.items():
            here = front[algos[front] == family]
            if here.size:
                ax.scatter(
                    coverage[here],
                    error[here],
                    s=14,
                    c=colour,
                    linewidths=0.4,
                    edgecolor="white",
                    zorder=4,
                )

        ax.axhline(budget, color="#c0392b", lw=1.3, ls="--", zorder=5)
        ax.annotate(
            f"E = {budget:.4g} {unit}",
            xy=(0.03, budget),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="#c0392b",
        )

        row = retained.filter(pl.col("channel") == channel).to_dicts()[0]
        if row["admissible"]:
            ax.scatter(
                [row["coverage_pct"]],
                [row["worst_segment_error"] * factor],
                marker="*",
                s=110,
                c="white",
                edgecolor="black",
                linewidths=1.0,
                zorder=6,
            )
        # Twice the budget: the front keeps climbing well past that, and letting
        # it set the scale would compress the admissible region into a sliver.
        ax.set_ylim(0, budget * 2.0)
        ax.set_xlim(left=0)
        ax.set_title(
            f"{LABEL[channel]} — {row['coverage_pct']:.1f} %"
            if row["admissible"]
            else f"{LABEL[channel]} — empty",
            fontsize=8,
        )
        ax.set_xlabel("committed coverage (%)")
        ax.set_ylabel(f"worst segment error ({unit})")
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=7, color=colour, label=name)
        for name, colour in FAMILY.items()
    ] + [
        plt.Line2D([], [], marker="o", ls="", ms=5, color="#9ecae1", label="within budget"),
        plt.Line2D([], [], marker="o", ls="", ms=4, color="#dddddd", label="near the front"),
        plt.Line2D([], [], color="#666666", lw=1.1, label="attainable front"),
        plt.Line2D([], [], ls="--", color="#c0392b", lw=1.3, label="budget E"),
        plt.Line2D(
            [],
            [],
            marker="*",
            ls="",
            ms=15,
            color="white",
            markeredgecolor="black",
            label="retained point",
        ),
    ]
    for spare in axes[len(channels) :]:
        spare.axis("off")
    legend_band.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=8,
        ncol=1,
        labelspacing=0.9,
    )
    # No suptitle. A 90-character title sets the canvas width on its own, and
    # bbox_inches="tight" then crops to it, leaving the panels centred between
    # two empty margins. The manuscript caption says the same thing.
    fig.savefig(FIGURES / "fig_front_zoom.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_family_domains() -> None:
    """Which family holds the front, and over which stretch of it.

    On the front, coverage and error move together — a point further right is
    also a point further up — so one axis says both: where a family holds the
    front at low coverage it holds it at low error too.

    The band is the front's own sequence, coloured by the family that reached
    each of its points, drawn against coverage. Above it, the margin: how much
    less error the winning family reached than the best of the others at that
    coverage. A wide band with no margin is a family that wins by a hair; a
    narrow one with a large margin is a family that wins where it matters.
    """
    sweep = pl.read_csv(OUTPUTS / "sweep.csv")
    retained = pl.read_csv(OUTPUTS / "retained.csv")
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8"))

    channels = [
        c
        for c in ("alt", "gamma", "vz", "mach", "cas")
        if c in sweep["channel"].unique().to_list()
    ]
    fig, axes = plt.subplots(len(channels), 1, figsize=(11.5, 1.35 * len(channels)), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, channel in zip(axes, channels):
        scored = sweep.filter(pl.col("channel") == channel)
        unit, factor = DISPLAY[channel]
        budget = budgets[channel] * factor
        error = scored["worst_segment_error"].to_numpy() * factor
        coverage = scored["coverage_pct"].to_numpy()
        algos = scored["algo"].to_numpy()
        front = _pareto(coverage, error)

        # One stripe per front point, spanning to the next: the front is a
        # sequence of coverage levels, and what the eye should read is which
        # family owns each stretch of it.
        edges = np.append(coverage[front], coverage[front][-1] * 1.02)
        for k, index in enumerate(front):
            ax.axvspan(edges[k], edges[k + 1], color=FAMILY[algos[index]], alpha=0.85, lw=0)

        # Where the budget cuts the front, so the stretch the rule could
        # actually choose from is visible rather than inferred.
        admissible = front[error[front] <= budget]
        if admissible.size:
            ax.axvline(coverage[admissible][-1], color="#c0392b", lw=1.4, ls="--", zorder=3)

        row = retained.filter(pl.col("channel") == channel).to_dicts()[0]
        if row["admissible"]:
            ax.plot(
                [row["coverage_pct"]],
                [0.5],
                marker="*",
                ms=15,
                color="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=4,
            )

        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(coverage[front].max() * 1.02, 1.0))
        ax.set_ylabel(LABEL[channel], rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xlabel("committed coverage along the front (%)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        # The error at each end of the front, so the band carries both axes.
        ax.annotate(
            f"{error[front][0]:.3g} {unit}",
            xy=(0.004, 1.06),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#555555",
        )
        ax.annotate(
            f"{error[front][-1]:.3g} {unit}",
            xy=(0.996, 1.06),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#555555",
            ha="right",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour, alpha=0.85, label=name)
        for name, colour in FAMILY.items()
    ] + [
        plt.Line2D(
            [], [], ls="--", color="#c0392b", lw=1.4, label="where the budget cuts the front"
        ),
        plt.Line2D(
            [],
            [],
            marker="*",
            ls="",
            ms=13,
            color="white",
            markeredgecolor="black",
            label="retained point",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Which detector family holds the attainable front, and where — "
        "low coverage is also low error",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURES / "fig_family_domains.png")
    plt.close(fig)


def fig_cascades_side_by_side(kind: str = "clean") -> None:
    """Both cascades on one flight, vertical left and speed right.

    Stacked, the two figures repeat the same x axis and cost two page-widths of
    height. Side by side they cost one: the flight is the same, the minutes are
    the same, and a reader compares what altitude claims against what Mach
    claims over the same stretch of time rather than turning a page.

    The columns hold three panels and two, so the speed column occupies the
    first two rows and the third is left empty rather than stretching them: a
    CAS panel half again the height of the Mach panel above it would read as an
    importance the cascade does not carry.
    """
    trace = pl.read_csv(OUTPUTS / f"example_trace_{kind}.csv")
    segments = pl.read_csv(OUTPUTS / f"example_segments_{kind}.csv")
    minutes = trace["minutes"].to_numpy()
    size = trace.height

    def mask_of(channel: str) -> np.ndarray:
        out = np.zeros(size, dtype=bool)
        for row in segments.filter(pl.col("channel") == channel).iter_rows(named=True):
            out[row["start_idx"] : row["end_idx"] + 1] = True
        return out

    def committed_of(channel: str, scale: float) -> np.ndarray:
        out = np.full(size, np.nan)
        for row in segments.filter(pl.col("channel") == channel).iter_rows(named=True):
            out[row["start_idx"] : row["end_idx"] + 1] = row["value"] * scale
        return out

    vertical_panels = (
        ("alt", "alt_ft", M_TO_FT, "Altitude [ft]", "raw altitude", "upper left"),
        ("gamma", "gamma_rad", 1.0, "$\\gamma$ [rad]", "raw $\\gamma$", "upper right"),
        ("vz", "vz_ftmin", 1.0, "$v_z$ [ft/min]", "raw vertical speed", "upper right"),
    )
    speed_panels = (
        ("mach", "mach", 1.0, "Mach", "raw Mach (BDS)", "lower right"),
        ("cas", "cas_kt", 1.0, "CAS [kt]", "raw CAS/IAS (BDS)", "lower right"),
    )

    # Two rows of three, the same shape as the front figure: the vertical
    # cascade fills the top row, the speed cascade the bottom two cells, and the
    # sixth is left empty. Three columns of one panel each are wider than two
    # columns of three stacked, so every trace gets the full page width instead
    # of half of it. Column titles are dropped — at this width they set the
    # canvas and the caption already names both cascades.
    # Five panels stacked in one column. These are 110-minute time series, not
    # point clouds: laid out three to a row each trace is 1.8 in wide and an
    # altitude plateau collapses to a tick. One full-width column per channel
    # keeps the x axis long enough to read a cruise, and the two cascades are
    # separated by the shared time axis rather than by a column break.
    # Three rows, two columns: vertical cascade down the left, speed down the
    # right, the sixth cell empty. Wider than three panels to a row — these are
    # 110-minute series and need the x extent — and shorter than one column of
    # five. Margins are set explicitly and no column title is drawn: a title
    # long enough to name a cascade sets the canvas width on its own, and
    # bbox_inches then crops to it, leaving the panels centred in empty margins.
    fig = plt.figure(figsize=(TEXT_IN, 5.4))
    grid = fig.add_gridspec(3, 2, hspace=0.20, wspace=0.24, left=0.10, right=0.985)
    top = np.array([fig.add_subplot(grid[i, 0]) for i in range(3)])
    bottom = np.array([fig.add_subplot(grid[i, 1]) for i in range(2)])
    _draw_cascade(
        top,
        VERTICAL,
        vertical_panels,
        "",
        trace,
        segments,
        minutes,
        {c: mask_of(c) for c in VERTICAL},
        committed_of,
        panel_legends=False,
    )
    _draw_cascade(
        bottom,
        SPEED,
        speed_panels,
        "",
        trace,
        segments,
        minutes,
        {c: mask_of(c) for c in SPEED},
        committed_of,
        panel_legends=False,
    )
    for ax in (*top[:-1], bottom[0]):
        ax.set_xlabel("")

    # Two legends in the free cell, one per cascade, rather than five boxes
    # sitting on top of the data they explain. Separated because which channel
    # belongs to which cascade is the thing a reader has to hold: the vertical
    # one arbitrates three channels, the speed one two, and reading them as a
    # single list of five hides the structure the figure exists to show.
    _cascade_key(fig, grid[2, 1], mask_of)

    fig.savefig(FIGURES / "fig_cascades.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _cascade_key(fig, cell, mask_of) -> None:
    """The shared two-column key, drawn into a spare grid cell.

    Both cascade figures carry the same key, and a second copy of it would
    drift from the first the next time a colour or a label changes.
    """
    key = fig.add_subplot(cell)
    key.axis("off")

    def held(channel: str) -> str:
        return f"{LABEL[channel]} ({len(_runs(mask_of(channel)))})"

    def swatch(channel: str) -> plt.Line2D:
        return plt.Line2D([], [], color=LINE[channel], lw=2.6, label=held(channel))

    # One legend in two columns, one cascade each, rather than two legend
    # artists: two legends anchored in the same cell overlap at any pair of
    # offsets that also keeps them both inside it. A multi-column legend fills
    # COLUMN BY COLUMN, so the two lists are concatenated and must hold the
    # same number of entries: three each, the shared raw-signal key balancing
    # the speed cascade's missing third channel.
    left = [swatch(c) for c in VERTICAL]
    right = [
        *(swatch(c) for c in SPEED),
        plt.Line2D([], [], color=RAW, lw=0.9, label="raw signal"),
    ]
    key.legend(handles=[*left, *right], loc="upper left",
               bbox_to_anchor=(-0.02, 0.62), ncol=2, columnspacing=1.0,
               frameon=False, fontsize=7, labelspacing=0.5,
               handlelength=1.7, alignment="left")
    # The two cascade names are drawn as text, not as handle-less legend
    # entries: every entry reserves `handlelength` before its label, so a
    # header inside the legend starts where the channel names start and reads
    # as centred over its column rather than titling it. Placed as text they
    # sit flush with the column's left edge.
    key.text(-0.02, 0.72, "Vertical cascade", transform=key.transAxes,
             fontsize=7, fontweight="bold", ha="left", va="baseline")
    key.text(0.52, 0.72, "Speed cascade", transform=key.transAxes,
             fontsize=7, fontweight="bold", ha="left", va="baseline")



#: The stretch of cruise the broken-axis variant elides. An hour of a 116-minute
#: flight, held at one altitude and one Mach, costs as much page width as the
#: climb and the descent where both cascades actually arbitrate. Segments
#: crossing the break are drawn on both sides rather than split; segments lying
#: wholly inside it are not drawn at all, so the figure shows fewer plateaux
#: than the flight commits and the counts in the key are the flight's, not the
#: figure's.
BREAK = (30.0, 90.0)


def fig_cascades_broken(kind: str = "clean", window: tuple[float, float] = BREAK) -> None:
    """The side-by-side cascades with a stretch of cruise elided.

    A 116-minute flight spends its middle hour holding one altitude and one
    Mach, and that hour costs the same page width as the climb and the descent
    where both cascades actually arbitrate. Cutting it out and MARKING the cut
    buys the two ends more room at no cost in honesty: the axis says a stretch
    is missing, so nobody measures a plateau's duration across the break.

    Silently rescaling the axis instead would put a false duration on a figure
    whose subject is how long the committed segments run.
    """
    trace = pl.read_csv(OUTPUTS / f"example_trace_{kind}.csv")
    segments = pl.read_csv(OUTPUTS / f"example_segments_{kind}.csv")
    minutes = trace["minutes"].to_numpy()
    size = trace.height
    lo, hi = window

    def mask_of(channel: str) -> np.ndarray:
        out = np.zeros(size, dtype=bool)
        for row in segments.filter(pl.col("channel") == channel).iter_rows(named=True):
            out[row["start_idx"] : row["end_idx"] + 1] = True
        return out

    def committed_of(channel: str, scale: float) -> np.ndarray:
        out = np.full(size, np.nan)
        for row in segments.filter(pl.col("channel") == channel).iter_rows(named=True):
            out[row["start_idx"] : row["end_idx"] + 1] = row["value"] * scale
        return out

    vertical_panels = (
        ("alt", "alt_ft", M_TO_FT, "Altitude [ft]", "raw altitude", "upper left"),
        ("gamma", "gamma_rad", 1.0, "$\\gamma$ [rad]", "raw $\\gamma$", "upper right"),
        ("vz", "vz_ftmin", 1.0, "$v_z$ [ft/min]", "raw vertical speed", "upper right"),
    )
    speed_panels = (
        ("mach", "mach", 1.0, "Mach", "raw Mach (BDS)", "lower right"),
        ("cas", "cas_kt", 1.0, "CAS [kt]", "raw CAS/IAS (BDS)", "lower right"),
    )

    # Each channel gets a PAIR of axes sharing a y axis, the left one holding
    # everything before the break and the right one everything after, their
    # widths in proportion to the time each covers so a minute is the same
    # length on both sides. Drawing the full series into both and clipping by
    # xlim is what keeps a segment spanning the break continuous: it is drawn
    # twice, once either side, rather than cut into two plateaux.
    span_l, span_r = lo - minutes[0], minutes[-1] - hi
    fig = plt.figure(figsize=(TEXT_IN, 5.4))
    outer = fig.add_gridspec(3, 2, hspace=0.20, wspace=0.24, left=0.10, right=0.985)

    def pair(cell) -> tuple:
        inner = cell.subgridspec(1, 2, width_ratios=[span_l, span_r], wspace=0.045)
        left = fig.add_subplot(inner[0])
        right = fig.add_subplot(inner[1], sharey=left)
        left.set_xlim(minutes[0], lo)
        right.set_xlim(hi, minutes[-1])
        # The facing spines go, and two slanted ticks mark where time was cut.
        left.spines["right"].set_visible(False)
        right.spines["left"].set_visible(False)
        right.tick_params(labelleft=False, left=False)
        return left, right

    pairs_v = [pair(outer[i, 0]) for i in range(3)]
    pairs_s = [pair(outer[i, 1]) for i in range(2)]

    for panels, order, pairs in (
        (vertical_panels, VERTICAL, pairs_v),
        (speed_panels, SPEED, pairs_s),
    ):
        masks = {c: mask_of(c) for c in order}
        for half in (0, 1):
            axes = np.array([p[half] for p in pairs])
            _draw_cascade(axes, order, panels, "", trace, segments, minutes,
                          masks, committed_of, panel_legends=False)
            if half == 1:
                for ax in axes:
                    ax.set_ylabel("")

    for pair_ in (*pairs_v, *pairs_s):
        for ax in pair_:
            ax.set_xlabel("")
    # Only the bottom panel of each column keeps its tick labels and its title.
    for column in (pairs_v[:-1], pairs_s[:1]):
        for left, right in column:
            left.tick_params(labelbottom=False)
            right.tick_params(labelbottom=False)
    for left, _ in (pairs_v[-1], pairs_s[-1]):
        left.set_xlabel("Time [min]", loc="right")

    _mark_breaks([*pairs_v, *pairs_s])
    _trim_facing_ticks([*pairs_v, *pairs_s], lo, hi)
    _cascade_key(fig, outer[2, 1], mask_of)
    fig.savefig(FIGURES / "fig_cascades2.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _mark_breaks(pairs) -> None:
    """Slanted ticks on the facing edges of every broken pair.

    Drawn at both the top and the bottom of each pair, in axes coordinates, so
    the marks sit on the spines the break interrupts rather than floating at
    mid-height.
    """
    kw = {"marker": [(-1, -0.6), (1, 0.6)], "markersize": 5, "linestyle": "none",
          "color": "0.3", "mec": "0.3", "mew": 0.9, "clip_on": False}
    for left, right in pairs:
        left.plot([1, 1], [0, 1], transform=left.transAxes, **kw)
        right.plot([0, 0], [0, 1], transform=right.transAxes, **kw)


def _trim_facing_ticks(pairs, lo: float, hi: float) -> None:
    """Drop the tick each half puts right at the cut.

    Left and right halves label their own limits, so the two ticks bracketing
    the break land within a couple of points of each other and overprint. Each
    half keeps only the ticks strictly inside its own span.
    """
    for left, right in pairs:
        first, last = left.get_xlim()[0], right.get_xlim()[1]
        left.set_xticks([t for t in left.get_xticks() if first <= t < lo])
        right.set_xticks([t for t in right.get_xticks() if hi < t <= last])


def fig_cascades() -> None:
    """The two cascades on a clean and a noisy flight, plus a shared-speed one."""
    for kind, suffix, note in (
        ("clean", "", " — 3 altitude plateaux, 73 % covered"),
        ("noisy", "_noisy", " — 45 plateaux, altitude wandering"),
    ):
        _cascade_panels(
            VERTICAL,
            (
                ("alt", "alt_ft", M_TO_FT, "Altitude [ft]", "raw altitude", "upper left"),
                ("gamma", "gamma_rad", 1.0, "$\\gamma$ [rad]", "raw $\\gamma$", "upper right"),
                ("vz", "vz_ftmin", 1.0, "$v_z$ [ft/min]", "raw vertical speed", "upper right"),
            ),
            f"Vertical cascade: alt-hold > $\\gamma$ > $v_z${note}",
            FIGURES / f"fig_cascade_vertical{suffix}.png",
            (7.6, 6.8),
            kind,
        )
        _cascade_panels(
            SPEED,
            (
                ("mach", "mach", 1.0, "Mach", "raw Mach (BDS)", "lower right"),
                ("cas", "cas_kt", 1.0, "CAS [kt]", "raw CAS/IAS (BDS)", "lower right"),
            ),
            "Speed cascade: Mach takes priority, CAS/IAS keeps the residual",
            FIGURES / f"fig_cascade_speed{suffix}.png",
            (7.6, 5.0),
            kind,
        )

    # Both cascades on a flight where the speed channels are broadcast
    # throughout. On the two flights above Mach holds most of the cruise, so
    # CAS/IAS is left with almost nothing and the speed panel shows a cascade
    # that never has to share — true of those flights, misleading as an
    # illustration. The vertical panel is shown on the same flight so the two
    # orderings can be read against one trajectory rather than three.
    _cascade_panels(
        VERTICAL,
        (
            ("alt", "alt_ft", M_TO_FT, "Altitude [ft]", "raw altitude", "upper left"),
            ("gamma", "gamma_rad", 1.0, "$\\gamma$ [rad]", "raw $\\gamma$", "upper right"),
            ("vz", "vz_ftmin", 1.0, "$v_z$ [ft/min]", "raw vertical speed", "upper right"),
        ),
        "Vertical cascade on the same flight: alt-hold commits 60.5 %, "
        "$\\gamma$ 2.1 % and $v_z$ 5.1 % of what is left",
        FIGURES / "fig_cascade_vertical_shared.png",
        (7.6, 6.8),
        "speed",
    )
    _cascade_panels(
        SPEED,
        (
            ("mach", "mach", 1.0, "Mach", "raw Mach (BDS)", "lower right"),
            ("cas", "cas_kt", 1.0, "CAS [kt]", "raw CAS/IAS (BDS)", "lower right"),
        ),
        "Speed cascade where both channels are broadcast throughout: "
        "Mach commits 49.7 %, CAS/IAS 38.2 % of the residual",
        FIGURES / "fig_cascade_speed_shared.png",
        (7.6, 5.0),
        "speed",
    )
    return


def _unused_fig_cascades() -> None:
    """The vertical and speed cascades, one figure each."""
    _cascade_panels(
        VERTICAL,
        (
            ("alt", "alt_ft", M_TO_FT, "Altitude [ft]", "raw altitude", "upper left"),
            ("gamma", "gamma_rad", 1.0, "$\\gamma$ [rad]", "raw $\\gamma$", "upper right"),
            ("vz", "vz_ftmin", 1.0, "$v_z$ [ft/min]", "raw vertical speed", "upper right"),
        ),
        "Vertical cascade: alt-hold > $\\gamma$ > $v_z$, at the retained points",
        FIGURES / "fig_cascade_vertical.png",
        (7.6, 6.8),
    )
    _cascade_panels(
        SPEED,
        (
            ("mach", "mach", 1.0, "Mach", "raw Mach (BDS)", "lower right"),
            ("cas", "cas_kt", 1.0, "CAS [kt]", "raw CAS/IAS (BDS)", "lower right"),
        ),
        "Speed cascade: Mach takes priority, CAS/IAS keeps the residual",
        FIGURES / "fig_cascade_speed.png",
        (7.6, 5.0),
    )


def main() -> int:
    _style()
    FIGURES.mkdir(exist_ok=True)
    write_example_segments()
    fig_frontier()
    fig_front_zoom()
    fig_family_domains()
    fig_cascades()
    fig_cascades_side_by_side()
    fig_cascades_broken()
    print(f"wrote {FIGURES / 'fig_frontier.png'}")
    print(f"wrote {FIGURES / 'fig_front_zoom.png'}")
    print(f"wrote {FIGURES / 'fig_family_domains.png'}")
    print(f"wrote {FIGURES / 'fig_cascade_vertical.png'}")
    print(f"wrote {FIGURES / 'fig_cascade_speed.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

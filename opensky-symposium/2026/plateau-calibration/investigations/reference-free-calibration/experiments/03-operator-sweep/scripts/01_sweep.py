"""Walk the grid, score every candidate, write the frontier.

For each channel and each candidate setting, over the calibration cohort:

  * committed coverage — the share of samples inside a segment;
  * the WORST segment's reconstruction error, over every segment of every
    flight, which is what the budget constrains;
  * the segment count and median duration, which say whether a candidate
    respects the budget by detecting plateaux or by cutting them up;
  * the worst drift, reported and not constrained (see the manifest's limits).

Detection may run on a filtered view; scoring always reads the RAW signal. A
candidate must not be able to buy conformity with smoothing.

Reproduce:
    uv run --active python scripts/01_sweep.py
    uv run --active python scripts/01_sweep.py --smoke 10   # scratch, 10 flights
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


import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PAPER_ROOT = project_root(__file__)
REPO_ROOT = project_root(__file__)
COHORT = data_dir() / 'flights.delta'
BUDGETS = EXPERIMENT_DIR / "inputs" / "budgets_full_step.json"
GRIDS = EXPERIMENT_DIR / "inputs" / "grids.json"
OUTPUTS = EXPERIMENT_DIR / "outputs"


from opensky26_plateau import densify_front as front  # noqa: E402
from opensky26_plateau import transfer_scoring as T  # noqa: E402

DATE = "2025-11-01"
CUTOFFS = (0, 120, 180, 300)
DT_S = 4.0


def candidates(channel: str, grids: dict) -> list[tuple[str, dict]]:
    """Every (family, knobs) the grid declares for *channel*.

    Built once and in a fixed order so two runs walk the same grid in the same
    sequence — the frontier's row order is part of what the lock attests.
    """
    out: list[tuple[str, dict]] = []
    # The grid's own key names. Reading a key the file does not carry would
    # silently fall back on a default and sweep a domain nobody declared — the
    # first version asked for "window" and "penalty", which the file spells
    # "win" and "pen", and it swept a hard-coded window list while producing
    # zero PELT candidates. Indexing rather than .get() so a renamed key raises
    # instead of vanishing.
    slope = sorted(grids["slope"][channel])
    sigma_r = sorted(grids["sigma"][channel])
    sigma_s = sorted(grids["sigma_s"][channel])
    std = sorted(grids["std"][channel])
    # `win` is one global list; every other key is a dict keyed by channel.
    windows = sorted(grids["win"])
    penalties = sorted(grids["pen"][channel])

    for cutoff in CUTOFFS:
        for tol in slope:
            for radius in sigma_r:
                for spatial in sigma_s:
                    out.append(("bilateral", {
                        "sigma_s": spatial, "sigma_r": radius,
                        "slope_tol": tol, "butter": cutoff,
                    }))
            for window in windows:
                out.append(("savgol", {
                    "window": window, "slope_tol": tol, "butter": cutoff,
                }))
        for threshold in std:
            for window in windows:
                out.append(("varthresh", {
                    "std_tol": threshold, "window": window, "butter": cutoff,
                }))
        for penalty in penalties:
            out.append(("pelt", {"penalty": penalty, "butter": cutoff}))
    return out


def score(channel: str, algo: str, knobs: dict, raws: list[np.ndarray],
          filtered: list[np.ndarray], masks: list[np.ndarray],
          total: int) -> dict | None:
    """One candidate over the cohort.

    The error of a segment is the mean absolute deviation of its RAW samples
    from the value it announces, and the candidate's statistic is the worst such
    error anywhere — not a per-flight average, and not a percentile. A plateau
    that violates the budget is a violation whichever flight holds it.
    """
    worst_error = 0.0
    worst_drift = 0.0
    committed = 0
    n_segments = 0
    lengths: list[int] = []
    n_flights_with_segments = 0

    for raw, view, mask in zip(raws, filtered, masks):
        segments = T.detect(algo, channel, view, knobs, mask)
        if not segments:
            continue
        seen_here = False
        for segment in segments:
            i, j = segment["start_idx"], segment["end_idx"]
            piece = raw[i:j + 1]
            finite = np.isfinite(piece)
            if not finite.any():
                continue
            values = piece[finite]
            deviation = float(np.abs(values - values.mean()).mean())
            worst_error = max(worst_error, deviation)
            # Reported, never constrained: median of the last third minus the
            # first third, robust to the spikes least squares would trip on.
            third = max(2, values.size // 3)
            drift = float(abs(np.median(values[-third:]) - np.median(values[:third])))
            worst_drift = max(worst_drift, drift)
            committed += int(finite.sum())
            n_segments += 1
            lengths.append(j - i + 1)
            seen_here = True
        if seen_here:
            n_flights_with_segments += 1

    if not n_segments:
        return None
    return {
        "channel": channel,
        "algo": algo,
        "butter": int(knobs["butter"]),
        "knobs": ",".join(
            f"{k}={v:g}" for k, v in sorted(knobs.items()) if k != "butter"
        ),
        # The knobs as the grid built them, carried alongside their "%g"
        # rendering. The cascade re-runs the retained detector to build the
        # exclusion it hands down, and re-parsing six significant digits would
        # run a slightly different detector than the one that was scored. This
        # key is dropped before the frontier is written.
        "_knobs": dict(knobs),
        "coverage_pct": committed / total * 100.0,
        "worst_segment_error": worst_error,
        "worst_drift": worst_drift,
        "n_segments": n_segments,
        "segments_per_flight": n_segments / max(n_flights_with_segments, 1),
        "median_duration_s": float(np.median(lengths)) * DT_S,
        "n_flights_with_segments": n_flights_with_segments,
    }


#: Coverage within this of the maximum is a tie. Declared in SOURCES.md, and
#: applied here as well as in 02_select.py because the cascade needs a channel's
#: point before the next one can be swept — the two must agree, so the rule
#: lives in one constant read by both.
TIE_PTS = 0.05


def _retain(scored: list[dict], budget: float) -> dict | None:
    """The point the selection rule retains, or None if none is admissible.

    Same rule as `02_select.py`, applied mid-sweep because a downstream channel
    cannot be swept until its priors have chosen. The two implementations must
    not drift: `02_select.py` re-derives the retained points from the frontier
    this run writes, and a disagreement between them is a defect.
    """
    admissible = [r for r in scored if r["worst_segment_error"] <= budget]
    if not admissible:
        return None
    ceiling = max(r["coverage_pct"] for r in admissible)
    tied = [r for r in admissible if r["coverage_pct"] >= ceiling - TIE_PTS]
    return min(tied, key=lambda r: (r["worst_segment_error"],
                                    r["segments_per_flight"]))


def _knobs_of(row: dict, channel: str, grids: dict) -> dict:
    """The grid point a resumed frontier row names, at the grid's own precision.

    A frontier writes its knobs with ``%g`` — six significant digits — so
    ``slope_tol=3.6342411857`` is stored as ``3.63424``. Running the detector at
    the rounded value is running a different detector from the one that was
    scored, and the exclusion it hands down would differ from a live sweep's.
    ``resolve_frozen`` maps each stored knob back to the grid value it can only
    have come from, and refuses when the match is ambiguous.
    """
    knobs = {
        k: float(v) for k, v in (p.split("=") for p in row["knobs"].split(","))
    }
    knobs["butter"] = int(row["butter"])
    resolved = T.resolve_frozen(knobs, channel, row["algo"], grids)
    resolved["butter"] = int(row["butter"])
    return resolved


def _committed_mask(channel: str, algo: str, knobs: dict, raw: np.ndarray,
                    view: np.ndarray, exclusion: np.ndarray) -> np.ndarray:
    """The samples the retained point commits on one flight.

    This is what a channel hands down the cascade. It marks the ENVELOPE of each
    segment rather than only its finite samples: a downstream channel must not
    slip into the NaN holes inside a stretch its prior has claimed, or it will
    commit micro-plateaux in gaps that belong to someone else.

    The knobs arrive as the dict the grid built, never re-parsed from the
    ``%g`` string the frontier stores: six significant digits are not the grid's
    own precision, and a detector run at a rounded tolerance is a different
    detector from the one that was scored.
    """
    out = np.zeros(raw.size, dtype=bool)
    for segment in T.detect(algo, channel, view, knobs, exclusion):
        out[segment["start_idx"]:segment["end_idx"] + 1] = True
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0,
                        help="restrict to N flights and write to outputs/scratch/")
    # Timing a single channel on the full cohort measures what extrapolating
    # from a handful of flights only guesses. The cascade is sequential, so a
    # partial run is only meaningful from the top: --channels alt, then alt
    # gamma, and so on. A run that skips a prior would sweep a downstream
    # channel on terrain that channel will never see.
    parser.add_argument("--channels", nargs="+", default=None,
                        help="sweep only these channels, in cascade order")
    # Resume. A channel already scored is read back from its file rather than
    # recomputed: the cascade needs its RETAINED POINT to build the exclusion it
    # hands down, and that point is a function of the frontier, not of the run.
    # Re-deriving it from a stored frontier gives the identical mask for a
    # fraction of the cost — altitude alone took three hours on the full cohort.
    parser.add_argument("--resume-from", nargs="+", default=None,
                        metavar="CSV",
                        help="frontier files whose channels are already scored")
    args = parser.parse_args()

    budgets = json.loads(BUDGETS.read_text(encoding="utf-8"))
    grids = json.loads(GRIDS.read_text(encoding="utf-8"))

    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    )
    frame = frame.filter(pl.col("_date") == DATE)
    ids = frame["meta_flight_id"].unique(maintain_order=True).to_list()
    if args.smoke:
        ids = ids[:args.smoke]
    flights = [
        frame.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
        for fid in ids
    ]
    signals = [T.signals(f) for f in flights]
    print(f"{len(flights)} flights on {DATE}")

    rows: list[dict] = []
    # The cascade is SEQUENTIAL, and the order below is the cascade's own.
    # A downstream channel must be swept on the residual its priors leave, not
    # on the whole trajectory: gamma tuned over a cruise that altitude has
    # already claimed would be tuned for terrain it never sees at use time. So
    # each channel is selected before the next is swept, and its exclusion mask
    # is built from the point actually retained — the same mechanic `mach_mask`
    # already gives CAS, applied to the vertical stages too.
    exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
    retained_upstream: dict[str, dict] = {}

    # Frontiers already computed, read back so their retained points can build
    # the exclusion without re-scoring the grid behind them.
    resumed: dict[str, pl.DataFrame] = {}
    for path in (args.resume_from or []):
        table = pl.read_csv(path)
        for channel in table["channel"].unique().to_list():
            resumed[channel] = table.filter(pl.col("channel") == channel)
            print(f"resumed {channel} from {path} "
                  f"({resumed[channel].height} candidates)")

    cascade_order = ("alt", "gamma", "vz", "mach", "cas")
    wanted = tuple(args.channels) if args.channels else cascade_order
    for channel in cascade_order:
        if channel in resumed and channel not in wanted:
            # Re-derive the point without touching the cohort, and hand its
            # residual down exactly as a live sweep would.
            scored = resumed[channel].to_dicts()
            chosen = _retain(scored, budgets[channel])
            if chosen is None:
                print(f"  {channel}: resumed, EMPTY")
                continue
            chosen["_knobs"] = _knobs_of(chosen, channel, grids)
            print(f"  {channel}: resumed, retained {chosen['algo']} "
                  f"b={chosen['butter']} {chosen['knobs']} — "
                  f"{chosen['coverage_pct']:.2f} % coverage")
            retained_upstream[channel] = chosen
            # The speed cascade is an INDEPENDENT ordering over the same
            # trajectory: mach owes nothing to the vertical stages. The live
            # branch resets the exclusion when it reaches mach; this branch
            # `continue`s past that reset, so without this line a resumed run
            # sweeps cas behind altitude's and gamma's residual as well as
            # mach's mask — measured at 17.88 % coverage against the 35.00 %
            # the same point actually reaches.
            if channel == "mach":
                exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
            if channel in ("alt", "gamma"):
                raws = [s[channel] for s in signals]
                views_here = [T.filtered(r, int(chosen["butter"])) for r in raws]
                exclusion = [
                    m | _committed_mask(channel, chosen["algo"], chosen["_knobs"],
                                        raw, view, m)
                    for m, raw, view in zip(exclusion, raws, views_here)
                ]
            continue
        if channel not in wanted:
            continue
        started = time.monotonic()
        raws = [s[channel] for s in signals]
        # FINITE samples, not array rows. `signals` turns the frozen ends into
        # NaN without shortening the array, and each channel carries its own
        # missing-data rate (gamma about 13 %). Dividing by the row count would
        # make coverage a share of a ceiling no candidate can reach, and a
        # different ceiling per channel — so gamma's coverage would not be
        # comparable to altitude's.
        total = sum(int(np.isfinite(r).sum()) for r in raws)

        if channel == "mach":
            # Mach opens the speed cascade and owes nothing to the vertical one:
            # the two are independent orderings over the same trajectory.
            exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
        elif channel == "cas":
            # CAS takes what Mach left, at the point Mach just retained — read
            # from `retained_upstream`, which the loop has just filled, and NOT
            # from a package constant. Until 2026-09-01 this line called
            # `T.mach_config()`, which returned `front.MACH` — sigma_r = 0.08
            # against the retained 0.0025 — so every CAS candidate competed for
            # a residual 15.4 points smaller than the one the cascade declares.
            # The comment above already said "at the point Mach just retained";
            # the code had never done it.
            mach = retained_upstream.get("mach")
            if mach is None:
                raise SystemExit(
                    "cas cannot be swept before mach is retained: the cascade "
                    "is sequential and the mask needs mach's retained point"
                )
            mach_config = {
                "sigma_r": mach["_knobs"]["sigma_r"],
                "slope_tol": mach["_knobs"]["slope_tol"],
                "flat_tol": front.CFG["mach"]["flat"],
                "min_len": front.CFG["mach"]["min_len"],
            }
            exclusion = [
                m | T.mach_mask(f, mach_config)
                for m, f in zip(exclusion, flights)
            ]
        masks = exclusion

        # Filtering is the expensive part and depends only on the cutoff, so it
        # is done once per cutoff rather than once per candidate.
        views = {c: [T.filtered(r, c) for r in raws] for c in CUTOFFS}

        grid = candidates(channel, grids)
        scored: list[dict] = []
        for index, (algo, knobs) in enumerate(grid):
            row = score(channel, algo, knobs, raws, views[knobs["butter"]],
                        masks, total)
            if row is not None:
                scored.append(row)
            if (index + 1) % 500 == 0:
                print(f"  {channel}: {index + 1}/{len(grid)} "
                      f"({time.monotonic() - started:.0f} s)")
        rows.extend(scored)

        chosen = _retain(scored, budgets[channel])
        admissible = sum(1 for r in scored
                         if r["worst_segment_error"] <= budgets[channel])
        print(f"  {channel}: {len(grid)} candidates, {admissible} admissible, "
              f"{time.monotonic() - started:.0f} s")
        if chosen is None:
            # A channel the rule cannot serve claims nothing, so the ones below
            # it in the cascade see the trajectory its priors left untouched.
            print(f"  {channel}: EMPTY — no candidate respects the budget")
            continue
        retained_upstream[channel] = chosen
        print(f"  {channel}: retained {chosen['algo']} b={chosen['butter']} "
              f"{chosen['knobs']} — {chosen['coverage_pct']:.2f} % coverage")

        # Hand the residual down. `mach_mask` is applied at the cas step above
        # rather than here, since Mach's own exclusion is computed by the shared
        # module and not from a segment list.
        if channel in ("alt", "gamma"):
            exclusion = [
                m | _committed_mask(channel, chosen["algo"], chosen["_knobs"],
                                    raw, view, mask)
                for m, raw, view, mask in zip(
                    exclusion, raws, views[int(chosen["butter"])], masks)
            ]

    # The raw knobs are a working value for the cascade, not a measurement.
    # A resumed channel belongs to the frontier as much as a freshly swept one:
    # the manifest commits to publishing every scored candidate, and where a row
    # was computed is not a property of the measurement.
    for channel, table in resumed.items():
        if channel not in wanted:
            rows.extend(table.to_dicts())

    table = pl.DataFrame([{k: v for k, v in r.items() if k != "_knobs"}
                          for r in rows])
    # A partial run never lands on the frontier the lock attests: outputs/
    # holds the complete sweep, and anything less goes to scratch/ under a name
    # that says what it is. A partial file that later looks complete is worse
    # than no file.
    # Complete means every channel of the cascade is present, whether it was
    # swept now or resumed from a stored frontier. A run restricted to a subset
    # is partial however it was assembled.
    covered = set(wanted) | set(resumed)
    partial = bool(args.smoke) or covered != set(cascade_order)
    destination = OUTPUTS / "scratch" if partial else OUTPUTS
    destination.mkdir(parents=True, exist_ok=True)
    name = "sweep.csv"
    if partial and args.channels:
        name = f"sweep_{'_'.join(wanted)}.csv"
    table.write_csv(destination / name)
    print(f"wrote {destination / name} — {table.height} scored candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

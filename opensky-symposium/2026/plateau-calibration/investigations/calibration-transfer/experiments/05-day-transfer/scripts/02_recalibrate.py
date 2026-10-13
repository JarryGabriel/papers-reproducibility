"""What the rule would have retained on each date alone.

The transport cost proper: the difference between the coverage the frozen point
reaches on a date and the coverage the selection rule would have reached had it
been applied to that day by itself. Reported with its sign — positive where
re-fitting would have helped, negative where the frozen point already beats what
that day's own rule would have chosen.

Two things this deliberately does NOT do.

It does not re-derive the budgets — they arrive from experiment 02 by digest,
and a budget adjusted per date would make the cost a function of the adjustment.
On a day-transfer experiment that matters twice over: a budget widened after
seeing a breach would make the breach disappear by construction.

And it does not re-select the cascade jointly. Each channel is swept against the
FROZEN upstream, exactly as the full sweep did: altitude first, then gamma over
what the frozen altitude left, then vz over what both left. Re-selecting the
whole cascade per date would change what each stage sees and the two coverages
would no longer be comparable.

Reproduce:
    uv run --active python scripts/02_recalibrate.py
    uv run --active python scripts/02_recalibrate.py --dates 2025-02-17
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
import os
import sys
import time
from pathlib import Path

#: Cap the numeric thread pools BEFORE numpy/polars are imported — they size
#: their pools at import time and ignore the variables afterwards. The sweep is
#: a Python loop over candidates, so the pools buy almost nothing here while
#: happily taking every core; leaving headroom keeps the machine usable.
_THREADS = os.environ.get("FDM_THREADS", "5")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "POLARS_MAX_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PAPER_ROOT = project_root(__file__)
REPO_ROOT = project_root(__file__)
COHORT = data_dir() / 'flights.delta'
INPUTS = EXPERIMENT_DIR / "inputs"
OUTPUTS = EXPERIMENT_DIR / "outputs"


from opensky26_plateau import transfer_scoring as T  # noqa: E402

DATES = ("2025-02-17", "2025-05-21", "2025-08-29", "2025-11-01")
CALIBRATION = "2025-11-01"
DT_S = 4.0
CUTOFFS = (0, 120, 180, 300)
VERTICAL = ("alt", "gamma", "vz")
SPEED = ("mach", "cas")
#: The sweep's own tie-break, so the two selections are the same rule.
TIE_PTS = 0.05


def candidates(channel: str, grids: dict) -> list[tuple[str, dict]]:
    """The same grid the sweep walked, in the same order."""
    out: list[tuple[str, dict]] = []
    slope = sorted(grids["slope"][channel])
    sigma_r = sorted(grids["sigma"][channel])
    sigma_s = sorted(grids["sigma_s"][channel])
    std = sorted(grids["std"][channel])
    windows = sorted(grids["win"])
    penalties = sorted(grids["pen"][channel])
    for cutoff in CUTOFFS:
        for tol in slope:
            for radius in sigma_r:
                for spatial in sigma_s:
                    out.append(("bilateral", {"sigma_s": spatial, "sigma_r": radius,
                                              "slope_tol": tol, "butter": cutoff}))
            for window in windows:
                out.append(("savgol", {"window": window, "slope_tol": tol,
                                       "butter": cutoff}))
        for threshold in std:
            for window in windows:
                out.append(("varthresh", {"std_tol": threshold, "window": window,
                                          "butter": cutoff}))
        for penalty in penalties:
            out.append(("pelt", {"penalty": penalty, "butter": cutoff}))
    return out


def score(channel: str, algo: str, knobs: dict, raws: list[np.ndarray],
          views: list[np.ndarray], masks: list[np.ndarray],
          total: int) -> dict | None:
    """One candidate on one fold — the sweep's scoring, unchanged."""
    worst = 0.0
    committed = n_segments = n_with = 0
    lengths: list[int] = []
    for raw, view, mask in zip(raws, views, masks):
        segments = T.detect(algo, channel, view, knobs, mask)
        if not segments:
            continue
        seen = False
        for segment in segments:
            i, j = segment["start_idx"], segment["end_idx"]
            piece = raw[i:j + 1]
            finite = np.isfinite(piece)
            if not finite.any():
                continue
            values = piece[finite]
            worst = max(worst, float(np.abs(values - values.mean()).mean()))
            committed += int(finite.sum())
            n_segments += 1
            lengths.append(j - i + 1)
            seen = True
        if seen:
            n_with += 1
    if not n_segments:
        return None
    return {
        "algo": algo,
        "butter": int(knobs["butter"]),
        "knobs": ",".join(f"{k}={v:g}" for k, v in sorted(knobs.items())
                          if k != "butter"),
        "_knobs": dict(knobs),
        "coverage_pct": committed / total * 100.0,
        "worst_segment_error": worst,
        "segments_per_flight": n_segments / max(n_with, 1),
        "median_duration_s": float(np.median(lengths)) * DT_S,
    }


def retain(scored: list[dict], budget: float) -> dict | None:
    """The sweep's selection rule, applied to one fold's frontier."""
    admissible = [r for r in scored if r["worst_segment_error"] <= budget]
    if not admissible:
        return None
    ceiling = max(r["coverage_pct"] for r in admissible)
    tied = [r for r in admissible if r["coverage_pct"] >= ceiling - TIE_PTS]
    return min(tied, key=lambda r: (r["worst_segment_error"],
                                    r["segments_per_flight"]))


def frozen_points() -> dict[str, tuple[str, dict]]:
    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    out: dict[str, tuple[str, dict]] = {}
    for row in pl.read_csv(INPUTS / "retained.csv").iter_rows(named=True):
        if not row["admissible"]:
            continue
        knobs = {k: float(v) for k, v in (p.split("=")
                                          for p in row["knobs"].split(","))}
        knobs["butter"] = int(row["butter"])
        resolved = T.resolve_frozen(knobs, row["channel"], row["algo"], grids)
        resolved["butter"] = int(row["butter"])
        out[row["channel"]] = (row["algo"], resolved)
    return out


def committed_mask(channel: str, algo: str, knobs: dict, raw: np.ndarray,
                   view: np.ndarray, exclusion: np.ndarray) -> np.ndarray:
    """The envelope of what a setting commits, to hand down the cascade."""
    out = np.zeros(raw.size, dtype=bool)
    for segment in T.detect(algo, channel, view, knobs, exclusion):
        out[segment["start_idx"]:segment["end_idx"] + 1] = True
    return out


def checkpoint_front(rows: list[dict]) -> None:
    """Persist one date's WHOLE scored front, not just the point retained.

    The selection rule is a filter and an argmax over these rows, so keeping
    them makes a change of budget a re-read instead of a re-run. The first
    version of this script scored 82 289 candidates and wrote only the 15
    winners; when the budget later needed correcting, none of that could be
    re-selected. The file is a few MB against ~36 h of compute.
    """
    OUTPUTS.mkdir(exist_ok=True)
    target = OUTPUTS / "fronts.csv"
    fresh = pl.DataFrame(rows)
    if target.exists():
        # Keyed on (date, channel), not date alone: a run restricted with
        # --channels writes only the channels it swept, and dropping the whole
        # date would delete the four it deliberately left alone.
        fresh_keys = set(zip(fresh["date"].to_list(), fresh["channel"].to_list()))
        kept = pl.read_csv(target).filter(
            ~pl.struct(["date", "channel"]).map_elements(
                lambda r: (r["date"], r["channel"]) in fresh_keys,
                return_dtype=pl.Boolean))
        if kept.height:
            fresh = pl.concat([kept, fresh.select(kept.columns)], how="vertical")
    fresh.write_csv(target)


def checkpoint(rows: list[dict]) -> None:
    """Persist one date's rows, merging with the dates already on disk.

    Written after EVERY date, not once at the end: four dates against the full
    grid is a night of compute, and a machine that shuts down at the third date
    would otherwise lose all three. Rows are keyed by date, so re-running a date
    replaces its own rows and leaves the others standing.
    """
    OUTPUTS.mkdir(exist_ok=True)
    target = OUTPUTS / "recalibrated.csv"
    fresh = pl.DataFrame(rows)
    if target.exists():
        # Keyed on (date, channel), not date alone: a run restricted with
        # --channels writes only the channels it swept, and dropping the whole
        # date would delete the four it deliberately left alone.
        fresh_keys = set(zip(fresh["date"].to_list(), fresh["channel"].to_list()))
        kept = pl.read_csv(target).filter(
            ~pl.struct(["date", "channel"]).map_elements(
                lambda r: (r["date"], r["channel"]) in fresh_keys,
                return_dtype=pl.Boolean))
        if kept.height:
            fresh = pl.concat([kept, fresh.select(kept.columns)], how="vertical")
    fresh.sort(["date", "channel"]).write_csv(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", default=None)
    parser.add_argument(
        "--channels", nargs="+", default=None,
        help="re-sweep only these channels. The cascade still runs in full — a "
             "skipped channel is detected at its frozen point so it hands the "
             "same residual down — but its grid is not walked. Correcting one "
             "channel costs its own sweep instead of all five: on one date "
             "gamma alone is 5 h 30 of the 14 h.")
    args = parser.parse_args()

    grids = json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))
    budgets = json.loads((INPUTS / "budgets_full_step.json").read_text())
    points = frozen_points()
    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    )
    mach_config = T.mach_config(INPUTS / "retained.csv", grids)

    wanted = tuple(args.dates) if args.dates else DATES
    wanted_channels = set(args.channels) if args.channels else None
    rows: list[dict] = []
    front_rows: list[dict] = []
    for date in wanted:
        here = frame.filter(pl.col("_date") == date)
        ids = here["meta_flight_id"].unique(maintain_order=True).to_list()
        flights = [here.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
                   for fid in ids]
        signals = [T.signals(f) for f in flights]
        tag = "calibration" if date == CALIBRATION else "held out"
        print(f"{date} ({tag}): {len(flights)} flights")

        for order in (VERTICAL, SPEED):
            exclusion = [np.zeros(s["alt"].size, dtype=bool) for s in signals]
            for channel in order:
                started = time.monotonic()
                raws = [s[channel] for s in signals]
                total = sum(int(np.isfinite(r).sum()) for r in raws)
                masks = exclusion
                if channel == "cas":
                    masks = [m | T.mach_mask(f, mach_config)
                             for m, f in zip(exclusion, flights)]

                # A channel outside --channels is NOT skipped: it is detected at
                # its frozen point so the cascade hands the same residual down.
                # Only its grid is left unwalked, which is where the hours are.
                if wanted_channels and channel not in wanted_channels:
                    algo_f, knobs_f = points[channel]
                    view_f = [T.filtered(r, int(knobs_f["butter"])) for r in raws]
                    exclusion = [
                        m | committed_mask(channel, algo_f, knobs_f, raw, v, m)
                        for m, raw, v in zip(exclusion, raws, view_f)
                    ] if channel in ("alt", "gamma", "mach") else exclusion
                    print(f"  {channel:6s} not swept, frozen point handed down")
                    continue

                views = {c: [T.filtered(r, c) for r in raws] for c in CUTOFFS}

                scored = []
                for algo, knobs in candidates(channel, grids):
                    row = score(channel, algo, knobs, raws,
                                views[knobs["butter"]], masks, total)
                    if row is not None:
                        scored.append(row)
                best = retain(scored, budgets[channel])
                front_rows += [
                    {"date": date, "channel": channel, "algo": r["algo"],
                     "butter": r["butter"], "knobs": r["knobs"],
                     "coverage_pct": r["coverage_pct"],
                     "worst_segment_error": r["worst_segment_error"],
                     "segments_per_flight": r["segments_per_flight"],
                     "median_duration_s": r["median_duration_s"]}
                    for r in scored
                ]

                # The frozen point's own coverage on this fold, so the deficit
                # is a difference between two numbers measured the same way.
                algo_f, knobs_f = points[channel]
                frozen = score(channel, algo_f, knobs_f, raws,
                               views[knobs_f["butter"]], masks, total)

                rows.append({
                    "date": date, "in_sample": date == CALIBRATION,
                    "channel": channel,
                    "frozen_algo": algo_f,
                    "frozen_knobs": ",".join(f"{k}={v:g}" for k, v in
                                             sorted(knobs_f.items()) if k != "butter"),
                    "frozen_coverage_pct": frozen["coverage_pct"] if frozen else 0.0,
                    "refit_algo": best["algo"] if best else None,
                    "refit_butter": best["butter"] if best else None,
                    "refit_knobs": best["knobs"] if best else None,
                    "refit_coverage_pct": best["coverage_pct"] if best else None,
                    "deficit_pts": (best["coverage_pct"] - frozen["coverage_pct"]
                                    if best and frozen else None),
                    "same_setting": bool(
                        best and frozen
                        and best["algo"] == algo_f
                        and best["butter"] == int(knobs_f["butter"])
                        and best["knobs"] == rows_knobs(knobs_f)
                    ),
                    "n_admissible": sum(
                        1 for r in scored
                        if r["worst_segment_error"] <= budgets[channel]),
                    "n_scored": len(scored),
                })
                print(f"  {channel:6s} frozen {rows[-1]['frozen_coverage_pct']:6.2f} % "
                      f"refit {rows[-1]['refit_coverage_pct'] or 0:6.2f} % "
                      f"deficit {rows[-1]['deficit_pts'] or 0:+5.2f} pts "
                      f"({time.monotonic() - started:.0f} s)"
                      + ("  same" if rows[-1]["same_setting"] else ""))

                # Hand the residual down at the FROZEN point, not the refit one:
                # what a downstream channel sees in production is the frozen
                # cascade, and the refit is a counterfactual for this stage only.
                if channel in ("alt", "gamma"):
                    exclusion = [
                        m | committed_mask(channel, algo_f, knobs_f, raw, view, m)
                        for m, raw, view in zip(
                            exclusion, raws, views[int(knobs_f["butter"])])
                    ]

        checkpoint([r for r in rows if r["date"] == date])
        checkpoint_front([r for r in front_rows if r["date"] == date])
        print(f"  {date} checkpointed "
              f"({sum(1 for r in front_rows if r['date'] == date)} front rows)")

    print(f"wrote {OUTPUTS / 'recalibrated.csv'} "
          f"({len(rows)} rows, dates {sorted({r['date'] for r in rows})})")
    return 0


def rows_knobs(knobs: dict) -> str:
    return ",".join(f"{k}={v:g}" for k, v in sorted(knobs.items()) if k != "butter")


if __name__ == "__main__":
    raise SystemExit(main())

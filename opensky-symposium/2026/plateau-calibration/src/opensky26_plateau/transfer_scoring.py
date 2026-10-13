"""The detector as experiment 02 scored it, extracted so it cannot fork.

Experiment 02 (the operator sweep) defined its trim, its filter, its Mach
cascade and its per-segment scoring inside ``scripts/01_sweep.py``. Experiment
03 measures what the setting 02 retained COSTS when transported to other
aircraft — which is only meaningful if the detection and the scoring are the
same operation. Two copies of that code would answer a different question the
moment one of them drifted, and the drift would be silent.

So the functions live here, once, and both experiments import them. The sweep's
own script is left untouched: it is locked evidence, and its lock records the
digest of its bytes. What is here is the same logic, extracted verbatim rather
than reworked — an "improvement" made during the extraction would invalidate the
comparison it exists to enable.

Two orderings are load-bearing and must not be rearranged:

  * **trim before filter.** ``butter_lowpass`` interpolates NaNs, so filtering
    first would reconstruct the frozen tail the trim exists to remove.
  * **cascade after detection.** Every constraint is applied to the segments a
    family returned, never through its signature. A constraint passed as a
    detector argument reaches only the families whose signature accepts it, and
    the silence is undetectable — that defect made run 2 of archived experiment
    14 meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

_PACKAGE = Path(__file__).resolve().parent
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from opensky26_plateau import densify_front as front  # noqa: E402
from opensky26_plateau import detectors as D  # noqa: E402
from opensky26_plateau import smoothing as S  # noqa: E402

__all__ = [
    "CHANNELS",
    "DT_S",
    "candidate_grid",
    "detect",
    "filtered",
    "mach_config",
    "mach_mask",
    "plateau_errors",
    "resolve_frozen",
    "score_segments",
    "signals",
]

DT_S = 4.0
CHANNELS = ("alt", "vz", "gamma", "cas", "mach")

#: A frozen run shorter than this is not a dead transmitter. Fifteen samples at
#: 4 s is a minute of every channel repeating to the last decimal.
_MIN_FROZEN_RUN = 15


def _frozen_ends(columns: list[np.ndarray]) -> tuple[int, int]:
    """First and last index that is not part of a frozen run at either end.

    A dead signal shows as every channel repeating its previous value at the
    same time: on flight 4bb270_THY30B_s0 the last eight minutes hold altitude
    at 125 ft, ground speed at 95 kt and track at 95.41186915231704 — fifteen
    identical decimals. No taxiing aircraft reproduces a heading to the
    fifteenth decimal; a decoder repeating its last frame does.

    One channel alone must never trigger this. A flat altitude is a level-off
    and a flat Mach is a held Mach — those are what the detector exists to
    find, and trimming on a single flat channel would delete the phenomenon
    under study. Interior frozen stretches are left alone: they are dropouts
    worth seeing, not ends worth cutting.

    A null does not break a run. An absent value is not the signal resuming,
    and a flight whose tail decays into nulls would otherwise reset the counter
    and keep the whole frozen stretch.
    """
    n = len(columns[0])

    def frozen_at(i: int) -> bool:
        for col in columns:
            a, b = col[i - 1], col[i]
            if np.isnan(a) or np.isnan(b):
                continue
            if a != b:
                return False
        return True

    tail = n - 1
    while tail > 0 and all(np.isnan(col[tail]) for col in columns):
        tail -= 1

    last, run = tail, 0
    for i in range(tail, 0, -1):
        if frozen_at(i):
            run += 1
        else:
            if run >= _MIN_FROZEN_RUN:
                last = i
            break

    first, run = 0, 0
    for i in range(1, n):
        if frozen_at(i):
            run += 1
        else:
            if run >= _MIN_FROZEN_RUN:
                first = i - 1
            break

    return first, min(last, n - 1)


def signals(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """The five channel signals, trimmed at their frozen ends.

    Trimmed samples become NaN rather than disappearing: the time base stays
    intact, so a segment's indices still map onto the trajectory, and the gap
    stays visible instead of the flight silently getting shorter.
    """
    # Copied: _signal hands back a read-only view onto the Polars column,
    # and the trim below writes NaN into the ends.
    out = {
        ch: np.array(front._signal(ch, frame), dtype=np.float64, copy=True)
        for ch in CHANNELS
    }
    # Position included, and it is the channel that decides. Altitude, ground
    # speed and track are QUANTISED — altitude to 25 ft, ground speed to the
    # knot — so a stabilised cruise can repeat all three for minutes while the
    # aircraft covers tens of kilometres. Measured over 100 flights, trimming
    # on those three alone cut six stretches where the position had moved by up
    # to 0.385 degrees (~40 km) at 34 000 ft: real flight, deleted. A dead
    # transmitter repeats its position too; a flying aircraft never does.
    reference = [
        np.asarray(frame["raw_alt_ft"], dtype=np.float64),
        np.asarray(frame["raw_gs_kt"], dtype=np.float64),
        np.asarray(frame["raw_track_deg"], dtype=np.float64),
        np.asarray(frame["raw_lat_deg"], dtype=np.float64),
        np.asarray(frame["raw_lon_deg"], dtype=np.float64),
    ]
    first, last = _frozen_ends(reference)
    for signal in out.values():
        signal[:first] = np.nan
        signal[last + 1:] = np.nan
    return out


def filtered(signal: np.ndarray, cutoff_s: int) -> np.ndarray:
    """The signal a cutoff of *cutoff_s* seconds presents to the detector.

    DETECTION ONLY. What a caller scores against is its own decision, and the
    two are not the same choice. Scoring on this signal would let a candidate
    buy conformity with smoothing — the smoother the view, the flatter
    everything looks — without improving what its segments claim about the
    trajectory. Experiment 03 therefore detects on this and scores on the raw
    signal, and declares that the asymmetry disfavours a non-zero cutoff.

    A cutoff of 0 leaves the signal untouched.
    """
    if not cutoff_s:
        return signal
    finite = np.isfinite(signal)
    if finite.sum() < 16:
        return signal
    out = np.full_like(signal, np.nan)
    out[finite] = S.butter_lowpass(signal[finite], cutoff_s=float(cutoff_s), dt=DT_S)
    return out


def retained_point(channel: str, retained_csv: Path, grids: dict) -> dict:
    """The point the sweep retained for a channel, resolved onto its grid.

    Reads `retained.csv` rather than repeating its values: a constant copied
    out of that file is a second source of truth, and the one this function
    replaced had drifted for over a month without anyone noticing.
    """
    for row in pl.read_csv(retained_csv).iter_rows(named=True):
        if row["channel"] != channel or not row["admissible"]:
            continue
        knobs = {k: float(v) for k, v in
                 (pair.split("=") for pair in row["knobs"].split(","))}
        knobs["butter"] = int(row["butter"])
        resolved = resolve_frozen(knobs, channel, row["algo"], grids)
        resolved["butter"] = int(row["butter"])
        return resolved
    raise ValueError(f"no admissible {channel} point in {retained_csv}")


def mach_config(retained_csv: Path | None = None,
                grids: dict | None = None) -> dict:
    """The Mach point the CAS/IAS cascade runs behind.

    CAS is swept on the residual Mach leaves, so the mask must be built with
    the Mach setting the sweep RETAINED. Until 2026-09-01 it was built from
    `front.MACH` — a constant of the July reproduction package, from a round
    since archived, carrying sigma_r = 0.08 against the retained 0.0025 and
    slope_tol = 6.5e-4 against 3.162e-4.

    The docstring then claimed that constant was "the production setting every
    CAS candidate is scored behind", which described a design nobody had
    chosen: the same sweep that retained a Mach point ignored it one channel
    later. Measured on 120 flights, the stale mask excluded 68.4 % of the
    finite CAS samples where the retained point excludes 53.0 % — 15.4 points
    of residual CAS never got to compete for.
    """
    config = front.CFG["mach"]
    if retained_csv is None or grids is None:
        raise ValueError(
            "mach_config now requires the sweep's retained.csv and its grids: "
            "the CAS cascade must run behind the Mach point the sweep RETAINED, "
            "and reading it is what stops a copy of it drifting again"
        )
    point = retained_point("mach", Path(retained_csv), grids)
    return {
        "sigma_r": point["sigma_r"],
        "slope_tol": point["slope_tol"],
        "flat_tol": config["flat"],
        "min_len": config["min_len"],
    }


def mach_mask(frame: pl.DataFrame, config: dict) -> np.ndarray:
    """Samples Mach has claimed, which CAS/IAS may not take.

    Mach takes priority WITHOUT conditions: no 0.5 floor, no altitude gate.
    The paper observes a Mach plateau and calls it a Mach plateau; it does not
    filter one out on what a command would plausibly be, having disclaimed any
    independent knowledge of what was commanded.
    """
    mach = front._signal("mach", frame)
    segments = D.detect_vz_plateaus_from_bilat(
        mach, np.zeros(mach.size, dtype=bool), sigma_s=8,
        sigma_r=config["sigma_r"], slope_tol=config["slope_tol"],
        flat_tol=config["flat_tol"], min_len=config["min_len"])
    mask = np.zeros(mach.size, dtype=bool)
    for segment in segments:
        mask[segment["start_idx"]: segment["end_idx"] + 1] = True
    return mask


def detect(algo: str, channel: str, signal: np.ndarray, knobs: dict,
           exclusion: np.ndarray) -> list[dict]:
    """One family at one operating point, with the cascade applied after."""
    config = front.CFG[channel]
    common = {"flat_tol": config["flat"], "min_len": config["min_len"]}
    sigma_s = knobs.get("sigma_s", 6)

    if algo == "bilateral":
        if channel == "gamma":
            # abs_min is 0: the cascade already excludes what it excluded, since
            # altitude claims the level samples before gamma ever sees them.
            segments = D.detect_gamma_plateaus_from_bilat(
                signal, exclusion, sigma_s=sigma_s, sigma_r=knobs["sigma_r"],
                slope_tol=knobs["slope_tol"], abs_min=0.0, **common)
        else:
            segments = D.detect_vz_plateaus_from_bilat(
                signal, exclusion, sigma_s=sigma_s, sigma_r=knobs["sigma_r"],
                slope_tol=knobs["slope_tol"], **common)
    elif algo == "savgol":
        segments = D.detect_savgol_plateaus(
            signal, exclusion, window=int(knobs["window"]),
            slope_tol=knobs["slope_tol"], **common)
    elif algo == "varthresh":
        segments = D.detect_variance_threshold(
            signal, window=int(knobs["window"]), std_tol=knobs["std_tol"],
            min_len=config["min_len"])
    else:
        segments = D.detect_pelt_plateaus(
            signal, exclusion, penalty=knobs["penalty"], **common)

    # The one constraint that survives, applied after detection so that all four
    # families face it alike. min_abs and abs_min are gone; the floor and the
    # gate are gone. What remains is the cascade itself: Mach has claimed its
    # samples, and CAS/IAS takes the residual.
    if exclusion.any():
        segments = [
            s for s in segments
            if not exclusion[s["start_idx"]: s["end_idx"] + 1].any()
        ]
    return _trim_to_finite(segments, signal)


def _trim_to_finite(segments: list[dict], signal: np.ndarray) -> list[dict]:
    """Clip each segment to the ground where the signal actually exists.

    Three of the four families call ``interpolate_nans`` before smoothing, so
    the holes ``signals`` cut — a dead transmitter's frozen tail, a dropout —
    come back as a straight line whose derivative is exactly constant. That is
    ``flat=True`` by construction, and a segment is then committed on ground the
    trim removed.

    Measured over the calibration cohort at the retained points: **23.3 % of
    altitude segments and 22.9 % of Mach segments** extended past their signal,
    claiming 121 644 and 150 927 interpolated samples. Coverage never counted
    them — ``score_segments`` and ``plateau_errors`` sum over finite samples
    only — so what such a segment inflates is its own extent, the value it
    announces where the interpolation drags the mean, and the segment count.

    Clipping rather than dropping: the finite part of the segment is a real
    detection and the invented tail is not, so the tail goes and the rest stays.
    A segment left with fewer than two finite samples has nothing to announce
    and is dropped.
    """
    out: list[dict] = []
    for segment in segments:
        i, j = segment["start_idx"], segment["end_idx"]
        finite = np.flatnonzero(np.isfinite(signal[i:j + 1]))
        if finite.size < 2:
            continue
        clipped = dict(segment)
        clipped["start_idx"] = i + int(finite[0])
        clipped["end_idx"] = i + int(finite[-1])
        out.append(clipped)
    return out


#: The variance-threshold windows the sweep enumerated, in samples at 4 s.
_VARTHRESH_WINDOWS = (5, 9, 15, 21, 31, 41, 51, 61)

#: Fallback when a channel declares no sigma_s row, as in the sweep.
_SIGMA_S_FALLBACK = (4, 6, 8, 10)


def candidate_grid(channel: str, algo: str, cutoff: int, grids: dict) -> list[dict]:
    """Every operating point of one family on one channel, at a fixed cutoff.

    The sweep's own enumeration, restricted to a single family and a single
    Butterworth cutoff — which is what a transfer experiment needs: the family
    and the cutoff are frozen by the sweep, and only the continuous knobs move.
    """
    out: list[dict] = []
    if algo == "bilateral":
        sigma_s_values = grids.get("sigma_s", {}).get(channel, _SIGMA_S_FALLBACK)
        for sigma_s in sigma_s_values:
            for sigma_r in grids["sigma"][channel]:
                for slope in grids["slope"][channel]:
                    out.append({
                        "sigma_s": sigma_s, "sigma_r": sigma_r,
                        "slope_tol": slope, "butter": cutoff})
    elif algo == "savgol":
        for window in grids["win"]:
            for slope in grids["slope"][channel]:
                out.append({"window": window, "slope_tol": slope, "butter": cutoff})
    elif algo == "varthresh":
        for std in grids["std"][channel]:
            for window in _VARTHRESH_WINDOWS:
                out.append({"std_tol": std, "window": window, "butter": cutoff})
    else:
        for penalty in grids["pen"][channel]:
            out.append({"penalty": penalty, "butter": cutoff})
    return out


def resolve_frozen(knobs: dict, channel: str, algo: str, grids: dict) -> dict:
    """The grid point a retained.csv row names, at the grid's own precision.

    ``retained.csv`` writes its knobs with ``%g`` — six significant digits — so
    the sweep's ``slope_tol=8.9020365071`` is stored as ``8.90204`` and
    ``std_tol=43.194383185129524`` as ``43.1944``. Four of the five retained
    points are affected.

    Detecting at the truncated value scores a point the sweep never retained.
    The shift is tiny, but altitude and CAS/IAS sit within 1% of their error
    budget, and that is precisely where a tiny shift decides whether a fold
    counts as a breach. Comparing the truncated value against the grid is worse
    still: nothing matches, and every fold reads as having re-selected something
    DIFFERENT for a formatting reason.

    Each knob is resolved onto its grid by nearest neighbour, and the match must
    be unambiguous — within 1e-4 relative, and ten times closer than the runner
    up. A knob that cannot be resolved stops the run rather than being
    approximated.
    """
    grid = candidate_grid(channel, algo, int(knobs["butter"]), grids)
    resolved: dict[str, float] = {}
    for knob, value in knobs.items():
        values = sorted({float(c[knob]) for c in grid if knob in c})
        if not values:
            raise ValueError(f"knob {knob!r} is absent from the {channel} grid")
        target = float(value)
        ordered = sorted(values, key=lambda v: abs(v - target))
        best = ordered[0]
        if abs(best - target) / max(abs(target), 1e-12) > 1e-4:
            raise ValueError(
                f"{channel}: knob {knob}={target:g} matches no grid value "
                f"(nearest {best:g})"
            )
        if len(ordered) > 1 and abs(best - target) * 10 > abs(ordered[1] - target):
            raise ValueError(
                f"{channel}: knob {knob}={target:g} is ambiguous between "
                f"{best:g} and {ordered[1]:g}"
            )
        resolved[knob] = best
    return resolved


def score_segments(signal: np.ndarray, segments: list[dict]) -> tuple[int, float, list[int]]:
    """Committed samples, summed absolute error, and segment lengths.

    The segment value is recomputed over the FINITE samples rather than read
    from the detector. ``_make_segment`` uses ``np.mean`` on the raw slice, so a
    single missing sample makes ``var_mean`` NaN, which makes the whole grid
    point's MAE NaN, which drops it from the budget — silently. gamma carries
    13% NaN before anything is trimmed, so reading the detector's value
    eliminated 100% of the variance-threshold family on gamma and mach: a
    competitor removed by arithmetic rather than by performance, which is
    exactly the defect that made run 2 of archived experiment 14 meaningless.

    Returns the three quantities unaggregated so a caller can pool them over
    whatever unit it is measuring — the whole cohort, one fold, one airframe.
    """
    committed = 0
    abs_error = 0.0
    lengths: list[int] = []
    for s in segments:
        i, j = s["start_idx"], s["end_idx"]
        piece = signal[i:j + 1]
        finite = np.isfinite(piece)
        if not finite.any():
            continue
        value = float(piece[finite].mean())
        lengths.append(j - i + 1)
        committed += int(finite.sum())
        abs_error += float(np.abs(piece[finite] - value).sum())
    return committed, abs_error, lengths


def plateau_errors(signal: np.ndarray, segments: list[dict]) -> tuple[int, list[float]]:
    """Committed samples, and the MAE of each plateau taken on its own.

    ``score_segments`` sums the absolute error over the whole flight and the
    caller divides by the committed count, which makes the budget a check on a
    SAMPLE-WEIGHTED MEAN: a long easy plateau pays for a short bad one. Measured
    on 2026-08-26, that is not a corner case but the population — the median
    per-flight worst plateau exceeds budget on all five channels, and 22 % to
    47 % of plateaux are over budget individually while the flights containing
    them report a comfortable mean.

    The mechanism, for the record. The vertical-speed point retained under that
    rule is ``varthresh`` at ``butter=120``. On a descent from 3 200 to 2 010
    ft/min the Butterworth drops the local standard deviation from 67.1 to 23.7,
    below the ``std_tol`` of 43.19 — a variance threshold tests LOCAL flatness
    over five samples, and the filter removes exactly the noise that betrayed
    the slope. That segment's own MAE is 255.5 ft/min against a budget of 50,
    and the flight still reports 19.4 because 207 minutes of cruise at MAE ≈ 0
    dominate the mean.

    So the errors come back **unaggregated, one per plateau**, for the same
    reason ``score_segments`` returns its three quantities separately: the unit
    of aggregation is the caller's decision, not this function's. Taking the max
    here would force every caller to the same choice and would throw away the
    distribution the selection rule needs.

    A plateau whose error is exactly zero is a real plateau and is REPORTED AS
    ZERO. It must not be dropped: altitude is quantised to 25 ft, so a stabilised
    level repeats one value exactly and its plateau errs by nothing at all.
    Dropping those zeros would compute the cohort's p95 over the flights that
    happen to be noisy, which is how a first pass at this measurement produced a
    p95 of 7e-12 on altitude — a statistic taken over the wrong population.

    Returns:
        The committed sample count (identical to ``score_segments``'s), and one
        MAE per plateau that carried at least one finite sample.
    """
    committed = 0
    errors: list[float] = []
    for s in segments:
        i, j = s["start_idx"], s["end_idx"]
        piece = signal[i:j + 1]
        finite = np.isfinite(piece)
        if not finite.any():
            continue
        value = float(piece[finite].mean())
        committed += int(finite.sum())
        errors.append(float(np.abs(piece[finite] - value).mean()))
    return committed, errors

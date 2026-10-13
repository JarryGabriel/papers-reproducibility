"""Measure what the signal imposes, channel by channel.

Four measurements, none of which reads a detected plateau:

  * the quantisation step of every channel, from the distribution of non-zero
    absolute first differences. A hard floor at the low quantiles means the
    channel is quantised.
  * the ERA5 reconstruction error on the two speed channels, against the
    broadcast register on the samples where both exist.
  * whether that error tracks the wind -- the hypothesis this experiment was
    proposed under, and which exploratory measurement refuted.
  * the propagated budget for gamma, which is computed rather than measured.

The filter and the statistic each budget reads are fixed in inputs/SOURCES.md
and committed before this ran.
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

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PAPER = project_root(__file__)
REPO = project_root(__file__)


from opensky26_plateau import transfer_scoring as T  # noqa: E402

DATE = "2025-11-01"
DELTA = data_dir() / 'flights.delta'

FT_MIN_TO_MS = 0.00508
KT_TO_MS = 0.5144444444444445

#: Declared in SOURCES.md before the run.
ALT_FLOOR_FT = 10_000.0
IAS_LOW_KT, IAS_HIGH_KT = 100.0, 400.0

#: The columns the DETECTOR reads, from ``densify_front._signal`` — not the
#: pipeline's other speed columns. `cas` is `bds_ias_kt_clean`, and altitude is
#: read in feet here even though the detector converts it to metres, because a
#: quantisation step is a property of the encoding and 25 ft is the number the
#: standard names. The conversion happens where the budget is assembled.
CHANNEL_COLUMN = {
    "alt": "raw_alt_ft",
    "vz": "raw_vz_ftmin",
    "cas": "bds_ias_kt_clean",
    "mach": "bds_mach_clean",
}
#: The unit each channel is MEASURED in above, which is not always the unit the
#: detector scores in. Altitude is the only one that differs.
MEASURED_UNIT = {"alt": "ft", "vz": "ft/min", "cas": "kt", "mach": "-"}
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def robust_sigma(x: np.ndarray) -> float:
    """1.4826 x MAD -- a dispersion the 119 kt tail cannot move.

    The raw comparison has a p99 of 119.5 kt, which is a decoder artefact
    rather than a reanalysis error. A standard deviation would be decided by
    those samples; the MAD is not, and SOURCES.md fixes this as the estimator
    the speed budgets read.
    """
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def quantisation_step(gaps: np.ndarray) -> tuple[float, float]:
    """The step every gap is a multiple of, and the share that actually are.

    NOT the mode and NOT a low percentile, and both failures are instructive.
    The mode of the altitude gaps is 75 ft, because a climbing aircraft crosses
    several steps between two samples — 75, 50, 25, 100 and 125 each take 10 to
    20 % of the mass, and only their common divisor is the step. A low
    percentile fails the other way: `resample_flight` interpolates onto a 4 s
    grid, which manufactures intermediate values, so vz's p1 is 9.14 ft/min —
    a resampling artefact that a percentile mistakes for the floor while 71 %
    of the gaps are exact multiples of 64.

    So the step is the largest candidate that divides most of the distribution.
    Candidates are taken from the frequent gaps themselves rather than from a
    grid of our choosing, and the largest one dividing at least half the gaps
    wins. A channel where no candidate reaches half is not quantised.

    Returns:
        The step, and the share of gaps that are integer multiples of it. A
        share below 0.5 means the channel carries no step and the value is the
        best candidate rather than a measurement.
    """
    rounded = np.round(gaps, 9)
    values, counts = np.unique(rounded, return_counts=True)
    # Only gaps that recur can be a step; a unique value is noise.
    candidates = values[counts >= max(3, int(0.005 * gaps.size))]
    candidates = candidates[candidates > 0]
    if candidates.size == 0:
        return float(np.min(rounded)), 0.0

    best_step, best_share = float(candidates[0]), 0.0
    for candidate in np.sort(candidates):
        ratio = rounded / candidate
        share = float(np.mean(np.abs(ratio - np.round(ratio)) < 1e-6))
        # Largest step clearing the bar: a step's own multiples also divide the
        # distribution, so scanning upward and keeping the last winner returns
        # the coarsest true step rather than its smallest submultiple.
        if share >= 0.5:
            best_step, best_share = float(candidate), share
        elif share > best_share and best_share < 0.5:
            best_step, best_share = float(candidate), share
    return best_step, best_share


def load() -> pl.DataFrame:
    frame = pl.read_delta(str(DELTA)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    )
    return frame.filter(pl.col("_date") == DATE)


def measure_quantisation(frame: pl.DataFrame) -> pl.DataFrame:
    """The effective step of each channel, per flight then pooled.

    Measured on the signal the detector sees -- trimmed at its frozen ends, so
    a dead transmitter repeating its last frame cannot manufacture a floor of
    zero. The gaps are pooled across flights because a step is a property of
    the encoding, not of a trajectory.
    """
    rows = []
    ids = frame["meta_flight_id"].unique(maintain_order=True).to_list()
    per_channel: dict[str, list[np.ndarray]] = {c: [] for c in CHANNEL_COLUMN}
    distinct: dict[str, list[float]] = {c: [] for c in CHANNEL_COLUMN}
    for flight_id in ids:
        flight = frame.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
        signals = T.signals(flight)
        for channel, column in CHANNEL_COLUMN.items():
            # gamma is derived, so it has no step of its own: excluded here and
            # handled by the propagation instead.
            raw = np.asarray(flight[column], dtype=np.float64)
            first = np.isfinite(signals["alt"])
            if not first.any():
                continue
            lo, hi = int(np.argmax(first)), int(len(first) - np.argmax(first[::-1]))
            piece = raw[lo:hi]
            piece = piece[np.isfinite(piece)]
            if piece.size < 100:
                continue
            gaps = np.abs(np.diff(piece))
            gaps = gaps[gaps > 0]
            if gaps.size:
                per_channel[channel].append(gaps)
            distinct[channel].append(float(np.unique(piece).size / piece.size))

    for channel, chunks in per_channel.items():
        pooled = np.concatenate(chunks)
        quantiles = {f"p{q}": float(np.percentile(pooled, q)) for q in QUANTILES}
        step, share = quantisation_step(pooled)
        rows.append({
            "channel": channel,
            "column": CHANNEL_COLUMN[channel],
            "unit": MEASURED_UNIT[channel],
            "n_gaps": int(pooled.size),
            "n_flights": len(chunks),
            "distinct_frac": float(np.mean(distinct[channel])),
            "step": step,
            "share_multiple_of_step": share,
            "quantised": bool(share >= 0.5),
            **quantiles,
        })
    return pl.DataFrame(rows)


def paired(frame: pl.DataFrame) -> pl.DataFrame:
    """The samples where register and reanalysis both exist, filtered as declared."""
    return frame.filter(
        pl.col("bds_ias_kt").is_not_null()
        & pl.col("era_cas_kt").is_not_null()
        & pl.col("bds_mach").is_not_null()
        & pl.col("era_mach").is_not_null()
        & (pl.col("raw_alt_ft") > ALT_FLOOR_FT)
        & (pl.col("bds_ias_kt") > IAS_LOW_KT)
        & (pl.col("bds_ias_kt") < IAS_HIGH_KT)
    )


def measure_reconstruction(frame: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """The ERA5 error against the register, and how much of the channel it fills."""
    n_rows = frame.height
    rows = []
    for channel, bds_col, era_col in (("cas", "bds_ias_kt", "era_cas_kt"),
                                      ("mach", "bds_mach", "era_mach")):
        register = np.asarray(pairs[bds_col], dtype=np.float64)
        reanalysis = np.asarray(pairs[era_col], dtype=np.float64)
        ok = np.isfinite(register) & np.isfinite(reanalysis)
        error = reanalysis[ok] - register[ok]
        absolute = np.abs(error)
        # How much of the channel is reconstructed rather than measured.
        clean_col = f"bds_{'ias_kt' if channel == 'cas' else 'mach'}_clean"
        have = frame[clean_col].is_not_null().sum()
        substituted = frame.filter(
            pl.col(bds_col).is_null() & pl.col(clean_col).is_not_null()
        ).height
        rows.append({
            "channel": channel,
            "register_column": bds_col,
            "reanalysis_column": era_col,
            "n_paired": int(error.size),
            "n_rows_total": int(n_rows),
            "register_present_pct": float(frame[bds_col].is_not_null().sum() / n_rows * 100),
            "reconstructed_pct": float(substituted / have * 100) if have else float("nan"),
            "bias": float(error.mean()),
            "mae": float(absolute.mean()),
            "robust_sigma": robust_sigma(error),
            **{f"p{q}": float(np.percentile(absolute, q)) for q in QUANTILES},
        })
    return pl.DataFrame(rows)


def measure_wind_dependence(pairs: pl.DataFrame) -> pl.DataFrame:
    """Does the reconstruction error track the wind? Reported either way.

    This is the hypothesis the experiment was proposed under. It is reported
    with its correlation and its binned medians whatever it shows, because a
    reader reaching for the same intuition needs the number that settles it.
    """
    u = np.asarray(pairs["era_u_wind_ms"], dtype=np.float64)
    v = np.asarray(pairs["era_v_wind_ms"], dtype=np.float64)
    wind_kt = np.hypot(u, v) / KT_TO_MS
    errors = {
        "cas": np.asarray(pairs["era_cas_kt"], dtype=np.float64)
        - np.asarray(pairs["bds_ias_kt"], dtype=np.float64),
        "mach": np.asarray(pairs["era_mach"], dtype=np.float64)
        - np.asarray(pairs["bds_mach"], dtype=np.float64),
    }
    altitude = np.asarray(pairs["raw_alt_ft"], dtype=np.float64)

    rows = []
    edges = [0, 10, 20, 30, 40, 60, 80, 200]
    for channel, error in errors.items():
        ok = np.isfinite(error) & np.isfinite(wind_kt)
        correlation = float(np.corrcoef(wind_kt[ok], np.abs(error[ok]))[0, 1])
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = ok & (wind_kt >= lo) & (wind_kt < hi)
            if inside.sum() < 200:
                continue
            rows.append({
                "channel": channel, "binned_on": "wind_kt",
                "bin_low": float(lo), "bin_high": float(hi),
                # Where the samples ACTUALLY sit. The last bin spans 80-200 kt
                # and its centre, 140, is a wind speed nothing in it flies at —
                # plotting there would draw a rise that is an artefact of the
                # bin edges rather than a measurement.
                "bin_centre": float(np.median(wind_kt[inside])),
                "n": int(inside.sum()),
                "median_abs_error": float(np.median(np.abs(error[inside]))),
                "pearson_r_full_range": correlation,
            })
        # Altitude as the comparison predictor: ERA5 is better constrained aloft.
        for lo, hi in ((10, 20), (20, 30), (30, 35), (35, 40), (40, 50)):
            inside = ok & (altitude >= lo * 1000) & (altitude < hi * 1000)
            if inside.sum() < 200:
                continue
            rows.append({
                "channel": channel, "binned_on": "alt_kft",
                "bin_low": float(lo), "bin_high": float(hi),
                "bin_centre": float(np.median(altitude[inside]) / 1000.0),
                "n": int(inside.sum()),
                "median_abs_error": float(np.median(np.abs(error[inside]))),
                "pearson_r_full_range": float(
                    np.corrcoef(altitude[ok], np.abs(error[ok]))[0, 1]
                ),
            })
    return pl.DataFrame(rows)


def measure_propagation(frame: pl.DataFrame, d_vz_ftmin: float,
                        d_speed_kt: float) -> pl.DataFrame:
    """The gamma budget implied by the vz and TAS budgets.

    gamma is computed as arcsin(vz/TAS), so its budget is not a free choice:

        dgamma = 1/sqrt(1-(vz/V)^2) * sqrt( (dvz/V)^2 + (vz*dV/V^2)^2 )

    Evaluated sample by sample on the real speed distribution rather than at an
    assumed cruise speed, because dgamma scales as 1/V and the cohort spans a
    factor of three in V.
    """
    vz = np.asarray(frame["raw_vz_ftmin"], dtype=np.float64) * FT_MIN_TO_MS
    speed = np.asarray(frame["fdm_tas_from_cas_kt"], dtype=np.float64) * KT_TO_MS
    ok = np.isfinite(vz) & np.isfinite(speed) & (speed > 50.0)
    vz, speed = vz[ok], speed[ok]

    d_vz = d_vz_ftmin * FT_MIN_TO_MS
    d_speed = d_speed_kt * KT_TO_MS
    ratio = np.clip(vz / speed, -0.999, 0.999)
    jacobian = 1.0 / np.sqrt(1.0 - ratio ** 2)
    term_vz = (d_vz / speed) ** 2
    term_speed = (vz * d_speed / speed ** 2) ** 2
    propagated = jacobian * np.sqrt(term_vz + term_speed)

    return pl.DataFrame([{
        "channel": "gamma",
        "n_samples": int(propagated.size),
        "d_vz_ftmin": d_vz_ftmin,
        "d_speed_kt": d_speed_kt,
        "term_vz_median": float(np.sqrt(np.median(term_vz))),
        "term_speed_median": float(np.sqrt(np.median(term_speed))),
        "term_ratio": float(np.sqrt(np.median(term_vz) / np.median(term_speed))),
        "speed_p5_ms": float(np.percentile(speed, 5)),
        "speed_p50_ms": float(np.percentile(speed, 50)),
        "speed_p95_ms": float(np.percentile(speed, 95)),
        **{f"p{q}": float(np.percentile(propagated, q)) for q in QUANTILES},
    }])


def main() -> None:
    out = EXP / "outputs"
    out.mkdir(exist_ok=True)
    frame = load()
    print(f"{frame.height:,} rows on {DATE}")

    quantisation = measure_quantisation(frame)
    quantisation.write_csv(out / "quantisation.csv")
    print(f"  quantisation: {quantisation.height} channels")

    pairs = paired(frame)
    print(f"  paired samples after the declared filter: {pairs.height:,}")

    reconstruction = measure_reconstruction(frame, pairs)
    reconstruction.write_csv(out / "reconstruction.csv")

    wind = measure_wind_dependence(pairs)
    wind.write_csv(out / "wind_dependence.csv")

    # The vz budget feeds the propagation, so it is read from the quantisation
    # measured above rather than hard-coded: step / 2, the declared convention.
    vz_step = float(quantisation.filter(pl.col("channel") == "vz")["step"][0])
    cas_sigma = float(reconstruction.filter(pl.col("channel") == "cas")["robust_sigma"][0])
    propagation = measure_propagation(frame, vz_step / 2.0, cas_sigma)
    propagation.write_csv(out / "propagation.csv")
    print(f"  propagation: dvz={vz_step / 2:.4g} ft/min, dV={cas_sigma:.4g} kt")


if __name__ == "__main__":
    main()

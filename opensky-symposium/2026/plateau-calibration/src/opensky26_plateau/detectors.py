"""Plateau detectors for Paper 0-A — self-contained, pure numpy/scipy.

Copied verbatim from node-fdm-data/segments.py (functions `_smooth` ..
`detect_vz_plateaus_from_bilat`) so this repository runs standalone without
installing the library. The bilateral / Savitzky-Golay / variance-threshold
front-ends are the algorithms compared on the coverage x self-consistency
Pareto front. See ../README.md.

Each detector returns a list of segment dicts {start, end, value}; the value
is the per-run constant (mean of the detector input over the run).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from opensky26_plateau.smoothing import bilateral_1d, butter_lowpass, interpolate_nans

__all__ = [
    "detect_alt_hold_from_vz",
    "detect_cas_plateaus_bilat",
    "detect_gamma_plateaus_from_bilat",
    "detect_mach_plateaus_bilat",
    "detect_pelt_plateaus",
    "detect_savgol_plateaus",
    "detect_variance_threshold",
    "detect_vz_plateaus_from_bilat",
]


def detect_variance_threshold(
    y: np.ndarray, *, window: int, std_tol: float, min_len: int
) -> list[dict[str, Any]]:
    """Trivial baseline: a sample is flat when its rolling std < ``std_tol``;
    a maximal flat run >= ``min_len`` is a plateau with a single constant value
    (mean of the raw signal over the run). No bilateral, no Hampel. Isolates
    the value of the edge-preserving front-end. Same segment-dict interface as
    the bilateral detectors."""
    n = y.size
    h = window // 2
    flat = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        w = y[lo:hi]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            flat[i] = w.std() < std_tol
    segs: list[dict[str, Any]] = []
    s = None
    for i in range(n):
        if flat[i]:
            if s is None:
                s = i
        else:
            if s is not None and i - s >= min_len:
                segs.append(_make_segment(s, i - 1, y))
            s = None
    if s is not None and n - s >= min_len:
        segs.append(_make_segment(s, n - 1, y))
    return segs


def detect_savgol_plateaus(
    y: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    window: int,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
    abs_min: float | None = None,
) -> list[dict[str, Any]]:
    """Savitzky-Golay front-end variant of the plateau detector: replaces the
    bilateral smoothing by a SavGol low-pass, then the same slope-flat +
    run-validator machinery. Lets the front-end be swapped on the same
    detection logic for the sensitivity analysis."""
    yi = interpolate_nans(y)
    if yi is None:
        return []
    ys = _smooth(yi, window, "savgol")
    flat = _slope_flat_mask(ys, slope_tol) & ~exclusion_mask
    return _bilateral_plateau_runs(ys, flat, flat_tol, min_len, abs_min)


def detect_pelt_plateaus(
    y: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    penalty: float,
    flat_tol: float,
    min_len: int,
    model: str = "l2",
    jump: int = 40,
    abs_min: float | None = None,
) -> list[dict[str, Any]]:
    """External PELT baseline (Killick et al. 2012, via ``ruptures``).

    PELT partitions the (NaN-interpolated) signal into change-point segments with
    the given ``penalty``; each segment is then a plateau candidate held to the
    SAME tolerances as the in-house front-ends (``flat_tol`` spread, ``min_len``,
    optional ``abs_min``) and excluded where ``exclusion_mask`` is set, so the
    comparison on the coverage x rf x parsimony axes is fair. PELT brings no
    channel physics (no cascade gate, no near-zero floor) beyond what the shared
    validator applies — which is exactly the point of an off-the-shelf baseline.

    ``jump`` subsamples the candidate change-point grid (ruptures' speed knob): at
    the 4 s sampling, jump=40 = 160 s resolution. PELT's l2 cost scales poorly on
    the long CAS/Mach cruise signals (jump=1 is quadratic, jump=5 still ~hours on
    the cohort); jump=40 makes it tractable (~minutes). A held set-point lasts far
    longer than 160 s, so the coarse grid is fine for a baseline whose role is to
    sit dominated on the front, not to be finely optimised.
    """
    import ruptures as rpt  # local import: optional heavy dependency

    yi = interpolate_nans(y)
    if yi is None or yi.size < max(2 * min_len, 2):
        return []
    algo = rpt.Pelt(model=model, min_size=max(2, min_len), jump=jump).fit(yi)
    bkps = algo.predict(pen=penalty)  # sorted change-points, last == n
    excl = np.asarray(exclusion_mask, dtype=bool)
    segments: list[dict[str, Any]] = []
    start = 0
    for end in bkps:  # segment [start, end-1] inclusive
        e = end - 1
        # drop the segment if it overlaps any excluded sample (kept simple and
        # identical in spirit to the other fronts' masking)
        if not excl[start : e + 1].any():
            seg = _validate_plateau_run(yi, start, e, flat_tol, min_len, abs_min)
            if seg is not None:
                segments.append(seg)
        start = end
    return segments


def _smooth(y: np.ndarray, window: int, method: str) -> np.ndarray:
    if method == "savgol":
        win = min(window, len(y) - (len(y) % 2 == 0))
        if win >= 3:
            smoothed: np.ndarray = savgol_filter(y, window_length=win, polyorder=2, mode="interp")
            return smoothed
        return y
    return np.convolve(y, np.ones(window) / window, mode="same")


def _make_segment(start: int, end_idx: int, y: np.ndarray) -> dict[str, Any]:
    return {
        "start_idx": start,
        "end_idx": end_idx,
        "var_mean": float(np.mean(y[start : end_idx + 1])),
    }


def _smooth_bilateral(
    values: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    n_passes: int,
) -> np.ndarray | None:
    """Interpolate NaNs then apply ``n_passes`` of 1D bilateral smoothing.

    Returns ``None`` if ``values`` is fully NaN (no signal to recover).
    """
    work = np.asarray(values, dtype=np.float64)
    nan_mask = np.isnan(work)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return None
        work = interpolate_nans(work)
    for _ in range(max(0, int(n_passes))):
        work = bilateral_1d(work, sigma_s, sigma_r)
    return work


def _slope_flat_mask(values: np.ndarray, slope_tol: float) -> np.ndarray:
    """Boolean mask of samples whose edge-prepended finite diff is below ``slope_tol``."""
    mask: np.ndarray = np.abs(np.diff(values, prepend=values[0])) < slope_tol
    return mask


def _flat_runs(flat: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Inclusive ``[start, end]`` runs of True in ``flat`` with length >= ``min_len``."""
    runs: list[tuple[int, int]] = []
    n = len(flat)
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j))
        i = j + 1
    return runs


def _below_tol_mask(values: np.ndarray, tol: float) -> np.ndarray:
    """Boolean mask of samples whose absolute value is strictly below ``tol``."""
    mask: np.ndarray = np.abs(values) < tol
    return mask


def detect_alt_hold_from_vz(
    vz_ftmin: np.ndarray,
    alt_ft: np.ndarray,
    *,
    sigma_s: float = 6.0,
    sigma_r: float = 350.0,
    n_passes: int = 2,
    tol_ftmin: float = 150.0,
    min_len: int = 6,
) -> list[dict[str, Any]]:
    """Detect altitude-hold plateaus from vertical speed.

    Linearly interpolates NaNs in ``vz_ftmin`` (returns ``[]`` if fully
    NaN), applies ``n_passes`` of a 1D bilateral filter, then keeps
    runs where ``|vz_bilat| < tol_ftmin`` of length ``>= min_len``.
    Each segment's ``var_mean`` is the mean of ``alt_ft`` over the run
    so the result can feed :func:`add_segment_column` directly into
    ``fdm_alt_sel_ft``.
    """
    vz = np.asarray(vz_ftmin, dtype=np.float64).copy()
    alt = np.asarray(alt_ft, dtype=np.float64)
    vz_bilat = _smooth_bilateral(vz, sigma_s, sigma_r, n_passes)
    if vz_bilat is None:
        return []
    mask = _below_tol_mask(vz_bilat, tol_ftmin)
    segments: list[dict[str, Any]] = []
    start: int | None = None
    n = len(mask)
    for i in range(n):
        if mask[i]:
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= min_len:
                segments.append(_make_segment(start, i - 1, alt))
            start = None
    if start is not None and n - start >= min_len:
        segments.append(_make_segment(start, n - 1, alt))
    return segments


def _runs_below_tol(mask: np.ndarray, alt: np.ndarray, min_len: int) -> list[dict[str, Any]]:
    """Maximal True-runs of ``mask`` >= min_len, each labelled with the mean of
    ``alt`` over the run. Shared hold-run collector for all alt-hold fronts."""
    segs: list[dict[str, Any]] = []
    s = None
    n = len(mask)
    for i in range(n):
        if mask[i]:
            if s is None:
                s = i
        else:
            if s is not None and i - s >= min_len:
                segs.append(_make_segment(s, i - 1, alt))
            s = None
    if s is not None and n - s >= min_len:
        segs.append(_make_segment(s, n - 1, alt))
    return segs


def detect_alt_hold(
    vz_ftmin: np.ndarray,
    alt_ft: np.ndarray,
    *,
    mode: str,
    tol_ftmin: float,
    min_len: int,
    sigma_s: float = 6.0,
    sigma_r: float = 350.0,
    n_passes: int = 2,
    window: int = 11,
    std_window: int = 15,
) -> list[dict[str, Any]]:
    """Altitude-hold detection sharing one criterion across three v_z smoothing
    front-ends. A hold is a run where the SMOOTHED v_z stays within
    ``tol_ftmin`` of zero; its value is the mean raw altitude over the run.

    Detection is on v_z (prod-faithful: altitude has encoding round-off, v_z is
    the clean hold signal). ``mode`` selects the front-end:
      - "bilateral": bilateral-smoothed v_z (prod).
      - "savgol":    Savitzky-Golay-smoothed v_z.
      - "varthresh": a hold where the rolling std of v_z is below tol_ftmin
                     (no pre-smoothing) — the trivial front.
    """
    vz = np.asarray(vz_ftmin, dtype=np.float64).copy()
    alt = np.asarray(alt_ft, dtype=np.float64)
    if mode == "bilateral":
        sm = _smooth_bilateral(vz, sigma_s, sigma_r, n_passes)
        if sm is None:
            return []
        mask = _below_tol_mask(sm, tol_ftmin)
    elif mode == "savgol":
        vzi = interpolate_nans(vz)
        if vzi is None:
            return []
        sm = _smooth(vzi, window, "savgol")
        mask = _below_tol_mask(sm, tol_ftmin)
    elif mode == "varthresh":
        h = std_window // 2
        n = vz.size
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            w = vz[max(0, i - h) : min(n, i + h + 1)]
            w = w[np.isfinite(w)]
            if w.size >= 3:
                mask[i] = w.std() < tol_ftmin and abs(np.mean(w)) < tol_ftmin
    else:
        raise ValueError(f"unknown alt-hold mode {mode!r}")
    return _runs_below_tol(mask, alt, min_len)


def _validate_plateau_run(
    y_bilat: np.ndarray,
    s: int,
    e_inclusive: int,
    flat_tol: float,
    min_len: int,
    abs_min: float | None,
) -> dict[str, Any] | None:
    """Return a segment dict if ``[s, e_inclusive]`` satisfies tolerances."""
    if e_inclusive - s + 1 < min_len:
        return None
    chunk = y_bilat[s : e_inclusive + 1]
    if (chunk.max() - chunk.min()) > 2.0 * flat_tol:
        return None
    mean = float(np.mean(chunk))
    if abs_min is not None and abs(mean) < abs_min:
        return None
    return {"start_idx": s, "end_idx": e_inclusive, "var_mean": mean}


def _bilateral_plateau_runs(
    y_bilat: np.ndarray,
    flat: np.ndarray,
    flat_tol: float,
    min_len: int,
    abs_min: float | None,
) -> list[dict[str, Any]]:
    """Walk runs of ``flat=True`` and accept those satisfying tolerances."""
    segments: list[dict[str, Any]] = []
    n = len(flat)
    start: int | None = None

    for i in range(n):
        if flat[i]:
            if start is None:
                start = i
            continue
        if start is None:
            continue
        seg = _validate_plateau_run(y_bilat, start, i - 1, flat_tol, min_len, abs_min)
        if seg is not None:
            segments.append(seg)
        start = None

    if start is not None:
        seg = _validate_plateau_run(y_bilat, start, n - 1, flat_tol, min_len, abs_min)
        if seg is not None:
            segments.append(seg)
    return segments


def detect_gamma_plateaus_from_bilat(
    gamma_raw: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    slope_tol: float,
    flat_tol: float,
    abs_min: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect gamma plateaus from a bilateral-smoothed gamma signal.

    The exclusion mask (True = exclude) lets the caller enforce cascade
    priority (e.g. mask out alt-hold regions before searching for gamma
    plateaus). Returns segment dicts ``{start_idx, end_idx, var_mean}``
    where ``var_mean`` is the mean of the bilateral gamma over the run.
    """
    g = np.asarray(gamma_raw, dtype=np.float64).copy()
    nan_mask = np.isnan(g)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        g = interpolate_nans(g)
    g_bilat = bilateral_1d(bilateral_1d(g, sigma_s, sigma_r), sigma_s, sigma_r)
    dgamma = np.abs(np.diff(g_bilat, prepend=g_bilat[0]))
    excl = np.asarray(exclusion_mask, dtype=bool)
    flat = (dgamma < slope_tol) & (~excl)
    return _bilateral_plateau_runs(g_bilat, flat, flat_tol, min_len, abs_min)


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Walk a boolean mask and return inclusive [start, end] runs of True."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    start: int | None = None
    for i in range(n):
        if mask[i]:
            if start is None:
                start = i
            continue
        if start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    return runs


def _passes_alt_gate(
    start: int,
    end: int,
    alt_segs: list[tuple[int, int]],
    alt_mask: np.ndarray,
) -> bool:
    """Endpoint inside any alt plateau OR plateau ``[start, end]`` fully
    contains an alt plateau."""
    if alt_mask[start] or alt_mask[min(end, len(alt_mask) - 1)]:
        return True
    return any(start <= a and b <= end for a, b in alt_segs)


def _finite_segment_mean(raw: np.ndarray, start: int, end_idx: int) -> float | None:
    """Return the mean finite raw value in a segment, or ``None`` if empty."""
    values = raw[start : end_idx + 1]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def detect_mach_plateaus_bilat(
    mach_raw: np.ndarray,
    alt_plateau_mask: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    n_passes: int = 2,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect Mach plateaus from a bilateral-smoothed Mach signal.

    Plateaus are kept only when their endpoint sits inside an altitude
    plateau (per ``alt_plateau_mask``) OR they fully contain at least one
    altitude plateau. ``var_mean`` is the mean of the **raw** Mach over
    the run (not the bilateral-smoothed value).
    """
    raw = np.asarray(mach_raw, dtype=np.float64)
    smooth = _smooth_bilateral(raw, sigma_s, sigma_r, n_passes)
    if smooth is None:
        return []
    flat = _slope_flat_mask(smooth, slope_tol)
    alt_mask = np.asarray(alt_plateau_mask, dtype=bool)
    alt_segs = _mask_to_segments(alt_mask)
    segments: list[dict[str, Any]] = []
    for i, j in _flat_runs(flat, min_len):
        seg = smooth[i : j + 1]
        if seg.max() - seg.min() > flat_tol:
            continue
        if not _passes_alt_gate(i, j, alt_segs, alt_mask):
            continue
        mean = _finite_segment_mean(raw, i, j)
        if mean is None:
            continue
        segments.append({"start_idx": i, "end_idx": j, "var_mean": mean})
    return segments


def detect_cas_plateaus_bilat(
    cas_raw: np.ndarray,
    mach_mask: np.ndarray,
    *,
    cutoff_s: float,
    sigma_s: float,
    sigma_r: float,
    n_passes: int = 2,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect CAS plateaus from a Butterworth+bilateral-smoothed CAS signal.

    Samples flagged in ``mach_mask`` are excluded so the detector does not
    fire inside Mach-plateau zones. ``var_mean`` is the mean of the raw
    CAS over the run.
    """
    raw = np.asarray(cas_raw, dtype=np.float64)
    work = raw.copy()
    nan_mask = np.isnan(work)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        work = interpolate_nans(work)
    work = butter_lowpass(work, cutoff_s)
    smooth = work
    for _ in range(max(0, int(n_passes))):
        smooth = bilateral_1d(smooth, sigma_s, sigma_r)
    dcas = np.abs(np.diff(smooth, prepend=smooth[0]))
    excl = np.asarray(mach_mask, dtype=bool)
    flat = (dcas < slope_tol) & (~excl)
    segments: list[dict[str, Any]] = []
    n = len(smooth)
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            seg = smooth[i : j + 1]
            if (seg.max() - seg.min()) <= flat_tol:
                mean = _finite_segment_mean(raw, i, j)
                if mean is not None:
                    segments.append({"start_idx": i, "end_idx": j, "var_mean": mean})
        i = j + 1
    return segments


def detect_vz_plateaus_from_bilat(
    vz_ftmin: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect vz plateaus from a bilateral-smoothed vz signal."""
    vz = np.asarray(vz_ftmin, dtype=np.float64).copy()
    nan_mask = np.isnan(vz)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        vz = interpolate_nans(vz)
    vz_bilat = bilateral_1d(bilateral_1d(vz, sigma_s, sigma_r), sigma_s, sigma_r)
    dvz = np.abs(np.diff(vz_bilat, prepend=vz_bilat[0]))
    excl = np.asarray(exclusion_mask, dtype=bool)
    flat = (dvz < slope_tol) & (~excl)
    return _bilateral_plateau_runs(vz_bilat, flat, flat_tol, min_len, abs_min=None)

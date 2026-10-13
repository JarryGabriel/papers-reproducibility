"""Ground-truth-free detector metrics for Paper 0-A.

Two quantities, both computed entirely on the detector's input signal (no
QAR / FCU ground truth):

- coverage:          fraction of samples the detector commits a plateau on.
- self_consistency:  among committed samples, fraction where the raw cleaned
                     signal lies within ``tol`` of the assigned plateau value.

The calibration objective is bi-objective (coverage x self_consistency) so a
single scalar cannot be Goodharted by shrinking the commit set. See README.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "coverage",
    "evaluate",
    "noise_level",
    "rf_star",
    "segments_to_prediction",
    "self_consistency",
]


def _plateau_runs(pred: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of finite (committed) samples in ``pred``."""
    fin = np.isfinite(pred)
    runs, i, n = [], 0, len(pred)
    while i < n:
        if not fin[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fin[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _sigma_local(seg: np.ndarray, *, min_len: int = 8,
                 slope_k: float = 3.0) -> float:
    """Robust local noise of a plateau: 1.4826 * MAD of the DETRENDED raw signal.

    Guards (anti-gaming, per the metric design):
    - too short to estimate noise (n < min_len)        -> 0 (tolerance = dial)
    - plateau sits on a ramp (trend dominates noise)   -> 0 (no noise credit)
    Returns the robust sigma in signal units (0 means: fall back to the dial-step).
    """
    y = seg[np.isfinite(seg)]
    n = y.size
    if n < min_len:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    # least-squares detrend so a tilted set-point is not counted as noise
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = 1.4826 * mad
    if sigma <= 0.0:
        return 0.0
    # flatness guard: total trend across the plateau must be small vs the noise
    if abs(a) * n > slope_k * sigma:
        return 0.0
    return sigma


def rf_star(raw: np.ndarray, pred: np.ndarray, dial: float, *,
            k: float = 2.0, angular: bool = False,
            shrink_sigma: float | None = None,
            shrink_n0: float = 20.0) -> dict[str, float]:
    """Noise-relative reconstruction fidelity.

    For each committed plateau j, the tolerance is ``tol_j = max(dial, k*sigma_j)``
    where ``sigma_j`` is the robust detrended-MAD local noise (0 -> tol = dial).
    A committed sample passes if ``|raw - pred| < tol_j``. Reduces EXACTLY to the
    fixed-tolerance rf when every sigma_j is below dial/k.

    ``shrink_sigma`` (optional): a phase/flight-level sigma to shrink each
    per-plateau sigma toward (empirical-Bayes-style, weight n0/(n0+n_j)); pass
    None to use the raw per-plateau sigma.

    Returns dict: rf_star, n_commit, n_hit, plus median plateau noise (kt-units)
    and the fraction of plateaus where the noise term won (tol > dial).
    """
    runs = _plateau_runs(pred)
    n_commit = n_hit = 0
    noise_won = 0
    sig_list: list[float] = []
    for a, b in runs:
        seg_raw = raw[a:b + 1]
        seg_pred = pred[a:b + 1]
        el = np.isfinite(seg_raw) & np.isfinite(seg_pred)
        if not el.any():
            continue
        sigma = _sigma_local(seg_raw)
        if shrink_sigma is not None and sigma > 0.0:
            n_j = int(el.sum())
            w = shrink_n0 / (shrink_n0 + n_j)
            sigma = float(np.exp((1 - w) * np.log(sigma) + w * np.log(shrink_sigma)))
        tol = max(dial, k * sigma)
        if tol > dial + 1e-12:
            noise_won += 1
        if sigma > 0:
            sig_list.append(sigma)
        if angular:
            err = np.abs((seg_pred[el] - seg_raw[el] + 180.0) % 360.0 - 180.0)
        else:
            err = np.abs(seg_pred[el] - seg_raw[el])
        n_commit += int(el.sum())
        n_hit += int((err < tol).sum())
    rf = n_hit / n_commit if n_commit else float("nan")
    return {
        "rf_star": rf,
        "n_commit": n_commit,
        "n_hit": n_hit,
        "noise_won_frac": noise_won / max(len(runs), 1),
        "median_plateau_sigma": float(np.median(sig_list)) if sig_list else 0.0,
    }


def noise_level(raw: np.ndarray, pred: np.ndarray) -> float:
    """Median per-plateau residual noise (detrended-MAD), the absolute noise
    level reported ALONGSIDE rf_star so heteroscedasticity stays visible."""
    runs = _plateau_runs(pred)
    sig = [s for a, b in runs if (s := _sigma_local(raw[a:b + 1])) > 0]
    return float(np.median(sig)) if sig else 0.0


def segments_to_prediction(segments: list[dict], n: int) -> np.ndarray:
    """Expand a list of {start_idx, end_idx, var_mean} into a per-sample
    prediction array (constant value over each run, NaN elsewhere)."""
    pred = np.full(n, np.nan)
    for s in segments:
        pred[s["start_idx"] : s["end_idx"] + 1] = s["var_mean"]
    return pred


def coverage(pred: np.ndarray) -> float:
    """Fraction of samples with a committed plateau value."""
    return float(np.isfinite(pred).mean())


def self_consistency(
    raw: np.ndarray, pred: np.ndarray, tol: float, *, angular: bool = False
) -> float:
    """Among committed samples, fraction where |raw - pred| < tol."""
    el = np.isfinite(pred) & np.isfinite(raw)
    if not el.any():
        return float("nan")
    if angular:
        d = (pred[el] - raw[el] + 180.0) % 360.0 - 180.0
        err = np.abs(d)
    else:
        err = np.abs(pred[el] - raw[el])
    return float((err < tol).mean())


def evaluate(
    raw: np.ndarray, pred: np.ndarray, tol: float, *, angular: bool = False
) -> dict[str, float]:
    """coverage + self-consistency + committed MAE, in one call."""
    el = np.isfinite(pred) & np.isfinite(raw)
    n_commit = int(el.sum())
    if n_commit:
        if angular:
            d = (pred[el] - raw[el] + 180.0) % 360.0 - 180.0
            err = np.abs(d)
        else:
            err = np.abs(pred[el] - raw[el])
        mae = float(err.mean())
        sc = float((err < tol).mean())
    else:
        mae = float("nan")
        sc = float("nan")
    return {
        "coverage": coverage(pred),
        "self_consistency": sc,
        "mae": mae,
        "n_commit": n_commit,
    }

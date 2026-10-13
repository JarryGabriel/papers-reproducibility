"""Densify the slope_tol grid per channel to test whether front "gaps" are real
or just under-sampling artefacts (confirmed artefact for CAS). Replicates each
channel's detector + gate exactly (mirrors sweep_<ch>.py) and sweeps a FINE
slope_tol grid at the low-sigma_r branch that forms the front.

Prints cov / rf(@1 dial-step) / MAE / plat-per-flight for each slope_tol so we
can read a continuous coverage<->MAE curve and pick by tolerated MAE.

Does NOT touch analysis/ or analysis2/.

Usage::
    cd <repo>/papers/paper_opensky26/analysis3_metric/src
    uv run python densify_front.py <channel> [--max-flights N] [--sigma-r S]
    channel in: vz alt gamma cas mach
"""

from __future__ import annotations

import argparse

from opensky26_plateau import detectors as D
from opensky26_plateau import metrics as M
import numpy as np
from opensky26_plateau import paths as P
import polars as pl

FT_TO_M = 0.3048
FT_MIN_TO_MS = 0.3048 / 60.0
KT_TO_MS = 0.514444

# per-channel: dial-step (rf tolerance), fixed gates, default front sigma_r, and
# a FINE slope_tol grid spanning the production grid's coarse steps.
CFG = {
    "vz": dict(dial=100.0, flat=100.0, min_len=8, sigma_r=100.0, min_abs=100.0,
               slope=[10, 20, 30, 40, 50, 70, 90, 120, 160, 220, 300]),
    "alt": dict(dial=100.0 * FT_TO_M, flat=100.0 * FT_TO_M, min_len=8,
                sigma_r=20.0, min_abs=0.0,
                slope=[1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30]),
    "gamma": dict(dial=0.1 * np.pi / 180.0, flat=2e-3, min_len=10, sigma_r=0.002,
                  abs_min=0.005,
                  slope=[5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4, 6e-4, 8e-4,
                         1.2e-3, 2e-3, 3e-3]),
    "cas": dict(dial=1.0, flat=20.0, min_len=5, sigma_r=3.0, min_abs=0.0,
                slope=[0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.18,
                       0.22, 0.25]),
    "mach": dict(dial=0.01, flat=0.05, min_len=15, sigma_r=0.01, min_abs=0.0,
                 slope=[1e-4, 2e-4, 3e-4, 4e-4, 6.5e-4, 1e-3, 1.5e-3, 3e-3,
                        6e-3, 1.2e-2, 3e-2]),
}
ALT_SIGMA_R, ALT_SLOPE_TOL = 20.0, 2.0
ALT_FLAT_TOL = 100.0 * FT_TO_M
MACH = {"sigma_r": 0.08, "slope_tol": 6.5e-4, "flat_tol": 0.05, "min_len": 15}


def _signal(ch, g):
    if ch == "alt":
        return g["raw_alt_ft"].cast(pl.Float64).to_numpy() * FT_TO_M
    if ch == "vz":
        return g["raw_vz_ftmin"].cast(pl.Float64).to_numpy()
    if ch == "cas":
        return g["bds_ias_kt_clean"].cast(pl.Float64).to_numpy()
    if ch == "mach":
        return g["bds_mach_clean"].cast(pl.Float64).to_numpy()
    vz = g["raw_vz_ftmin"].cast(pl.Float64).to_numpy() * FT_MIN_TO_MS
    tas = g["fdm_tas_from_cas_kt"].cast(pl.Float64).to_numpy() * KT_TO_MS
    return np.arcsin(np.clip(vz / np.clip(tas, 1e-6, None), -1.0, 1.0))


def _alt_mask(g):
    alt = g["raw_alt_ft"].cast(pl.Float64).to_numpy() * FT_TO_M
    segs = D.detect_vz_plateaus_from_bilat(
        alt, np.zeros(alt.size, bool), sigma_s=6, sigma_r=ALT_SIGMA_R,
        slope_tol=ALT_SLOPE_TOL, flat_tol=ALT_FLAT_TOL, min_len=8)
    m = np.zeros(alt.size, bool)
    for s in segs:
        m[s["start_idx"]: s["end_idx"] + 1] = True
    return m


def _mach_mask(g):
    mach = g["bds_mach_clean"].cast(pl.Float64).to_numpy()
    am = _alt_mask(g)
    segs = D.detect_mach_plateaus_bilat(
        mach, am, sigma_s=8, sigma_r=MACH["sigma_r"], n_passes=2,
        slope_tol=MACH["slope_tol"], flat_tol=MACH["flat_tol"], min_len=MACH["min_len"])
    m = np.zeros(mach.size, bool)
    for s in segs:
        if s["var_mean"] >= 0.5:
            m[s["start_idx"]: s["end_idx"] + 1] = True
    return m


def _detect(ch, sig, g, sr, sl, c):
    if ch == "gamma":
        return D.detect_gamma_plateaus_from_bilat(
            sig, np.zeros(sig.size, bool), sigma_s=6, sigma_r=sr, slope_tol=sl,
            flat_tol=c["flat"], abs_min=c["abs_min"], min_len=c["min_len"])
    excl = _mach_mask(g) if ch == "cas" else np.zeros(sig.size, bool)
    segs = D.detect_vz_plateaus_from_bilat(
        sig, excl, sigma_s=8 if ch in ("cas", "mach") else 6, sigma_r=sr,
        slope_tol=sl, flat_tol=c["flat"], min_len=c["min_len"])
    if ch == "vz":
        segs = [s for s in segs if abs(s["var_mean"]) >= c["min_abs"]]
    if ch == "mach":
        am = _alt_mask(g)
        alt_segs = D._mask_to_segments(am)
        segs = [s for s in segs if s["var_mean"] >= 0.5
                and D._passes_alt_gate(s["start_idx"], s["end_idx"], alt_segs, am)]
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", choices=list(CFG))
    ap.add_argument("--max-flights", type=int, default=None)
    ap.add_argument("--sigma-r", type=float, default=None)
    args = ap.parse_args()
    ch = args.channel
    c = CFG[ch]
    sr = args.sigma_r if args.sigma_r is not None else c["sigma_r"]
    dial = c["dial"]
    min_abs = c.get("min_abs", 0.0)

    df = pl.read_parquet(P.DATA)
    flights = df["meta_flight_id"].unique(maintain_order=True).to_list()
    if args.max_flights:
        flights = flights[: args.max_flights]
    gs = [df.filter(pl.col("meta_flight_id") == f).sort("raw_timestamp")
          for f in flights]
    sigs = [_signal(ch, g) for g in gs]
    print(f"{ch}: {len(gs)} flights, sigma_r={sr:g}, dial={dial:g}, "
          f"rf@1dial\n")
    print(f"{'slope_tol':>10} {'cov%':>6} {'rf%':>6} {'MAE':>9} {'plat/fl':>8}")
    print("-" * 46)
    for sl in c["slope"]:
        tot = commit = hit = nplat = 0
        abs_err = 0.0
        for sig, g in zip(sigs, gs, strict=True):
            segs = _detect(ch, sig, g, sr, sl, c)
            pred = M.segments_to_prediction(segs, sig.size)
            el = np.isfinite(pred) & np.isfinite(sig)
            if min_abs > 0:
                el = el & (np.abs(pred) > min_abs)
            tot += sig.size
            commit += int(el.sum())
            err = np.abs(pred[el] - sig[el])
            hit += int((err < dial).sum())
            abs_err += float(err.sum())
            nplat += len(segs)
        cov = 100 * commit / max(tot, 1)
        rf = 100 * hit / max(commit, 1)
        mae = abs_err / max(commit, 1)
        ppf = nplat / max(len(gs), 1)
        print(f"{sl:10g} {cov:6.1f} {rf:6.1f} {mae:9.4g} {ppf:8.2f}")


if __name__ == "__main__":
    main()

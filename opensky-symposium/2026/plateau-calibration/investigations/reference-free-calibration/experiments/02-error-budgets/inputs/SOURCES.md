# Sources, and the rules fixed before the measurement

## The cohort

`data/paper_opensky26/flights.delta`, restricted to **2025-11-01** — the
calibration date. 1 108 921 rows, the same date the sweep will run on.

The other three dates (February, May, August) exist in the Delta and are
**deliberately not read here**. They are the only data no measurement has
touched, and experiment 05 measures transfer against them. Spending them on a
budget derivation would leave nothing untouched to test transfer with.

## What is measured on what

| channel | column | nature |
|---|---|---|
| altitude | `raw_alt_ft` | broadcast, quantised |
| vertical speed | `raw_vz_ftmin` | broadcast, quantised |
| CAS/IAS | `fdm_tas_from_cas_kt` | **reconstructed** — see below |
| Mach | `bds_mach_clean` | **reconstructed** — see below |
| flight-path angle | derived | `arcsin(vz/TAS)`, computed by `_gamma_raw_from_vz_tas` |

**The two speed channels are 70.47 % reconstructed.** `bds_ias_kt` and
`bds_mach` are present on 29.53 % of rows; `_fill_with_era` in
`node_fdm_data.preprocessing.clean_speeds` fills the rest from the ERA5-derived
`era_cas_kt` / `era_mach`. What the detector sees on those samples is a
reanalysis product, not a broadcast measurement.

`fdm_tas_from_cas_kt` adds a second ERA5 dependence: `_derive_tas_from_cas`
converts the cleaned CAS to TAS using `era_temp_K`. So even the 29.53 % that
carry a register are converted with a reanalysis temperature.

## The plausibility filter on the paired samples

Fixed here, before the measurement, because the raw comparison has a p99 of
119.5 kt — physically impossible, and a budget fitted to that would be a budget
fitted to decoder artefacts.

A paired sample enters the reconstruction measurement when **all** of:

- `bds_ias_kt` and `era_cas_kt` are both non-null and finite;
- `bds_mach` and `era_mach` are both non-null and finite;
- `raw_alt_ft > 10000` — airborne and above the terminal area, where ERA5's
  boundary layer and the aircraft's own manoeuvring both stop dominating;
- `100 < bds_ias_kt < 400` — a plausible airspeed for this fleet.

Nothing else is filtered. In particular the tail is NOT trimmed by a quantile
rule: the quantiles are reported and the robust sigma is what the budget reads,
which is a decision about the estimator rather than a decision about which
samples exist.

## Which number each budget reads, fixed before the results

Declared here so the choice cannot be made after seeing which gives a
convenient answer.

| channel | budget reads | why |
|---|---|---|
| altitude | **quantisation step / 2** | a plateau spread over one step of a quantised signal has a mean absolute deviation of about half a step. The step is measured, the halving is the declared convention. |
| vertical speed | **quantisation step / 2** | same rule, same reason. |
| CAS/IAS | **robust sigma of the reconstruction error** (1.4826 × MAD) | 70 % of the channel is a reanalysis product; the detector cannot resolve a plateau finer than the error of the signal it is given. A median would be exceeded by half the samples by construction; a sigma is a dispersion. |
| Mach | **robust sigma of the reconstruction error** | same. |
| flight-path angle | **median of the propagated budget** | gamma is not measured, so its budget is not a free choice: it follows from vz and TAS through the propagation of uncertainty, evaluated sample by sample. The median names the regime the detector works in; the distribution is reported beside it. |

**The halving convention, stated as a convention.** Splitting the quantisation
step in two is not derived — it is the dispersion of a plateau whose values
occupy one step, under the assumption that they occupy it roughly uniformly.
A plateau sitting exactly on one step errs by nothing; one alternating between
two steps errs by half a step. Half is the middle of that range and it is a
choice, declared rather than justified as a measurement.

## What is NOT used, and why

**No detected plateau.** Not one quantity here is measured on a segment the
detector committed to. Measuring the tolerable error on the detector's own
output and then constraining the detector with it would make the constraint
pass by construction — the circularity this experiment exists to avoid.

**No broadcast set-point.** The MCP selected-altitude register would give a
third route to the altitude budget, and it is deliberately left alone:
experiment 06 of round 2 used it as an independent second observer, and a
budget derived from it would spend that independence.

# Calibrating plateau detectors in ADS-B data without ground truth

**Gabriel Jarry, Ramon Dalmau and Philippe Very**  
OpenSky Symposium, 2026

This repository contains the code and supporting results for calibrating
plateau detectors in ADS-B trajectories without a complete reference for the
commanded values. It covers altitude, flight-path angle, vertical speed, Mach
and calibrated airspeed.

## Explore the results

- [Detector settings](investigations/reference-free-calibration/experiments/03-operator-sweep/outputs/retained.csv) and [coverage–error trade-offs](investigations/reference-free-calibration/experiments/03-operator-sweep/figures/fig_front_zoom.png).
- [Transfer across aircraft and dates](investigations/calibration-transfer/experiments/05-day-transfer/figures/fig_transfer.png).
- [Comparison with broadcast selected altitude](investigations/selected-altitude-agreement/experiments/07-setpoint-agreement/figures/fig_difference.png).

The nine experiments are grouped under `investigations/` by research question.
Each experiment contains its inputs, scripts, result tables and figures.
Shared detector code is in `src/opensky26_plateau/`.

## Run the examples

With Python 3.12 or newer and [uv](https://docs.astral.sh/uv/), start from the
repository root:

```bash
cd opensky-symposium/2026/plateau-calibration
uv sync --locked
```

Regenerate figures from the included result tables:

```bash
uv run --locked python investigations/calibration-transfer/experiments/04-aircraft-transfer/scripts/04_figures.py
uv run --locked python investigations/calibration-transfer/experiments/08-error-distribution/scripts/02_figures.py
```

These commands replace the images in the corresponding `figures/` folders.
They use the supplied measurements; they do not repeat data acquisition.

## Data

The study uses a fixed plan of 200 aircraft over four dates in 2025. The aircraft
list, flight plan, result tables and figures are included. Raw OpenSky observations
and ERA5 weather caches are not: rebuilding those requires provider access.

See [local data and configuration](PUBLIC_PATHS.md) to use your own data directory.
Acquisition dependencies can be installed with `uv sync --locked --extra replay`.

## License

[EUPL-1.2](LICENSE). Third-party data remain subject to their providers' terms.

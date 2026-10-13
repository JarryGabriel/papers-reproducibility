# Research reproduction packages

I share the code, results and figures behind my papers here, so you can explore
and reproduce the work.

## Available projects

| Venue / year | Project | What you can explore |
|---|---|---|
| OpenSky Symposium / 2026 | [Calibrating plateau detectors in ADS-B data without ground truth](opensky-symposium/2026/plateau-calibration/README.md) | Detector calibration, transfer across aircraft and dates, and comparison with broadcast selected altitude. |

## Getting started

Open a project's README first. You can browse the included figures and result tables
without installing software. To run code, enter that project's directory and follow
its installation instructions; there is no shared Python environment at the repository root.

Projects are arranged as `<venue>/<year>/<project>/`. Within each project,
experiments are grouped by scientific question. Their `inputs/`, `scripts/`,
`outputs/` and `figures/` folders keep the data used, analysis code and reported
results together.

Recreating figures from included tables is different from rebuilding results from
the original observations. Some analyses require external data, provider access or
substantial computation. Each project explains what is included and what must be
obtained separately.

## Reuse and questions

Check the `LICENSE` in each project before reusing its contents. Access to third-party
datasets remains subject to their providers' terms.

For questions about the research, contact me at gabriel.jarry@eurocontrol.int.
To report a problem with the code or reproduction instructions, open an issue.

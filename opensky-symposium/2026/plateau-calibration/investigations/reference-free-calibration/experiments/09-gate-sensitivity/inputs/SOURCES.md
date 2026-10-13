# Inputs — 09-gate-sensitivity

Four artefacts, all inherited from `03-operator-sweep` and copied here rather
than read across folders or recomputed. Each is pinned by digest in
`manifest.yaml`; re-pin with `experiment_repin` if the upstream folder moves,
never by editing a digest.

| file | from | why this one |
|---|---|---|
| `sweep.csv` | `03-operator-sweep/outputs/sweep.csv` | the scored frontier, one row per candidate. This experiment reads the head of each channel's admissible set from it — the candidates that could plausibly take the argmax. |
| `retained.csv` | `03-operator-sweep/outputs/retained.csv` | the incumbent under test on each channel, and the source of the cascade exclusion each channel must be scored behind. |
| `grids.json` | `03-operator-sweep/inputs/grids.json` | the candidate grid. `sweep.csv` writes knobs with `%g`, six significant digits, so `slope_tol=3.6342411857` comes back as `3.63424`; every candidate is resolved onto this grid before being re-detected, or the detector run is not the detector that was scored. |
| `budgets_full_step.json` | `03-operator-sweep/inputs/budgets_full_step.json` | the per-channel budgets measured by `02-error-budgets`. Admissibility here is decided exactly as the sweep decided it. |

## The cohort

The scripts also read the cohort at `data/paper_opensky26/flights.delta` of the
sibling `node-fdm-v2` workspace, which is not versioned and not copied here —
the same convention `03-operator-sweep` follows, for the same reason: it is a
regenerable derived dataset of several gigabytes, and its provenance is stated
in that folder's `inputs/SOURCES.md`. Rebuilding it needs an OpenSky Trino
account and ERA5 access through `fastmeteo`, and runs through the `fdm`
pipeline of the sibling workspace.

Re-detection is unavoidable here. `sweep.csv` carries one aggregate row per
candidate, computed AFTER the gate has already dropped the runs it drops, so
the gate's effect cannot be recovered from it — which is precisely why this
experiment exists as a folder rather than as a re-read of the frontier.

## Retrieval

```bash
cp ../03-operator-sweep/outputs/sweep.csv       inputs/
cp ../03-operator-sweep/outputs/retained.csv    inputs/
cp ../03-operator-sweep/inputs/grids.json       inputs/
cp ../03-operator-sweep/inputs/budgets_full_step.json inputs/
```

Copied 2026-09-09, from the state of `03-operator-sweep` locked at `2fbf079`.

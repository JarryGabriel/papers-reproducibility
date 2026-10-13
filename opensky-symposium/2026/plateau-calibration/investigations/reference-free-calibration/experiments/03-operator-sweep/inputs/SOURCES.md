# Sources, and the rules fixed before the grid is walked

## The cohort

`data/paper_opensky26/flights.delta`, restricted to **2025-11-01** — the
calibration date, 1 507 flights over ten typecodes.

The other three dates (February, May, August) are **not read here**. They are
the only data no selection has touched, and the transfer experiments measure
against them. A sweep that consumed them would leave nothing to test transport
with.

## The upstream input

`inputs/budgets_full_step.json`, from experiment 02, pinned by digest
`c389faa2e870b8fdce80cf604af58998336f2e28c3462e11475ee7c0a1434d8e`.

| channel | E | unit | basis |
|---|---|---|---|
| altitude | 7.62 | m (25 ft) | quantisation step |
| vertical speed | 64 | ft/min | quantisation step |
| flight-path angle | 1.4288e-3 | rad | propagated from vz and TAS |
| CAS/IAS | 2.5727 | kt | robust sigma of the ERA5 reconstruction error |
| Mach | 6.6725e-3 | — | robust sigma of the ERA5 reconstruction error |

**E is not swept.** It is an input, derived upstream from the signal, and moving
it here would make the budget a function of what the grid happens to contain.

## The selection rule, fixed before any candidate is scored

> Among the settings whose reconstruction error respects the budget on **every**
> committed segment of **every** flight, retain the one committing the largest
> share of the cohort's samples.

Per segment, the error is

    mean(|sample - segment mean|)

over the segment's finite samples, measured on the **raw** signal — the one the
segment claims to describe — with the segment's announced value recomputed over
those same samples.

**Ties.** Coverage within 0.05 percentage points of the maximum is a tie, and it
is broken by the lower worst-segment error, then by the fewer segments per
flight. An argmax on hundredths of a point is choosing on noise, and the
tie-break is declared here so the choice cannot be made after seeing which
candidate it favours.

**A channel with no admissible candidate is reported empty**, together with the
smallest worst-segment error the grid reaches and the factor by which it misses.
The budget is not widened to produce an answer.

## The grid

`inputs/grids.json`, the same ranges the earlier round swept, carried over
unchanged so the comparison between rules is not confounded by a change of
domain. Four families:

| family | knobs |
|---|---|
| bilateral | `sigma_s`, `sigma_r`, `slope_tol` |
| Savitzky-Golay | `window`, `slope_tol` |
| variance threshold | `window`, `std_tol` |
| PELT | `penalty` |

crossed with the low-pass cutoff `{0, 120, 180, 300}` seconds. A candidate
detects on the filtered view and is scored on the raw signal — see the section
below, and note that this **structurally disfavours a non-zero cutoff**: the more
a candidate smooths, the longer the segments it forms, and the larger the error
those segments carry once measured against what was actually transmitted.

That is a property of the criterion, not a finding about the data. If the sweep
retains `butter = 0` on every channel, the honest reading is that the rule
penalises smoothing, not that the data prefer an unfiltered view. The count of
admissible candidates per cutoff is reported so the effect is visible rather
than inferred.

**What is not swept, and why.**

- **`min_len`** — 8 samples, a property of the 4 s sampling rate rather than a
  knob: shorter than half a minute is not a held value at this resolution.
- **The cascade order** — altitude before gamma before vz, Mach before CAS. The
  paper argues for it on physical grounds; making it a swept dimension would
  make the argument decorative.
- **`min_abs` and `abs_min`** — removed outright rather than tuned. The cascade
  already excludes what they excluded, and filtering twice distorts the
  denominator rather than protecting it.

## What the scoring reads, and one thing it deliberately does not

The error is measured on the **raw** signal even when a candidate detects on a
filtered one. Detection may use a smoothed view; the committed value is judged
against what was actually transmitted. Scoring on the filtered signal would let
a candidate buy conformity with smoothing — the smoother the view, the flatter
everything looks — without improving what the segment claims about the
trajectory.

**Trim before filter**, and the ordering is load-bearing: the low-pass
interpolates NaNs, so filtering first would reconstruct the frozen tail the trim
exists to remove.

## What is NOT used

**No broadcast set-point.** The MCP selected-altitude register exists on 30 % of
samples and is deliberately left out of the selection. The moment a set-point
informs which tolerance is retained, the agreement measured against it stops
being independent evidence.

**No flight plan, no manufacturer table, no domain rule about what a crew would
plausibly select.** The paper claims a calibration needing no reference, and a
filter justified by "no aircraft holds Mach below 0.5" is exactly such a
reference used while being disclaimed.

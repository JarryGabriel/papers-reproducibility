# Sources, and what one row of the output represents

## The four dates

`data/paper_opensky26/flights.delta`, the same four dates experiments 05 and 06
used:

| date | flights | airframes | status |
|---|---|---|---|
| 2025-02-17 | 1 168 | 161 | held out of the sweep |
| 2025-05-21 | 1 354 | 176 | held out of the sweep |
| 2025-08-29 | 1 434 | 179 | held out of the sweep |
| 2025-11-01 | 1 507 | 200 | calibration |

**Nothing is held out here and nothing needs to be.** This experiment fits
nothing and selects nothing: it records a quantity the earlier runs computed and
discarded. Reading all four dates is the point, since the comparison between the
calibration date and the three others is measurement 4.

## The frozen points

Exactly what experiment 03 retained, resolved back to the grid values the stored
`%g` strings can only have come from. All five are `bilateral` with `butter=0`:

| channel | knobs | budget |
|---|---|---|
| altitude | σr=5, σs=20, slope=3.6342 | 7.62 m |
| flight-path angle | σr=0.002, σs=4, slope=1.4423e-4 | 1.4288e-3 rad |
| vertical speed | σr=50, σs=4, slope=21.938 | 64 ft/min |
| Mach | σr=0.0025, σs=8, slope=3.1623e-4 | 6.6725e-3 |
| CAS/IAS | σr=1.5, σs=8, slope=0.353439 | 2.5727 kt |

**Not re-selected, not re-tuned, not swept.** The cascade runs exactly as the
sweep ran it — sequential, each channel seeing only the residual its priors
leave, with the priors at their frozen points. The detector is bit-for-bit the
one experiments 04 and 05 carried.

## What one row is

**One committed segment.** Not one channel, not one date, not one flight: every
segment the frozen detector commits to, on every flight, on every date.

Each row carries the segment's own mean absolute deviation from the value it
announces — the same quantity `worst_segment_error` takes a maximum of — plus
what is needed to weight it and to locate it:

| field | why it is there |
|---|---|
| `error` | the per-segment MAE, the quantity under examination |
| `error_over_budget` | the same as a share of the channel's budget, so five units compare |
| `n_samples` | a breach on a segment holding 0.01 % of the coverage is not a breach holding 5 %; it is also the ONLY correct weight, since a span can hold gaps |
| `span_s`, `covered_s` | the segment's extent and the time it actually covers — they differ on about 4 % of segments, which hold interior gaps |
| `drift` | the SECOND statistic 03 and 05 constrained, and of which they also kept only the maximum |
| `mean_value` | to locate a breaching segment in a flight |
| `flight_id`, `icao24` | the clustering unit, so a share can be reported per airframe as well as pooled |

Everything the analysis reports is derived from this file. A reader who
disagrees with the statistics chosen can compute their own from the same rows.

## Why this experiment exists

The paper constrains and reports a **maximum** over segments, and never recorded
what the maximum is a maximum of. Two consequences follow arithmetically rather
than statistically, and an independent review of the evidence found both:

**A max over a partition is bounded by the max over the whole.** Experiment 04
reports "no channel breaches its budget on any fold" — over five folds of a
cohort whose maximum already respects the budget by construction. On four
channels of five exactly one fold attains the cohort maximum and the other four
fall below it. The result was arithmetically forced.

**A max fitted to a sample is exceeded on a fresh one.** Four channels were
retained at 95.7 to 97.7 % of their budget on the date the budget was checked
on. Carried to three unseen dates, they breach on 12 channel-dates of 15 — which
is what a maximum selected at its own sample's edge does, with no seasonal
effect required.

Neither observation says the detector transports badly. Neither says it
transports well. They say **the statistic cannot tell**, and that is what this
experiment sets out to replace.

## What is NOT done

**No re-selection and no new rule.** The frozen points are carried unchanged.
Whether a rule constraining a percentile would retain different points is a
different question and costs a full re-sweep: `sweep.csv` records only the
maximum for each of its 27 414 candidates.

**No verdict on the experiments upstream.** 03, 04 and 05 measured what they
declared and their locks stand. This experiment adds a measurement they did not
make; what follows from it is read afterwards.

**No threshold.** How many segments in breach would be too many is not declared,
because declaring it would invent the convention this paper exists to remove —
and would do it on the very statistic under suspicion.

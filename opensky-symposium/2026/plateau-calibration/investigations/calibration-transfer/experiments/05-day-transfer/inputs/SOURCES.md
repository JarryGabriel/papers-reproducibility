# Sources, and the rules fixed before the three untouched dates are read

## The four dates

`data/paper_opensky26/flights.delta`, all four dates the cohort carries:

| date | flights | airframes | rows | status |
|---|---|---|---|---|
| 2025-02-17 | 1 168 | 161 | 858 670 | **untouched** |
| 2025-05-21 | 1 354 | 176 | 993 980 | **untouched** |
| 2025-08-29 | 1 434 | 179 | 1 093 076 | **untouched** |
| 2025-11-01 | 1 507 | 200 | 1 108 921 | calibration, in-sample reference |

**Three of these have been read by no SELECTION in this paper**, and by no
measurement that enters a budget or an operating point. Not the budget
derivation, not the sweep, not the aircraft transfer — each of those declared
2025-11-01 and filtered to it.

One earlier measurement did read all four: experiment 01 counted the
selected-altitude registers per date, and its `outputs/coverage_by_date.csv`
carries rows for every one of them. That is a property of the Mode-S stream
which enters no budget, no grid and no retained point, so it cannot steer what
is carried here. But it means the reserve was never untouched in the literal
sense, and saying so plainly costs nothing — an overstated reserve is the kind
of claim a reader checks. Spending the dates HERE is still a one-way move for
the purpose that matters, and it is why this file exists before they are opened.

**The calibration date is included as the in-sample reference**, not as a fourth
observation. Its numbers are what the sweep already reported and it is there so
a reader can see the out-of-sample dates against the day the point was chosen
on, rather than against nothing.

**Spread across the year on purpose.** February, May, August and November carry
different synoptic weather, different traffic and different seasons. The
archived round's external cohort was three *consecutive* September dates, which
three reviewers flagged independently: consecutive days share their weather, so
day and weather could not be separated. These four cannot be accused of that.

## The upstream inputs

| file | from | digest |
|---|---|---|
| `retained.csv` | 03-operator-sweep | `2654c9b6…` |
| `grids.json` | 03-operator-sweep | `55a80354…` |
| `budgets_full_step.json` | 02-error-budgets | `c389faa2…` |

The five frozen points, in the units the detector scores in:

| channel | family | cutoff | knobs | budget |
|---|---|---|---|---|
| altitude | bilateral | 0 | σr=5, σs=20, slope=3.6342 | 7.62 m |
| flight-path angle | bilateral | 0 | σr=0.002, σs=4, slope=1.4423e-4 | 1.4288e-3 rad |
| vertical speed | bilateral | 0 | σr=50, σs=4, slope=21.938 | 64 ft/min |
| Mach | bilateral | 0 | σr=0.0025, σs=8, slope=3.1623e-4 | 6.6725e-3 |
| CAS/IAS | bilateral | 0 | σr=1.5, σs=8, slope=0.353439 | 2.5727 kt |

**Neither the points nor the budgets are re-derived here.** They arrive pinned by
digest. A budget adjusted after seeing a breach would make the breach a function
of the adjustment, which is the one thing this experiment must not do.

## What is measured on each date

For each channel and each date, over that date's flights:

**At the frozen point** — committed coverage, the worst committed segment's mean
absolute deviation from its own announced value, the segment count and the
median duration. The same quantities the sweep selected on, so the four dates
are comparable to each other and to the calibration.

**At the date's own optimum** — the coverage the selection rule would have
reached had it been applied to that date alone, and which setting it would have
retained. The difference is the re-calibration deficit, reported with its sign:
positive where re-fitting would have helped, negative where the frozen point
already beats what that day's own rule would have chosen.

The cascade runs exactly as the sweep ran it: sequential, each channel seeing
only the residual its priors leave, with the priors at their frozen points. The
re-calibration re-selects one channel at a time against the frozen upstream, not
the whole cascade jointly — the same greedy structure, so the two coverages are
comparable.

## The ratio this experiment exists to test the durability of

Per channel:

```
dispersion / margin  =  (spread of the worst-segment error across dates)
                        ---------------------------------------------
                        (budget − the frozen point's error, as a share)
```

The archived round found this ordered its five channels' failures without
exception — 2.32 on vertical speed, 1.30 on CAS/IAS, 0.61 on Mach, 0.19 on the
flight-path angle, with altitude having negative margin and breaching outright.

It is reported here **whether or not it does so again**. A regularity that fails
to reproduce on data it did not come from is a finding about the regularity.

## What is NOT done

**No re-derivation of E**, as above.

**No leave-one-day-out.** Calibrating on three dates and measuring on the fourth,
four times over, would be statistically richer and would cost four full sweeps —
but it would also draw the three untouched dates into the calibration, leaving
nothing unread for anything after this. The frozen point carried outward is what
a user of this paper would actually do.

**No claim that four dates characterise a year.** They say whether a setting
survives a change of season. They do not say what the distribution of days
looks like.

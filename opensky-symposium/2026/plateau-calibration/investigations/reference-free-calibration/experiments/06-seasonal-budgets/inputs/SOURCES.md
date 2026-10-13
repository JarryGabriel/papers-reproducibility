# Sources, and what each measurement targets

## The four dates

`data/paper_opensky26/flights.delta`, all four dates the cohort carries:

| date | flights | airframes | rows |
|---|---|---|---|
| 2025-02-17 | 1 168 | 161 | 858 670 |
| 2025-05-21 | 1 354 | 176 | 993 980 |
| 2025-08-29 | 1 434 | 179 | 1 093 076 |
| 2025-11-01 | 1 507 | 200 | 1 108 921 |

**None of these is held out here, and that is not a loss.** Experiment 05 already
read the three unseen dates, so the reserve is spent. This experiment measures a
property of the signal rather than fitting anything to it: there is no selection
to protect from in-sample optimism, and reading all four is the point.

**No date is privileged.** November is the calibration date of every experiment
upstream, and it is measured here exactly like the other three — the question is
whether it is representative, which cannot be asked while treating it as the
reference.

## What each measurement targets

**Quantisation step**, per channel and per date. The largest value most gaps
between consecutive distinct readings are a multiple of — not the modal gap,
which on altitude is 75 ft rather than 25, and not a percentile, which the 4 s
resampling contaminates with interpolated values. Altitude and vertical speed
are broadcast on a fixed grid, so this is a property of the transponder.

**Reconstruction error**, on the two speed channels. The robust sigma of the
difference between the broadcast speed and the one rebuilt from ground speed and
the ERA5 wind field. About 70 % of both speed channels is that reconstruction
rather than a broadcast measurement, so this is where a seasonal effect would
show if there is one: the reanalysis field is not the same in February and
August.

**Propagation**, for the flight-path angle. It carries no budget of its own —
its budget is propagated from the vertical-speed step and the true airspeed, so
it is derived per date from that date's own inputs rather than assumed.

**Wind dependence** is measured and reported because experiment 02 proposed the
wind as the driver of the reconstruction error and measured r = −0.036 on
November, refuting it. Whether that holds on a February with a different jet
stream is worth one column, and it is reported whichever way it falls.

## The method is experiment 02's, unchanged

`scripts/01_measure_seasonal.py` is a copy of `02-error-budgets/scripts/01_measure.py`
with one change: it loops over the four dates and stamps a `date` column instead
of filtering to one. The estimators, the filters and the conventions are
identical, deliberately — a difference between this experiment's numbers and
experiment 02's must come from the data, not from a re-implementation.

The budget derivation follows the same conventions experiment 02 locked: a
quantised channel takes its full step, a reconstructed channel takes its robust
sigma, and the flight-path angle takes the propagated value.

## What is NOT done

**No budget is retrofitted.** The budgets consumed by experiments 03, 04 and 05
arrive there by digest from experiment 02 and are not touched by anything
measured here. If this experiment finds the budgets should have been wider, that
is reported as a finding about the budgets, not applied backwards to the results
that breached them.

**No pooled budget is derived.** Averaging the four dates into one better
estimate would answer a different question, and would lose exactly the
per-date spread this experiment exists to report.

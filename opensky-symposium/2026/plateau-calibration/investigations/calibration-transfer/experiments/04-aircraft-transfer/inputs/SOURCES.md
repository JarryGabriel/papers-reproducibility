# Sources, and the rules fixed before the measurement

## The cohort

`data/paper_opensky26/flights.delta`, restricted to **2025-11-01** — the same
date experiment 03 swept, 1 507 flights over 200 airframes and ten typecodes.

**One date on purpose.** Every fold draws from the same day, the same
meteorology and the same reanalysis, so the only thing that differs between them
is which aircraft they contain. A difference measured here is a difference
between aircraft populations, and cannot be a difference between days. The day
transfer is a separate experiment and consumes separate dates.

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

**Neither the points nor the budgets are re-derived here.** They arrive pinned
by digest, and this experiment measures what carrying them costs.

## How the folds are built

Five folds, assigned once and committed to `inputs/folds.csv` rather than
recomputed at each run. A selection regenerated per run can drift between
executions, and every number reported here rests on the split being identical
every time.

Two properties, and both are load-bearing:

**Grouped by airframe.** Every `raw_icao24` lands in exactly one fold, so the
subsets are disjoint, but all contributed to global calibration. Repeated timestamps of one aircraft are not
independent units — a split over samples would put the same airframe on both
sides and measure memorisation rather than transfer.

**Stratified by typecode.** Each fold gets its share of every type, so a fold
cannot come out badly merely for having drawn the turboprops. With 200 airframes
over 10 types, that is 4 airframes per type per fold.

The assignment is a deterministic function of the airframe address and a
versioned salt — no random state to share, no seed to trust:

```
rank = position of SHA256(salt + ':' + raw_icao24) within its typecode
fold = rank mod 5
```

with `salt = "04-aircraft-transfer/v1"`. **Changing the split means changing the
salt and saying so**, never reshuffling in place.

## What is measured on each fold

For each channel and each fold, on the fold's airframes only:

**At the frozen point** — committed coverage and the worst committed segment's
mean absolute deviation from its own announced value, the same two quantities
experiment 03 selected on. Reported per fold, and as the spread across folds.

**At the fold's own optimum** — the coverage the selection rule would have
reached had it been applied to that fold alone, and which setting it would have
retained. The difference between the two coverages is the transport cost.

The cascade runs exactly as in the sweep: sequential, each channel seeing only
the residual its priors leave, with the priors at their frozen points. A fold's
re-selection therefore re-selects one channel at a time against the frozen
upstream, not the whole cascade jointly — the same greedy structure, so the two
numbers are comparable.

## What is NOT done

**No re-derivation of E.** The budgets come from experiment 02 and are inherited.
A budget adjusted to make a fold pass would make the transport cost a function of
the adjustment.

**No pass mark on re-selection.** Whether a fold recovers the frozen setting is
counted and reported; no threshold says how many folds must. On a grid this
dense two adjacent settings differ by hundredths of a coverage point, so a count
of exact matches measures the grid's resolution as much as the rule's stability.

**No second date.** February, May and August exist in the Delta and are not read
here. They are the only data no selection has touched, and the day transfer needs
them intact.

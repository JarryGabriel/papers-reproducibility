# Sources, and the pairing rule fixed before the comparison is made

## The cohort

`data/paper_opensky26/flights.delta`, the calibration date **2025-11-01**: 1 507
flights, 200 airframes, 1 108 921 rows.

The calibration date and not another, deliberately. The detector's operating
point was chosen on this day, so the agreement measured here is the agreement in
the most favourable case. Measuring it on a held-out date would confound two
questions — how the detector relates to the register, and how it transports —
and experiment 05 already answers the second.

## The register

| column | non-null rows | share |
|---|---|---|
| `bds_mcp_alt_sel_ft` | 328 817 | 29.65 % |
| `bds_fms_alt_sel_ft` | 117 225 | 10.57 % |

**MCP/FCU is the reference used here**, not FMS. It is the value dialled on the
mode control panel — the level the crew has selected — and it is three times as
available. The FMS value is a different object (the flight plan's next
constraint) and is reported only as a count.

Its availability is uneven and that unevenness is a property of the fleet:

| | |
|---|---|
| airframes broadcasting it at all | **137 of 200** |
| flights carrying at least one value | **565 of 1 507** |

**63 airframes broadcast nothing.** An aircraft that does not broadcast the
register is not a random draw — it reflects avionics fit and operator
configuration — so the comparable plateaux are not a random sample of all
plateaux. This bounds every number this experiment produces and is reported
beside them rather than in a footnote.

## Provenance, and why it may be compared against

Experiment 01 established three things about this register: the stream carries
it (29.5 % of Comm-B messages decode as BDS 4,0, and every one carries a
selected altitude), no pipeline stage removes it, and the published zero was a
decoder regression rather than a property of the data. Two independent decoders
— pyModeS 3.6.0 and the rs1090 path — agree to 0.01 %.

**The register entered no part of the selection**, and this is checked rather
than asserted: `bds_mcp_alt_sel_ft` and `bds_fms_alt_sel_ft` appear nowhere in
experiments 02 through 06, nor in `src/`. The budgets came from quantisation
steps and reanalysis error; the sweep scored self-consistency; the transport
experiments carried frozen points. If the register had informed any of them,
this comparison would be circular.

## How a plateau is paired with a broadcast value

Fixed here, before the distribution is seen.

**The plateaux are the ones the frozen detector commits to.** The altitude
channel, at the point experiment 03 retained: bilateral, `butter=0`, σr=5,
σs=20, slope=3.6342. Not re-selected, not re-tuned for this comparison.

**A plateau's broadcast value is the median of the register over the plateau's
own extent**, taken on the rows where it exists. The median rather than the mean
because the register is quantised to 100 ft and a mean would invent values
between the steps; over the plateau's own extent rather than at its start
because a plateau is a claim about a stretch, not about an instant.

**A plateau with no register value anywhere in its extent is UNPAIRED** and is
counted as such. It is not dropped silently: the share of plateaux that cannot
be compared is measurement 1 and bounds everything else.

**A plateau whose register CHANGES during its extent is kept**, flagged, and
reported separately. Dropping it would be the most tempting form of dishonesty
available here — those are the plateaux most likely to disagree, and they are
also the ones that say something about the register's semantics rather than the
detector's accuracy.

## The units the difference is reported in

**Feet, signed** — `plateau value - broadcast value`, so a systematic offset
reads as one rather than hiding in an absolute value.

**And in units of the 25 ft quantisation step.** A difference smaller than the
step the altitude is broadcast on is not a disagreement the data can resolve,
and the second expression makes that visible without a threshold being chosen.

**One flight level (100 ft) is used as a reporting bracket**, not as a pass mark.
It is the granularity the register itself is quantised to, so it is the natural
unit for "the same level" — but no claim in this experiment depends on a plateau
falling inside it.

## The ground stratum, declared before the full run

The smoke run over 40 flights surfaced a category this file had anticipated in
the abstract — "a crew may leave a level set after levelling elsewhere" — but
had not made recordable. Three plateaux at 144, 150 and 405 ft, against dialled
values of 3 000 and 5 000 ft, are an aircraft on the runway or at the gate with
the MCP still holding the approach level. They moved the 5th percentile of the
difference from -43 ft to -2 850 ft by themselves.

`below_1000ft` is therefore recorded per plateau, and every measurement is
reported twice: over all paired plateaux, and over those at or above 1 000 ft.

**The band is fixed at 1 000 ft here, with 40 flights seen and 1 507 unseen.**
It is a round altitude below which an airliner in cruise configuration does not
hold a commanded level, not a value tuned to the distribution — which is the
point of declaring it before the distribution exists.

**Neither reading is discarded and neither is the headline.** The all-plateaux
figure is what a consumer of the detector meets; the above-1 000 ft figure is
what the two observers say about each other where the register still describes
an intent. Reporting only the second would be filtering by disagreement with
extra steps, which the manifest names as dishonest.

## What is NOT done

**No re-selection.** The detector is at its frozen point. Tuning it to agree
better with the register would make the register part of the selection, which is
the one thing that would void this experiment.

**No pass rate.** The register is an observer, not a truth, and a rate implies
one of them is right. What is reported is a distribution.

**No causal attribution of disagreements.** A difference has at least three
possible sources — the detector, the register's dialled-not-held semantics, and
the timing between dialling and levelling — and this experiment separates none
of them. Measurement 4 describes the disagreeing set; it does not explain it.

**No other date, and no other channel.** No register carries a selected speed,
so four channels of five have no reference at all.

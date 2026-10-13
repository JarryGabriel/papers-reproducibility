"""Re-score a channel's leading candidates under a perturbed gate.

Both steps of this experiment ask the same question of a different constant, so
the scoring lives here once. What they share is the part that must not drift
between them: the cascade exclusion each channel is scored behind, the
resolution of a stored knob back onto the grid, and the selection rule itself,
which is the sweep's rule reused verbatim rather than restated.

The gate is reached by rebinding the detector's own per-channel configuration.
That is deliberate and it is what limits this experiment: it measures the gates
of the implementation as it stands, and a second copy of either constant reached
by another code path would be invisible to it.
"""
from __future__ import annotations

# Resolve this checkout by its marker, regardless of investigation nesting.
import sys as _sys
from pathlib import Path as _Path

_checkout = next((p for p in _Path(__file__).resolve().parents
                  if (p / "src/opensky26_plateau/project_paths.py").is_file()), None)
if _checkout is None:
    raise FileNotFoundError(f"No project source tree above {__file__}")
_sys.path.insert(0, str(_checkout / "src"))
from opensky26_plateau.project_paths import data_dir, experiment_dir, pipeline_config, project_root


import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PAPER_ROOT = project_root(__file__)
REPO_ROOT = project_root(__file__)
INPUTS = EXPERIMENT_DIR / "inputs"
OUTPUTS = EXPERIMENT_DIR / "outputs"
COHORT = data_dir() / 'flights.delta'


from opensky26_plateau import densify_front as front  # noqa: E402
from opensky26_plateau import transfer_scoring as T  # noqa: E402

#: The detector's per-channel configuration, which is where both gates live.
#: Exposed here so the two steps perturb the same object the detector reads,
#: rather than a copy that would leave the detector on its configured value.
CFG = front.CFG

#: The calibration date, and ONLY it. The sweep scored 1 507 flights of
#: 2025-11-01; the Delta table holds four dates and 5 463 flights. Scoring the
#: incumbent against its challengers on the other three would change the cohort
#: at the same time as the gate, and no permutation could then be attributed to
#: the gate — which is the one thing this experiment exists to attribute. The
#: freeze names that substitution as what would make this experiment dishonest.
DATE = "2025-11-01"

#: The two cascades, in the order the sweep walked them. A channel is scored on
#: the residual its priors leave, so the exclusion has to be rebuilt per flight
#: from the priors at THEIR retained points — not read from a mask.
VERTICAL = ("alt", "gamma", "vz")
SPEED = ("mach", "cas")
CHANNELS = (*VERTICAL, *SPEED)

#: The sweep's own coverage tie band. Reused unchanged so the counterfactual
#: rule is the same rule: a permutation inside this band is not a decision the
#: gate made, since the rule already treats those candidates as tied.
TIE_PTS = 0.05

#: How many of each channel's admissible candidates are re-scored. The question
#: is whether the ARGMAX moves, and the gate costs every candidate coverage, so
#: a candidate far below the winner cannot overtake it. Eight is well past the
#: spread the observed permutations occupy.
TOP_K = 8


def grids() -> dict:
    return json.loads((INPUTS / "grids.json").read_text(encoding="utf-8"))


def budgets() -> dict:
    return json.loads((INPUTS / "budgets_full_step.json").read_text(encoding="utf-8"))


def _resolve(knobs_text: str, channel: str, algo: str, butter: int, grid: dict) -> dict:
    """A stored knob string, back onto the grid value it can only have come from.

    `retained.csv` and `sweep.csv` both write knobs with `%g`, six significant
    digits, so a detector run on the stored string is not the detector that was
    scored.
    """
    knobs = {k: float(v) for k, v in (p.split("=") for p in knobs_text.split(","))}
    knobs["butter"] = int(butter)
    resolved = T.resolve_frozen(knobs, channel, algo, grid)
    resolved["butter"] = int(butter)
    return resolved


def retained_points(grid: dict) -> dict[str, tuple[str, dict, str]]:
    """Each channel's retained (family, knobs, knob string), from the artefact."""
    out: dict[str, tuple[str, dict, str]] = {}
    for row in pl.read_csv(INPUTS / "retained.csv").iter_rows(named=True):
        if not row["admissible"]:
            continue
        resolved = _resolve(row["knobs"], row["channel"], row["algo"], row["butter"], grid)
        out[row["channel"]] = (row["algo"], resolved, row["knobs"])
    return out


def leading_candidates(channel: str, grid: dict, budget: float) -> list[dict]:
    """The channel's TOP_K admissible candidates, highest coverage first."""
    sweep = pl.read_csv(INPUTS / "sweep.csv").filter(pl.col("channel") == channel)
    head = (
        sweep.filter(pl.col("worst_segment_error") <= budget)
        .sort("coverage_pct", descending=True)
        .head(TOP_K)
    )
    out = []
    for rank, row in enumerate(head.iter_rows(named=True)):
        out.append(
            {
                "sweep_rank": rank,
                "algo": row["algo"],
                "butter": int(row["butter"]),
                "knobs_text": row["knobs"],
                "knobs": _resolve(row["knobs"], channel, row["algo"], row["butter"], grid),
                "sweep_coverage_pct": row["coverage_pct"],
            }
        )
    return out


def _triple(candidate: dict) -> str:
    """The identity the falsifier compares — family, cutoff and knobs."""
    return f"{candidate['algo']}|{candidate['butter']}|{candidate['knobs_text']}"


def score_cohort(
    channel: str,
    candidates: list[dict],
    settings: list,
    apply_setting: Callable[[str, object], None],
    n_flights: int = 0,
) -> pl.DataFrame:
    """Every candidate at every setting, over the cohort, one row each.

    The whole cohort is walked once and every (setting, candidate) pair is
    scored on the same flight before moving on, so a flight is read and its
    signals derived once rather than once per pair.
    """
    grid = grids()
    points = retained_points(grid)
    order = VERTICAL if channel in VERTICAL else SPEED
    priors = order[: order.index(channel)]

    frame = pl.read_delta(str(COHORT)).with_columns(
        pl.col("raw_timestamp").dt.date().cast(pl.Utf8).alias("_date")
    )
    frame = frame.filter(pl.col("_date") == DATE)
    # `maintain_order=True` and a head slice, exactly as the sweep takes them:
    # a smoke run must probe the flights the sweep actually scored first, and
    # sorting would draw a different sample.
    ids = frame["meta_flight_id"].unique(maintain_order=True).to_list()
    if n_flights:
        ids = ids[:n_flights]
    mach_cfg = T.mach_config(INPUTS / "retained.csv", grid)

    keys = [(s, i) for s in settings for i in range(len(candidates))]
    acc = {
        k: {"committed": 0, "finite": 0, "worst": 0.0, "segments": 0, "flights": 0}
        for k in keys
    }
    flights_seen = 0

    for n, flight_id in enumerate(ids):
        if n % 250 == 0:
            print(f"  {channel}: {n}/{len(ids)}", flush=True)
        flight = frame.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
        if flight.height < 30:
            continue
        flights_seen += 1
        signals = T.signals(flight)
        size = signals[channel].size

        # The cascade exclusion, at the priors' OWN retained points and at their
        # own configured gates: this experiment perturbs one channel's gate, not
        # the cascade that feeds it.
        exclusion = np.zeros(size, dtype=bool)
        for prior in priors:
            palgo, pknobs, _ = points[prior]
            pview = T.filtered(signals[prior], pknobs["butter"])
            for seg in T.detect(palgo, prior, pview, pknobs, exclusion.copy()):
                exclusion[seg["start_idx"] : seg["end_idx"] + 1] = True
        if channel == "cas":
            exclusion = exclusion | T.mach_mask(flight, mach_cfg)

        for setting in settings:
            apply_setting(channel, setting)
            for index, candidate in enumerate(candidates):
                view = T.filtered(signals[channel], candidate["knobs"]["butter"])
                cell = acc[(setting, index)]
                cell["finite"] += int(np.isfinite(view).sum())
                seen_here = False
                for seg in T.detect(candidate["algo"], channel, view,
                                    candidate["knobs"], exclusion):
                    piece = view[seg["start_idx"] : seg["end_idx"] + 1]
                    finite = np.isfinite(piece)
                    if not finite.any():
                        continue
                    values = piece[finite]
                    cell["committed"] += int(finite.sum())
                    cell["segments"] += 1
                    cell["worst"] = max(
                        cell["worst"],
                        float(np.abs(values - values.mean()).mean()),
                    )
                    seen_here = True
                # The sweep divides by the flights that PRODUCED segments, per
                # candidate, and this quantity is the rule's second tie-break.
                # Dividing by every flight walked would put the column on a
                # different scale from sweep.csv's column of the same name.
                cell["flights"] += int(seen_here)

    rows = []
    for (setting, index), cell in acc.items():
        candidate = candidates[index]
        coverage = 100.0 * cell["committed"] / cell["finite"] if cell["finite"] else 0.0
        rows.append(
            {
                "channel": channel,
                "setting": setting,
                "sweep_rank": candidate["sweep_rank"],
                "triple": _triple(candidate),
                "algo": candidate["algo"],
                "butter": candidate["butter"],
                "knobs": candidate["knobs_text"],
                "coverage_pct": coverage,
                "worst_segment_error": cell["worst"],
                "n_segments": cell["segments"],
                "segments_per_flight": cell["segments"] / max(cell["flights"], 1),
                "n_flights_with_segments": cell["flights"],
                "sweep_coverage_pct": candidate["sweep_coverage_pct"],
                "n_flights": flights_seen,
            }
        )
    return pl.DataFrame(rows)


def retain(frame: pl.DataFrame, budget: float) -> dict | None:
    """The selection rule of 03-operator-sweep, applied to one setting's rows."""
    admissible = frame.filter(pl.col("worst_segment_error") <= budget)
    if not admissible.height:
        return None
    ceiling = admissible["coverage_pct"].max()
    tied = admissible.filter(pl.col("coverage_pct") >= ceiling - TIE_PTS).sort(
        ["worst_segment_error", "segments_per_flight"]
    )
    return tied.row(0, named=True)


def verdict_per_channel(scored: pl.DataFrame, budget: float, base_setting) -> dict:
    """Whether the argmax moved, and to what, against the configured setting."""
    base = retain(scored.filter(pl.col("setting") == base_setting), budget)
    if base is None:
        return {"outcome": "no-admissible-at-base", "incumbent": None}

    moved, indeterminate = [], []
    incumbent_over_budget, no_admissible = [], []
    for setting in sorted(scored["setting"].unique().to_list()):
        rows = scored.filter(pl.col("setting") == setting)
        here = retain(rows, budget)
        # The freeze refutes on either route: the ranking permutes, OR the
        # incumbent stops being admissible. Both are recorded, and both count.
        mine = rows.filter(pl.col("triple") == base["triple"])
        if mine.height and mine["worst_segment_error"][0] > budget:
            incumbent_over_budget.append(setting)
        if here is None:
            # Nothing admissible at all: the incumbent has definitionally
            # stopped being admissible, which the falsifier calls refutation.
            # Counting this as `survived` would report stability that the run
            # did not observe.
            no_admissible.append(setting)
            continue
        if here["triple"] == base["triple"]:
            continue
        gap = abs(here["coverage_pct"] - (mine["coverage_pct"][0] if mine.height else 0.0))
        (indeterminate if gap < TIE_PTS else moved).append(
            {"setting": setting, "to": here["triple"], "gap_pts": gap}
        )
    if moved or incumbent_over_budget or no_admissible:
        outcome = "refuted"
    elif indeterminate:
        outcome = "indeterminate"
    else:
        outcome = "survived"
    return {
        "outcome": outcome,
        "incumbent": base["triple"],
        "moved": moved,
        "indeterminate": indeterminate,
        "incumbent_over_budget": incumbent_over_budget,
        "no_admissible": no_admissible,
    }

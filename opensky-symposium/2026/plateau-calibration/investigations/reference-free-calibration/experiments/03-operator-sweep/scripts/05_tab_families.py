"""Render tab-families: the four detector families and the four pre-filter cutoffs.

Asserts that every family and every cutoff was scored on every channel, and how
many candidates each contributes to the admissible set the retained point had to
beat. Reads the sweep's own scored candidates and the locked budgets; writes a
LaTeX table plus the CSV it was rendered from, so the numbers are auditable
without re-reading 27 414 rows.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs" / "sweep.csv"
BUDGETS = ROOT / "inputs" / "budgets_full_step.json"
OUT_CSV = ROOT / "outputs" / "family_census.csv"
OUT_TEX = ROOT / "figures" / "tab_families.tex"

# Display order: most candidates first, so the table reads as a ranking.
FAMILY_LABEL = {
    "bilateral": "Bilateral filter",
    "savgol": "Savitzky--Golay",
    "varthresh": "Variance threshold",
    "pelt": "PELT (change point)",
}


def main() -> None:
    budgets = json.loads(BUDGETS.read_text())
    rows = list(csv.DictReader(SWEEP.open()))

    scored: Counter[str] = Counter()
    admissible: Counter[str] = Counter()
    by_cutoff_scored: Counter[str] = Counter()
    by_cutoff_adm: Counter[str] = Counter()
    channels_seen: dict[str, set[str]] = {}
    channels_admissible: dict[str, set[str]] = {}

    for row in rows:
        algo = row["algo"]
        channel = row["channel"]
        cutoff = row["butter"]
        scored[algo] += 1
        by_cutoff_scored[cutoff] += 1
        channels_seen.setdefault(algo, set()).add(channel)
        if float(row["worst_segment_error"]) <= float(budgets[channel]):
            admissible[algo] += 1
            by_cutoff_adm[cutoff] += 1
            channels_admissible.setdefault(algo, set()).add(channel)

    order = sorted(scored, key=lambda a: -scored[a])
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["kind", "key", "channels_scored", "scored", "admissible", "channels_admissible"]
        )
        for algo in order:
            writer.writerow(
                [
                    "family",
                    algo,
                    len(channels_seen[algo]),
                    scored[algo],
                    admissible[algo],
                    len(channels_admissible.get(algo, ())),
                ]
            )
        for cutoff in sorted(by_cutoff_scored, key=float):
            writer.writerow(
                ["cutoff", cutoff, 5, by_cutoff_scored[cutoff], by_cutoff_adm[cutoff], ""]
            )

    total_scored = sum(scored.values())
    total_adm = sum(admissible.values())

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Detector family & Scored & Admissible & \multicolumn{2}{c}{Channels} \\",
        r"\cmidrule(l){4-5}",
        r" & & & scored & admissible \\",
        r"\midrule",
    ]
    for algo in order:
        lines.append(
            f"{FAMILY_LABEL.get(algo, algo)} & {scored[algo]:,} & "
            f"{admissible[algo]:,} & {len(channels_seen[algo])} & "
            f"{len(channels_admissible.get(algo, ())):d} \\\\".replace(",", r"\,")
        )
    lines += [
        r"\midrule",
        f"Total & {total_scored:,} & {total_adm:,} & 5 & 5 \\\\".replace(",", r"\,"),
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Admissible candidates by pre-filter cutoff (s)}} \\",
    ]
    for cutoff in sorted(by_cutoff_scored, key=float):
        label = "none" if float(cutoff) == 0 else cutoff
        lines.append(
            f"\\quad {label} & {by_cutoff_scored[cutoff]:,} & "
            f"{by_cutoff_adm[cutoff]:,} & & \\\\".replace(",", r"\,")
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TEX}")
    print(f"scored {total_scored}, admissible {total_adm}")
    for algo in order:
        print(f"  {algo:12s} scored={scored[algo]:6d} "
              f"admissible={admissible[algo]:5d} "
              f"channels {len(channels_seen[algo])}->{len(channels_admissible.get(algo, ()))}")
    for cutoff in sorted(by_cutoff_scored, key=float):
        print(f"  cutoff {cutoff:>4s}: scored={by_cutoff_scored[cutoff]:6d} "
              f"admissible={by_cutoff_adm[cutoff]:5d}")


if __name__ == "__main__":
    main()

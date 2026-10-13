"""The cohort table the Data section carries.

The venue treats the data section as a chapter and expects it to be heavy: the
dates written out, the fleet counted, and the register's availability given per
date rather than as a single share. This renders that table from the same CSVs
the analysis reads, so the manuscript never retypes a value.

Reproduce:
    uv run --active python scripts/05_tab_cohort.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = EXPERIMENT_DIR / "outputs"
FIGURES = EXPERIMENT_DIR / "figures"

#: The dates as the manuscript writes them, keyed by the ISO date in the CSV.
WRITTEN = {
    "2025-02-17": "17 Feb 2025",
    "2025-05-21": "21 May 2025",
    "2025-08-29": "29 Aug 2025",
    "2025-11-01": "1 Nov 2025",
}


def main() -> int:
    by_date = pl.read_csv(OUTPUTS / "coverage_by_date.csv")

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Date & Flights & Airframes & Samples & MCP & FMS \\",
        r" & & & & (\%) & (\%) \\",
        r"\midrule",
    ]
    for row in by_date.iter_rows(named=True):
        lines.append(
            f"{WRITTEN[row['date']]} & {row['flights']:,} & {row['airframes']} & "
            f"{row['rows']:,} & {row['mcp_pct']:.1f} & {row['fms_pct']:.1f} "
            r"\\".replace(",", r"\,")
        )
    lines += [
        r"\midrule",
        f"Cohort & {by_date['flights'].sum():,} & 200 & {by_date['rows'].sum():,} & "
        r"--- & --- \\".replace(",", r"\,"),
        r"\bottomrule",
        r"\end{tabular}",
    ]

    FIGURES.mkdir(exist_ok=True)
    target = FIGURES / "tab_cohort.tex"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(by_date.select(["date", "flights", "airframes", "mcp_pct", "fms_pct"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

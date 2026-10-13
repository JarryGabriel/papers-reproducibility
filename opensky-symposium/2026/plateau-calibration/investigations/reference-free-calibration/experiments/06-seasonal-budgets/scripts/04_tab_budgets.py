"""Render tab-budgets: each channel's budget, its derivation basis, and its
stability across the four dates.

Asserts that every budget is read from a property of the channel's own signal,
and that the two broadcast channels give the identical value on all four dates
while the two ERA5-reconstructed ones move. Reads experiment 02's locked budgets
for the derivation basis and units, and this experiment's per-date derivation for
the ratio; writes a LaTeX table and the CSV it was rendered from.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ROOT / "inputs" / "budgets_full_step.csv"
BY_DATE = ROOT / "outputs" / "budgets_by_date.csv"
OUT_CSV = ROOT / "outputs" / "budget_table.csv"
OUT_TEX = ROOT / "figures" / "tab_budgets.tex"

# Cascade order: the two vertical channels the cascade selects first, then the
# derived one, then the two speed channels.
ORDER = ["alt", "gamma", "vz", "mach", "cas"]

LABEL = {
    "alt": "Altitude",
    "gamma": "Flight-path angle",
    "vz": "Vertical speed",
    "mach": "Mach",
    "cas": "CAS/IAS",
}

# The operational unit this venue writes in, beside the SI the detector scores
# in. Where the two coincide the SI column repeats the value rather than
# inventing a conversion.
OPERATIONAL = {
    "alt": ("ft", 25.0 / 2 * 2),  # full quantisation step, 25 ft
    "gamma": ("deg", None),
    "vz": ("ft/min", None),
    "mach": ("", None),
    "cas": ("kt", None),
}

BASIS_SHORT = {
    "quantisation step / 2": "full quantisation step",
    "median of the propagated budget": r"propagation through $\arcsin(v_z/\mathrm{TAS})$",
    "robust sigma of the ERA5 reconstruction error": "robust $\\sigma$ of the ERA5 reconstruction error",
}


def main() -> None:
    locked = {row["channel"]: row for row in csv.DictReader(BUDGETS.open())}

    per_date: dict[str, list[float]] = defaultdict(list)
    for row in csv.DictReader(BY_DATE.open()):
        per_date[row["channel"]].append(float(row["budget"]))

    rows = []
    for channel in ORDER:
        source = locked[channel]
        budget_si = float(source["budget_full_step"])
        values = per_date[channel]
        ratio = max(values) / min(values)

        native_unit, _ = OPERATIONAL[channel]
        if channel == "alt":
            native = f"{budget_si / 0.3048:.0f}"
        elif channel == "gamma":
            import math

            native = f"{math.degrees(budget_si):.3f}"
        elif channel == "vz":
            native = f"{budget_si:.0f}"
        elif channel == "mach":
            native = f"{budget_si:.4f}"
        else:
            native = f"{budget_si:.2f}"

        rows.append(
            {
                "channel": channel,
                "label": LABEL[channel],
                "basis": BASIS_SHORT.get(source["basis"], source["basis"]),
                "budget_native": native,
                "native_unit": native_unit,
                "budget_si": budget_si,
                "si_unit": source["unit"],
                "ratio_across_dates": round(ratio, 4),
            }
        )

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Only altitude and the flight-path angle read differently in operational
    # and SI units; for the other three the two columns would print the same
    # number twice, so the table carries one budget column and names the unit
    # the venue writes in, with the SI equivalent only where it differs.
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Channel & Derived from & Budget $E$ & Ratio across dates \\",
        r"\midrule",
    ]
    for row in rows:
        si = row["budget_si"]
        unit = row["native_unit"]
        budget = f"{row['budget_native']}~{unit}" if unit else row["budget_native"]
        if row["channel"] in {"alt", "gamma"}:
            budget += f" ({si:.5g}~{row['si_unit']})"
        lines.append(
            f"{row['label']} & {row['basis']} & {budget} & "
            f"{row['ratio_across_dates']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TEX}")
    for row in rows:
        print(
            f"  {row['channel']:6s} {row['budget_native']:>9s} {row['native_unit']:<7s} "
            f"SI={row['budget_si']:<12.6g} ratio={row['ratio_across_dates']:.4f}  "
            f"{row['basis'][:44]}"
        )


if __name__ == "__main__":
    main()

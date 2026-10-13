"""Render tab-retained: the operating point retained on each channel.

Asserts what the selection rule returned — one point per channel, its family and
settings, the coverage it commits, how much of its budget it uses, and how many
admissible candidates it was chosen from. Reads the selection's own output and
writes a LaTeX table beside the CSV it was rendered from.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "outputs" / "retained.csv"
OUT_TEX = ROOT / "figures" / "tab_retained.tex"

# Cascade order: vertical first (altitude, then the flight-path angle on what it
# leaves, then vertical speed), then the speed cascade.
ORDER = ["alt", "gamma", "vz", "mach", "cas"]

LABEL = {
    "alt": "Altitude",
    "gamma": "Flight-path angle",
    "vz": "Vertical speed",
    "mach": "Mach",
    "cas": "CAS/IAS",
}
FAMILY = {
    "bilateral": "bilateral",
    "savgol": "Savitzky--Golay",
    "varthresh": "variance threshold",
    "pelt": "PELT",
}


def _settings(knobs: str) -> str:
    """The retained knobs, rendered for a table cell."""
    parts = []
    for item in knobs.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        symbol = {
            "sigma_r": r"$\sigma_r$",
            "sigma_s": r"$\sigma_s$",
            "slope_tol": "slope tol.",
        }.get(key, key.replace("_", r"\_"))
        # Three significant figures. The grid resolves the slope tolerance to
        # more digits than that, and printing 0.000144225 claims a resolution
        # the measurement does not have.
        parts.append(f"{symbol}~{float(value):.3g}")
    return ", ".join(parts)


def main() -> None:
    rows = {row["channel"]: row for row in csv.DictReader(RETAINED.open())}

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Channel & Retained setting & Coverage & Budget & Admissible \\",
        r" & & (\%) & used (\%) & candidates \\",
        r"\midrule",
    ]
    for channel in ORDER:
        row = rows[channel]
        family = FAMILY.get(row["algo"], row["algo"])
        cutoff = float(row["butter"])
        prefilter = "no pre-filter" if cutoff == 0 else f"cutoff {cutoff:g}~s"
        lines.append(
            f"{LABEL[channel]} & {family}, {prefilter}, {_settings(row['knobs'])} & "
            f"{float(row['coverage_pct']):.1f} & "
            f"{float(row['budget_used_pct']):.1f} & "
            f"{int(row['n_admissible'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEX.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_TEX}")
    for channel in ORDER:
        row = rows[channel]
        print(
            f"  {channel:6s} {row['algo']:10s} butter={row['butter']:>3s} "
            f"cov={float(row['coverage_pct']):6.2f} % "
            f"budget_used={float(row['budget_used_pct']):6.2f} % "
            f"admissible={row['n_admissible']}"
        )


if __name__ == "__main__":
    main()

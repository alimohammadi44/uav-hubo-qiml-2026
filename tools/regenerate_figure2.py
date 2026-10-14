#!/usr/bin/env python3
"""Regenerate Figure 2 with AAAI-readable fonts and grayscale-safe styling."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "unified_comparison_with_cpsat.csv"
OUT = ROOT / "paper" / "fig_unified_success_with_cpsat.png"

def pct(v):
    v = v.strip()
    if v.lower() == "yes":
        return 100.0
    if v.lower() == "no":
        return 0.0
    return float(v.rstrip("%"))

rows = list(csv.DictReader(CSV.open(newline="", encoding="utf-8")))
labels = [r["Method"].replace("HUBO trajectory-SA", "HUBO-SA").replace("HUBO->QUBO/neal", "QUBO/neal") for r in rows]
metrics = [
    ("Feasible", [pct(r["Feasible"]) for r in rows], "///", "0.85"),
    ("Goal arrival", [pct(r["Goal arrival"]) for r in rows], "...", "0.60"),
    ("Full success", [pct(r["Full success"]) for r in rows], "xxx", "0.35"),
]

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
})
fig, ax = plt.subplots(figsize=(3.35, 2.75), dpi=300)
y = np.arange(len(labels))
h = 0.22
for i, (name, values, hatch, shade) in enumerate(metrics):
    ax.barh(y + (i - 1) * h, values, height=h, label=name,
            color=shade, edgecolor="black", linewidth=0.65, hatch=hatch)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 105)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Rate (%)")
ax.grid(axis="x", linewidth=0.35, linestyle=":", color="0.55")
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
          frameon=False, handlelength=1.6, columnspacing=0.8)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.subplots_adjust(left=0.37, right=0.985, bottom=0.16, top=0.84)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(OUT)

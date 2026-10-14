"""Emit tab:gap: the script disparity before and after treating both conditions.

"Not condition-specific" is not the same as "does not reduce the disparity". The old
98-percent framing compared a treated Dev--Dev against an untreated Lat--Lat, which is a
legitimate comparison but says nothing about parity under equal treatment. Treating both,
the gap shrinks in three of four arms and grows slightly in Marathi.

The untreated columns carry paired intervals because Dev--Dev and Lat--Lat are two
conditions of one behavioural run. The treated column cannot: treated Dev--Dev comes from
the window run and treated Lat--Lat from a separate control run, and the control's
per-item file was overwritten by the later site runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
CI = Path(__file__).resolve().parents[2] / "results_mr" / "gap_intervals.csv"

# arm: (Dev--Dev untreated, Lat--Lat untreated, Dev--Dev treated, Lat--Lat treated)
ARMS = {
    "Marathi": (0.065, 0.340, 0.335, 0.629),
    "English, mr corpus": (0.283, 0.529, 0.590, 0.756),
    "Hindi": (0.122, 0.401, 0.546, 0.690),
    "English, hi corpus": (0.260, 0.521, 0.585, 0.743),
}

ci = pd.read_csv(CI) if CI.exists() else None


def lookup(arm: str, quantity: str):
    if ci is None:
        return None
    r = ci[(ci.arm == arm) & (ci.quantity == quantity)]
    return r.iloc[0] if len(r) else None


rows, shrink = [], []
for arm, (dd, ll, tdd, tll) in ARMS.items():
    before, after = (ll - dd) * 100, (tll - tdd) * 100
    rows.append((arm, before, after, after - before))
    if after < before:
        shrink.append(arm)

widens = [a for a in ARMS if a not in shrink]

print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{The script disparity in positive-pair accuracy, in points, before any "
      r"intervention and with the same frozen vector applied to both conditions. The "
      r"untreated columns carry paired item-clustered intervals over 2000 draws, since "
      r"Dev--Dev and Lat--Lat are two conditions of one run; the treated column is a point "
      r"estimate, because treated Dev--Dev and treated Lat--Lat come from separate "
      r"evaluations and the entities cannot be paired across them from the files we "
      r"retain. Treating both conditions equally, the disparity narrows in "
      + ("three of the four arms" if len(shrink) == 3 else f"{len(shrink)} of four arms")
      + r" and widens slightly in " + ", ".join(widens) + r". This is the comparison the "
      r"earlier framing of a percentage closure did not make: that figure set a treated "
      r"Dev--Dev against an untreated Lat--Lat, which measures movement rather than "
      r"parity. The intervention improves both conditions and does not eliminate the "
      r"disparity between them.}")
print(r"\label{tab:gap}")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{Positive-pair gap} & & Balanced gap\\")
print(r"\cmidrule(lr){2-3}")
print(r"Arm & untreated & both treated & Change & untreated\\")
print(r"\midrule")
for arm, before, after, delta in rows:
    mark = r"\textbf{%+.1f}" % delta if delta > 0 else "%+.1f" % delta
    p = lookup(arm, "untreated gap truth=1")
    untreated = (f"{before:.1f} [{p.lo:+.1f}, {p.hi:+.1f}]" if p is not None
                 else f"{before:.1f}")
    b = lookup(arm, "untreated balanced gap")
    balanced = f"{b.value:+.1f} [{b.lo:+.1f}, {b.hi:+.1f}]" if b is not None else "---"
    print(f"{arm} & {untreated} & {after:.1f} & {mark} & {balanced}\\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

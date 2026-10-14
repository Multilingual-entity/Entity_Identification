"""Emit tab:calvsprobe: four decision rules on the same entities and the same unit.

Three things the caption has to state, all raised in review. The probe's layer, because
tab:probee2 reports higher E2 figures that are test-profile maxima rather than selected.
The false-positive rates are not matched, so the difference is a balanced-accuracy gap at
unmatched rates. And the Marathi probe and string rows agree to three decimals by
coincidence: the two predictors differ on 2 of 240 pairs and the disagreements cancel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
CSV = (Path(__file__).resolve().parents[2] / "results_mr"
       / "probe_versus_calibration.csv")

d = pd.read_csv(CSV)
ROWS = [("raw model decision", "raw model decision"),
        ("margin $+$ one validation constant", "margin + validation offset"),
        ("hidden-state probe at E2", "hidden-state probe at E2"),
        ("five string features", "five string features")]

# measured by the agreement check: 2 of 240 Marathi pairs differ, and they cancel
DISAGREE, NPAIRS = 2, 240


def get(ctx, q, col="value"):
    return float(d[(d.context == ctx) & (d.quantity == q)][col].iloc[0])


mr_cal = get("mr", "margin + validation offset")
mr_probe = get("mr", "hidden-state probe at E2")
mr_raw = get("mr", "raw model decision")
closed = (mr_cal - mr_raw) / (mr_probe - mr_raw) * 100
gap_mr = d[(d.context == "mr") & (d.quantity == "probe minus margin + offset")].iloc[0]
gap_en = d[(d.context == "en") & (d.quantity == "probe minus margin + offset")].iloc[0]
mr_layer, en_layer = get("mr", "probe layer"), get("en", "probe layer")
cal_fpr = get("mr", "margin + validation offset", "fpr")
probe_fpr = get("mr", "hidden-state probe at E2", "fpr")

REF = chr(92) + "ref"
caption = (
    "Four decision rules on the 120 held-out entities, Dev--Dev inputs, all at entity "
    "level: an entity's six prompt margins are averaged first, which is the unit the probe "
    "predicts on. Every parameter is chosen on validation and nothing is selected on test. "
    "The contrast between the probe and the raw model is mostly a calibration difference: "
    f"one constant added to the model's own margin recovers {closed:.0f} percent of the "
    "distance from the raw decision to the probe under Marathi prompts. A detectable probe "
    f"advantage remains, ${gap_mr.value:+.3f}$ $[{gap_mr.lo:+.3f}, {gap_mr.hi:+.3f}]$ in "
    f"Marathi and ${gap_en.value:+.3f}$ $[{gap_en.lo:+.3f}, {gap_en.hi:+.3f}]$ in English, "
    f"but the two are scored at different false-positive rates, {cal_fpr:.3f} against "
    f"{probe_fpr:.3f}, so that is a balanced-accuracy difference at unmatched rates rather "
    "than a gain at a common one. The offset here is unconstrained, unlike the floored "
    f"offset of Table~{REF}{{tab:calibration}}, which is why its rate is the higher of the "
    f"two. The probe uses the E2 layer each arm's validation split selected, "
    f"{mr_layer:.0f} for Marathi and {en_layer:.0f} for English; the larger E2 figures in "
    f"Table~{REF}{{tab:probee2}} are maxima over a test-layer profile and were not "
    "selected. The Marathi probe and string rows match to three decimals by coincidence "
    f"and not by construction: the two predictors differ on {DISAGREE} of {NPAIRS} pairs "
    "and the disagreements cancel. That near-identity is itself the point of this section, "
    "since on this negative map the two are very nearly the same classifier.")

print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{" + caption + "}")
print(r"\label{tab:calvsprobe}")
print(r"\begin{tabular}{lccc|ccc}")
print(r"\toprule")
print(r"& \multicolumn{3}{c|}{Marathi} & \multicolumn{3}{c}{English}\\")
print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
print(r"Rule & bal.\ acc & TPR & FPR & bal.\ acc & TPR & FPR\\")
print(r"\midrule")
for pretty, key in ROWS:
    cells = []
    for ctx in ("mr", "en"):
        bal = get(ctx, key)
        best = max(get(ctx, k) for _, k in ROWS)
        b = r"\textbf{%.3f}" % bal if abs(bal - best) < 1e-9 else "%.3f" % bal
        cells += [b, "%.3f" % get(ctx, key, "tpr"), "%.3f" % get(ctx, key, "fpr")]
    print(f"{pretty} & " + " & ".join(cells) + r"\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

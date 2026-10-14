"""Emit tab:pertemplate, the appendix home for sec:repair's per-template numbers.

Those figures live only in the body at present, so no paragraph carrying them can be moved
without losing them. This table is the copy that makes the cut safe, and it also answers
the review's request for a template-stratified AUROC summary beside the entity-averaged one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[2] / "results_mr"

auroc = pd.read_csv(R / "per_template_auroc.csv")
points = pd.read_csv(R / "per_template_points.csv")
d = auroc.merge(points, on="template")

tuned = 0
beats = [int(r.template) for _, r in d.iterrows()
         if r.vector_tpr > max(r.threshold_tpr, r.offset_tpr) + 0.01]
ranks = [int(r.template) for _, r in d.iterrows() if r.lo > 0]

print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{The window intervention template by template, Marathi prompts, "
      r"Dev--Dev, 120 held-out entities. The AUROC columns average an entity's two answer "
      r"keys within a template and then rank, with paired item-clustered intervals over "
      r"2000 draws; the operating-point columns are prompt-level. The threshold column is "
      r"the oracle diagnostic of Table~\ref{tab:calibration}, placed per template on the "
      r"test curve; the offset column is chosen on the validation tuning entities under "
      r"that arm's own validation negative-pair accuracy and reads nothing from test. "
      f"Ranking improves on templates {' and '.join(str(t) for t in ranks)}, which a "
      r"constant offset cannot do. The vector beats both calibration columns on template "
      f"{beats[0] if beats else 'none'} alone, the one its scale was tuned on. "
      r"Template 2 carries an apparent tension: its ranking gain excludes zero while its "
      r"true-positive rate at the operating point falls, which is consistent because the "
      r"ranking gain lies away from the low-false-positive region the operating point "
      r"sits in.}")
print(r"\label{tab:pertemplate}")
print(r"\begin{tabular}{lccc|ccc}")
print(r"\toprule")
print(r"& \multicolumn{3}{c|}{Entity AUROC} & \multicolumn{3}{c}{TPR at the vector's FPR}\\")
print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
print(r"Template & baseline & $+$ vector & change & vector & threshold & offset\\")
print(r"\midrule")
for _, r in d.iterrows():
    star = r"\textbf{%+.3f}" % r.change if r.lo > 0 else "%+.3f" % r.change
    ci = f"$[{r.lo:+.3f}, {r.hi:+.3f}]$"
    vec = r"\textbf{%.3f}" % r.vector_tpr if int(r.template) in beats \
        else "%.3f" % r.vector_tpr
    print(f"{int(r.template)} & {r.baseline:.3f} & {r.with_vector:.3f} & "
          f"{star} {ci} & {vec} ({r.vector_fpr:.3f}) & "
          f"{r.threshold_tpr:.3f} & {r.offset_tpr:.3f} ({r.offset_fpr:.3f})\\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

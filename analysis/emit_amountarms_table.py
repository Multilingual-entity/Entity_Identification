"""Emit tab:amountarms, the appendix home for sec:mediator's amount-arm numbers.

Those six figures live only in main-text prose, so compressing that section would lose
them. "Six fold and language cells" means Marathi and Hindi, excluding English; on that
subset a fixed coefficient spans 3.55 to 7.00 and an item's own spans 2.52 to 6.62, which
is what the prose reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SEED, DRAWS = 20260813, 2000

d = pd.concat([pd.read_csv(ROOT / f"mediator_pull/{t}_arms/mediator_items.csv")
               for t in ("mr", "hi")])
LABEL = {"fixed": "one constant for every item", "matched": "the item's own",
         "shuffled": "another item's"}

md = d[(d.method == "meandiff") & (d.group == "gated")]
piv = md.groupby(["context", "fold", "coefficient"]).delta_mediated.mean().unstack()
native = piv.loc[[i for i in piv.index if i[0] in ("mr", "hi")]]

jc = d[(d.method == "jacobian") & (d.group == "gated")]
rng = np.random.default_rng(SEED)
rows = []
for (ctx, fold), g in jc.groupby(["context", "fold"]):
    p = g.pivot_table(index=["fact_id", "truth", "paraphrase_id"],
                      columns="coefficient", values="delta_mediated")
    p = p.dropna(subset=["matched", "shuffled"])
    diff = (p["matched"] - p["shuffled"]).to_numpy()
    ids = p.index.get_level_values(0).to_numpy()
    u = np.unique(ids)
    idx = {k: np.flatnonzero(ids == k) for k in u}
    draws = [diff[np.concatenate([idx[k] for k in rng.choice(u, len(u), True)])].mean()
             for _ in range(DRAWS)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rows.append((ctx, fold, diff.mean(), lo, hi, len(u)))
j = pd.DataFrame(rows, columns=["ctx", "fold", "diff", "lo", "hi", "n"])
worst = j.iloc[j["diff"].abs().idxmax()]
spans = int(((j.lo < 0) & (j.hi > 0)).sum())

print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{How much each item moves, holding the direction fixed. Mean signed margin "
      r"change on gated items, by fold and prompt language, at layer 16 and rank 1 on "
      r"validation. The upper block varies the coefficient for the mean-difference "
      r"direction: across the six Marathi and Hindi cells a single constant for every item "
      f"spans {native['fixed'].min():.2f} to {native['fixed'].max():.2f} and an item's own "
      f"coefficient spans {native['matched'].min():.2f} to {native['matched'].max():.2f}, "
      r"so a constant amount is not what limits recovery. The lower block gives the paired "
      r"difference between an item's own coefficient and another item's for the "
      f"sensitivity direction: it spans zero in {spans} of {len(j)} cells, the largest "
      f"magnitude being ${worst['diff']:+.2f}$ $[{worst.lo:+.2f}, {worst.hi:+.2f}]$ in "
      f"{worst.ctx} fold {worst.fold}. With {int(j.n.min())} to {int(j.n.max())} entities "
      r"per fold these intervals bound an advantage loosely rather than excluding one.}")
print(r"\label{tab:amountarms}")
print(r"\begin{tabular}{llccc}")
print(r"\toprule")
print(r"\multicolumn{5}{l}{\em Mean-difference direction, gated margin change}\\")
print(r"Prompts & Fold & " + " & ".join(LABEL[c] for c in ("fixed", "matched", "shuffled"))
      + r"\\")
print(r"\midrule")
for (ctx, fold), r in piv.iterrows():
    cells = " & ".join(f"{r[c]:.2f}" for c in ("fixed", "matched", "shuffled"))
    print(f"{ctx} & {fold} & {cells}\\\\")
print(r"\midrule")
print(r"\multicolumn{5}{l}{\em Sensitivity direction, own coefficient minus another "
      r"item's}\\")
print(r"Prompts & Fold & difference & interval & entities\\")
print(r"\midrule")
for _, r in j.iterrows():
    print(f"{r.ctx} & {r.fold} & ${r['diff']:+.2f}$ & "
          f"$[{r.lo:+.2f}, {r.hi:+.2f}]$ & {int(r.n)}\\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

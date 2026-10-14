"""Emit tab:jacfine: both correction directions on the finer validation grid.

The coarse grid of tab:jacsweep stepped 0.5, 1, 2, 3, which jumped straight over the
sensitivity direction's plateau and made it look like a one-step window. This resolves
0.5 to 1.5 at 0.1 and puts the mean-difference vector's own grid beside it, both scored
the same way on the same 60 validation items.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2] / "results_mr" / "qwen_mr"


def curve(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    g = d.groupby(["context", "alpha"]).apply(
        lambda f: pd.Series({
            "pos": (f[f.truth == 1].yes_minus_no_margin > 0).mean(),
            "neg": (f[f.truth == 0].yes_minus_no_margin < 0).mean()}),
        include_groups=False)
    g["bal"] = (g["pos"] + g["neg"]) / 2
    return g


sens = curve(ROOT / "jac_fine_grid.csv")
mean = curve(ROOT / "recovery_window" / "vector_tuning_validation.csv")


def rows(context: str) -> list[str]:
    s, m = sens.loc[context], mean.loc[context]
    s_best, m_best = s.bal.idxmax(), m.bal.idxmax()
    out = []
    for i in range(max(len(s), len(m))):
        left = right = " & & & "
        if i < len(s):
            a = s.index[i]
            r = s.loc[a]
            bold = r"\textbf{%.3f}" % r.bal if a == s_best else "%.3f" % r.bal
            left = f"{a:.1f} & {r.pos:.3f} & {r.neg:.3f} & {bold}"
        if i < len(m):
            a = m.index[i]
            r = m.loc[a]
            bold = r"\textbf{%.3f}" % r.bal if a == m_best else "%.3f" % r.bal
            right = f"{a:g} & {r.pos:.3f} & {r.neg:.3f} & {bold}"
        out.append(f"{left} & {right}\\\\")
    return out


mr_s, mr_m = sens.loc["mr"], mean.loc["mr"]
en_s, en_m = sens.loc["en"], mean.loc["en"]
near = int(((mr_s.bal >= mr_s.bal.max() - 0.025) & (mr_s.index >= 1.0)).sum())

caption = (
    "Both correction directions on the validation tuning items, selected by balanced "
    "accuracy exactly as in Table~\\ref{tab:winsweep}. Scales are not comparable across "
    "the two panels: each direction is unit length, so its own scale carries all "
    "magnitude, and the grids are the ones each was searched on. Two things the coarse "
    f"grid of Table~\\ref{{tab:jacsweep}} could not show. The sensitivity direction has a "
    f"plateau rather than a one-step window: {near} consecutive Marathi scales from "
    f"$\\alpha{{=}}1.0$ to 1.5 sit within 0.025 of its maximum. And under the same rule it "
    f"reaches higher balanced accuracy than the mean-difference vector in both languages, "
    f"{mr_s.bal.max():.3f} against {mr_m.bal.max():.3f} in Marathi and "
    f"{en_s.bal.max():.3f} against {en_m.bal.max():.3f} in English, the Marathi gap "
    f"because the mean-difference vector never lifts positive-pair accuracy above "
    f"{mr_m.pos.max():.3f} at any scale while the sensitivity direction reaches "
    f"{mr_s.pos.max():.3f}. It is fitted on 31 gated training items against the other's "
    f"361. No held-out test evaluation was run for the sensitivity direction, so these "
    f"figures cannot be set beside the test numbers of Table~\\ref{{tab:wintest}}."
)

print(r"\begin{table*}[t]")
print(r"\centering\small")
print(r"\caption{" + caption + "}")
print(r"\label{tab:jacfine}")
print(r"\begin{tabular}{cccc|cccc}")
print(r"\toprule")
print(r"& \multicolumn{3}{c|}{Local sensitivity} & "
      r"& \multicolumn{3}{c}{Mean difference}\\")
print(r"\cmidrule(lr){1-4}\cmidrule(lr){5-8}")
print(r"$\alpha$ & positives & negatives & balanced & "
      r"$\alpha$ & positives & negatives & balanced\\")
for lang, ctx in (("Marathi", "mr"), ("English", "en")):
    print(r"\midrule")
    print(r"\multicolumn{8}{l}{\em " + lang + r" prompts}\\")
    for line in rows(ctx):
        print(line)
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")

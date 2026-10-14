"""Emit the LaTeX comparing the two correction directions across the scale grid.

    python analysis/emit_jacobian_table.py > jacobian_table.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MEAN = ROOT / "results_mr" / "overnight_20260930_0537" / "1d_tuning_grid.csv"
JAC = ROOT / "results_mr" / "qwen_mr" / "jacobian_tuning_grid.csv"
BS = chr(92)
NL = BS + BS


def grid(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    return t.groupby(["context", "alpha", "truth"]).correct.mean().unstack("truth") * 100


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    m, j = grid(MEAN), grid(JAC)
    alphas = sorted(set(m.loc["mr"].index) & set(j.loc["mr"].index))

    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{The two correction directions across the same scale grid, "
          "Marathi prompts, 60 validation tuning items, both applied across layers "
          "12--17 and both rescaled to unit length. " + 'At their selected scales the sensitivity direction is the better of the two on both classes, 68.3 positives at 98.3 negatives against 38.3 at 97.5, so neither is exclusively effective under a negative-pair constraint. What separates them is the width of the usable range. The mean-difference vector holds negative-pair accuracy at 100.0 through $\\alpha{=}8$ and 93.3 at the largest scale searched, so its operating region is broad. The sensitivity direction runs 100.0, 98.3, 80.0, 39.2 and 0.8 across $\\alpha \\in \\{0.5, 1, 1.5, 2, 3\\}$, so its selected scale of 1.0 sits one grid step from collapse. Both reach a higher balanced accuracy than the single-layer sweep of Appendix~\\ref{app:recovery}, and \\S\\ref{sec:repair} shows that for the mean-difference vector the gain is largely a movement of the decision criterion rather than of discrimination.' + "}")
    print(BS + "label{tab:jacsweep}")
    print(BS + "begin{tabular}{lcc|cc}")
    print(BS + "toprule")
    print(" & " + BS + "multicolumn{2}{c|}{Mean difference} & "
          + BS + "multicolumn{2}{c}{Local sensitivity}" + NL)
    print(BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-5}")
    print("Scale & positives & negatives & positives & negatives" + NL)
    print(BS + "midrule")
    for a in alphas:
        mp, mn = m.loc[("mr", a)][1], m.loc[("mr", a)][0]
        jp, jn = j.loc[("mr", a)][1], j.loc[("mr", a)][0]
        cells = [f"${BS}alpha{{=}}{a:g}$", f"{mp:.1f}", f"{mn:.1f}",
                 f"{jp:.1f}", f"{jn:.1f}"]
        # mark each column's selected operating point
        if abs(a - 12.0) < 1e-9:
            cells[1] = BS + "textbf{" + cells[1] + "}"
        if abs(a - 1.0) < 1e-9:
            cells[3] = BS + "textbf{" + cells[3] + "}"
        print(" & ".join(cells) + NL)
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()
    for label, g, a in (("mean difference", m, 12.0), ("local sensitivity", j, 1.0)):
        p, n = g.loc[("mr", a)][1], g.loc[("mr", a)][0]
        print(f"% {label} at alpha {a:g}: positives {p:.1f}, negatives {n:.1f}, "
              f"balanced {(p + n) / 2:.1f}")
    print(f"% sensitivity at alpha 3: positives {j.loc[('mr', 3.0)][1]:.1f}, "
          f"negatives {j.loc[('mr', 3.0)][0]:.1f}")
    print(f"% mean difference at alpha 32: negatives {m.loc[('mr', 32.0)][0]:.1f}")
    for ctx in ("hi", "en"):
        if ctx in j.index.get_level_values(0):
            sel = j.loc[ctx]
            print(f"% sensitivity {ctx}: best balanced "
                  f"{((sel[1] + sel[0]) / 2).max():.1f}")


if __name__ == "__main__":
    main()

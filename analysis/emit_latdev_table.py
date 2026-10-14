"""Lat--Dev minus Lat--Lat AUROC per template, for the appendix.

Pooling can hide within-template ranking changes, as \\S7 shows, so this is reported per
template rather than pooled.

    python analysis/emit_latdev_table.py > latdev_table.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)
NL = BS + BS
ARMS = [("Marathi", "results_mr", "qwen_mr", "mr"),
        ("English", "results_mr", "qwen_mr", "en"),
        ("Hindi", "results_hi", "qwen_hi", "hi")]


def auc(p, n):
    v = np.concatenate([p, n])
    r = pd.Series(v).rank(method="average").to_numpy()
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def paired(d):
    m = d.groupby(["fact_id", "truth"], as_index=False).yes_minus_no_margin.mean()
    p = m[m.truth == 1].set_index("fact_id").yes_minus_no_margin
    n = m[m.truth == 0].set_index("fact_id").yes_minus_no_margin
    s = p.index.intersection(n.index)
    return p.loc[s].to_numpy(), n.loc[s].to_numpy()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{Lat--Dev minus Lat--Lat AUROC of the yes-minus-no margin, per "
          "question template, held-out entities, item-clustered bootstrap over 2000 "
          "draws. Reported per template rather than pooled because pooling averages each "
          "entity's margin across templates with different distributions and can hide "
          "within-template ranking changes, as "
          + BS + "S" + BS + "ref{sec:repair} finds for the correction vector. A negative "
          "value means Lat--Dev ranks positives against negatives less well than "
          "Lat--Lat does, while Table~" + BS + "ref{tab:behaviour} shows it is the more "
          "accurate condition under English and Hindi prompts.}")
    print(BS + "label{tab:latdev}")
    print(BS + "begin{tabular}{lccc}")
    print(BS + "toprule")
    print("Prompt & template 0 & template 1 & template 2" + NL)
    print(BS + "midrule")
    for label, tree, run, ctx in ARMS:
        c = pd.read_csv(ROOT / tree / run / "data" / "corpus.csv", dtype={"fact_id": str})
        b = pd.read_csv(ROOT / tree / run / "tables" / "behavior.csv",
                        dtype={"fact_id": str})
        test = set(c[c.split == "test"].fact_id)
        cells = []
        for t in (0, 1, 2):
            sel = lambda cond: paired(b[(b.context == ctx) & (b.condition == cond)
                                        & (b.paraphrase_id == t)
                                        & b.fact_id.isin(test)])
            pdv, ndv = sel("LATDEV")
            pl, nl = sel("LATLAT")
            m = min(len(pdv), len(pl))
            rng = np.random.default_rng(20260813 + t)
            draws = [auc(pdv[i], ndv[i]) - auc(pl[i], nl[i])
                     for i in (rng.integers(0, m, m) for _ in range(2000))]
            lo, hi = np.quantile(draws, [0.025, 0.975])
            pt = auc(pdv, ndv) - auc(pl, nl)
            cell = f"${pt:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$"
            if lo > 0 or hi < 0:
                cell = BS + "textbf{" + cell + "}"
            cells.append(cell)
        print(label + " & " + " & ".join(cells) + NL)
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")


if __name__ == "__main__":
    main()

"""Emit the four-arm replication table for the window repair.

Positives alone flatter the Hindi arm, which buys them with 13 points of negative-pair
accuracy. Balanced accuracy prices that in, and is what the table leads with.

    python analysis/emit_replication_table.py > replication_table.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)
NL = BS + BS

ARMS = [
    ("Marathi", "results_mr", "qwen_mr", "mr",
     "results_mr/qwen_mr/recovery_window/vector_recovery_test_summary.csv"),
    ("English, Marathi corpus", "results_mr", "qwen_mr", "en",
     "results_mr/qwen_mr/recovery_window/vector_recovery_test_summary.csv"),
    ("Hindi", "results_hi", "qwen_hi", "hi",
     "results_hi/qwen_hi/hindi_window_test_summary.csv"),
    ("English, Hindi corpus", "results_hi", "qwen_hi", "en",
     "results_hi/qwen_hi/hindi_window_test_summary.csv"),
]


def row(*cells: str) -> str:
    return " & ".join(cells) + NL


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{The window intervention across all four arms, held-out test split, 120 "
          "items each, {" + BS + "em additive} vector applied across layers 12--17 at the "
          "scale validation selected. Balanced accuracy is given because positive-pair "
          "accuracy alone flatters the Hindi arm: it reaches 0.546 against a 0.401 "
          "reference, but pays 13 points of negative-pair accuracy for it, against under "
          "3 in every other arm. Priced that way, all four arms land at or just above "
          "their Lat--Lat reference, which is the replication claim. The scales are "
          "$" + BS + "alpha{=}12$ Marathi, 24 English on the Marathi corpus, 16 Hindi "
          "and 32 English on the Hindi corpus.}")
    print(BS + "label{tab:winreplicate}")
    print(BS + "begin{tabular}{lccc|ccc}")
    print(BS + "toprule")
    print(row("", BS + "multicolumn{3}{c|}{Balanced accuracy}",
              BS + "multicolumn{3}{c}{Positives / negatives, repaired}").rstrip(NL) + NL)
    print(BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}")
    print(row("Arm", "baseline", "repaired", "Lat--Lat",
              "positives", "negatives", "reference"))
    print(BS + "midrule")
    for label, tree, run, ctx, summ in ARMS:
        c = pd.read_csv(ROOT / tree / run / "data" / "corpus.csv", dtype={"fact_id": str})
        b = pd.read_csv(ROOT / tree / run / "tables" / "behavior.csv",
                        dtype={"fact_id": str})
        test = set(c[c.split == "test"].fact_id)
        b = b[b.fact_id.isin(test) & (b.context == ctx)]

        def acc(cond: str, truth: int) -> float:
            s = b[(b.condition == cond) & (b.truth == truth)]
            return s.correct.mean() * 100

        v = pd.read_csv(ROOT / summ)

        def vec(truth: int) -> float:
            s = v[(v.context == ctx) & (v.truth == truth)
                  & (v.vector_kind == "additive")]
            return float(s.accuracy.iloc[0]) * 100

        base = (acc("DEVDEV", 1) + acc("DEVDEV", 0)) / 2
        rep = (vec(1) + vec(0)) / 2
        ceil = (acc("LATLAT", 1) + acc("LATLAT", 0)) / 2
        bold = BS + "textbf{" + f"{rep:.1f}" + "}"
        print(row(label, f"{base:.1f}", bold, f"{ceil:.1f}",
                  f"{vec(1):.1f}", f"{vec(0):.1f}", f"{acc('LATLAT', 1):.1f}"))
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()
    for label, tree, run, ctx, summ in ARMS:
        c = pd.read_csv(ROOT / tree / run / "data" / "corpus.csv", dtype={"fact_id": str})
        b = pd.read_csv(ROOT / tree / run / "tables" / "behavior.csv",
                        dtype={"fact_id": str})
        test = set(c[c.split == "test"].fact_id)
        b = b[b.fact_id.isin(test) & (b.context == ctx)]
        v = pd.read_csv(ROOT / summ)
        p = float(v[(v.context == ctx) & (v.truth == 1)
                    & (v.vector_kind == "additive")].accuracy.iloc[0]) * 100
        n = float(v[(v.context == ctx) & (v.truth == 0)
                    & (v.vector_kind == "additive")].accuracy.iloc[0]) * 100
        dd1 = b[(b.condition == "DEVDEV") & (b.truth == 1)].correct.mean() * 100
        dd0 = b[(b.condition == "DEVDEV") & (b.truth == 0)].correct.mean() * 100
        ll1 = b[(b.condition == "LATLAT") & (b.truth == 1)].correct.mean() * 100
        ll0 = b[(b.condition == "LATLAT") & (b.truth == 0)].correct.mean() * 100
        pos_pct = 100 * (p - dd1) / (ll1 - dd1)
        bal_pct = 100 * ((p + n) / 2 - (dd1 + dd0) / 2) / ((ll1 + ll0) / 2 - (dd1 + dd0) / 2)
        print(f"% {label}: positives {pos_pct:.0f} percent of the gap, "
              f"balanced {bal_pct:.0f} percent, negatives cost {dd0 - n:.1f} points")


if __name__ == "__main__":
    main()

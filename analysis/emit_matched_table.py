"""Emit the LaTeX for the similarity-matched negatives result.

    python analysis/emit_matched_table.py > matched_table.tex
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BS = chr(92)
NL = BS + BS

_spec = importlib.util.spec_from_file_location("sb", HERE / "surface_baseline.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


def classifier(corpus, negmap, script, keep):
    d = sb.pair_frame(corpus, negmap, script)
    d = d[d.fact_id.isin(keep)]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    clf = LogisticRegression(max_iter=2000).fit(tr[sb.FEATURES], tr.y)
    return balanced_accuracy_score(te.y, clf.predict(te[sb.FEATURES])), len(te)


def behaviour(path, keep, ctx, cond):
    d = pd.read_csv(path, dtype={"fact_id": str})
    d = d[d.fact_id.isin(keep) & (d.context == ctx) & (d.condition == cond)]
    pos, neg = d[d.truth == 1], d[d.truth == 0]
    auc = (mannwhitneyu(pos.yes_minus_no_margin, neg.yes_minus_no_margin,
                        alternative="greater").statistic / (len(pos) * len(neg)))
    return pos.correct.mean() * 100, neg.correct.mean() * 100, auc


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = pd.read_csv(ROOT / "results_mr/qwen_mr/data/corpus.csv", dtype={"fact_id": str})
    maps = {
        "shipped": pd.read_csv(ROOT / "results_mr/qwen_mr/data/hard_negative_map.csv",
                               dtype={"fact_id": str, "negative_fact_id": str}),
        "matched": pd.read_csv(ROOT / "results_mr/matched_mr/data/hard_negative_map.csv",
                               dtype={"fact_id": str, "negative_fact_id": str}),
    }
    beh = {"shipped": ROOT / "results_mr/qwen_mr/tables/behavior.csv",
           "matched": ROOT / "results_mr/matched_mr/tables/behavior.csv"}
    q = pd.read_csv(ROOT / "results_mr/matched_mr/corpora/match_quality.csv",
                    dtype={"fact_id": str})
    keep = set(q[q.tight].fact_id)

    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{Similarity-matched negatives, Qwen, on the 144 items where the "
          "matching succeeded to within 0.05 (positive similarity 0.409, negative 0.406). "
          "The classifier is the one of Table~" + BS + "ref{tab:surface}, reading only "
          "the Devanagari strings; on this subset it falls from 0.800 to 0.680, losing "
          "two fifths of its margin over chance, and on the full corpus from 0.933 to "
          "0.796. The model does not move with it: positive-pair accuracy, the script "
          "gap and the threshold-free separation all hold to within 0.01, and Dev--Dev "
          "negatives stay at 100.0 under both maps. Whatever the model is using, it is "
          "not what the classifier lost.}")
    print(BS + "label{tab:matched}")
    print(BS + "begin{tabular}{lccccc}")
    print(BS + "toprule")
    print(" & String & " + BS + "multicolumn{2}{c}{Dev--Dev} & "
          + BS + "multicolumn{2}{c}{Lat--Lat}" + NL)
    print(BS + "cmidrule(lr){3-4}" + BS + "cmidrule(lr){5-6}")
    print("Negatives & classifier & positives & AUC & positives & AUC" + NL)
    print(BS + "midrule")
    for label in ("shipped", "matched"):
        bal, _ = classifier(corpus, maps[label], "dev", keep)
        dd_p, _, dd_a = behaviour(beh[label], keep, "mr", "DEVDEV")
        ll_p, _, ll_a = behaviour(beh[label], keep, "mr", "LATLAT")
        name = ("Occupation and country" if label == "shipped"
                else BS + "textbf{Also string similarity}")
        cells = [name, f"{bal:.3f}", f"{dd_p:.1f}", f"{dd_a:.3f}",
                 f"{ll_p:.1f}", f"{ll_a:.3f}"]
        if label == "matched":
            cells[1] = BS + "textbf{" + cells[1] + "}"
        print(" & ".join(cells) + NL)
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()
    # numbers quoted in the prose, so they cannot be mistyped
    for label in ("shipped", "matched"):
        for script in ("dev", "lat"):
            bal, n = classifier(corpus, maps[label], script, keep)
            print(f"% {label} {script}: classifier {bal:.3f} on n={n}")
        neg = behaviour(beh[label], keep, "mr", "DEVDEV")[1]
        print(f"% {label}: Marathi Dev--Dev negatives {neg:.1f}")


if __name__ == "__main__":
    main()

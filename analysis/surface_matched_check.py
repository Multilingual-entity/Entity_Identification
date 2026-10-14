"""Fit the surface classifier on similarity-matched negatives and see what is left.

surface_baseline.py compares the classifier against the probe and needs probe outputs the
matched run does not have. This runs only the classifier half, with the same five
features and the same train/test protocol, so the two numbers are comparable, and reports
it on the tightly matched subset as well as on everything.

    python analysis/surface_matched_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Import the feature code rather than restating it, so the two cannot diverge.
_spec = importlib.util.spec_from_file_location("sb", HERE / "surface_baseline.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


def fit(corpus: pd.DataFrame, negatives: pd.DataFrame, script: str,
        keep: set[str] | None) -> tuple[float, float, int]:
    d = sb.pair_frame(corpus, negatives, script)
    if keep is not None:
        d = d[d.fact_id.isin(keep)]
    train, test = d[d.split == "train"], d[d.split == "test"]
    if len(test) < 10 or train.y.nunique() < 2:
        return float("nan"), float("nan"), len(test)
    clf = LogisticRegression(max_iter=2000).fit(train[sb.FEATURES], train.y)
    pred = clf.predict(test[sb.FEATURES])
    return (balanced_accuracy_score(test.y, pred),
            roc_auc_score(test.y, clf.predict_proba(test[sb.FEATURES])[:, 1]),
            len(test))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = pd.read_csv(ROOT / "results_mr/qwen_mr/data/corpus.csv", dtype={"fact_id": str})
    shipped = pd.read_csv(ROOT / "results_mr/qwen_mr/data/hard_negative_map.csv",
                          dtype={"fact_id": str, "negative_fact_id": str})
    matched = pd.read_csv(ROOT / "results_mr/matched_mr/data/hard_negative_map.csv",
                          dtype={"fact_id": str, "negative_fact_id": str})
    q = pd.read_csv(ROOT / "results_mr/matched_mr/corpora/match_quality.csv",
                    dtype={"fact_id": str})
    tight = set(q[q.tight].fact_id)

    print("A classifier reading only the two strings, held-out test pairs.")
    print("0.500 balanced accuracy means the strings carry no usable signal.\n")
    print(f"{'negatives':12}{'subset':14}{'script':8}{'bal acc':>9}{'AUC':>8}{'n':>6}")
    for nlabel, nmap in (("shipped", shipped), ("matched", matched)):
        for slabel, keep in (("all 601", None), ("tight 144", tight)):
            for script in ("dev", "lat"):
                bal, auc, n = fit(corpus, nmap, script, keep)
                print(f"{nlabel:12}{slabel:14}{script:8}{bal:>9.3f}{auc:>8.3f}{n:>6}")
    print()
    print("The tight subset is the one the paper should quote: it is where the two")
    print("classes were matched to within 0.05, so a classifier at chance there means")
    print("the surface reading has nothing left to explain the behaviour with.")


if __name__ == "__main__":
    main()

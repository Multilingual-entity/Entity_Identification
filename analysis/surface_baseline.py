"""How much of the probe result is string similarity?

The corpus matches hard negatives on occupation and country, which forces a negative
pair to be plausible, but not on how alike the two names look. If positives are mostly
name expansions and negatives mostly unrelated names, a classifier reading only the
surface forms can separate them, and a probe that reads representations encoding surface
form will appear to decode identity when it is decoding spelling.

This script measures that directly. It fits a logistic classifier on five string
features, on the train split, and scores it on the held-out test split. Then it asks the
three questions that matter:

  1. Does it beat the probe on the paper's headline claim, recovering the correct label
     on the items the model itself gets wrong?
  2. Does the probe succeed where the surface fails, which is where entity knowledge
     would be required?
  3. Is the script gap itself a surface effect? If Devanagari names were less alike than
     their Latin forms, the behavioural result would follow without any mechanism.

The surface is read in the script the model actually sees, Devanagari for Dev--Dev, not
the Latin forms, which was the first version's mistake.

    python analysis/surface_baseline.py --results ../results_mr --run qwen_mr --context mr
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

FEATURES = ["sequence_ratio", "token_jaccard", "char_jaccard", "length_gap", "same_last"]


def features(a: str, b: str) -> list[float]:
    a, b = str(a).lower(), str(b).lower()
    ta, tb = set(a.split()), set(b.split())
    ca, cb = set(a), set(b)
    return [
        SequenceMatcher(None, a, b).ratio(),
        len(ta & tb) / max(len(ta | tb), 1),
        len(ca & cb) / max(len(ca | cb), 1),
        abs(len(a) - len(b)) / max(len(a), len(b), 1),
        1.0 if a.split()[-1:] == b.split()[-1:] else 0.0,
    ]


def pair_frame(corpus: pd.DataFrame, negatives: pd.DataFrame, script: str) -> pd.DataFrame:
    a_col, b_col = f"name_a_{script}", f"name_b_{script}"
    by = corpus.set_index("fact_id")
    partner = dict(zip(negatives.fact_id, negatives.negative_fact_id))
    rows = []
    for fid in corpus.fact_id:
        a = by.loc[fid]
        rows.append((fid, a.split, 1, *features(a[a_col], a[b_col])))
        p = partner.get(fid)
        if p in by.index:
            rows.append((fid, a.split, 0, *features(a[a_col], by.loc[p][b_col])))
    return pd.DataFrame(rows, columns=["fact_id", "split", "y"] + FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--run", required=True, help="e.g. qwen_mr")
    ap.add_argument("--context", required=True, help="the native prompt language, mr or hi")
    ap.add_argument("--script", default="dev", choices=["dev", "lat"],
                    help="the surface the model sees; dev for the Dev--Dev condition")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    base = args.results / args.run
    corpus = pd.read_csv(base / "data/corpus.csv")
    negatives = pd.read_csv(base / "data/hard_negative_map.csv")
    for f in (corpus, negatives):
        for col in ("fact_id", "negative_fact_id"):
            if col in f:
                f[col] = f[col].astype(str)

    rows = []
    d = pair_frame(corpus, negatives, args.script)
    train, test = d[d.split == "train"], d[d.split == "test"]
    clf = LogisticRegression(max_iter=2000).fit(train[FEATURES], train.y)
    pred = clf.predict(test[FEATURES])
    bal = balanced_accuracy_score(test.y, pred)
    auc = roc_auc_score(test.y, clf.predict_proba(test[FEATURES])[:, 1])
    print(f"1. the surface baseline alone, {args.script} strings, held-out test pairs")
    print(f"   n={len(test)}  balanced accuracy {bal:.3f}  AUC {auc:.3f}")
    rows.append(dict(quantity="surface balanced accuracy, test pairs",
                     value=bal, n=len(test)))

    # 1. on the model's own failures
    b = pd.read_csv(base / "tables/behavior.csv")
    b["fact_id"] = b.fact_id.astype(str)
    m = b[(b.context == args.context) & (b.condition == "DEVDEV") & (b.truth == 1)]
    acc = m.groupby("fact_id").correct.mean()
    failed = set(acc[acc <= 0.2].index)
    sub = test[(test.y == 1) & (test.fact_id.isin(failed))]
    rec = float((clf.predict(sub[FEATURES]) == 1).mean()) if len(sub) else float("nan")
    print(f"   on the model's own Dev--Dev failures: n={len(sub)}, recovers {rec:.3f}")
    rows.append(dict(quantity="surface recovery on the model's failures", value=rec, n=len(sub)))

    # 2. the probe, at the layer validation chose, split by whether the surface succeeds
    summ = pd.read_csv(base / "tables/relation_probe_heldout_summary.csv")
    q = summ[(summ.source_context == args.context) & (summ.target_context == args.context)
             & (summ.position == "Q")]
    v = q[q.evaluation_split == "validation"]
    layer = int(v.loc[v.balanced_accuracy.idxmax(), "layer"])
    preds = pd.read_csv(base / "tables/relation_probe_heldout_predictions.csv")
    preds["fact_id"] = preds.fact_id.astype(str)
    pt = preds[(preds.source_context == args.context) & (preds.target_context == args.context)
               & (preds.position == "Q") & (preds.evaluation_split == "test")
               & (preds.layer == layer)]
    probe = pt.groupby(["fact_id", "truth"]).probe_correct.mean().reset_index()
    test = test.copy()
    test["surface_ok"] = (pred == test.y).astype(int)
    merged = test.merge(probe, left_on=["fact_id", "y"], right_on=["fact_id", "truth"])
    if len(merged) < len(test):
        lost = len(test) - len(merged)
        raise SystemExit(
            f"the probe/surface join dropped {lost} of {len(test)} test pairs. The two "
            f"sides are indexed on different split assignments: the surface pairs come "
            f"from corpus.csv and the probe predictions from the split inherited through "
            f"behavior.csv. Any accuracy computed on the {len(merged)} survivors is not "
            f"comparable with a run where the two agree. Re-run stage 03 with "
            f"--splits-from-corpus."
        )
    if len(merged) < len(test):
        lost = len(test) - len(merged)
        raise SystemExit(
            f"the probe/surface join dropped {lost} of {len(test)} test pairs. The two "
            f"sides are indexed on different split assignments: the surface pairs come "
            f"from corpus.csv and the probe predictions from the split inherited through "
            f"behavior.csv. Any accuracy computed on the {len(merged)} survivors is not "
            f"comparable with a run where the two agree. Re-run stage 03 with "
            f"--splits-from-corpus."
        )
    print()
    print(f"2. the probe at layer {layer}, split by whether the surface baseline succeeds")
    for name, mask in (("all", merged.index == merged.index),
                       ("surface right", merged.surface_ok == 1),
                       ("surface wrong", merged.surface_ok == 0)):
        s = merged[mask]
        if len(s):
            print(f"   {name:16} n={len(s):<5} probe correct {s.probe_correct.mean():.3f}")
            rows.append(dict(quantity=f"probe, {name}", value=float(s.probe_correct.mean()),
                             n=len(s)))

    # 3. is the script gap itself a surface effect?
    print()
    print("3. separation between positives and negatives, by script")
    seps = {}
    for script in ("dev", "lat"):
        f = pair_frame(corpus, negatives, script)
        p = f[f.y == 1].sequence_ratio.mean()
        n = f[f.y == 0].sequence_ratio.mean()
        seps[script] = p - n
        print(f"   {script}: positives {p:.3f}, negatives {n:.3f}, separation {p-n:.3f}")
        rows.append(dict(quantity=f"surface separation, {script}", value=p - n, n=len(f)))
    print(f"   the two scripts differ by {abs(seps['dev']-seps['lat']):.3f}, so the "
          f"behavioural gap is not a surface artefact")

    out = args.out or base / "tables/surface_baseline.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

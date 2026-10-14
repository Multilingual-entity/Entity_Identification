"""Can the surface confound be answered on the corpus we already have?

A reviewer asks for two things the current corpus was not built to provide:

  A. similarity-matched negatives, so that a negative pair is as alike on the surface as
     a positive pair, which removes the string cue the Appendix Q classifier exploits;
  B. a low-overlap positive subset, where the two Devanagari forms are not obviously the
     same string, on which the probe and the patching comparison would be repeated.

Neither needs new data. Both are subsets or re-pairings of the audited corpus, so this
script asks only whether enough items survive to support the claims that would rest on
them. It writes nothing and changes no corpus file.

    python analysis/matched_subset_feasibility.py --results ../results_mr --run qwen_mr \
        --context mr
"""
from __future__ import annotations

import argparse
import ast
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


def overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def qids(value) -> set:
    """The qid columns are written as list literals; fall back to a pipe split."""
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return set(str(value).split("|"))
    if isinstance(parsed, (list, set, tuple)):
        return set(parsed)
    return {parsed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--fail-threshold", type=float, default=0.2,
                    help="per-item Dev--Dev SAME accuracy at or below which an item counts "
                         "as a failure, matching the probe analysis")
    ap.add_argument("--match-tolerance", type=float, default=0.05)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    base = args.results / args.run
    corpus = pd.read_csv(base / "data/corpus.csv")
    negmap = pd.read_csv(base / "data/hard_negative_map.csv")
    for frame in (corpus, negmap):
        for col in ("fact_id", "negative_fact_id"):
            if col in frame:
                frame[col] = frame[col].astype(str)

    by = corpus.set_index("fact_id")
    partner = dict(zip(negmap.fact_id, negmap.negative_fact_id))

    pos = {f: overlap(r.name_a_dev, r.name_b_dev) for f, r in by.iterrows()}
    neg = {f: overlap(by.loc[f].name_a_dev, by.loc[p].name_b_dev)
           for f, p in partner.items() if f in by.index and p in by.index}
    P = np.array(list(pos.values()))
    N = np.array(list(neg.values()))

    print("Devanagari surface overlap between the two names in a pair")
    for tag, v in (("positives", P), ("negatives", N)):
        print(f"  {tag:10} n={len(v):4d}  mean {v.mean():.3f}  median {np.median(v):.3f}"
              f"  p10 {np.percentile(v, 10):.3f}  p90 {np.percentile(v, 90):.3f}")
    print(f"  mean separation: {P.mean() - N.mean():+.3f}")

    print()
    print("A. similarity-matched negatives, re-paired within the audited corpus")
    occ = {f: qids(r.occupation_qids) for f, r in by.iterrows()}
    ctry = {f: qids(r.country_qids) for f, r in by.iterrows()}
    name_b = by.name_b_dev.to_dict()
    name_a = by.name_a_dev.to_dict()

    gaps = []
    for f in by.index:
        target = pos[f]
        best = min(
            (abs(overlap(name_a[f], name_b[g]) - target)
             for g in by.index
             if g != f and (occ[f] & occ[g]) and (ctry[f] & ctry[g])),
            default=None)
        if best is not None:
            gaps.append(best)
    G = np.array(gaps)
    within = int((G <= args.match_tolerance).sum()) if len(G) else 0
    print(f"  facts with any occupation+country-matched partner: {len(G)} of {len(by)}")
    if len(G):
        print(f"  best achievable overlap gap: median {np.median(G):.3f}  mean {G.mean():.3f}")
    print(f"  matchable within {args.match_tolerance}: {within} "
          f"({100 * within / max(len(by), 1):.0f}% of the corpus)")

    print()
    print("B. low-overlap positives, where the string cue is weak")
    threshold = float(np.median(N))
    low = {f for f in by.index if pos[f] <= threshold}
    print(f"  positives at or below the median negative overlap ({threshold:.3f}): {len(low)}")

    behaviour = pd.read_csv(base / "tables/behavior.csv")
    behaviour["fact_id"] = behaviour.fact_id.astype(str)
    same = behaviour[(behaviour.context == args.context)
                     & (behaviour.condition == "DEVDEV")
                     & (behaviour.truth == 1)]
    acc = same.groupby("fact_id").correct.mean()
    failed = set(acc[acc <= args.fail_threshold].index)
    print(f"  Dev--Dev SAME failures at accuracy <= {args.fail_threshold}: {len(failed)}")

    splits = by.split.to_dict()
    test = {f for f in by.index if splits.get(f) == "test"}
    print(f"  low-overlap, all splits:  {len(low)}   and also failing: {len(low & failed)}")
    print(f"  low-overlap, test split:  {len(low & test)}   "
          f"and also failing: {len(low & failed & test)}")

    print()
    print("C. what this leaves for each claim")
    print(f"  probe, fitted on train and read on failures: "
          f"{len(low & failed)} items corpus-wide, which is the arm that can carry it")
    print(f"  patching, which needs the causal gate on top of the failure gate: "
          f"at most {len(low & failed & test)} test items before gating, "
          f"against 12 today")


if __name__ == "__main__":
    main()

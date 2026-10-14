"""Does the amount carry item information, or only the direction?

The mediator result holds one direction fixed and lets each item move along it by the
amount its own Latin donor implies. That leaves a question the original design could not
answer: is the per-item amount doing work, or would any comparable amount do?

Three arms share the direction and differ only in the amount:

    matched    this item's own donor-implied coordinate
    shuffled   another item's, drawn within the same group, never its own
    fixed      one coordinate for every item, the mean over the training items

If matched and shuffled agree, the amount is a scale and the paper's claim reduces to
"a fixed direction with a tuned magnitude", which is the steering result. If matched
beats shuffled, the amount carries something about the item.

Margins are reported signed, toward the correct answer. Absolute change cannot
distinguish a uniform push from scattered movement, so it is reported alongside rather
than instead: an arm whose signed mean equals its absolute mean moved every item the
same way.

    python analysis/coefficient_arms.py --items ../mediator_pull/mr_test_/mediator_items.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
DRAWS = 2000


def per_fact(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("fact_id")[column].mean()


def boot(values: np.ndarray, rng, draws: int = DRAWS):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired(a: pd.Series, b: pd.Series, rng):
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return np.nan, np.nan, np.nan, 0
    diff = (a.loc[shared] - b.loc[shared]).to_numpy(float)
    m, lo, hi = boot(diff, rng)
    return m, lo, hi, len(shared)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--method", default="jacobian")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    for path in args.items:
        d = pd.read_csv(path)
        d["fact_id"] = d.fact_id.astype(str)
        if "coefficient" not in d.columns:
            print(f"{path}: no coefficient column, this file predates the arms")
            continue
        d["fold"] = d.get("fold", pd.Series([""] * len(d))).fillna("")

        for context in sorted(d.context.unique()):
            for fold in sorted(d[d.context == context].fold.unique()):
                here = d[(d.context == context) & (d.layer == args.layer)
                         & (d.fold == fold)]
                if here.empty:
                    continue
                sub = here[(here.method == args.method) & (here["rank"] == args.rank)]
                full = here[here.method == "fullpatch"]
                if sub.empty:
                    continue
                print()
                print("=" * 94)
                print(f"{path.parent.name}  {context}  {args.method}  layer {args.layer}"
                      f"  rank {args.rank}  fold {fold}"
                      f"  split {', '.join(sorted(here.split.unique()))}")
                print("=" * 94)
                print(f"{'arm':10}{'gated (signed)':>26}{'|abs|':>8}"
                      f"{'never':>9}{'negatives':>11}{'neg acc':>9}")

                rows = {}
                for arm in ("matched", "shuffled", "fixed"):
                    a = sub[sub.coefficient == arm]
                    if a.empty:
                        continue
                    g = a[a.group == "gated"]
                    rng = np.random.default_rng(args.seed)
                    rep = per_fact(g, "delta_mediated")
                    m, lo, hi = boot(rep.to_numpy(float), rng)
                    rows[arm] = rep
                    # Every column aggregates to one value per entity first. Mixing a
                    # per-entity numerator with a per-row denominator would weight
                    # entities by how many prompts survived the gate.
                    def cell(group: str, column: str) -> float:
                        part = a[a.group == group]
                        return per_fact(part, column).mean() if len(part) else np.nan

                    print(f"{arm:10}{f'{m:>8.2f} [{lo:+.2f},{hi:+.2f}]':>26}"
                          f"{per_fact(g, 'abs_change').mean():>8.2f}"
                          f"{cell('never', 'delta_mediated'):>9.2f}"
                          f"{cell('negatives', 'delta_mediated'):>11.2f}"
                          f"{cell('negatives', 'mediated_correct'):>9.2f}")

                if not full.empty:
                    fg = per_fact(full[full.group == "gated"], "delta_full")
                    rng = np.random.default_rng(args.seed)
                    m, lo, hi = boot(fg.to_numpy(float), rng)
                    print(f"{'whole':10}{f'{m:>8.2f} [{lo:+.2f},{hi:+.2f}]':>26}")
                    rows["whole"] = fg

                print()
                print("  the comparisons that decide it, paired on shared entities:")
                for a, b in (("matched", "shuffled"), ("matched", "fixed"),
                             ("shuffled", "fixed"), ("matched", "whole")):
                    if a not in rows or b not in rows:
                        continue
                    rng = np.random.default_rng(args.seed)
                    m, lo, hi, n = paired(rows[a], rows[b], rng)
                    verdict = "differ" if (lo > 0 or hi < 0) else "not separated"
                    print(f"    {a:9} minus {b:9} n={n:>3}  "
                          f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]   {verdict}")

                if "matched" in rows and "shuffled" in rows:
                    rng = np.random.default_rng(args.seed)
                    _, lo, hi, _ = paired(rows["matched"], rows["shuffled"], rng)
                    print()
                    if lo > 0:
                        print("  The amount carries item information: giving an item another "
                              "item's\n  coordinate along the same direction repairs less.")
                    elif lo <= 0 <= hi:
                        print("  The amount does not carry item information at this size. The "
                              "result is\n  a fixed direction with a scale, which is the "
                              "steering claim, not a mediator.")


if __name__ == "__main__":
    main()

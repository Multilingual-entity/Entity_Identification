"""What the subspace results support, stated as selectivity rather than raw closure.

Raw closure rewards a method for pushing every item toward yes. The mean-difference
direction does exactly that: it closes more of the gated gap than the whole patch does,
while moving items the patch is not supposed to touch almost as much. Closure alone
cannot tell that apart from a mediator.

The quantity that can is the ratio of repair on gated items to disturbance on items that
were already right or were never right. The whole patch provides the reference: a
subspace that is more selective than the patch it approximates is doing something the
patch does not.

    python analysis/mediator_selectivity.py --items ../mediator_pull/mr_test_/mediator_items.csv
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


def ratio_boot(repair: pd.Series, disturb: pd.Series, rng, draws: int = DRAWS):
    """Interval on repair/disturbance. The two groups are disjoint entities, so they are
    resampled independently; the ratio is formed inside each draw."""
    r, d = repair.to_numpy(float), disturb.to_numpy(float)
    if len(r) == 0 or len(d) == 0:
        return (np.nan,) * 3
    ri = rng.integers(0, len(r), size=(draws, len(r))).astype(np.intp)
    di = rng.integers(0, len(d), size=(draws, len(d))).astype(np.intp)
    num, den = r[ri].mean(axis=1), d[di].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(np.abs(den) > 1e-9, num / den, np.nan)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan,) * 3
    point = float(r.mean() / d.mean()) if abs(d.mean()) > 1e-9 else np.nan
    return point, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def inventory(path: Path) -> None:
    d = pd.read_csv(path)
    d["fold"] = d.get("fold", pd.Series([""] * len(d))).fillna("")
    print(f"\n{path.parent.name}/{path.name}  rows={len(d)}")
    for col in ("context", "split", "fold", "layer", "rank", "method"):
        if col in d:
            vals = sorted(map(str, d[col].unique()))
            print(f"   {col:9} {', '.join(vals[:14])}{' ...' if len(vals) > 14 else ''}")
    gated = d[d.group == "gated"]
    if len(gated):
        print(f"   gated entities per context: "
              f"{gated.groupby('context').fact_id.nunique().to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--control", default="never", choices=["never", "robust", "negatives"])
    ap.add_argument("--coefficient", default="matched",
                    choices=["matched", "shuffled", "fixed"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("=" * 88)
    print("INVENTORY")
    print("=" * 88)
    for p in args.items:
        inventory(p)

    for p in args.items:
        d = pd.read_csv(p)
        d["fact_id"] = d.fact_id.astype(str)
        d["fold"] = d.get("fold", pd.Series([""] * len(d))).fillna("")
        # Arms differ only in how far each item moves along the same direction. Averaging
        # them would blend an intervention with the controls that exist to bound it.
        if "coefficient" in d.columns:
            arms = sorted(d.coefficient.dropna().unique())
            d = d[d.coefficient == args.coefficient]
            print(f"\n{p.parent.name}: coefficient arms present {arms}, "
                  f"reporting {args.coefficient!r}")
            if d.empty:
                print(f"  no rows for the {args.coefficient!r} arm, skipping")
                continue
        for context in sorted(d.context.unique()):
            for fold in sorted(d[d.context == context].fold.unique()):
                here = d[(d.context == context) & (d.layer == args.layer)
                         & (d.fold == fold)]
                if here.empty:
                    continue
                split = ", ".join(sorted(here.split.unique()))
                print()
                print("=" * 88)
                print(f"{p.parent.name}  {context}  layer {args.layer}  rank {args.rank}"
                      f"  fold {fold}  split {split}")
                print("=" * 88)
                print(f"{'method':11}{'gated repair':>16}{'disturbance':>14}"
                      f"{'selectivity (95% CI)':>30}")

                order = ["fullpatch"] + sorted(
                    m for m in here.method.unique() if m != "fullpatch")
                for method in order:
                    rank = 0 if method == "fullpatch" else args.rank
                    sub = here[(here.method == method) & (here["rank"] == rank)]
                    if sub.empty:
                        continue
                    g = sub[sub.group == "gated"]
                    c = sub[sub.group == args.control]
                    if g.empty or c.empty:
                        continue
                    rng = np.random.default_rng(args.seed)
                    rep = per_fact(g, "delta_mediated")
                    dis = per_fact(c, "abs_change")
                    pt, lo, hi = ratio_boot(rep, dis, rng)
                    star = ""
                    if np.isfinite(lo) and lo > 1.0:
                        star = "  repairs more than it disturbs"
                    print(f"{method:11}{rep.mean():>10.2f} (n={len(rep):>2}){dis.mean():>10.2f}"
                          f"{f'{pt:>14.2f} [{lo:.2f},{hi:.2f}]':>30}{star}")


if __name__ == "__main__":
    main()

"""Item-clustered intervals for the subspace results, and the contrast the claim rests on.

The finding is not a single number, it is a comparison: two directions reach almost the
same repair on gated items while disturbing the control groups very differently. A
comparison needs an interval on the difference, not two intervals side by side, so the
contrasts here are computed paired on the entities present in both arms and resampled
once.

Every quantity is aggregated to one value per fact_id before any resampling. The six
prompts of one entity are not independent, so resampling rows would be pseudoreplication
and would produce intervals that are too narrow.

    python analysis/mediator_bootstrap.py --items ../results_mr/qwen_mr/mediator/mediator_items.csv
    python analysis/mediator_bootstrap.py --items ... --layer 17 --rank 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
DRAWS = 2000


def per_fact(frame: pd.DataFrame, column: str) -> pd.Series:
    """One value per entity, averaging over paraphrases and answer keys."""
    return frame.groupby("fact_id")[column].mean()


def boot(values: np.ndarray, rng: np.random.Generator, draws: int = DRAWS):
    """Mean and a 95 percent interval, resampling entities with replacement."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired(a: pd.Series, b: pd.Series, rng: np.random.Generator):
    """Interval on a minus b over the entities present in both."""
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return np.nan, np.nan, np.nan, 0
    diff = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=float)
    m, lo, hi = boot(diff, rng)
    return m, lo, hi, len(shared)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: every layer present")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--contexts", nargs="+", default=None)
    ap.add_argument("--fold", default=None,
                    help="one leave-one-template-out fold, e.g. t12e0. Default: report "
                         "each fold present separately, never pooled.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--coefficient", default="matched",
                    choices=["matched", "shuffled", "fixed"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    items = pd.read_csv(args.items)
    items["fact_id"] = items.fact_id.astype(str)
    # Arms differ only in how far each item moves along the same direction. Pooling them
    # would average an intervention with its own control.
    if "coefficient" in items.columns:
        arms = sorted(items.coefficient.dropna().unique())
        items = items[items.coefficient == args.coefficient]
        print(f"coefficient arms present: {arms}. Reporting {args.coefficient!r}.")
        if items.empty:
            raise SystemExit(f"no rows for the {args.coefficient!r} arm")
    # A file written before folds were recorded has one unnamed fold.
    if "fold" not in items.columns:
        items["fold"] = ""
    contexts = args.contexts or sorted(items.context.unique())
    layers = [args.layer] if args.layer is not None else sorted(items.layer.unique())
    folds = [args.fold] if args.fold else sorted(items.fold.fillna("").unique())
    if len(folds) > 1:
        print(f"{len(folds)} folds present: {folds}. Reported separately; pooling them "
              f"would average directions fitted on different templates.")
    rows = []

    for context in contexts:
      for fold in folds:
        for layer in layers:
            here = items[(items.context == context) & (items.layer == layer)
                         & (items.fold.fillna("") == fold)]
            if here.empty:
                continue
            # fullpatch is stored at rank 0; every subspace method at the asked-for rank.
            full = here[here.method == "fullpatch"]
            methods = [m for m in here.method.unique() if m != "fullpatch"]
            print()
            print("=" * 92)
            print(f"{context}, layer {layer}, rank {args.rank}"
                  + (f", fold {fold}" if fold else ""))
            print("=" * 92)
            print(f"{'method':11}{'group':11}{'n':>4}{'repair or disturbance':>24}"
                  f"{'   vs the whole patch, paired':>32}")

            for method in sorted(methods):
                sub = here[(here.method == method) & (here["rank"] == args.rank)]
                if sub.empty:
                    continue
                rng = np.random.default_rng(args.seed)
                for group in ("gated", "negatives", "robust", "never"):
                    g = sub[sub.group == group]
                    f = full[full.group == group]
                    if g.empty:
                        continue
                    # gated is the repair, measured toward the correct answer.
                    # the control groups are disturbance, so magnitude is what matters.
                    col = "delta_mediated" if group == "gated" else "abs_change"
                    fcol = "delta_full" if group == "gated" else "abs_change"
                    mine = per_fact(g, col)
                    theirs = per_fact(f, fcol) if not f.empty else pd.Series(dtype=float)
                    m, lo, hi = boot(mine.to_numpy(dtype=float), rng)
                    dm, dlo, dhi, n = paired(mine, theirs, rng)
                    excl = "" if np.isnan(dlo) else ("  excludes 0" if (dlo > 0) == (dhi > 0) else "")
                    print(f"{method:11}{group:11}{len(mine):>4}"
                          f"{m:>10.2f} [{lo:+.2f},{hi:+.2f}]"
                          f"{dm:>12.2f} [{dlo:+.2f},{dhi:+.2f}]{excl}")
                    rows.append(dict(context=context, fold=fold, layer=layer,
                                     rank=args.rank, method=method, group=group,
                                     n_items=len(mine), mean=m, ci_low=lo, ci_high=hi,
                                     vs_full=dm, vs_full_low=dlo, vs_full_high=dhi))

            # The contrast the claim rests on: two directions, same repair, different
            # disturbance. Paired on the entities both arms cover.
            if {"jacobian", "meandiff"} <= set(methods):
                print()
                print("  the contrast the claim rests on, Jacobian minus mean difference:")
                rng = np.random.default_rng(args.seed)
                for group, col in (("gated", "delta_mediated"), ("never", "abs_change"),
                                   ("robust", "abs_change"), ("negatives", "abs_change")):
                    j = here[(here.method == "jacobian") & (here["rank"] == args.rank)
                             & (here.group == group)]
                    d = here[(here.method == "meandiff") & (here["rank"] == args.rank)
                             & (here.group == group)]
                    if j.empty or d.empty:
                        continue
                    m, lo, hi, n = paired(per_fact(j, col), per_fact(d, col), rng)
                    verdict = ("the two differ" if (lo > 0) == (hi > 0) else "includes zero")
                    print(f"    {group:11} n={n:<4}{m:+7.2f} [{lo:+.2f}, {hi:+.2f}]   {verdict}")
                    rows.append(dict(context=context, fold=fold, layer=layer,
                                     rank=args.rank, method="jacobian_minus_meandiff",
                                     group=group, n_items=n, mean=m, ci_low=lo, ci_high=hi,
                                     vs_full=np.nan, vs_full_low=np.nan,
                                     vs_full_high=np.nan))

    if rows:
        out = args.out or args.items.with_name("mediator_bootstrap.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print()
        print(f"Saved: {out}")
        print(f"{DRAWS} draws, seed {args.seed}, resampling entities after averaging "
              f"each entity's prompts.")


if __name__ == "__main__":
    main()

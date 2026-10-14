"""What the coefficient arms establish, across every fold rather than one at a time.

Three questions, in the order they matter:

  1. Does an item's own amount beat a stranger's?  matched minus shuffled.
  2. Does an item-derived amount beat a constant one, on repair?  matched minus fixed.
  3. What does a constant amount cost?  disturbance on items the intervention is not
     meant to help, per unit of repair.

Folds are leave-one-template-out and are never pooled into a single interval, because
the directions are fitted on different templates. They are reported side by side, and
the summary counts how many folds agree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
DRAWS = 2000


def per_fact(f: pd.DataFrame, col: str) -> pd.Series:
    return f.groupby("fact_id")[col].mean()


def boot(v: np.ndarray, rng, draws: int = DRAWS):
    if len(v) == 0:
        return (np.nan,) * 3
    idx = rng.integers(0, len(v), size=(draws, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired(a: pd.Series, b: pd.Series, rng):
    sh = a.index.intersection(b.index)
    if len(sh) == 0:
        return np.nan, np.nan, np.nan, 0
    d = (a.loc[sh] - b.loc[sh]).to_numpy(float)
    m, lo, hi = boot(d, rng)
    return m, lo, hi, len(sh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--method", default="jacobian")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    frames = []
    for p in args.items:
        d = pd.read_csv(p)
        d["fact_id"] = d.fact_id.astype(str)
        d["source"] = p.parent.name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.layer == args.layer)]

    print(f"method {args.method}, layer {args.layer}, rank {args.rank}, "
          f"{d.split.unique()} split")
    print()
    print(f"{'ctx':4}{'fold':8}{'arm':10}{'gated':>17}{'never':>9}{'neg':>8}"
          f"{'never/gated':>13}")
    cells, contrasts = {}, []
    for ctx in sorted(d.context.unique()):
        for fold in sorted(d[d.context == ctx].fold.unique()):
            here = d[(d.context == ctx) & (d.fold == fold)]
            sub = here[(here.method == args.method) & (here["rank"] == args.rank)]
            if sub.empty:
                continue
            for arm in ("matched", "shuffled", "fixed"):
                a = sub[sub.coefficient == arm]
                if a.empty:
                    continue
                rng = np.random.default_rng(args.seed)
                g = per_fact(a[a.group == "gated"], "delta_mediated")
                m, lo, hi = boot(g.to_numpy(float), rng)
                # Aggregated per entity like the numerator, so the ratio compares like
                # with like rather than weighting entities by surviving prompt count.
                def cell(group: str) -> float:
                    part = a[a.group == group]
                    return per_fact(part, "delta_mediated").mean() if len(part) else np.nan

                nev = cell("never")
                neg = cell("negatives")
                ratio = nev / m if abs(m) > 1e-9 else np.nan
                cells[(ctx, fold, arm)] = (g, nev, neg, ratio)
                print(f"{ctx:4}{fold:8}{arm:10}"
                      f"{f'{m:>6.2f} [{lo:+.2f},{hi:+.2f}]':>17}"
                      f"{nev:>9.2f}{neg:>8.2f}{ratio:>13.2f}")
            print()

    print("=" * 78)
    print("1. does an item's own amount beat a stranger's?")
    print("=" * 78)
    for (ctx, fold, arm), _ in sorted(cells.items()):
        if arm != "matched" or (ctx, fold, "shuffled") not in cells:
            continue
        rng = np.random.default_rng(args.seed)
        m, lo, hi, n = paired(cells[(ctx, fold, "matched")][0],
                              cells[(ctx, fold, "shuffled")][0], rng)
        sep = lo > 0 or hi < 0
        contrasts.append(sep)
        print(f"  {ctx:4}{fold:8} n={n:>3}  {m:+.2f} [{lo:+.2f},{hi:+.2f}]"
              f"   {'differ' if sep else 'not separated'}")
    print(f"\n  folds where the item's own amount wins: {sum(contrasts)} of {len(contrasts)}")

    print()
    print("=" * 78)
    print("2. what a constant amount costs, per unit of repair")
    print("=" * 78)
    print(f"  {'ctx':4}{'fold':8}{'matched never/gated':>22}{'fixed never/gated':>20}")
    worse = []
    for ctx in sorted(d.context.unique()):
        for fold in sorted(d[d.context == ctx].fold.unique()):
            if (ctx, fold, "matched") not in cells or (ctx, fold, "fixed") not in cells:
                continue
            rm = cells[(ctx, fold, "matched")][3]
            rf = cells[(ctx, fold, "fixed")][3]
            worse.append(rf > rm)
            print(f"  {ctx:4}{fold:8}{rm:>22.2f}{rf:>20.2f}")
    print(f"\n  folds where a constant amount disturbs relatively more: "
          f"{sum(worse)} of {len(worse)}")


if __name__ == "__main__":
    main()

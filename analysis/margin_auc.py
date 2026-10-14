"""Threshold-free sensitivity: AUC of the yes-minus-no margin, per condition.

d' is computed from hit and false-alarm *rates*, and Qwen's false-alarm rate sits on the
floor: 100.0 against 99.6 percent in English is about 1 against 14 false alarms in 3606
trials. After the log-linear correction the z(FA) term dominates, so a d' gap there is
reading the floor rather than sensitivity. The margins are continuous and already
stored, so the same question can be asked without a threshold.

AUC here is the probability that a randomly drawn same-person item has a higher
correctness-oriented margin than a randomly drawn different-person item, computed within
each prompt language and script condition. The script gap is Lat--Lat minus Dev--Dev,
with a bootstrap over items rather than trials, since six prompts of one entity are not
independent.

This writes a new file and touches no existing result.

    python analysis/margin_auc.py --results ../results_mr --run qwen_mr --contexts mr en
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
DRAWS = 2000


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U as a probability, ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    vals = np.concatenate([pos, neg])
    # average ranks within ties
    frame = pd.DataFrame({"v": vals, "r": ranks})
    ranks = frame.groupby("v").r.transform("mean").to_numpy()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def item_means(d: pd.DataFrame) -> pd.DataFrame:
    """One margin per (item, truth), so the bootstrap resamples entities."""
    return (d.groupby(["fact_id", "truth"], as_index=False)
             .yes_minus_no_margin.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--contexts", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    b = pd.read_csv(args.results / args.run / "tables" / "behavior.csv")
    rows = []
    print(f"{args.run}: AUC of the margin, same-person against different-person\n")
    print(f"{'ctx':5}{'condition':11}{'AUC':>8}{'95% CI':>18}{'items':>8}")
    for ctx in args.contexts:
        store = {}
        for cond in ("DEVDEV", "DEVLAT", "LATDEV", "LATLAT"):
            sub = item_means(b[(b.context == ctx) & (b.condition == cond)])
            pos = sub[sub.truth == 1].set_index("fact_id").yes_minus_no_margin
            neg = sub[sub.truth == 0].set_index("fact_id").yes_minus_no_margin
            shared = pos.index.intersection(neg.index)
            pos, neg = pos.loc[shared], neg.loc[shared]
            point = auc(pos.to_numpy(), neg.to_numpy())
            rng = np.random.default_rng(args.seed)
            draws = []
            n = len(shared)
            for _ in range(DRAWS):
                idx = rng.integers(0, n, n)
                draws.append(auc(pos.to_numpy()[idx], neg.to_numpy()[idx]))
            lo, hi = np.quantile(draws, [0.025, 0.975])
            store[cond] = (pos.to_numpy(), neg.to_numpy())
            print(f"{ctx:5}{cond:11}{point:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}{n:>8}")
            rows.append(dict(context=ctx, condition=cond, auc=point,
                             ci_low=lo, ci_high=hi, n_items=n))

        # the script gap, resampled jointly so the two conditions share entities
        p1, n1 = store["LATLAT"]
        p0, n0 = store["DEVDEV"]
        m = min(len(p1), len(p0))
        rng = np.random.default_rng(args.seed + 1)
        diffs = []
        for _ in range(DRAWS):
            idx = rng.integers(0, m, m)
            diffs.append(auc(p1[idx], n1[idx]) - auc(p0[idx], n0[idx]))
        gap = auc(p1, n1) - auc(p0, n0)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        sig = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
        print(f"{ctx:5}{'gap LL-DD':11}{gap:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{m:>8}   {sig}")
        rows.append(dict(context=ctx, condition="gap_LATLAT_minus_DEVDEV", auc=gap,
                         ci_low=lo, ci_high=hi, n_items=m))
        print()

    out = args.out or (args.results / args.run / "tables" / "margin_auc.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()

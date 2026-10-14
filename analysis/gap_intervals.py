"""Paired item-clustered intervals on the script disparity, before and after treatment.

The review asks for intervals and balanced-accuracy gaps beside the positive-accuracy
gaps, because the point estimates alone cannot say whether the narrowing in three arms is
real. The disparity is Lat--Lat minus Dev--Dev on the same held-out entities, so the two
conditions are paired by entity and the bootstrap resamples entities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SEED = 20260813
DRAWS = 2000

ARMS = [
    ("Marathi", "results_mr", "qwen_mr", "mr",
     "results_mr/qwen_mr/recovery_window/vector_recovery_test.csv",
     "results_mr/qwen_mr/ctrl_latlat_summary.csv"),
    ("English, mr corpus", "results_mr", "qwen_mr", "en",
     "results_mr/qwen_mr/recovery_window/vector_recovery_test.csv",
     "results_mr/qwen_mr/ctrl_latlat_summary.csv"),
    ("Hindi", "results_hi", "qwen_hi", "hi",
     "results_hi/qwen_hi/hindi_window_test.csv",
     "results_mr/qwen_hi/hi_ctrl_latlat_hi.csv"),
    ("English, hi corpus", "results_hi", "qwen_hi", "en",
     "results_hi/qwen_hi/hindi_window_test.csv",
     "results_mr/qwen_hi/hi_ctrl_latlat_en.csv"),
]

OUT = []


def arm_rows(tree, run, ctx):
    c = pd.read_csv(ROOT / tree / run / "data/corpus.csv", dtype={"fact_id": str})
    b = pd.read_csv(ROOT / tree / run / "tables/behavior.csv", dtype={"fact_id": str})
    test = set(c[c.split == "test"].fact_id)
    b = b[b.fact_id.isin(test) & (b.context == ctx)]
    return b


def paired_gap(b, cond_hi, cond_lo, truth):
    """Per-entity accuracy difference between two conditions, for one truth value."""
    d = b[(b.truth == truth) & b.condition.isin([cond_hi, cond_lo])]
    g = d.groupby(["fact_id", "condition"]).correct.mean().unstack()
    g = g.dropna()
    return g[cond_hi].to_numpy() - g[cond_lo].to_numpy()


def boot(values, draws=DRAWS):
    rng = np.random.default_rng(SEED)
    n = len(values)
    point = float(values.mean())
    vals = [values[rng.integers(0, n, n)].mean() for _ in range(draws)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point * 100, lo * 100, hi * 100


print("Untreated script disparity, Lat--Lat minus Dev--Dev, paired by entity")
print("item-clustered bootstrap, 2000 draws, seed 20260813\n")
print(f"{'arm':22}{'positives':>24}{'negatives':>24}")
for label, tree, run, ctx, _, _ in ARMS:
    b = arm_rows(tree, run, ctx)
    cells = []
    for truth in (1, 0):
        v = paired_gap(b, "LATLAT", "DEVDEV", truth)
        p, lo, hi = boot(v)
        cells.append(f"{p:+6.1f} [{lo:+5.1f},{hi:+5.1f}]")
        OUT.append(dict(arm=label, quantity=f"untreated gap truth={truth}",
                        value=p, lo=lo, hi=hi, n=len(v)))
    print(f"{label:22}{cells[0]:>24}{cells[1]:>24}")

print("\nBalanced-accuracy disparity, untreated")
print(f"{'arm':22}{'gap':>24}")
for label, tree, run, ctx, _, _ in ARMS:
    b = arm_rows(tree, run, ctx)
    pos = paired_gap(b, "LATLAT", "DEVDEV", 1)
    neg = paired_gap(b, "LATLAT", "DEVDEV", 0)
    n = min(len(pos), len(neg))
    bal = (pos[:n] + neg[:n]) / 2
    p, lo, hi = boot(bal)
    print(f"{label:22}{f'{p:+6.1f} [{lo:+5.1f},{hi:+5.1f}]':>24}")
    OUT.append(dict(arm=label, quantity="untreated balanced gap",
                    value=p, lo=lo, hi=hi, n=n))

pd.DataFrame(OUT).to_csv(ROOT / "results_mr" / "gap_intervals.csv", index=False)
print(f"\nsaved {ROOT / 'results_mr' / 'gap_intervals.csv'}")
print("\nNote: the treated gap pairs a Dev--Dev run against a Lat--Lat control run, which")
print("are separate evaluations rather than two conditions of one run, so a paired")
print("interval on the treated gap is not available from these files.")

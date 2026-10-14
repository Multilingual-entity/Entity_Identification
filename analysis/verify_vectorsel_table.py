"""Recompute tab:vectorsel, the single-layer vector selection, from each run's test summary.

The Positives and Negatives columns are held-out test accuracies for the {\\em additive}
variant, not the validation figures in vector_hyperparameters_selected.csv. Those two differ
(Qwen English is 0.350 on test and 0.367 on validation), which is worth asserting so the
columns are not later read as the selection numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2] / "results_mr"

# (run, context): (layer, alpha, positives, negatives) exactly as the paper prints them
WANT = {
    ("qwen_mr", "mr"): (16, 6.0, 0.083, 0.993),
    ("qwen_mr", "en"): (12, 8.0, 0.350, 1.000),
    ("gemma_mr", "mr"): (10, 0.5, 0.925, 0.246),
    ("gemma_mr", "en"): (12, 3.0, 0.979, 0.231),
    ("llama_mr", "mr"): (11, 8.0, 0.354, 0.661),
    ("llama_mr", "en"): (21, 4.0, 0.478, 0.849),
}

failed: list[str] = []
skipped: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


for (run, ctx), (layer, alpha, wp, wn) in WANT.items():
    summary = ROOT / run / "recovery" / "vector_recovery_test_summary.csv"
    selected = ROOT / run / "recovery" / "vector_hyperparameters_selected.csv"
    if not summary.exists():
        skipped.append(f"{run}/{ctx}")
        print(f"  --    {run} {ctx}: no local recovery directory, not checked")
        continue
    a = pd.read_csv(summary)
    a = a[a.vector_kind == "additive"]
    pos = float(a[(a.context == ctx) & (a.truth == 1)].accuracy.iloc[0])
    neg = float(a[(a.context == ctx) & (a.truth == 0)].accuracy.iloc[0])
    check(f"tab:vectorsel {run} {ctx} = {wp}/{wn}",
          abs(pos - wp) < 0.0006 and abs(neg - wn) < 0.0006,
          f"{pos:.3f}/{neg:.3f}")

    s = pd.read_csv(selected)
    row = s[s.context == ctx]
    if len(row):
        check(f"tab:vectorsel {run} {ctx} layer {layer}, alpha {alpha}",
              int(row.layer.iloc[0]) == layer and abs(float(row.alpha.iloc[0]) - alpha) < 1e-9,
              f"layer {int(row.layer.iloc[0])}, alpha {float(row.alpha.iloc[0])}")

# the distinction that caused a false alarm when auditing this table
s = pd.read_csv(ROOT / "qwen_mr" / "recovery" / "vector_hyperparameters_selected.csv")
val = float(s[s.context == "en"].same_accuracy.iloc[0])
check("the table's 0.350 is the test figure, not validation's 0.367",
      abs(val - 0.3667) < 0.001, f"validation same_accuracy {val:.4f}")

print()
if skipped:
    print(f"not checked ({len(skipped)}): {', '.join(skipped)} "
          f"- the Hindi-corpus rows need those recovery directories pulled from the server")
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every locally checkable tab:vectorsel number matches the data")

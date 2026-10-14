"""Recompute tab:windows, which paper_tables.py declines and nothing has checked.

Its documented convention is pooled fact-level rows, truth=1, Lat--Lat donor. The E1+E2+Q
row reproduces for Marathi but not for English, and the margins in that row do not
reproduce for either, so every row is recomputed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ROWS = [
    ("Layers 9--11", "window_factorial", "9_11", "E2"),
    ("Layers 12--14", "window_factorial", "12_14", "E2"),
    ("Layers 15--17", "window_factorial", "15_17", "E2"),
    ("Layers 12--17", "window_factorial", "12_17", "E2"),
    ("Layers 18--20", "window_factorial", "18_20", "E2"),
    ("E1+E2, 12--17", "joint_12_17", "12_17", "E1+E2"),
    ("E2+Q, 12--17", "joint_12_17", "12_17", "E2+Q"),
    ("E1+E2+Q, 12--17", "joint_12_17", "12_17", "E1+E2+Q"),
]

STATED = {
    "Layers 9--11": ((0.77, 0.00), (0.04, 0.07)),
    "Layers 12--14": ((4.04, 0.04), (3.10, 0.22)),
    "Layers 15--17": ((9.08, 0.30), (6.91, 0.50)),
    "Layers 12--17": ((11.70, 0.43), (9.05, 0.64)),
    "Layers 18--20": ((0.56, 0.00), (0.28, 0.00)),
    "E1+E2, 12--17": ((12.23, 0.51), (9.99, 0.69)),
    "E2+Q, 12--17": ((12.29, 0.49), (10.96, 0.77)),
    "E1+E2+Q, 12--17": ((12.80, 0.57), (11.85, 0.77)),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_csv(ROOT / "results_mr/qwen_mr/patching/causal_patching.csv")
    d = d[(d.truth == 1) & (d.source_condition == "LATLAT")]
    print(f"{'row':20}{'Marathi paper':>16}{'Marathi data':>16}"
          f"{'English paper':>16}{'English data':>16}")
    bad = 0
    for label, exp, window, sites in ROWS:
        cells = []
        for k, ctx in enumerate(("mr", "en")):
            s = d[(d.experiment == exp) & (d.context == ctx)
                  & (d.layer_or_window.astype(str) == window)]
            if sites in ("E1+E2", "E2+Q", "E1+E2+Q"):
                s = s[d.site == sites] if "site" in s and sites in set(s.site) else s
            if s.empty:
                cells.append("--")
                continue
            m = s.groupby("fact_id").delta_margin.mean().mean()
            f = s.groupby("fact_id").patched_correct.max().mean()
            stated = STATED[label][k]
            ok = abs(m - stated[0]) < 0.02 and abs(f - stated[1]) < 0.006
            bad += not ok
            cells.append(f"{stated[0]:.2f} ({stated[1]:.2f})|{m:.2f} ({f:.2f})"
                         + ("" if ok else "  X"))
        print(f"{label:20}" + "   ".join(cells))
    print()
    print("sites present for joint_12_17:",
          sorted(d[d.experiment == "joint_12_17"].site.unique()))
    print(f"\n{bad} cell(s) disagree" if bad else "\nall rows reproduce")


if __name__ == "__main__":
    main()

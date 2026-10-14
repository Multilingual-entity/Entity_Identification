"""Recompute tab:probee2 from the probe summary that includes the second entity."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (ROOT / "results_mr" / "qwen_mr" / "tables"
           / "relation_probe_heldout_summary_e2.csv")

AT16 = {("mr", "E2"): 0.942, ("mr", "Q"): 0.925, ("mr", "READOUT"): 0.838,
        ("en", "E2"): 0.954, ("en", "Q"): 0.933, ("en", "READOUT"): 0.879}
BEST = {("mr", "E2"): (0.954, 17), ("mr", "Q"): (0.942, 6), ("mr", "READOUT"): (0.925, 7),
        ("en", "E2"): (0.958, 14), ("en", "Q"): (0.950, 18), ("en", "READOUT"): (0.950, 20)}

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


d = pd.read_csv(SUMMARY)
d = d[(d.evaluation_split == "test") & (d.source_context == d.target_context)]

for (ctx, pos), want in AT16.items():
    row = d[(d.source_context == ctx) & (d.position == pos) & (d.layer == 16)]
    got = float(row.balanced_accuracy.iloc[0])
    check(f"tab:probee2 {ctx} {pos} at layer 16 = {want}",
          abs(got - want) < 0.0005, f"{got:.3f}")

for (ctx, pos), (want_acc, want_layer) in BEST.items():
    g = d[(d.source_context == ctx) & (d.position == pos)]
    b = g.loc[g.balanced_accuracy.idxmax()]
    check(f"tab:probee2 {ctx} {pos} best = {want_acc} at layer {want_layer}",
          abs(b.balanced_accuracy - want_acc) < 0.0005 and int(b.layer) == want_layer,
          f"{b.balanced_accuracy:.3f} at {int(b.layer)}")

# the two claims the paragraph makes
for ctx in ("mr", "en"):
    at16 = {p: float(d[(d.source_context == ctx) & (d.position == p)
                       & (d.layer == 16)].balanced_accuracy.iloc[0])
            for p in ("E2", "Q", "READOUT")}
    check(f"{ctx}: E2 is the best position at layer 16",
          at16["E2"] == max(at16.values()), str({k: round(v, 3) for k, v in at16.items()}))

for ctx in ("mr", "en"):
    g = d[(d.source_context == ctx) & (d.position == "E2")]
    best = int(g.loc[g.balanced_accuracy.idxmax(), "layer"])
    check(f"{ctx}: the E2 optimum falls inside the 12--17 causal window",
          12 <= best <= 17, f"layer {best}")

g = d[(d.source_context == "mr") & (d.position.isin(["Q", "READOUT"]))]
early = [int(x.loc[x.balanced_accuracy.idxmax(), "layer"])
         for _, x in g.groupby("position")]
check("mr: the question mark and readout peak before the window",
      all(l < 12 for l in early), f"layers {sorted(early)}")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every tab:probee2 number matches the data")

"""Recompute tab:gap from the held-out accuracies, and guard the claim it corrects."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
PAPER = Path.home() / "Downloads" / "when_entities_align" / "final_draft.tex"

ARMS = {
    "Marathi": (0.065, 0.340, 0.335, 0.629),
    "English, mr corpus": (0.283, 0.529, 0.590, 0.756),
    "Hindi": (0.122, 0.401, 0.546, 0.690),
    "English, hi corpus": (0.260, 0.521, 0.585, 0.743),
}
WANT = {"Marathi": (27.5, 29.4), "English, mr corpus": (24.6, 16.6),
        "Hindi": (27.9, 14.4), "English, hi corpus": (26.1, 15.8)}

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


shrink = []
for arm, (dd, ll, tdd, tll) in ARMS.items():
    before, after = (ll - dd) * 100, (tll - tdd) * 100
    wb, wa = WANT[arm]
    check(f"tab:gap {arm} untreated = {wb}", abs(before - wb) < 0.06, f"{before:.1f}")
    check(f"tab:gap {arm} both treated = {wa}", abs(after - wa) < 0.06, f"{after:.1f}")
    if after < before:
        shrink.append(arm)

check("the disparity narrows in exactly three arms", len(shrink) == 3, f"{shrink}")
check("Marathi is the arm where it widens",
      "Marathi" not in shrink, f"widens in {[a for a in ARMS if a not in shrink]}")

paper = PAPER.read_text(encoding="utf-8")
flat = " ".join(paper.split())
# the per-arm figures live in tab:gap; sec:repair carries the claim in prose
check("sec:repair states the disparity narrows in three of the four arms",
      "three of the four arms" in flat, "")
check("sec:repair attributes the widening to Marathi",
      "1.9" in flat and "Marathi" in flat
      and ("rises by 1.9" in flat or "widens" in flat), "")
check("the per-arm changes are in tab:gap",
      all(f"{v:+.1f}" in flat or f"{v:.1f}" in flat
          for v in (-8.0, -13.5, -10.3, 1.9)), "")
check("the paper no longer claims the gap simply does not close",
      "does not eliminate the disparity between them" in paper, "")


# the intervals the review asked for, from the paired bootstrap
ci = pd.read_csv(ROOT / "results_mr" / "gap_intervals.csv") if (
    ROOT / "results_mr" / "gap_intervals.csv").exists() else None
check("gap_intervals.csv exists", ci is not None, "")
if ci is not None:
    for arm in ARMS:
        r = ci[(ci.arm == arm) & (ci.quantity == "untreated gap truth=1")]
        check(f"{arm}: untreated positive gap has a paired interval",
              len(r) == 1 and r.iloc[0].lo > 0, f"{r.iloc[0].lo:+.1f} lower bound"
              if len(r) else "missing")
        b = ci[(ci.arm == arm) & (ci.quantity == "untreated balanced gap")]
        check(f"{arm}: balanced gap interval excludes zero",
              len(b) == 1 and b.iloc[0].lo > 0,
              f"{b.iloc[0].value:+.1f} [{b.iloc[0].lo:+.1f}, {b.iloc[0].hi:+.1f}]"
              if len(b) else "missing")
    check("the caption says the treated gap has no interval",
          "the entities cannot be paired across them" in paper, "")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every tab:gap number matches the held-out accuracies")

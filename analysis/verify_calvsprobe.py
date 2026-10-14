"""Recompute tab:calvsprobe from the saved comparison and check the sec:probe claims."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results_mr" / "probe_versus_calibration.csv"
PAPER = Path.home() / "Downloads" / "when_entities_align" / "final_draft.tex"

WANT = {
    ("mr", "raw model decision"): 0.525,
    ("mr", "margin + validation offset"): 0.879,
    ("mr", "hidden-state probe at E2"): 0.933,
    ("mr", "five string features"): 0.933,
    ("en", "raw model decision"): 0.633,
    ("en", "margin + validation offset"): 0.904,
    ("en", "hidden-state probe at E2"): 0.954,
    ("en", "five string features"): 0.933,
}

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


d = pd.read_csv(CSV)


def get(ctx, q, col="value"):
    return float(d[(d.context == ctx) & (d.quantity == q)][col].iloc[0])


for (ctx, q), want in WANT.items():
    check(f"tab:calvsprobe {ctx} {q} = {want}", abs(get(ctx, q) - want) < 0.0006,
          f"{get(ctx, q):.3f}")

for ctx, want, lo_w, hi_w in (("mr", 0.054, 0.017, 0.092), ("en", 0.050, 0.021, 0.079)):
    r = d[(d.context == ctx) & (d.quantity == "probe minus margin + offset")].iloc[0]
    check(f"{ctx} probe minus calibrated margin = {want:+.3f} [{lo_w:+.3f}, {hi_w:+.3f}]",
          abs(r.value - want) < 0.0015 and abs(r.lo - lo_w) < 0.002
          and abs(r.hi - hi_w) < 0.002,
          f"{r.value:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")
    check(f"{ctx}: the probe advantage survives calibration", r.lo > 0, f"lo {r.lo:+.3f}")

raw, cal, probe = (get("mr", "raw model decision"),
                   get("mr", "margin + validation offset"),
                   get("mr", "hidden-state probe at E2"))
closed = (cal - raw) / (probe - raw) * 100
check("sec:probe: calibration recovers 87 percent of the distance",
      abs(closed - 87) < 1.0, f"{closed:.1f} percent")

paper = PAPER.read_text(encoding="utf-8")
# anchored on the claim, not a sentence: the wording has been rephrased once already
check("sec:probe sets the probe against the calibrated margin",
      "87 percent of the distance to the probe" in paper, "")
flat = " ".join(paper.split())
check("sec:probe distinguishes the probe's two positions",
      "0.925 balanced accuracy at the question mark" in flat, "")
# the abstract writes this with the symbol; the caption's "87 percent" is a different
# sentence, and anchoring here on that one made this check pass without testing it
check("the abstract reports the calibration finding",
      "recovers 87" + chr(92) + "% of the distance" in paper, "")
check("the retired 'carries more than the surface' wording is gone",
      "carries more than the surface" not in paper, "")

# the three disclosures the fourth review asked for, checked on whitespace-normalised text
flat = " ".join(paper.split())
check("the caption names the probe's selected layers",
      "12 for Marathi and 21 for English" in flat, "")
# the rates may be stated in the caption or in app:protocol, and the wording varies;
# what matters is that both appear together with the mismatch named
check("the unmatched false-positive rates are disclosed somewhere",
      "0.050" in flat and "0.017" in flat
      and ("unmatched rates" in flat or "different false-positive rates" in flat
           or "rate runs higher" in flat), "")
check("the caption explains the identical probe and string rows",
      "differ on 2 of 240 pairs" in flat, "")
check("sec:probe gives the right fraction, an eighth and not a twentieth",
      "a twentieth" not in flat and "an eighth of what the uncalibrated" in flat, "")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every tab:calvsprobe number matches the saved comparison")

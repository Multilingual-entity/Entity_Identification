"""Recompute the split-disjointness figures quoted in app:probe.

Both reviews raise that the globally selected distractors are not split-disjoint. The
appendix now gives counts and a bound, so those numbers need checking like any other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2] / "results_mr" / "qwen_mr"
PAPER = Path.home() / "Downloads" / "when_entities_align" / "final_draft.tex"

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


c = pd.read_csv(ROOT / "data/corpus.csv", dtype={"fact_id": str})
n = pd.read_csv(ROOT / "data/hard_negative_map.csv",
                dtype={"fact_id": str, "negative_fact_id": str})
b = pd.read_csv(ROOT / "tables/behavior.csv", dtype={"fact_id": str})

sp = c.set_index("fact_id").split
n["own"] = n.fact_id.map(sp)
n["donor"] = n.negative_fact_id.map(sp)

t = n[n.own == "test"]
cross = int((t.donor != "test").sum())
from_train = int((t.donor == "train").sum())
v = n[n.own == "validation"]
v_cross = int((v.donor != "validation").sum())

check("100 of 120 test entities have a cross-split distractor", cross == 100, f"{cross}")
check("76 of those come from the training split", from_train == 76, f"{from_train}")
check("94 of 120 validation entities likewise", v_cross == 94, f"{v_cross}")

don = n.set_index("fact_id").donor
test = set(c[c.split == "test"].fact_id)
bt = b[b.fact_id.isin(test) & (b.condition == "DEVDEV")].copy()
bt["donor"] = bt.fact_id.map(don)
bt["seen"] = np.where(bt.donor == "train", "train", "heldout")

WANT = {("mr", 0, "train"): 99.1, ("mr", 0, "heldout"): 100.0,
        ("en", 0, "train"): 100.0, ("en", 0, "heldout"): 100.0,
        ("mr", 1, "train"): 8.3, ("mr", 1, "heldout"): 3.4}
for (ctx, truth, grp), want in WANT.items():
    got = bt[(bt.context == ctx) & (bt.truth == truth)
             & (bt.seen == grp)].correct.mean() * 100
    kind = "negatives" if truth == 0 else "positives"
    check(f"app:probe {ctx} {kind}, distractor {grp} = {want}",
          abs(got - want) < 0.06, f"{got:.1f}")

# the claim that rests on it
neg_train = bt[(bt.truth == 0) & (bt.seen == "train")].correct.mean()
neg_held = bt[(bt.truth == 0) & (bt.seen == "heldout")].correct.mean()
check("prior exposure gives no advantage on the class it could touch",
      abs(neg_train - neg_held) < 0.02, f"{neg_train*100:.1f} vs {neg_held*100:.1f}")

paper = PAPER.read_text(encoding="utf-8")
check("app:probe states the counts rather than only flagging the issue",
      "100 have a distractor taken" in paper, "")
check("app:probe explains the positive-side difference is not exposure",
      "the distractor is absent from positive prompts" in paper, "")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every app:probe disjointness number matches the corpus")

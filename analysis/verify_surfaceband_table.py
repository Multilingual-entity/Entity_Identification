"""Recompute tab:surfaceband: held-out Marathi positives split by Latin sequence ratio."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "results_mr"
PAPER_TEXT = (Path.home() / "Downloads" / "when_entities_align"
              / "final_draft.tex").read_text(encoding="utf-8")

spec = importlib.util.spec_from_file_location("sb", HERE / "surface_baseline.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

# (label, lower, upper): items, surface accuracy, probe accuracy, as printed
WANT = [
    ("below 0.3", -1.0, 0.3, 3, 0.000, 0.333),
    ("0.3 to 0.5", 0.3, 0.5, 15, 0.267, 0.400),
    ("0.5 to 0.7", 0.5, 0.7, 34, 1.000, 0.912),
    ("above 0.7", 0.7, 2.0, 68, 1.000, 0.971),
]

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


base = ROOT / "qwen_mr"
corpus = pd.read_csv(base / "data/corpus.csv")
negatives = pd.read_csv(base / "data/hard_negative_map.csv")
for f in (corpus, negatives):
    for col in ("fact_id", "negative_fact_id"):
        if col in f:
            f[col] = f[col].astype(str)

d = sb.pair_frame(corpus, negatives, "dev")
train, test = d[d.split == "train"], d[d.split == "test"]
clf = LogisticRegression(max_iter=2000).fit(train[sb.FEATURES], train.y)

# The bands are cut on the DEVANAGARI sequence ratio, the surface the model is shown in
# the Dev--Dev condition, not on the Latin forms. Cutting on Latin gives 7/8/27/78.
pos = test[test.y == 1].copy()
pos["surface_ok"] = (clf.predict(pos[sb.FEATURES]) == 1).astype(int)
pos["ratio"] = pos.sequence_ratio

summ = pd.read_csv(base / "tables/relation_probe_heldout_summary.csv")
q = summ[(summ.source_context == "mr") & (summ.target_context == "mr")
         & (summ.position == "Q") & (summ.evaluation_split == "validation")]
layer = int(q.loc[q.balanced_accuracy.idxmax(), "layer"])
preds = pd.read_csv(base / "tables/relation_probe_heldout_predictions.csv")
preds["fact_id"] = preds.fact_id.astype(str)
pt = preds[(preds.source_context == "mr") & (preds.target_context == "mr")
           & (preds.position == "Q") & (preds.evaluation_split == "test")
           & (preds.layer == layer) & (preds.truth == 1)]
probe = pt.groupby("fact_id").probe_correct.mean()
pos["probe_ok"] = [probe.get(f, float("nan")) for f in pos.fact_id]

print(f"held-out Marathi positives: {len(pos)}, probe layer {layer}\n")
for label, lo, hi, n_want, s_want, p_want in WANT:
    band = pos[(pos.ratio > lo) & (pos.ratio <= hi)] if lo >= 0 else pos[pos.ratio <= hi]
    check(f"tab:surfaceband {label!r} items = {n_want}", len(band) == n_want, f"{len(band)}")
    if len(band) != n_want:
        continue
    check(f"tab:surfaceband {label!r} surface = {s_want}",
          abs(band.surface_ok.mean() - s_want) < 0.0015, f"{band.surface_ok.mean():.3f}")
    check(f"tab:surfaceband {label!r} probe = {p_want}",
          abs(band.probe_ok.mean() - p_want) < 0.0015, f"{band.probe_ok.mean():.3f}")

low = pos[pos.ratio <= 0.5]   # the first two bands of the table
check("sec:probe: 18 held-out positives below 0.5", len(low) == 18, f"{len(low)}")
check("sec:probe: surface 22 percent on those",
      abs(low.surface_ok.mean() - 0.22) < 0.006, f"{low.surface_ok.mean():.3f}")
check("sec:probe: probe 39 percent on those",
      abs(low.probe_ok.mean() - 0.39) < 0.006, f"{low.probe_ok.mean():.3f}")

# The band edges are inclusive at the top. Stated as strict inequalities the counts
# change, so the paper says "0.5 or below" rather than "below 0.5".
strict = len(pos[pos.ratio < 0.5])
check("the 0.5 edge is inclusive: strict gives 16, not 18", strict == 16, f"{strict}")
check("the paper does not say 'below 0.5' or 'below 0.3'",
      "ratio below 0.5" not in PAPER_TEXT and "below 0.3 &" not in PAPER_TEXT, "")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every tab:surfaceband number matches the data")

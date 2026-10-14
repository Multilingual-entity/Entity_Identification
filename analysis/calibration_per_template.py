"""Is the vector's gain over a threshold template-specific calibration?

Held out, per template: the vector's own operating point, the best threshold at that same
false-positive rate, and a per-template offset, three scalars chosen on the validation
tuning entities under the same floor. If per-template thresholds match the vector
everywhere, the effect is calibration that happens to vary by template.

    python analysis/calibration_per_template.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MR = ROOT / "results_mr" / "qwen_mr"


def mp(d, t):
    return d[d.truth == t].yes_minus_no_margin.to_numpy()


# the Marathi window arm negative-pair accuracy on validation at alpha=12
VALIDATION_NEG_FLOOR = 0.975
OUT = []


def best_offset(pos, neg, floor):
    best, bc = -1.0, 0.0
    for c in np.linspace(-20, 20, 8001):
        tn = float((neg + c < 0).mean())
        if tn < floor - 1e-12:
            continue
        bal = (float((pos + c > 0).mean()) + tn) / 2
        if bal > best:
            best, bc = bal, float(c)
    return bc


def tpr_at_fpr(pos, neg, target):
    best = 0.0
    for t in np.unique(np.concatenate([pos, neg])):
        if float((neg > t).mean()) <= target + 1e-12:
            best = max(best, float((pos > t).mean()))
    return best


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    c = pd.read_csv(MR / "data" / "corpus.csv", dtype={"fact_id": str})
    b = pd.read_csv(MR / "tables" / "behavior.csv", dtype={"fact_id": str})
    v = pd.read_csv(MR / "recovery_window" / "vector_recovery_test.csv",
                    dtype={"fact_id": str})
    tuning = [f for f in c[c.split == "validation"].fact_id][:60]
    test = set(c[c.split == "test"].fact_id)
    dd = b[(b.context == "mr") & (b.condition == "DEVDEV")]
    va = v[(v.context == "mr") & (v.vector_kind == "additive") & v.fact_id.isin(test)]

    print("Marathi, held-out split, per template.\n")
    print(f"{'tpl':4}{'vector TPR':>12}{'at FPR':>9}{'threshold':>11}"
          f"{'offset':>9}{'offset FPR':>12}   verdict")
    rows = []
    for p in (0, 1, 2):
        vt = va[va.paraphrase_id == p]
        bt = dd[dd.fact_id.isin(test) & (dd.paraphrase_id == p)]
        bv = dd[dd.fact_id.isin(tuning) & (dd.paraphrase_id == p)]

        tpr_v = float((mp(vt, 1) > 0).mean())
        fpr_v = 1.0 - float((mp(vt, 0) < 0).mean())
        thr = tpr_at_fpr(mp(bt, 1), mp(bt, 0), fpr_v)
        # One scalar per template, chosen on the validation tuning entities. The floor is
        # the arm own validation negative-pair accuracy at the selected scale, 0.975.
        # It must not be 1 - fpr_v: fpr_v is measured on the test split, so that form
        # lets the evaluation split set the selection constraint.
        co = best_offset(mp(bv, 1), mp(bv, 0), VALIDATION_NEG_FLOOR)
        tpr_o = float((mp(bt, 1) + co > 0).mean())
        fpr_o = 1.0 - float((mp(bt, 0) + co < 0).mean())
        beats = tpr_v > max(thr, tpr_o) + 0.01
        print(f"{p:<4}{tpr_v:>12.3f}{fpr_v:>9.3f}{thr:>11.3f}"
              f"{tpr_o:>9.3f}{fpr_o:>12.3f}   "
              f"{'vector beats both' if beats else 'calibration matches it'}")
        rows.append((p, tpr_v, fpr_v, thr, tpr_o))
        OUT.append(dict(template=p, vector_tpr=tpr_v, vector_fpr=fpr_v,
                        threshold_tpr=thr, offset_tpr=tpr_o, offset_fpr=fpr_o))

    print()
    tv = float(np.mean([r[1] for r in rows]))
    tt = float(np.mean([r[3] for r in rows]))
    to = float(np.mean([r[4] for r in rows]))
    print(f"mean over templates: vector {tv:.3f}, per-template threshold {tt:.3f}, "
          f"per-template offset {to:.3f}")
    print()
    print("For contrast, one global threshold pooled over templates:")
    bt_all = dd[dd.fact_id.isin(test)]
    fpr_all = 1.0 - float((mp(va, 0) < 0).mean())
    print(f"  vector {float((mp(va, 1) > 0).mean()):.3f} at FPR {fpr_all:.3f}, "
          f"one threshold {tpr_at_fpr(mp(bt_all, 1), mp(bt_all, 0), fpr_all):.3f}")


if __name__ == "__main__":
    main()

import pandas as _pd
_pd.DataFrame(OUT).to_csv(
    Path(__file__).resolve().parents[2] / "results_mr" / "per_template_points.csv",
    index=False)
print("saved per_template_points.csv")

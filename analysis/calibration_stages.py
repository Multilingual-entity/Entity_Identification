"""Compare the vector, a scalar offset, and a bare threshold at both stages.

The validation and held-out comparisons were drawn on different populations and it was
not clear the offset had been held to the same floor as the vector. This puts all three
methods on one population at each stage and reports the matched-rate column at both.

    python analysis/calibration_stages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MR = ROOT / "results_mr" / "qwen_mr"


def margins(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (d[d.truth == 1].yes_minus_no_margin.to_numpy(),
            d[d.truth == 0].yes_minus_no_margin.to_numpy())


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
    grid = pd.read_csv(ROOT / "results_mr" / "overnight_20260930_0537" / "1d_tuning_grid.csv",
                       dtype={"fact_id": str})
    vt = pd.read_csv(MR / "recovery_window" / "vector_recovery_test.csv",
                     dtype={"fact_id": str})

    tuning = [f for f in c[c.split == "validation"].fact_id][:60]
    test = set(c[c.split == "test"].fact_id)
    dd = b[(b.context == "mr") & (b.condition == "DEVDEV")]

    print("Marathi. Both stages, three methods, one population per stage.\n")

    # ---- validation: 60 tuning entities, template 0, as the sweep used -------
    base_v = dd[(dd.paraphrase_id == 0) & dd.fact_id.isin(tuning)]
    pv, nv = margins(base_v)
    g = grid[(grid.context == "mr") & (grid.alpha == 12.0)]
    vec_tpr = float((g[g.truth == 1].correct).mean())
    vec_fpr = 1.0 - float((g[g.truth == 0].correct).mean())
    off_c = best_offset(pv, nv, 1.0 - vec_fpr)
    print("VALIDATION, 60 entities, template 0")
    print(f"  baseline            TPR {float((pv > 0).mean()):.3f} at FPR "
          f"{1.0 - float((nv < 0).mean()):.3f}")
    print(f"  vector, alpha 12    TPR {vec_tpr:.3f} at FPR {vec_fpr:.3f}")
    print(f"  offset c={off_c:+.2f}       TPR {float((pv + off_c > 0).mean()):.3f} at FPR "
          f"{1.0 - float((nv + off_c < 0).mean()):.3f}   (floor matched to the vector)")
    print(f"  threshold only      TPR {tpr_at_fpr(pv, nv, vec_fpr):.3f} at FPR "
          f"{vec_fpr:.3f}   (matched by construction)")

    # ---- held out: 120 entities, three templates ----------------------------
    base_t = dd[dd.fact_id.isin(test)]
    pt, nt = margins(base_t)
    va = vt[(vt.context == "mr") & (vt.vector_kind == "additive")
            & vt.fact_id.isin(test)]
    pva, nva = margins(va)
    t_fpr = 1.0 - float((nva < 0).mean())
    off_t = best_offset(pv, nv, 1.0 - vec_fpr)  # chosen on validation, as it must be
    print("\nHELD OUT, 120 entities, three templates")
    print(f"  baseline            TPR {float((pt > 0).mean()):.3f} at FPR "
          f"{1.0 - float((nt < 0).mean()):.3f}")
    print(f"  vector, alpha 12    TPR {float((pva > 0).mean()):.3f} at FPR {t_fpr:.3f}")
    print(f"  offset c={off_t:+.2f}       TPR {float((pt + off_t > 0).mean()):.3f} at FPR "
          f"{1.0 - float((nt + off_t < 0).mean()):.3f}")
    print(f"  threshold only      TPR {tpr_at_fpr(pt, nt, t_fpr):.3f} at FPR {t_fpr:.3f}")

    # ---- the same validation population, all three templates ----------------
    base_v3 = dd[dd.fact_id.isin(tuning)]
    p3, n3 = margins(base_v3)
    off3 = best_offset(p3, n3, 1.0 - vec_fpr)
    print("\nVALIDATION entities but all three templates, to separate population from stage")
    print(f"  baseline            TPR {float((p3 > 0).mean()):.3f}")
    print(f"  offset c={off3:+.2f}       TPR {float((p3 + off3 > 0).mean()):.3f} at FPR "
          f"{1.0 - float((n3 + off3 < 0).mean()):.3f}")
    print(f"  threshold only      TPR {tpr_at_fpr(p3, n3, vec_fpr):.3f} at FPR {vec_fpr:.3f}")


if __name__ == "__main__":
    main()

"""Is the correction a change in discrimination, or a change in criterion?

Two quantities, each computed the way the section that already reports it computes it, so
the new table cannot disagree with the old ones:

  AUROC follows \\S4.1 and analysis/margin_auc.py: the six prompts of an item are averaged
  first, so the unit is the entity, and the bootstrap resamples entities. Using raw
  prompts instead inflates every AUC by roughly a factor of three and would contradict
  the gaps already reported there.

  True- and false-positive rates follow the accuracy tables: one prompt is one trial, and
  the false-positive rate is one minus the negative-pair accuracy those tables report.

    python analysis/calibration_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260813
DRAWS = 2000

# The last field is the negative-pair floor the offset is selected under: each arm's own
# validation negative-pair accuracy at the scale the sweep chose, read from
# vector_hyperparameters_selected.csv and hindi_window_grid.csv respectively.
ARMS = [
    ("Marathi", "results_mr", "qwen_mr", "mr",
     "results_mr/qwen_mr/recovery_window/vector_recovery_test.csv", 0.975),
    ("English (mr corpus)", "results_mr", "qwen_mr", "en",
     "results_mr/qwen_mr/recovery_window/vector_recovery_test.csv", 0.975),
    ("Hindi", "results_hi", "qwen_hi", "hi",
     "results_hi/qwen_hi/hindi_window_test.csv", 0.925),
]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U as a probability, ties counted as half. Same as margin_auc.py."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    ranks = pd.Series(vals).rank(method="average").to_numpy()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Item-level means, restricted to entities present in both truth classes."""
    m = d.groupby(["fact_id", "truth"], as_index=False).yes_minus_no_margin.mean()
    pos = m[m.truth == 1].set_index("fact_id").yes_minus_no_margin
    neg = m[m.truth == 0].set_index("fact_id").yes_minus_no_margin
    shared = pos.index.intersection(neg.index)
    return pos.loc[shared].to_numpy(), neg.loc[shared].to_numpy(), list(shared)


def auc_delta_ci(p1, n1, p0, n0, seed: int) -> tuple[float, float, float]:
    """Paired item-clustered bootstrap on an AUC difference, as margin_auc.py does."""
    m = min(len(p1), len(p0))
    rng = np.random.default_rng(seed)
    draws = [auc(p1[i], n1[i]) - auc(p0[i], n0[i])
             for i in (rng.integers(0, m, m) for _ in range(DRAWS))]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return auc(p1, n1) - auc(p0, n0), float(lo), float(hi)


def best_offset(pos: np.ndarray, neg: np.ndarray, neg_floor: float) -> float:
    """Offset maximising balanced accuracy subject to a negative-pair floor.

    The same objective and the same floor the scale sweep used, so the two interventions
    are selected under one rule rather than two.
    """
    best, best_c = -1.0, 0.0
    for c in np.linspace(-20, 20, 4001):
        tn = float((neg + c < 0).mean())
        if tn < neg_floor:
            continue
        bal = (float((pos + c > 0).mean()) + tn) / 2
        if bal > best:
            best, best_c = bal, float(c)
    return best_c


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, target: float) -> float:
    best = 0.0
    for t in np.unique(np.concatenate([pos, neg])):
        if float((neg > t).mean()) <= target + 1e-12:
            best = max(best, float((pos > t).mean()))
    return best


def arm(label, tree, run, ctx, vpath, neg_floor) -> dict:
    c = pd.read_csv(ROOT / tree / run / "data" / "corpus.csv", dtype={"fact_id": str})
    b = pd.read_csv(ROOT / tree / run / "tables" / "behavior.csv", dtype={"fact_id": str})
    v = pd.read_csv(ROOT / vpath, dtype={"fact_id": str})
    test = set(c[c.split == "test"].fact_id)
    val = set(c[c.split == "validation"].fact_id)
    cond = lambda name: b[(b.context == ctx) & (b.condition == name)]
    vt = v[(v.context == ctx) & (v.vector_kind == "additive")]

    out = {"label": label}
    # AUROC, entity level, every condition plus the intervention
    for name, frame in (("DEVDEV", cond("DEVDEV")[cond("DEVDEV").fact_id.isin(test)]),
                        ("LATLAT", cond("LATLAT")[cond("LATLAT").fact_id.isin(test)]),
                        ("LATDEV", cond("LATDEV")[cond("LATDEV").fact_id.isin(test)]),
                        ("DEVLAT", cond("DEVLAT")[cond("DEVLAT").fact_id.isin(test)]),
                        ("vector", vt[vt.fact_id.isin(test)])):
        p, n, _ = paired(frame)
        out[f"auc_{name}"] = auc(p, n)
        out[f"_pn_{name}"] = (p, n)

    pv, nv = out["_pn_vector"]
    p0, n0 = out["_pn_DEVDEV"]
    pl, nl = out["_pn_LATLAT"]
    out["d_vector"], out["d_vector_lo"], out["d_vector_hi"] = auc_delta_ci(
        pv, nv, p0, n0, SEED + 7)
    out["d_script"], out["d_script_lo"], out["d_script_hi"] = auc_delta_ci(
        pl, nl, p0, n0, SEED + 1)
    pld, nld = out["_pn_LATDEV"]
    out["d_latdev"], out["d_latdev_lo"], out["d_latdev_hi"] = auc_delta_ci(
        pld, nld, pl, nl, SEED + 3)

    # operating points, prompt level, matching the accuracy tables
    dd_t = cond("DEVDEV")[cond("DEVDEV").fact_id.isin(test)]
    dd_v = cond("DEVDEV")[cond("DEVDEV").fact_id.isin(val)]
    vt_t = vt[vt.fact_id.isin(test)]
    mp = lambda d, t: d[d.truth == t].yes_minus_no_margin.to_numpy()
    out["tpr_vec"] = float((mp(vt_t, 1) > 0).mean())
    out["fpr_vec"] = 1.0 - float((mp(vt_t, 0) < 0).mean())
    # The 60 template-0 tuning entities, and a floor from the vector's own validation
    # negative-pair accuracy. Neither quantity comes from the test split; using
    # 1 - out["fpr_vec"] here would, because fpr_vec is measured on test.
    tuning = [f for f in c[c.split == "validation"].fact_id][:60]
    sel = cond("DEVDEV")
    sel = sel[(sel.paraphrase_id == 0) & sel.fact_id.isin(tuning)]
    out["offset_floor"] = neg_floor
    c_off = best_offset(mp(sel, 1), mp(sel, 0), neg_floor)
    out["offset"] = c_off
    out["tpr_off"] = float((mp(dd_t, 1) + c_off > 0).mean())
    out["fpr_off"] = 1.0 - float((mp(dd_t, 0) + c_off < 0).mean())
    out["tpr_thr"] = tpr_at_fpr(mp(dd_t, 1), mp(dd_t, 0), out["fpr_vec"])
    return out


def sensitivity_versus_offset() -> None:
    r"""The sensitivity direction's operating point against a scalar offset.

    \S7 contrasts two directions. Once calibration is the baseline, the comparison that
    matters is whether either beats a constant on the same 60 validation tuning items
    under the same negative-pair floor. The tuning loop uses paraphrase 0 and both answer
    keys, so the baseline is drawn the same way.
    """
    c = pd.read_csv(ROOT / "results_mr/qwen_mr/data/corpus.csv", dtype={"fact_id": str})
    tuning = [f for f in c[c.split == "validation"].fact_id][:60]
    b = pd.read_csv(ROOT / "results_mr/qwen_mr/tables/behavior.csv", dtype={"fact_id": str})
    base = b[(b.context == "mr") & (b.condition == "DEVDEV")
             & (b.paraphrase_id == 0) & b.fact_id.isin(tuning)]
    pos = base[base.truth == 1].yes_minus_no_margin.to_numpy()
    neg = base[base.truth == 0].yes_minus_no_margin.to_numpy()
    print("=== the two directions against a scalar offset, Marathi validation tuning set ===")
    print(f"  baseline, no intervention: positives {float((pos > 0).mean()):.3f}, "
          f"negatives {float((neg < 0).mean()):.3f}  (n={len(pos)} each)")
    for label, tpr, tnr in (("sensitivity direction, alpha 1", 0.683, 0.983),
                            ("mean difference, alpha 12", 0.383, 0.975)):
        c_off = best_offset(pos, neg, tnr)
        got_t = float((pos + c_off > 0).mean())
        got_n = float((neg + c_off < 0).mean())
        verdict = "offset matches or beats it" if got_t >= tpr - 0.005 else "it beats the offset"
        print(f"  {label}: {tpr:.3f} / {tnr:.3f}")
        print(f"    offset c={c_off:+.2f} under the same floor: {got_t:.3f} / {got_n:.3f}"
              f"   -> {verdict}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for a in (arm(*x) for x in ARMS):
        print(f"=== {a['label']} ===")
        print(f"  AUROC, entity level: Dev--Dev {a['auc_DEVDEV']:.3f}   "
              f"+vector {a['auc_vector']:.3f}   Lat--Lat {a['auc_LATLAT']:.3f}   "
              f"Lat--Dev {a['auc_LATDEV']:.3f}   Dev--Lat {a['auc_DEVLAT']:.3f}")
        print(f"  vector minus baseline  {a['d_vector']:+.3f} "
              f"[{a['d_vector_lo']:+.3f}, {a['d_vector_hi']:+.3f}]"
              f"   {'excludes 0' if a['d_vector_lo'] > 0 or a['d_vector_hi'] < 0 else 'INCLUDES 0'}")
        print(f"  script gap LL-DD       {a['d_script']:+.3f} "
              f"[{a['d_script_lo']:+.3f}, {a['d_script_hi']:+.3f}]")
        print(f"  operating points: vector TPR {a['tpr_vec']:.3f} at FPR {a['fpr_vec']:.3f}"
              f" | offset c={a['offset']:+.2f} TPR {a['tpr_off']:.3f} at FPR {a['fpr_off']:.3f}"
              f" | threshold only TPR {a['tpr_thr']:.3f} at the same FPR")
        print()
    sensitivity_versus_offset()


if __name__ == "__main__":
    main()

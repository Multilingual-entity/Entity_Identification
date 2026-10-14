"""Four decision rules on the same examples and the same prediction unit.

The question this answers: how much better is a hidden-state probe than a properly
calibrated classifier using only the model's own answer margin? If the probe still leads,
the representation-behaviour dissociation stands. If calibration closes most of it, the
finding is about how available output evidence is converted into a decision.

Everything is at entity level, averaging an entity's six prompt margins first, which is the
unit the probe predicts on and the unit tab:calibration's AUROC column uses. Selection is
on validation for every rule that has a parameter; nothing is chosen on test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

spec = importlib.util.spec_from_file_location("sb", HERE / "surface_baseline.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

OUT = []
SEED = 20260813
DRAWS = 2000


def entity_margins(b: pd.DataFrame, ctx: str, cond: str) -> pd.DataFrame:
    """One margin per (entity, truth): the mean over that entity's six prompts."""
    d = b[(b.context == ctx) & (b.condition == cond)]
    g = d.groupby(["fact_id", "truth"]).yes_minus_no_margin.mean().reset_index()
    return g.rename(columns={"yes_minus_no_margin": "margin"})


def tpr_fpr(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(pred[y == 1].mean()), float(pred[y == 0].mean())


def best_offset(m: np.ndarray, y: np.ndarray) -> float:
    """Offset maximising balanced accuracy, no floor, so every rule is unconstrained."""
    best, best_c = -1.0, 0.0
    for c in np.linspace(-20, 20, 4001):
        p = (m + c > 0).astype(int)
        bal = (p[y == 1].mean() + (1 - p[y == 0]).mean()) / 2
        if bal > best:
            best, best_c = bal, float(c)
    return best_c


def boot_diff(a: np.ndarray, b: np.ndarray, y: np.ndarray, ids: np.ndarray):
    """Paired item-clustered interval on the balanced-accuracy difference a - b."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(ids)
    idx = {u: np.flatnonzero(ids == u) for u in uniq}
    point = (balanced_accuracy_score(y, a) - balanced_accuracy_score(y, b))
    vals = []
    for _ in range(DRAWS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        vals.append(balanced_accuracy_score(y[sel], a[sel])
                    - balanced_accuracy_score(y[sel], b[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


for ctx, label in (("mr", "Marathi"), ("en", "English")):
    base = ROOT / "results_mr" / "qwen_mr"
    corpus = pd.read_csv(base / "data/corpus.csv", dtype={"fact_id": str})
    negatives = pd.read_csv(base / "data/hard_negative_map.csv", dtype={"fact_id": str})
    b = pd.read_csv(base / "tables/behavior.csv", dtype={"fact_id": str})

    test = set(corpus[corpus.split == "test"].fact_id)
    val = set(corpus[corpus.split == "validation"].fact_id)

    em = entity_margins(b, ctx, "DEVDEV")
    em["fact_id"] = em.fact_id.astype(str)
    te = em[em.fact_id.isin(test)].reset_index(drop=True)
    va = em[em.fact_id.isin(val)].reset_index(drop=True)
    y = te.truth.to_numpy(dtype=int)
    ids = te.fact_id.to_numpy()

    # 1. raw model decision
    raw = (te.margin.to_numpy() > 0).astype(int)

    # 2. the same margin with one constant chosen on validation
    c = best_offset(va.margin.to_numpy(), va.truth.to_numpy(dtype=int))
    cal = (te.margin.to_numpy() + c > 0).astype(int)

    # 3. the hidden-state probe, layer chosen on validation
    # the E2 position is only in the run that added it; see verify_probe_e2.py
    summ = pd.read_csv(base / "tables/relation_probe_heldout_summary_e2.csv")
    q = summ[(summ.source_context == ctx) & (summ.target_context == ctx)
             & (summ.position == "E2") & (summ.evaluation_split == "validation")]
    layer = int(q.loc[q.balanced_accuracy.idxmax(), "layer"])
    pr = pd.read_csv(base / "tables/relation_probe_heldout_predictions_e2.csv",
                     dtype={"fact_id": str})
    pr = pr[(pr.source_context == ctx) & (pr.target_context == ctx)
            & (pr.position == "E2") & (pr.evaluation_split == "test")
            & (pr.layer == layer)]
    pm = pr.groupby(["fact_id", "truth"]).probe_pred.mean().reset_index()
    probe = te.merge(pm, on=["fact_id", "truth"], how="left")
    assert probe.probe_pred.notna().all(), "probe does not cover every test entity"
    probe = (probe.probe_pred.to_numpy() > 0.5).astype(int)

    # 4. the five string features, refit on the training split
    d = sb.pair_frame(corpus, negatives, "dev")
    clf = LogisticRegression(max_iter=2000).fit(
        d[d.split == "train"][sb.FEATURES], d[d.split == "train"].y)
    sp = d[d.split == "test"][["fact_id", "y"] + sb.FEATURES].copy()
    sp["pred"] = clf.predict(sp[sb.FEATURES])
    sp = sp.groupby(["fact_id", "y"]).pred.mean().reset_index()
    sp = te.merge(sp, left_on=["fact_id", "truth"], right_on=["fact_id", "y"], how="left")
    assert sp.pred.notna().all(), "string baseline does not cover every test entity"
    string = (sp.pred.to_numpy() > 0.5).astype(int)

    print(f"=== {label} prompts, Dev--Dev, {len(te)} held-out entity-truth pairs ===")
    print(f"    entity-averaged margins; validation offset c={c:+.2f}; probe layer {layer}")
    print(f"    {'rule':30}{'bal acc':>9}{'TPR':>8}{'FPR':>8}")
    rules = [("raw model decision", raw), ("margin + validation offset", cal),
             ("hidden-state probe at E2", probe), ("five string features", string)]
    for name, pred in rules:
        bal = balanced_accuracy_score(y, pred)
        tpr, fpr = tpr_fpr(pred, y)
        print(f"    {name:30}{bal:>9.3f}{tpr:>8.3f}{fpr:>8.3f}")
    auc = roc_auc_score(y, te.margin.to_numpy())
    print(f"    (entity-averaged AUROC of the raw margin: {auc:.3f})")

    for other, oname in ((cal, "margin + offset"), (string, "string features")):
        pt, lo, hi = boot_diff(probe, other, y, ids)
        verdict = "probe ahead" if lo > 0 else ("behind" if hi < 0 else "includes zero")
        print(f"    probe minus {oname:18} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")
        OUT.append(dict(context=ctx, quantity=f"probe minus {oname}",
                        value=pt, lo=lo, hi=hi))
    for name, pred in rules:
        bal = balanced_accuracy_score(y, pred)
        tpr, fpr = tpr_fpr(pred, y)
        OUT.append(dict(context=ctx, quantity=name, value=bal, tpr=tpr, fpr=fpr))
    OUT.append(dict(context=ctx, quantity="raw margin entity AUROC", value=auc))
    OUT.append(dict(context=ctx, quantity="validation offset", value=c))
    OUT.append(dict(context=ctx, quantity="probe layer", value=layer))
    print()

pd.DataFrame(OUT).to_csv(ROOT / "results_mr" / "probe_versus_calibration.csv", index=False)
print(f"saved {ROOT / 'results_mr' / 'probe_versus_calibration.csv'}")

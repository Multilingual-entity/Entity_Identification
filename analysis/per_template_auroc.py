"""Per-template AUROC change under the window intervention, with paired intervals.

Two reasons this exists. The review asks for a template-stratified AUROC summary beside
the entity-averaged one. And the three ranking gains sec:repair quotes, +0.034, +0.020 and
+0.003, are computed by no script in the repository: they appear only in the prose. This
recomputes them and writes a CSV so the appendix table and the verifier share one source.

Within a template an entity has two prompts, one per answer key, so the margins are
averaged over those before ranking. The bootstrap resamples entities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2] / "results_mr" / "qwen_mr"
SEED = 20260813
DRAWS = 2000


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def entity_scores(d: pd.DataFrame) -> pd.DataFrame:
    """One score per (entity, truth), averaging the two answer keys."""
    return (d.groupby(["fact_id", "truth"]).yes_minus_no_margin.mean()
            .reset_index())


b = pd.read_csv(ROOT / "tables" / "behavior.csv", dtype={"fact_id": str})
v = pd.read_csv(ROOT / "recovery_window" / "vector_recovery_test.csv",
                dtype={"fact_id": str})
c = pd.read_csv(ROOT / "data" / "corpus.csv", dtype={"fact_id": str})
test = set(c[c.split == "test"].fact_id)

OUT = []
print("Per-template entity AUROC, Marathi prompts, Dev--Dev, held-out test")
print("paired item-clustered bootstrap, 2000 draws, seed 20260813\n")
print(f"{'tpl':5}{'baseline':>10}{'+vector':>10}{'change':>10}{'interval':>22}{'n':>5}")

for p in (0, 1, 2):
    base = b[(b.context == "mr") & (b.condition == "DEVDEV")
             & (b.paraphrase_id == p) & b.fact_id.isin(test)]
    vec = v[(v.context == "mr") & (v.vector_kind == "additive")
            & (v.paraphrase_id == p) & v.fact_id.isin(test)]
    eb, ev = entity_scores(base), entity_scores(vec)
    m = eb.merge(ev, on=["fact_id", "truth"], suffixes=("_b", "_v"))
    ids = m.fact_id.to_numpy()
    y = m.truth.to_numpy()

    def delta(sel):
        s = m.iloc[sel]
        pb, nb = (s[s.truth == 1].yes_minus_no_margin_b.to_numpy(),
                  s[s.truth == 0].yes_minus_no_margin_b.to_numpy())
        pv, nv = (s[s.truth == 1].yes_minus_no_margin_v.to_numpy(),
                  s[s.truth == 0].yes_minus_no_margin_v.to_numpy())
        if not len(pb) or not len(nb):
            return np.nan
        return auc(pv, nv) - auc(pb, nb)

    a_b = auc(m[m.truth == 1].yes_minus_no_margin_b.to_numpy(),
              m[m.truth == 0].yes_minus_no_margin_b.to_numpy())
    a_v = auc(m[m.truth == 1].yes_minus_no_margin_v.to_numpy(),
              m[m.truth == 0].yes_minus_no_margin_v.to_numpy())
    point = a_v - a_b

    rng = np.random.default_rng(SEED)
    uniq = np.unique(ids)
    idx = {u: np.flatnonzero(ids == u) for u in uniq}
    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        d = delta(np.concatenate([idx[u] for u in pick]))
        if d == d:
            draws.append(d)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"{p:<5}{a_b:>10.3f}{a_v:>10.3f}{point:>+10.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{len(uniq):>5}")
    OUT.append(dict(template=p, baseline=a_b, with_vector=a_v,
                    change=point, lo=lo, hi=hi, n_entities=len(uniq)))

df = pd.DataFrame(OUT)
df.to_csv(ROOT.parent / "per_template_auroc.csv", index=False)
print(f"\nsaved {ROOT.parent / 'per_template_auroc.csv'}")
excl = df[(df.lo > 0)].template.tolist()
print(f"templates whose interval excludes zero: {excl}")

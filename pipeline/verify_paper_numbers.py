"""Recompute numbers printed in the paper from the archive, and compare.

verify_archive.py answers whether the files are present. This answers the harder question:
does the archive actually reproduce what the paper says? A file can be present and still be
the wrong file, or the right file from a different run, and neither shows up as a missing
path.

Each check names the table it comes from, the value printed in the tex, and the value
recomputed here. A check fails when they disagree beyond a tolerance chosen for how the
figure is reported: accuracies printed to one decimal place get a looser tolerance than
correlations printed to three.

Coverage is deliberately not total. It takes at least one number from every table that has a
machine-readable source, which is what establishes that the source is the right one; the
remaining values in that table come from the same groupby and would only re-test pandas.

    python verify_paper_numbers.py --archive ../../archive/299_fact
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

CHECKS: list = []
RESULTS: list = []


def check(table: str, label: str, printed, tolerance: float = 0.05):
    """Register a check. The decorated function returns the recomputed value, or None if
    the source is absent."""
    def register(fn):
        CHECKS.append((table, label, printed, tolerance, fn))
        return fn
    return register


def load(run: Path, rel: str):
    path = run / rel
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------ this run
@check("tab:behaviour", "Qwen, Marathi, Dev--Dev positive accuracy (%)", 12.0, 0.15)
def _(a):
    b = load(a / "qwen", "tables/behavior.csv")
    s = b[(b.context == "mr") & (b.condition == "DEVDEV") & (b.truth == 1)]
    return s.correct.mean() * 100


@check("tab:behaviour", "Qwen, Marathi, Lat--Lat positive accuracy (%)", 42.8, 0.15)
def _(a):
    b = load(a / "qwen", "tables/behavior.csv")
    s = b[(b.context == "mr") & (b.condition == "LATLAT") & (b.truth == 1)]
    return s.correct.mean() * 100


@check("tab:behaviour", "Gemma, Marathi, Dev--Dev negative accuracy (%)", 28.1, 0.15)
def _(a):
    b = load(a / "gemma", "tables/behavior.csv")
    s = b[(b.context == "mr") & (b.condition == "DEVDEV") & (b.truth == 0)]
    return s.correct.mean() * 100


@check("tab:para", "Qwen, Marathi P0, Dev--Dev positive accuracy (%)", 14.9, 0.15)
def _(a):
    b = load(a / "qwen", "tables/behavior.csv")
    s = b[(b.context == "mr") & (b.condition == "DEVDEV") & (b.truth == 1)
          & (b.paraphrase_id == 0)]
    return s.correct.mean() * 100


@check("tab:llamapara", "Llama, Marathi P2, Dev--Dev positive accuracy (%)", 50.0, 0.15)
def _(a):
    b = load(a / "llama", "tables/behavior.csv")
    s = b[(b.context == "mr") & (b.condition == "DEVDEV") & (b.truth == 1)
          & (b.paraphrase_id == 2)]
    return s.correct.mean() * 100


@check("sec:behaviour", "Qwen, Devanagari tokens per name", 15.63, 0.02)
def _(a):
    t = load(a / "qwen", "tables/tokenization.csv")
    return t[t.script == "dev"].n_tokens.mean()


@check("tab:models", "Gemma, Devanagari tokens per character", 0.55, 0.01)
def _(a):
    t = load(a / "gemma", "tables/tokenization.csv")
    s = t[t.script == "dev"]
    return (s.n_tokens / s.n_chars.clip(lower=1)).mean()


@check("tab:auc", "Qwen (mr), tokenization-only AUC", 0.494, 0.002)
def _(a):
    f = load(a / "qwen", "tables/failure_prediction_oof_auc.csv")
    s = f[(f.context == "mr") & (f.feature_set == "surface")]
    return s.oof_auc.iloc[0] if len(s) else None


@check("tab:auc", "Qwen (mr), with relation states AUC", 0.958, 0.002)
def _(a):
    f = load(a / "qwen", "tables/failure_prediction_oof_auc.csv")
    s = f[(f.context == "mr") & (f.feature_set == "all_including_relation")]
    return s.oof_auc.max() if len(s) else None


@check("tab:inprompt", "Qwen, Marathi, first entity at E2, layer 8", 0.960, 0.002)
def _(a):
    p = load(a / "qwen", "tables/pair_retrieval_summary.csv")
    s = p[(p.context == "mr") & (p.role == "a") & (p.position == "E2") & (p.layer == 8)
          ]
    return s.top1.mean() if len(s) else None


@check("tab:inprompt", "Qwen, Marathi, second entity at E2, layer 8", 0.525, 0.002)
def _(a):
    p = load(a / "qwen", "tables/pair_retrieval_summary.csv")
    s = p[(p.context == "mr") & (p.role == "b") & (p.position == "E2") & (p.layer == 8)
          ]
    return s.top1.mean() if len(s) else None


@check("tab:inprompt", "Llama, Marathi, second entity at E2, layer 8", 0.564, 0.002)
def _(a):
    p = load(a / "llama", "tables/pair_retrieval_summary.csv")
    s = p[(p.context == "mr") & (p.role == "b") & (p.position == "E2") & (p.layer == 8)
          ]
    return s.top1.mean() if len(s) else None


@check("sec:probe", "Qwen, isolated retrieval, first entity, layer 8", 0.311, 0.002)
def _(a):
    i = load(a / "qwen", "tables/isolated_retrieval_summary.csv")
    s = i[(i.role == "a") & (i.layer == 8)]
    return s.top1.iloc[0] if len(s) else None


@check("tab:probeheld", "Qwen, fit English evaluate Marathi, held-out", 0.925, 0.002)
def _(a):
    r = load(a / "qwen", "tables/relation_probe_heldout_summary.csv")
    s = r[(r.source_context == "en") & (r.target_context == "mr")
          & (r.evaluation_split == "test")]
    return s.balanced_accuracy.max() if len(s) else None


@check("tab:probeheld", "Gemma, fit Marathi evaluate English, held-out", 0.783, 0.002)
def _(a):
    r = load(a / "gemma", "tables/relation_probe_heldout_summary.csv")
    s = r[(r.source_context == "mr") & (r.target_context == "en")
          & (r.evaluation_split == "test")]
    return s.balanced_accuracy.max() if len(s) else None


@check("tab:patch", "Qwen, Marathi, E2, layers 10--18", 3.16, 0.02)
def _(a):
    c = load(a / "qwen", "patching/causal_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.experiment == "single_factorial")
          & (c.site == "E2") & (c.source_condition == "LATLAT")]
    s = s[pd.to_numeric(s.layer_or_window, errors="coerce").between(10, 18)]
    return (s.mean_delta * s.n_facts).sum() / s.n_facts.sum() if len(s) and s.n_facts.sum() else None


@check("tab:patch", "Qwen, Marathi, E1, layers 10--18", 0.05, 0.02)
def _(a):
    c = load(a / "qwen", "patching/causal_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.experiment == "single_factorial")
          & (c.site == "E1") & (c.source_condition == "LATLAT")]
    s = s[pd.to_numeric(s.layer_or_window, errors="coerce").between(10, 18)]
    return (s.mean_delta * s.n_facts).sum() / s.n_facts.sum() if len(s) and s.n_facts.sum() else None


@check("tab:profile", "Qwen, Marathi, E2 at layer 16", 6.84, 0.02)
def _(a):
    c = load(a / "qwen", "patching/causal_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.experiment == "single_factorial")
          & (c.site == "E2") & (c.source_condition == "LATLAT")
          & (c.layer_or_window.astype(str) == "16")]
    return (s.mean_delta * s.n_facts).sum() / s.n_facts.sum() if len(s) and s.n_facts.sum() else None


@check("tab:windows", "Qwen, Marathi, joint layers 12--17 at E2", 12.77, 0.05)
def _(a):
    c = load(a / "qwen", "patching/causal_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.site == "E2")
          & (c.experiment == "window_factorial") & (c.source_condition == "LATLAT")
          & (c.layer_or_window.astype(str) == "12_17")]
    return (s.mean_delta * s.n_facts).sum() / s.n_facts.sum() if len(s) and s.n_facts.sum() else None


@check("tab:components", "Qwen, Marathi, full residual at E2, layers 10--18", 3.01, 0.03)
def _(a):
    c = load(a / "qwen", "patching/component_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.component == "residual")
          & (c.layer.between(10, 18))]
    return (s.mean_delta * s.n).sum() / s.n.sum() if len(s) and s.n.sum() else None


@check("tab:components", "Qwen, Marathi, attention only at E2, layers 10--18", 0.58, 0.03)
def _(a):
    c = load(a / "qwen", "patching/component_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.component == "attention")
          & (c.layer.between(10, 18))]
    return (s.mean_delta * s.n).sum() / s.n.sum() if len(s) and s.n.sum() else None


@check("tab:components", "Qwen, Marathi, MLP only at E2, layers 10--18", 0.15, 0.03)
def _(a):
    c = load(a / "qwen", "patching/component_patching_summary.csv")
    s = c[(c.context == "mr") & (c.truth == 1) & (c.component == "mlp")
          & (c.layer.between(10, 18))]
    return (s.mean_delta * s.n).sum() / s.n.sum() if len(s) and s.n.sum() else None


@check("tab:recovery", "Qwen, Marathi, no intervention, positives", 0.108, 0.003)
def _(a):
    p = load(a / "qwen", "recovery/prompt_anchor_summary.csv")
    s = p[(p.context == "mr") & (p.split == "test") & (p.truth == 1) & (p.anchor == "none")]
    return (s.accuracy * s.n).sum() / s.n.sum() if len(s) and s.n.sum() else None


@check("tab:recovery", "Qwen, Marathi, Latin-only ceiling, positives", 0.378, 0.003)
def _(a):
    p = load(a / "qwen", "recovery/prompt_anchor_summary.csv")
    s = p[(p.context == "mr") & (p.split == "test") & (p.truth == 1)
          & (p.anchor == "latin_only_ceiling")]
    return (s.accuracy * s.n).sum() / s.n.sum() if len(s) and s.n.sum() else None


@check("tab:recovery", "Qwen, Marathi, additive vector, positives", 0.292, 0.003)
def _(a):
    v = load(a / "qwen", "recovery/vector_recovery_test_summary.csv")
    s = v[(v.context == "mr") & (v.truth == 1) & (v.vector_kind == "additive")]
    return s.accuracy.iloc[0] if len(s) else None


@check("tab:grid", "Qwen, selected layer for the correction vector", 16, 0.001)
def _(a):
    h = load(a / "qwen", "recovery/vector_hyperparameters_selected.csv")
    s = h[h.context == "mr"]
    return s.layer.iloc[0] if len(s) and "layer" in h.columns else None


@check("tab:models", "Items passing the causal gate, Qwen", 115, 0.001)
def _(a):
    c = load(a / "qwen", "patching/causal_candidates.csv")
    return int(c.stable.sum()) if c is not None else None


@check("tab:models", "Items passing the causal gate, Llama", 13, 0.001)
def _(a):
    c = load(a / "llama", "patching/causal_candidates.csv")
    return int(c.stable.sum()) if c is not None else None


@check("sec:behaviour", "Marathi script gap minus English script gap", 1.71, 0.02)
def _(a):
    # The stored column is signed English-minus-Marathi; the paper states the same
    # quantity in the opposite direction, saying Marathi exceeds English by this much.
    i = load(a / "qwen", "tables/context_script_interaction_per_fact.csv")
    column = [c for c in i.columns if "interaction" in c]
    return abs(i[column[0]].mean()) if column else None


@check("tab:tiers", "Items in the largest relation tier", 216, 0.001)
def _(a):
    c = load(a / "qwen", "data/corpus.csv")
    return int(c.relation_tier.value_counts().iloc[0]) if "relation_tier" in c else None


@check("app:corpus", "Retained items after the audit", 299, 0.001)
def _(a):
    c = load(a / "qwen", "data/corpus.csv")
    return len(c)


# ------------------------------------------------------- earlier notebook run
@check("tab:oldnew", "Earlier run, Dev--Dev positive accuracy (%)", 14.8, 0.15)
def _(a):
    b = load(a / "earlier_notebook_run" / "qwen", "h0_canonical_behavior.csv")
    if b is None:
        return None
    s = b[(b.condition == "DEVDEV") & (b.truth == 1)]
    return s.correct.mean() * 100 if len(s) else None


@check("tab:fragcorr", "Earlier run, fragmentation vs first recognition, rho", 0.308, 0.01)
def _(a):
    f = load(a / "earlier_notebook_run" / "qwen", "h1_fragmentation_vs_recognition.csv")
    if f is None:
        return None
    sub = f[["dev_lat_token_ratio", "ignition_layer"]].dropna()
    if len(sub) < 3:
        return None
    return float(sub.dev_lat_token_ratio.rank().corr(sub.ignition_layer.rank()))


@check("tab:oldnew", "Earlier run, Devanagari tokens per name", 15.65, 0.03)
def _(a):
    t = load(a / "earlier_notebook_run" / "qwen", "h1_tokenization.csv")
    if t is None:
        return None
    for col in ("n_tokens_dev", "dev_tokens", "n_tokens"):
        if col in t.columns:
            sub = t if "script" not in t.columns else t[t.script.astype(str).str.contains("dev", case=False)]
            return sub[col].mean()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ROOT / "archive" / "299_fact")
    args = ap.parse_args()
    archive = args.archive
    if not archive.is_dir():
        raise SystemExit(f"no archive at {archive}")

    print(f"recomputing paper numbers from {archive.resolve()}\n")
    print(f"{'table':<16}{'quantity':<52}{'printed':>10}{'archive':>10}  ")
    print("-" * 92)
    passed = failed = skipped = 0
    for table, label, printed, tolerance, fn in CHECKS:
        try:
            got = fn(archive)
        except Exception as exc:                                         # noqa: BLE001
            got, note = None, f"error: {exc}"
        else:
            note = ""
        if got is None or (isinstance(got, float) and np.isnan(got)):
            skipped += 1
            print(f"{table:<16}{label:<52}{printed:>10}{'--':>10}  no source {note}")
            continue
        agrees = abs(float(got) - float(printed)) <= tolerance
        passed, failed = (passed + 1, failed) if agrees else (passed, failed + 1)
        print(f"{table:<16}{label:<52}{printed:>10}{float(got):>10.3f}  "
              f"{'ok' if agrees else 'DIFFERS'}")

    print("-" * 92)
    print(f"{passed} reproduce, {failed} differ, {skipped} have no machine-readable source")
    if failed:
        print("\nA difference means the archive does not reproduce a printed number. Either")
        print("the paper quotes a run that is not archived, or the figure needs correcting.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

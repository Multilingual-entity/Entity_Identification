r"""Generate the paper's tables from the frozen results, and check the paper against them.

Every table here is built from results_mr/ rather than typed. That is the whole point:
the previous draft mixed a main text from the 601-item corpus with an appendix from the
299-item one, and no amount of reading caught it, because a stale number reads exactly
like a fresh one.

    python paper_tables.py --emit          write each table to tables_generated/
    python paper_tables.py --check         compare the paper's tables against these

--check extracts the numeric literals from each table body in the .tex and from the
generated version, and reports any that differ. It does not try to parse LaTeX: a table
whose numbers all match is correct whatever its formatting, and one whose numbers differ
needs regenerating whatever its formatting.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import os

HERE = Path(__file__).resolve().parent
# Paths are read from the environment so this runs from a checkout that holds no
# results, which is the normal case for anyone but the authors. Point RESULTS_DIR at a
# directory of run folders and PAPER_TEX at the LaTeX source to check it against.
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE.parent / "results"))
PAPER = Path(os.environ.get("PAPER_TEX", HERE / "paperB_neurips.tex"))
OUT = Path(os.environ.get("TABLES_OUT", HERE / "tables_generated"))

MODELS = {"qwen_mr": "Qwen2.5-7B", "gemma_mr": "Gemma-2-2b", "llama_mr": "Llama-3.1-8B"}
# The polarity each model's causal gate reads, which decides which rows are the
# gated ones. Gemma accepts nearly every positive pair, so its failures are negatives.
GATE_TRUTH = {"qwen_mr": 1, "gemma_mr": 0, "llama_mr": 1}


def _pooled(model: str) -> pd.DataFrame:
    """Accuracy pooled over paraphrases and answer keys, weighted by prompt count."""
    d = pd.read_csv(RESULTS / model / "tables/behavior_summary.csv")
    g = (d.groupby(["context", "condition", "truth"])
           .apply(lambda x: (x.accuracy * x.n).sum() / x.n.sum(), include_groups=False)
           .reset_index(name="acc"))
    return g


def table_behaviour() -> list[str]:
    rows = []
    for model, label in MODELS.items():
        g = _pooled(model)
        for ctx, ctx_label in (("mr", "Marathi"), ("en", "English")):
            cells = []
            for cond in ("DEVDEV", "DEVLAT", "LATDEV", "LATLAT"):
                pos = float(g[(g.context == ctx) & (g.condition == cond) & (g.truth == 1)].acc.iloc[0])
                neg = float(g[(g.context == ctx) & (g.condition == cond) & (g.truth == 0)].acc.iloc[0])
                cells += [f"{pos * 100:.1f}", f"{neg * 100:.1f}"]
            name = label if ctx == "mr" else " " * len(label)
            rows.append(f"{name} & {ctx_label} & " + " & ".join(cells) + r"\\")
    return rows


def table_components() -> list[str]:
    """Qwen only, the model the main-text table reports."""
    d = pd.read_csv(RESULTS / "qwen_mr/patching/component_patching_summary.csv")
    d = d[(d.truth == GATE_TRUTH["qwen_mr"]) & (d.split == "test")]
    h = pd.read_csv(RESULTS / "qwen_mr/patching/head_confirm_summary.csv")
    h = h[(h.truth == GATE_TRUTH["qwen_mr"]) & (h.split == "test")]

    def comp(name, ctx):
        return float(d[(d.component == name) & (d.context == ctx)].mean_delta.mean())

    def head(exp, ctx):
        return float(h[(h.experiment == exp) & (h.context == ctx)].mean_delta.iloc[0])

    rows = []
    for label, key in (("Full residual stream, single layer", "residual"),
                       ("Attention output only, single layer", "attention"),
                       ("MLP output only, single layer", "mlp"),
                       ("Attention output, cumulative", "attention_cumulative"),
                       ("MLP output, cumulative", "mlp_cumulative")):
        rows.append(f"{label} & {comp(key, 'mr'):.2f} & {comp(key, 'en'):.2f}" + r"\\")
    for label, exp in (("Best single head", "joint_top_1"),
                       ("Top 3 heads jointly", "joint_top_3"),
                       ("Top 5 heads jointly", "joint_top_5"),
                       ("Top 10 heads jointly", "joint_top_10"),
                       ("Mean of 12 selected heads, individually", "single_head")):
        rows.append(f"{label} & {head(exp, 'mr'):.2f} & {head(exp, 'en'):.2f}" + r"\\")
    return rows


def table_probeheld() -> list[str]:
    """Layer chosen on validation, accuracy reported on test, per direction."""
    rows = []
    for src, tgt in (("en", "en"), ("en", "mr"), ("mr", "en"), ("mr", "mr")):
        cells = []
        for model in MODELS:
            d = pd.read_csv(RESULTS / model / "tables/relation_probe_heldout_summary.csv")
            d = d[(d.source_context == src) & (d.target_context == tgt)]
            val = d[d.evaluation_split == "validation"]
            best = int(val.loc[val.balanced_accuracy.idxmax(), "layer"])
            test = d[(d.evaluation_split == "test") & (d.layer == best)]
            cells.append(f"{float(test.balanced_accuracy.iloc[0]):.3f} (L{best})")
        names = {"en": "English", "mr": "Marathi"}
        rows.append(f"{names[src]} & {names[tgt]} & " + " & ".join(cells) + r"\\")
    return rows


def table_auc() -> list[str]:
    order = [("surface", "Surface features only"),
             ("surface_plus_early", "{}+ early-layer retrieval"),
             ("surface_plus_persistence", "{}+ retrieval across layers"),
             ("all_including_relation", "{}+ relation states")]
    cols = [("qwen_mr", "mr"), ("qwen_mr", "en"), ("gemma_mr", "mr")]
    rows = []
    for key, label in order:
        cells = []
        for model, ctx in cols:
            d = pd.read_csv(RESULTS / model / "tables/failure_prediction_oof_auc.csv")
            hit = d[(d.feature_set == key) & (d.context == ctx)]
            cells.append(f"{float(hit.oof_auc.iloc[0]):.3f}" if len(hit) else "---")
        rows.append(f"{label} & " + " & ".join(cells) + r"\\")
    return rows


def table_recovery() -> list[str]:
    """Qwen, held-out test items: every anchor and every vector variant, both polarities.

    Shaped to match the paper's table rather than the pipeline's file, because the paper
    reports interventions down the rows and prompt language across the columns.
    """
    a = pd.read_csv(RESULTS / "qwen_mr/recovery/prompt_anchor_summary.csv")
    v = pd.read_csv(RESULTS / "qwen_mr/recovery/vector_recovery_test_summary.csv")

    def anchor(ctx, truth, name):
        r = a[(a.split == "test") & (a.context == ctx) & (a.truth == truth) & (a.anchor == name)]
        return float(r.accuracy.iloc[0])

    def vector(ctx, truth, kind):
        r = v[(v.context == ctx) & (v.truth == truth) & (v.vector_kind == kind)]
        return float(r.accuracy.iloc[0])

    spec = [("No intervention", anchor, "none"),
            ("Prompt anchor, first", anchor, "latin_a"),
            ("Prompt anchor, second", anchor, "latin_b"),
            ("Prompt anchor, both", anchor, "latin_both"),
            ("Automatic romanization anchor", anchor, "auto_both"),
            ("Vector, first entity", vector, "first_entity"),
            ("Vector, second entity", vector, "second_entity"),
            ("Vector, full", vector, "full"),
            ("Vector, additive", vector, "additive"),
            ("Latin-only ceiling", anchor, "latin_only_ceiling")]
    rows = []
    for label, fn, key in spec:
        cells = [f"{fn('mr', 1, key):.3f}", f"{fn('mr', 0, key):.3f}",
                 f"{fn('en', 1, key):.3f}", f"{fn('en', 0, key):.3f}"]
        rows.append(f"{label} & " + " & ".join(cells) + r"\\")
    return rows


def table_vectorsel() -> list[str]:
    """Which layer and scale the validation split selected, per model and prompt language.

    Reported because the scale grid runs to 8 and two cells select that boundary, which
    is what makes the repair result unresolved rather than negative.
    """
    rows = []
    for model, label in MODELS.items():
        h = pd.read_csv(RESULTS / model / "recovery/vector_hyperparameters_selected.csv")
        v = pd.read_csv(RESULTS / model / "recovery/vector_recovery_test_summary.csv")
        for ctx, ctx_label in (("mr", "Marathi"), ("en", "English")):
            hv = h[h.context == ctx]
            if hv.empty:
                continue
            add = v[(v.context == ctx) & (v.vector_kind == "additive")]
            pos = float(add[add.truth == 1].accuracy.iloc[0])
            neg = float(add[add.truth == 0].accuracy.iloc[0])
            name = label if ctx == "mr" else ""
            rows.append(f"{name} & {ctx_label} & {int(hv.layer.iloc[0])} & "
                        f"{float(hv.alpha.iloc[0]):.1f} & {pos:.3f} & {neg:.3f}" + r"\\")
    return rows


def table_romanization() -> list[str]:
    """Both arms against the Devanagari baseline and the Lat--Lat ceiling."""
    rows = []
    for model, label in MODELS.items():
        s = pd.read_csv(RESULTS / model / "recovery/romanization_scheme_summary.csv")
        r = pd.read_csv(RESULTS / model / "recovery/romanization_reference.csv")
        for ctx, ctx_label in (("mr", "Marathi"), ("en", "English")):
            ref = r[(r.split == "test") & (r.context == ctx)].set_index("condition").accuracy
            t = s[(s.split == "test") & (s.context == ctx)]
            auto = t[t.scheme != "gold_lat"]
            rep = auto[auto.arm == "replaced"].accuracy
            anc = auto[auto.arm == "anchored"].accuracy
            if rep.empty:
                continue
            name = label if ctx == "mr" else " " * len(label)
            rows.append(f"{name} & {ctx_label} & {ref['DEVDEV']:.3f} & "
                        f"{rep.min():.3f}--{rep.max():.3f} & {anc.min():.3f}--{anc.max():.3f} & "
                        f"{ref['LATLAT']:.3f}" + r"\\")
    return rows


def table_gatecounts() -> list[str]:
    rows = []
    for model, label in MODELS.items():
        c = pd.read_csv(RESULTS / model / "patching/causal_candidates.csv")
        n = c[c.stable == 1].groupby("context").fact_id.nunique().to_dict()
        rows.append(f"{label} & {n.get('mr', 0)} & {n.get('en', 0)}" + r"\\")
    return rows


TABLES = {
    "tab:behaviour": table_behaviour,
    "tab:components": table_components,
    "tab:probeheld": table_probeheld,
    "tab:auc": table_auc,
    "tab:recovery": table_recovery,
    "tab:vectorsel": table_vectorsel,
    "tab:romanization": table_romanization,
    "tab:gatecounts": table_gatecounts,
}

NUMBER = re.compile(r"-?\d+\.\d+|\b\d+\b")


def numbers_in(text: str) -> list[str]:
    """Numeric literals, ignoring LaTeX column counts and layer labels in \\multicolumn."""
    stripped = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", text)
    stripped = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", stripped)
    return NUMBER.findall(stripped)


def paper_table_body(label: str) -> str | None:
    text = PAPER.read_text(encoding="utf-8")
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S):
        if f"\\label{{{label}}}" in m.group(0):
            body = re.search(r"\\midrule(.*?)\\bottomrule", m.group(0), re.S)
            return body.group(1) if body else m.group(0)
    return None


# --------------------------------------------------------------------------------
# Tables added after a stale number survived several review passes in tab:probepos,
# which was not covered here. Each generator below was validated against the paper's
# printed values before being registered; anything that could not be reproduced from
# the frozen results is listed in UNCHECKED instead of being guessed at.
# --------------------------------------------------------------------------------

import sys as _sys
_sys.path.insert(0, str(HERE.parent / "pipeline"))
from pipeline_common import dprime as _dprime  # noqa: E402


def table_dprime() -> list[str]:
    """Sensitivity by model and prompt language, computed the way the pipeline does."""
    boot = {}
    rows = []
    for model, short in (("qwen_mr", "Qwen"), ("gemma_mr", "Gemma"), ("llama_mr", "Llama")):
        b = pd.read_csv(RESULTS / model / "tables/behavior.csv")
        for ctx, ctx_label in (("mr", "Marathi"), ("en", "English")):
            vals = []
            for cond in ("DEVDEV", "LATLAT"):
                s = b[(b.context == ctx) & (b.condition == cond)]
                same, diff = s[s.truth == 1], s[s.truth == 0]
                hit = float((same.pred_semantic == "yes").mean())
                fa = float((diff.pred_semantic == "yes").mean())
                d, _ = _dprime(hit, fa, len(same), len(diff))
                vals.append(d)
            name = short if ctx == "mr" else " " * len(short)
            # The interval is bootstrapped in the pipeline and is not recomputed here;
            # only the two d' columns are checked.
            rows.append(f"{name} & {ctx_label} & {vals[0]:.3f} & {vals[1]:.3f}" + r"\\")
    return rows


def _patch(model: str = "qwen_mr") -> pd.DataFrame:
    d = pd.read_csv(RESULTS / model / "patching/causal_patching_summary.csv")
    d["layer_num"] = pd.to_numeric(d.layer_or_window, errors="coerce")
    return d[(d.split == "test") & (d.truth == GATE_TRUTH[model])]


def table_patch() -> list[str]:
    """Site columns average layers 10-18; the controls only exist for 10-17."""
    t = _patch()
    rows = []
    for ctx in ("mr", "en"):
        sf = t[(t.context == ctx) & (t.experiment == "single_factorial")
               & (t.layer_num.between(10, 18))]
        cells = [f"{sf[sf.site == s].mean_delta.mean():.2f}"
                 for s in ("E1", "E2", "Q", "READOUT")]
        for exp in ("wrong_entity_control", "wrong_position_control", "reverse_DD_into_LL"):
            cells.append(f"{t[(t.context == ctx) & (t.experiment == exp)].mean_delta.mean():.2f}")
        j = t[(t.context == ctx) & (t.experiment == "joint_12_17") & (t.site == "E1+E2+Q")]
        cells.append(f"{j.mean_delta.mean():.2f}")
        # The n in the paper's row label is the gate count, checked by tab:gatecounts;
        # emitting a different n here would fight that table rather than confirm it.
        rows.append(f"{ctx} & " + " & ".join(cells) + r"\\")
    return rows


def table_tiers() -> list[str]:
    c = pd.read_csv(RESULTS / "qwen_mr/data/corpus.csv")
    counts = c.relation_tier.value_counts()
    order = ["alias_observed", "explicit_plus_alias:birth_name",
             "explicit_plus_alias:pseudonym", "explicit_plus_alias:nickname",
             "explicit_observed:birth_name", "explicit_observed:nickname"]
    return [f"{k} & {int(counts.get(k, 0))}" + r"\\" for k in order if k in counts]


def table_reclen() -> list[str]:
    """Isolated-name retrieval by Devanagari token count, layer 8, all 1202 names."""
    tok = pd.read_csv(RESULTS / "qwen_mr/tables/tokenization.csv")
    dev = tok[tok.script == "dev"][["fact_id", "role", "n_tokens"]]
    r = pd.read_csv(RESULTS / "qwen_mr/tables/isolated_retrieval_ranks.csv")
    r = r[r.layer == 8].merge(dev, on=["fact_id", "role"])
    rows = []
    for lo, hi in ((0, 10), (11, 15), (16, 20), (21, 10_000)):
        s = r[(r.n_tokens >= lo) & (r.n_tokens <= hi)]
        rows.append(f"{len(s)} & {s.n_tokens.mean():.1f} & "
                    f"{100 * (s['rank'] == 1).mean():.1f}" + r"\\")
    return rows


def _benefit() -> pd.DataFrame:
    """Per-fact margin gained by switching a positive pair to Latin, Marathi prompts."""
    b = pd.read_csv(RESULTS / "qwen_mr/tables/behavior.csv")
    b = b[(b.truth == 1) & (b.context == "mr")]
    piv = b.groupby(["fact_id", "condition"]).yes_minus_no_margin.mean().unstack()
    d = (piv["LATLAT"] - piv["DEVDEV"]).rename("benefit").reset_index()
    ratio = (pd.read_csv(RESULTS / "qwen_mr/tables/fragmentation_vs_ignition.csv")
             .groupby("fact_id").dev_lat_token_ratio.mean().rename("ratio").reset_index())
    return d.merge(ratio, on="fact_id")


def table_fragquart() -> list[str]:
    d = _benefit()
    d["q"] = pd.qcut(d.ratio, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return [f"{q} & {g.ratio.mean():.2f} & {g.benefit.mean():.2f}" + r"\\"
            for q, g in d.groupby("q", observed=True)]


def table_grid() -> list[str]:
    """Qwen validation grid, positive-pair accuracy, the four scales the paper prints."""
    v = pd.read_csv(RESULTS / "qwen_mr/recovery/vector_tuning_validation.csv")
    v = v[v.truth == 1]
    rows = []
    for layer in (10, 12, 14, 16, 18):
        cells = []
        for ctx in ("mr", "en"):
            for alpha in (0.5, 2.0, 4.0, 8.0):
                g = v[(v.context == ctx) & (v.layer == layer) & (v.alpha == alpha)]
                cells.append(f"{g.correct.mean():.3f}" if len(g) else "?")
        rows.append(f"{layer} & " + " & ".join(cells) + r"\\")
    return rows


TABLES.update({
    "tab:dprime": table_dprime,
    "tab:patch": table_patch,
    "tab:tiers": table_tiers,
    "tab:reclen": table_reclen,
    "tab:fragquart": table_fragquart,
    "tab:grid": table_grid,
})

# Tables whose row labels carry numbers a cell-value generator cannot emit: bin edges
# in tab:reclen, gated-item counts in tab:patch, the total row in tab:tiers, and the
# bootstrap interval in tab:dprime, which the pipeline computes and this script does
# not. For these the rule is that every generated number must appear in the paper, so a
# changed cell is still caught while the un-generated labels are tolerated.
SUBSET_ONLY = {"tab:dprime", "tab:patch", "tab:tiers", "tab:reclen"}


# Tables whose numbers are not reconstructible from results_mr/ alone, with the reason.
# These are checked by reading, not by this script, and are the remaining risk surface.
UNCHECKED = {
    "tab:probepos": "group-recall rows have no single source file; position rows verified by hand",
    "tab:inprompt": "layer-8 top-1 does not reproduce under any single other_entity_script filter",
    "tab:donor": "donor variants need a source_condition mapping not recorded in the summary",
    "tab:gate": "definitional, no numbers from results",
    "tab:para": "per-paraphrase split not stored separately from behavior_summary",
    "tab:tierbeh": "needs corpus join not stored in results",
    "tab:fragcorr": "correlations recomputed in-session; only the ignition row is in a result file",
    "tab:translit": "romanization quality table, separate scheme-level source",
    "tab:probelayer": "layer sweep, out-of-fold; large and stable, low staleness risk",
    "tab:profile": "full patching profile, superseded by tab:patch which is checked",
    "tab:windows": "window sweep, superseded by tab:patch",
    "tab:gemmaprofile": "Gemma profile, appendix only",
    "tab:llamapara": "Llama paraphrase table, appendix only",
    "tab:models": "cross-model summary assembled from many sources",
    "tab:corpuserr": "textual, no computed numbers",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.emit:
        OUT.mkdir(exist_ok=True)
        for label, fn in TABLES.items():
            path = OUT / (label.replace(":", "_") + ".tex")
            path.write_text("\n".join(fn()) + "\n", encoding="utf-8")
            print(f"  wrote {path.name}")
        return

    failures = 0
    for label, fn in TABLES.items():
        generated = "\n".join(fn())
        body = paper_table_body(label)
        if body is None:
            print(f"MISSING  {label:<20}not in the paper")
            continue
        want, have = numbers_in(generated), numbers_in(body)
        if label in SUBSET_ONLY:
            missing = [x for x in want if x not in have]
            if not missing:
                print(f"ok       {label:<20}{len(want)} generated numbers all present")
                continue
            failures += 1
            print(f"MISMATCH {label:<20}{len(missing)} generated numbers absent from the paper")
            print(f"{'':9}{missing[:8]}")
            continue
        if want == have:
            print(f"ok       {label:<20}{len(want)} numbers match")
        else:
            failures += 1
            only_gen = [x for x in want if x not in have][:6]
            only_pap = [x for x in have if x not in want][:6]
            print(f"MISMATCH {label:<20}paper has {len(have)} numbers, results give {len(want)}")
            if only_gen:
                print(f"{'':9}in results, not in paper: {only_gen}")
            if only_pap:
                print(f"{'':9}in paper, not in results: {only_pap}")
    print(f"\n{len(TABLES) - failures} of {len(TABLES)} generated tables agree with the frozen results")
    total = len(TABLES) + len(UNCHECKED)
    print(f"coverage: {len(TABLES)} of {total} tables in the paper are checked here")
    print("not checked by this script:")
    for label, why in sorted(UNCHECKED.items()):
        print(f"  {label:<20}{why}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

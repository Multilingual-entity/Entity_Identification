r"""Generate the paper's tables from the frozen results, and check the paper against them.

Every table here is built from the result files rather than typed. That is the whole
point: an earlier draft mixed a main text from the 601-item corpus with an appendix from
the 299-item one, and no amount of reading caught it, because a stale number reads
exactly like a fresh one.

    python paper_tables.py --check --results ../results_mr --results ../results_hi \
                           --paper ../paper/final_draft.tex
    python paper_tables.py --emit  --results ... --out tables_generated/

Results may live in more than one root; pass --results once per root, or set
RESULTS_DIR to a list separated by the OS path separator. A run folder such as
`qwen_hi` is looked up in each root in turn. The paper is given by --paper or PAPER_TEX.

--check extracts the numeric literals from each table body in the .tex and from the
generated version, and reports any that differ. It does not try to parse LaTeX: a table
whose numbers all match is correct whatever its formatting, and one whose numbers differ
needs regenerating whatever its formatting.

Every generator below reproduces the paper's layout: Marathi and English rows come from
the Marathi run, Hindi rows from the Hindi run, and English is never taken from the
Hindi run unless the paper's row says so.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from pipeline_common import dprime as _dprime  # noqa: E402

# Filled in by main() from --results / RESULTS_DIR. Kept module-level so the table
# functions stay simple.
RESULTS_ROOTS: list[Path] = []
PAPER: Path | None = None

# The three model families, and the run that holds each prompt language.
FAMILIES = {"qwen": "Qwen2.5-7B", "gemma": "Gemma-2-2b", "llama": "Llama-3.1-8B"}
# The polarity each model's causal gate reads, which decides which rows are the gated
# ones. Gemma accepts nearly every positive pair, so its failures are negatives.
GATE_TRUTH = {"qwen": 1, "gemma": 0, "llama": 1}
# Prompt language -> (run suffix, context code). English is read from the Marathi run.
LANG = {"Marathi": ("mr", "mr"), "Hindi": ("hi", "hi"), "English": ("mr", "en")}
# Models that appear in the behaviour table only, with the prompt languages they were
# run on. Sarvam-M has no Hindi corpus run and no causal analysis, so it contributes two
# rows to tab:behaviour and nothing else.
BEHAVIOUR_ONLY = {"mistral24": ("Mistral-Small 24B", ("Marathi", "English")),
                  "sarvam": ("Sarvam-M 24B", ("Marathi", "English"))}


def run_dir(run: str) -> Path:
    for root in RESULTS_ROOTS:
        if (root / run).is_dir():
            return root / run
    raise SystemExit(f"run folder {run!r} not found under any results root: "
                     f"{[str(r) for r in RESULTS_ROOTS]}")


def csv(run: str, rel: str) -> pd.DataFrame:
    return pd.read_csv(run_dir(run) / rel)


def gate(run: str) -> int:
    return GATE_TRUTH[run.split("_")[0]]


# --------------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------------

def _pooled(run: str) -> pd.DataFrame:
    """Accuracy pooled over paraphrases and answer keys, weighted by prompt count."""
    d = csv(run, "tables/behavior_summary.csv")
    return (d.groupby(["context", "condition", "truth"])
              .apply(lambda x: (x.accuracy * x.n).sum() / x.n.sum(), include_groups=False)
              .reset_index(name="acc"))


def table_behaviour() -> list[str]:
    rows = []
    for fam, label in FAMILIES.items():
        for lang, (suffix, ctx) in LANG.items():
            g = _pooled(f"{fam}_{suffix}")
            cells = []
            for cond in ("DEVDEV", "DEVLAT", "LATDEV", "LATLAT"):
                pos = float(g[(g.context == ctx) & (g.condition == cond) & (g.truth == 1)].acc.iloc[0])
                neg = float(g[(g.context == ctx) & (g.condition == cond) & (g.truth == 0)].acc.iloc[0])
                cells += [f"{pos * 100:.1f}", f"{neg * 100:.1f}"]
            name = label if lang == "Marathi" else " " * len(label)
            rows.append(f"{name} & {lang} & " + " & ".join(cells) + r"\\")
    for fam, (label, langs) in BEHAVIOUR_ONLY.items():
        for lang in langs:
            suffix, ctx = LANG[lang]
            g = _pooled(f"{fam}_{suffix}")
            cells = []
            for cond in ("DEVDEV", "DEVLAT", "LATDEV", "LATLAT"):
                pos = float(g[(g.context == ctx) & (g.condition == cond) & (g.truth == 1)].acc.iloc[0])
                neg = float(g[(g.context == ctx) & (g.condition == cond) & (g.truth == 0)].acc.iloc[0])
                cells += [f"{pos * 100:.1f}", f"{neg * 100:.1f}"]
            name = label if lang == langs[0] else " " * len(label)
            rows.append(f"{name} & {lang} & " + " & ".join(cells) + r"\\")
    return rows


def table_dprime() -> list[str]:
    """Sensitivity by model and prompt language, computed the way the pipeline does.

    The interval is bootstrapped by analysis/dprime_bootstrap.py and is not recomputed
    here; only the two d' columns are checked. Hindi rows are generated because the
    Hindi run exists; the paper acquires them under TODO 1.10, and until then this
    table reports them as absent.
    """
    rows = []
    for fam, short in (("qwen", "Qwen"), ("gemma", "Gemma"), ("llama", "Llama")):
        for lang, (suffix, ctx) in LANG.items():
            b = csv(f"{fam}_{suffix}", "tables/behavior.csv")
            vals = []
            for cond in ("DEVDEV", "LATLAT"):
                s = b[(b.context == ctx) & (b.condition == cond)]
                same, diff = s[s.truth == 1], s[s.truth == 0]
                hit = float((same.pred_semantic == "yes").mean())
                fa = float((diff.pred_semantic == "yes").mean())
                d, _ = _dprime(hit, fa, len(same), len(diff))
                vals.append(d)
            name = short if lang == "Marathi" else " " * len(short)
            rows.append(f"{name} & {lang} & {vals[0]:.3f} & {vals[1]:.3f}" + r"\\")
    return rows


# --------------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------------

def _patch(run: str) -> pd.DataFrame:
    d = csv(run, "patching/causal_patching_summary.csv")
    d["layer_num"] = pd.to_numeric(d.layer_or_window, errors="coerce")
    return d[(d.split == "test") & (d.truth == gate(run))]


def table_patch() -> list[str]:
    """Site columns average layers 10-18 over all three Latin-bearing donors; the
    control and reverse columns average every tested layer. Qwen only."""
    rows = []
    for lang, (suffix, ctx) in LANG.items():
        t = _patch(f"qwen_{suffix}")
        sf = t[(t.context == ctx) & (t.experiment == "single_factorial")
               & (t.layer_num.between(10, 18))]
        cells = [f"{sf[sf.site == s].mean_delta.mean():.2f}"
                 for s in ("E1", "E2", "Q", "READOUT")]
        for exp in ("wrong_entity_control", "wrong_position_control", "reverse_DD_into_LL"):
            cells.append(f"{t[(t.context == ctx) & (t.experiment == exp)].mean_delta.mean():.2f}")
        j = t[(t.context == ctx) & (t.experiment == "joint_12_17") & (t.site == "E1+E2+Q")]
        cells.append(f"{j.mean_delta.mean():.2f}")
        # The n in the paper's row label is the gate count, checked by tab:gatecounts.
        rows.append(f"{lang} & " + " & ".join(cells) + r"\\")
    return rows


def table_components() -> list[str]:
    """Qwen only, test split, columns Marathi, English, Hindi."""
    def comp(run, ctx, name):
        d = csv(run, "patching/component_patching_summary.csv")
        d = d[(d.truth == gate(run)) & (d.split == "test")]
        return float(d[(d.component == name) & (d.context == ctx)].mean_delta.mean())

    def head(run, ctx, exp):
        h = csv(run, "patching/head_confirm_summary.csv")
        h = h[(h.truth == gate(run)) & (h.split == "test")]
        return float(h[(h.experiment == exp) & (h.context == ctx)].mean_delta.iloc[0])

    cols = [("qwen_mr", "mr"), ("qwen_mr", "en"), ("qwen_hi", "hi")]
    rows = []
    for label, key in (("Full residual stream, single layer", "residual"),
                       ("Attention output only, single layer", "attention"),
                       ("MLP output only, single layer", "mlp"),
                       ("Attention output, cumulative", "attention_cumulative"),
                       ("MLP output, cumulative", "mlp_cumulative")):
        rows.append(f"{label} & " + " & ".join(f"{comp(r, c, key):.2f}" for r, c in cols) + r"\\")
    for label, exp in (("Best single head", "joint_top_1"),
                       ("Top 3 heads jointly", "joint_top_3"),
                       ("Top 5 heads jointly", "joint_top_5"),
                       ("Top 10 heads jointly", "joint_top_10"),
                       ("Mean of 12 selected heads, individually", "single_head")):
        rows.append(f"{label} & " + " & ".join(f"{head(r, c, exp):.2f}" for r, c in cols) + r"\\")
    return rows


def table_gatecounts() -> list[str]:
    """Gate = items passing the causal gate; Run = items actually patched after the cap
    of 60 per prompt language; Test = unique test-split items among them."""
    rows = []
    for fam, label in FAMILIES.items():
        for lang, suffix, ctx in (("Marathi", "mr", "mr"), ("Hindi", "hi", "hi"),
                                  ("English (Marathi run)", "mr", "en"),
                                  ("English (Hindi run)", "hi", "en")):
            run = f"{fam}_{suffix}"
            c = csv(run, "patching/causal_candidates.csv")
            p = csv(run, "patching/causal_patching.csv")
            g = c[(c.stable == 1) & (c.context == ctx)].fact_id.nunique()
            r = p[p.context == ctx].fact_id.nunique()
            t = p[(p.context == ctx) & (p.split == "test")].fact_id.nunique()
            name = label if lang == "Marathi" else " " * len(label)
            rows.append(f"{name} & {lang} & {g} & {r} & {t}" + r"\\")
    return rows


# --------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------

def table_probeheld() -> list[str]:
    """Layer chosen on validation, accuracy reported on test, question-mark position.

    The position filter matters: without it the validation argmax can land on the
    readout position and print a number the paper never reports.
    """
    rows = []
    for src, tgt in (("en", "en"), ("en", "mr"), ("mr", "en"), ("mr", "mr")):
        cells = []
        for fam in FAMILIES:
            d = csv(f"{fam}_mr", "tables/relation_probe_heldout_summary.csv")
            d = d[(d.source_context == src) & (d.target_context == tgt) & (d.position == "Q")]
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
    cols = [("qwen_mr", "mr"), ("qwen_mr", "en"), ("gemma_mr", "mr"),
            ("qwen_hi", "hi"), ("gemma_hi", "hi")]
    rows = []
    for key, label in order:
        cells = []
        for run, ctx in cols:
            d = csv(run, "tables/failure_prediction_oof_auc.csv")
            hit = d[(d.feature_set == key) & (d.context == ctx)]
            cells.append(f"{float(hit.oof_auc.iloc[0]):.3f}" if len(hit) else "---")
        rows.append(f"{label} & " + " & ".join(cells) + r"\\")
    return rows


# --------------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------------

def table_recovery() -> list[str]:
    """Qwen, held-out test items: every anchor and every vector variant, both
    polarities, columns Marathi, English, Hindi."""
    def anchor(run, ctx, truth, name):
        a = csv(run, "recovery/prompt_anchor_summary.csv")
        r = a[(a.split == "test") & (a.context == ctx) & (a.truth == truth) & (a.anchor == name)]
        return float(r.accuracy.iloc[0])

    def vector(run, ctx, truth, kind):
        v = csv(run, "recovery/vector_recovery_test_summary.csv")
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
            ("Lat--Lat reference", anchor, "latin_only_ceiling")]
    cols = [("qwen_mr", "mr"), ("qwen_mr", "en"), ("qwen_hi", "hi")]
    rows = []
    for label, fn, key in spec:
        cells = []
        for run, ctx in cols:
            cells += [f"{fn(run, ctx, 1, key):.3f}", f"{fn(run, ctx, 0, key):.3f}"]
        rows.append(f"{label} & " + " & ".join(cells) + r"\\")
    return rows


def table_vectorsel() -> list[str]:
    """Which layer and scale the validation split selected. The paper prints the
    Marathi run's two prompt languages first, then the Hindi run's two."""
    rows = []
    for suffix, native in (("mr", "Marathi"), ("hi", "Hindi")):
        for fam, label in FAMILIES.items():
            run = f"{fam}_{suffix}"
            h = csv(run, "recovery/vector_hyperparameters_selected.csv")
            v = csv(run, "recovery/vector_recovery_test_summary.csv")
            for ctx, ctx_label in ((suffix, native), ("en", "English")):
                hv = h[h.context == ctx]
                if hv.empty:
                    continue
                add = v[(v.context == ctx) & (v.vector_kind == "additive")]
                pos = float(add[add.truth == 1].accuracy.iloc[0])
                neg = float(add[add.truth == 0].accuracy.iloc[0])
                name = label if ctx == suffix else ""
                rows.append(f"{name} & {ctx_label} & {int(hv.layer.iloc[0])} & "
                            f"{float(hv.alpha.iloc[0]):.1f} & {pos:.3f} & {neg:.3f}" + r"\\")
    return rows


def table_grid() -> list[str]:
    """Qwen validation grid, positive-pair accuracy, the four scales the paper prints;
    Marathi corpus then Hindi corpus, native prompts then English."""
    rows = []
    for suffix in ("mr", "hi"):
        v = csv(f"qwen_{suffix}", "recovery/vector_tuning_validation.csv")
        v = v[v.truth == 1]
        for layer in (10, 12, 14, 16, 18):
            cells = []
            for ctx in (suffix, "en"):
                for alpha in (0.5, 2.0, 4.0, 8.0):
                    g = v[(v.context == ctx) & (v.layer == layer) & (v.alpha == alpha)]
                    cells.append(f"{g.correct.mean():.3f}" if len(g) else "?")
            rows.append(f"{layer} & " + " & ".join(cells) + r"\\")
    return rows


def table_romanization() -> list[str]:
    """Both arms against the Devanagari baseline and the Lat--Lat reference."""
    rows = []
    for fam, label in FAMILIES.items():
        for lang, (suffix, ctx) in LANG.items():
            run = f"{fam}_{suffix}"
            s = csv(run, "recovery/romanization_scheme_summary.csv")
            r = csv(run, "recovery/romanization_reference.csv")
            ref = r[(r.split == "test") & (r.context == ctx)].set_index("condition").accuracy
            t = s[(s.split == "test") & (s.context == ctx)]
            auto = t[t.scheme != "gold_lat"]
            rep = auto[auto.arm == "replaced"].accuracy
            anc = auto[auto.arm == "anchored"].accuracy
            if rep.empty:
                continue
            name = label if lang == "Marathi" else " " * len(label)
            rows.append(f"{name} & {lang} & {ref['DEVDEV']:.3f} & "
                        f"{rep.min():.3f}--{rep.max():.3f} & {anc.min():.3f}--{anc.max():.3f} & "
                        f"{ref['LATLAT']:.3f}" + r"\\")
    return rows


# --------------------------------------------------------------------------------
# corpus and fragmentation
# --------------------------------------------------------------------------------

def table_tiers() -> list[str]:
    c = csv("qwen_mr", "data/corpus.csv")
    counts = c.relation_tier.value_counts()
    order = ["alias_observed", "explicit_plus_alias:birth_name",
             "explicit_plus_alias:pseudonym", "explicit_plus_alias:nickname",
             "explicit_observed:birth_name", "explicit_observed:nickname"]
    return [f"{k} & {int(counts.get(k, 0))}" + r"\\" for k in order if k in counts]


def table_reclen() -> list[str]:
    """Isolated-name retrieval by Devanagari token count, layer 8, all 1202 names."""
    tok = csv("qwen_mr", "tables/tokenization.csv")
    dev = tok[tok.script == "dev"][["fact_id", "role", "n_tokens"]]
    r = csv("qwen_mr", "tables/isolated_retrieval_ranks.csv")
    r = r[r.layer == 8].merge(dev, on=["fact_id", "role"])
    rows = []
    for lo, hi in ((0, 10), (11, 15), (16, 20), (21, 10_000)):
        s = r[(r.n_tokens >= lo) & (r.n_tokens <= hi)]
        rows.append(f"{len(s)} & {s.n_tokens.mean():.1f} & "
                    f"{100 * (s['rank'] == 1).mean():.1f}" + r"\\")
    return rows


def _benefit(run: str, ctx: str) -> pd.DataFrame:
    """Per-fact margin gained by switching a positive pair to Latin, native prompts."""
    b = csv(run, "tables/behavior.csv")
    b = b[(b.truth == 1) & (b.context == ctx)]
    piv = b.groupby(["fact_id", "condition"]).yes_minus_no_margin.mean().unstack()
    d = (piv["LATLAT"] - piv["DEVDEV"]).rename("benefit").reset_index()
    ratio = (csv(run, "tables/fragmentation_vs_ignition.csv")
             .groupby("fact_id").dev_lat_token_ratio.mean().rename("ratio").reset_index())
    return d.merge(ratio, on="fact_id")


def table_fragquart() -> list[str]:
    rows = []
    for run, ctx in (("qwen_mr", "mr"), ("qwen_hi", "hi")):
        d = _benefit(run, ctx)
        d["q"] = pd.qcut(d.ratio, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        rows += [f"{q} & {g.ratio.mean():.2f} & {g.benefit.mean():.2f}" + r"\\"
                 for q, g in d.groupby("q", observed=True)]
    return rows


# --------------------------------------------------------------------------------
# registry and checker
# --------------------------------------------------------------------------------

TABLES = {
    "tab:behaviour": table_behaviour,
    "tab:dprime": table_dprime,
    "tab:patch": table_patch,
    "tab:components": table_components,
    "tab:gatecounts": table_gatecounts,
    "tab:probeheld": table_probeheld,
    "tab:auc": table_auc,
    "tab:recovery": table_recovery,
    "tab:vectorsel": table_vectorsel,
    "tab:grid": table_grid,
    "tab:romanization": table_romanization,
    "tab:tiers": table_tiers,
    "tab:reclen": table_reclen,
    "tab:fragquart": table_fragquart,
}

# Tables whose row labels carry numbers a cell-value generator cannot emit: bin edges
# in tab:reclen, gated-item counts in tab:patch, the total row in tab:tiers, and the
# bootstrap interval in tab:dprime. For these the rule is that every generated number
# must appear in the paper, so a changed cell is still caught while the un-generated
# labels are tolerated.
SUBSET_ONLY = {"tab:dprime", "tab:patch", "tab:tiers", "tab:reclen"}

# Tables whose numbers this script does not regenerate, with the reason. All were
# verified by hand against the result files on 2026-09-06 using the reductions recorded
# in TODO.md section 7; this list records only what --check cannot do on its own.
UNCHECKED = {
    "tab:probepos": "group-recall rows have no single source file; position rows verified by hand",
    "tab:inprompt": "layer-8 top-1 does not reproduce under any single other_entity_script filter",
    "tab:donor": "donor variants need a source_condition mapping not recorded in the summary",
    "tab:gate": "definitional, no numbers from results",
    "tab:para": "per-paraphrase split not stored separately from behavior_summary",
    "tab:tierbeh": "needs corpus join not stored in results",
    "tab:fragcorr": "correlations recomputed in-session; only the ignition row is in a result file",
    "tab:translit": "romanization quality table, separate scheme-level source",
    "tab:probelayer": "layer sweep, out-of-fold; source_context == target_context, position Q",
    "tab:profile": "Lat--Lat source, single layer, pooled over splits",
    "tab:windows": "pooled fact-level rows, truth=1, Lat--Lat donor",
    "tab:gemmaprofile": "as tab:profile, gated on negatives, margin negated",
    "tab:llamapara": "Llama paraphrase table, appendix only",
    "tab:models": "cross-model summary assembled from many sources",
    "tab:corpuserr": "textual, no computed numbers",
}

NUMBER = re.compile(r"-?\d+\.\d+|\b\d+\b")


def numbers_in(text: str) -> list[str]:
    """Numeric literals, ignoring LaTeX column counts and layer labels in \\multicolumn."""
    stripped = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", text)
    stripped = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", stripped)
    return NUMBER.findall(stripped)


def paper_table_body(label: str) -> str | None:
    r"""The rows of one table in the paper, whether or not it spans both columns.

    The backreference keeps \begin{table*} paired with \end{table*}; matching
    `table\*?` on both ends would let a starred opening close on an unstarred ending and
    swallow everything between two different tables.
    """
    text = PAPER.read_text(encoding="utf-8")
    for m in re.finditer(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", text, re.S):
        if f"\\label{{{label}}}" in m.group(0):
            body = re.search(r"\\midrule(.*?)\\bottomrule", m.group(0), re.S)
            return body.group(1) if body else m.group(0)
    return None


def main() -> None:
    global RESULTS_ROOTS, PAPER
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--results", action="append", type=Path,
                    help="results root holding run folders; repeatable. "
                         "Default: RESULTS_DIR, split on the OS path separator.")
    ap.add_argument("--paper", type=Path, help="LaTeX source to check. Default: PAPER_TEX.")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("TABLES_OUT", HERE / "tables_generated")))
    args = ap.parse_args()

    roots = args.results or [Path(p) for p in os.environ.get("RESULTS_DIR", "").split(os.pathsep) if p]
    if not roots:
        raise SystemExit("no results root: pass --results or set RESULTS_DIR")
    RESULTS_ROOTS = [r.resolve() for r in roots]
    paper = args.paper or (Path(os.environ["PAPER_TEX"]) if os.environ.get("PAPER_TEX") else None)
    if args.check and paper is None:
        raise SystemExit("no paper: pass --paper or set PAPER_TEX")
    PAPER = paper

    if args.emit:
        args.out.mkdir(exist_ok=True)
        for label, fn in TABLES.items():
            path = args.out / (label.replace(":", "_") + ".tex")
            path.write_text("\n".join(fn()) + "\n", encoding="utf-8")
            print(f"  wrote {path.name}")
        return

    failures = 0
    for label, fn in TABLES.items():
        generated = "\n".join(fn())
        body = paper_table_body(label)
        if body is None:
            # Counted as a failure. A table the checker cannot locate is unverified,
            # and saying so quietly is how tab:behaviour went unchecked for a while.
            failures += 1
            print(f"MISSING  {label:<20}not found in the paper")
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

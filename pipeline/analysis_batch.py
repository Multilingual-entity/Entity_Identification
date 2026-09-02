"""Analyses that need only the finished CSVs. No GPU, no model, no new runs.

Six items that were flagged during review of the paper and never resolved. Each either
corrects a number already reported, or extends a result from one model to three. Run
after the pipeline stages; reads whatever run directories it is pointed at.

    python analysis_batch.py --runs qwen=../../archive/299_fact/qwen \
                                    gemma=../../archive/299_fact/gemma \
                                    llama=../../archive/299_fact/llama
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load(run: Path, rel: str) -> pd.DataFrame | None:
    p = run / rel
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:                                                # noqa: BLE001
        return None


# ---------------------------------------------------------------- 1. auto_both
def itrans_is_clean(text: str) -> bool:
    """ITRANS leaves Devanagari characters in its output for names containing the
    vowel sign U+0949, so a fifth of the corpus received a mixed-script hybrid rather
    than a romanization. Those items cannot test what the condition claims to test."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return True
    out = transliterate(str(text), sanscript.DEVANAGARI, sanscript.ITRANS)
    return not any(ord(ch) > 127 for ch in out)


def auto_both_on_clean(run: Path) -> None:
    rule("1. auto_both recomputed on the names ITRANS handles cleanly")
    anchors = load(run, "recovery/prompt_anchor_recovery.csv")
    corpus = load(run, "data/corpus.csv")
    if anchors is None or corpus is None:
        print("  missing prompt_anchor_recovery.csv or corpus.csv"); return

    clean = {str(r.fact_id) for r in corpus.itertuples()
             if itrans_is_clean(r.name_a_dev) and itrans_is_clean(r.name_b_dev)}
    print(f"  {len(clean)} of {len(corpus)} names romanize to pure ASCII")

    a = anchors[(anchors.anchor == "auto_both") & (anchors.truth == 1)]
    base = load(run, "tables/behavior.csv")
    for ctx in sorted(a.context.unique()):
        sub = a[a.context == ctx]
        all_acc = sub.correct.mean()
        cl_acc = sub[sub.fact_id.astype(str).isin(clean)].correct.mean()
        row = [f"  {ctx}: all items {all_acc:.3f}", f"clean only {cl_acc:.3f}"]
        if base is not None:
            b = base[(base.context == ctx) & (base.condition == "DEVDEV") & (base.truth == 1)]
            bc = b[b.fact_id.astype(str).isin(clean)].correct.mean()
            row.append(f"Devanagari baseline, same items {bc:.3f}")
        print("   | ".join(row))
    print("  If the clean subset still sits below its Devanagari baseline, the")
    print("  romanization result holds on uncontaminated input.")


# ---------------------------------------------- 2. drop the known corpus errors
CORPUS_ERRORS = ["q485697", "q855252", "q213812", "q35811", "q61047",
                 "q11576", "q310817", "q36153", "q41532", "q13117994"]


"""Columns worth grouping by. Anything with many distinct values, such as layer, would
split the tables so finely that a single dropped item moves every cell, which says nothing
about whether a headline changes."""
GROUP_KEYS = ["context", "condition", "truth", "anchor", "vector_kind", "experiment",
              "site", "component", "position", "order", "scheme", "split"]
VALUE_KEYS = ["correct", "patched_correct", "baseline_correct", "accuracy", "delta_margin",
              "yes_minus_no_margin", "mean_delta", "recovery_rate", "flip_rate",
              "probe_same_recall_exact_failures", "top1"]


def without_corpus_errors(run: Path) -> None:
    rule("2. Behaviour with the ten mismatched-name items removed")
    beh = load(run, "tables/behavior.csv")
    if beh is None:
        print("  no behaviour table"); return
    beh["fact_id"] = beh.fact_id.astype(str)
    present = sorted(set(CORPUS_ERRORS) & set(beh.fact_id))
    print(f"  {len(present)} of the ten are in this run: {', '.join(present) or 'none'}")
    if not present:
        return
    keep = beh[~beh.fact_id.isin(present)]
    for ctx in sorted(beh.context.unique()):
        out = []
        for label, frame in (("with", beh), ("without", keep)):
            g = frame[(frame.context == ctx) & (frame.truth == 1)]
            acc = g.groupby("condition").correct.mean() * 100
            out.append(f"{label}: DD {acc.get('DEVDEV', np.nan):.1f} LL {acc.get('LATLAT', np.nan):.1f}")
        print(f"  {ctx}  " + "  |  ".join(out))


def all_aggregates_without_errors(run: Path, top: int = 6) -> None:
    """Every per-fact table in the run, recomputed with the mismatched items dropped.

    Behaviour is not the only place those items appear. They are in the probes, in the
    patching set, and in the recovery arms, and a claim only needs to move in one of them
    to matter. Rather than naming files, this walks every CSV carrying a fact_id, so a
    table added later is covered without anyone remembering to add it here.

    Reported as the largest absolute change per file, because the question is not what
    each cell becomes but whether any headline moves.
    """
    rule("2b. All per-fact tables recomputed without the mismatched items")
    errors = set(CORPUS_ERRORS)
    files = sorted(p for folder in ("tables", "patching", "recovery")
                   for p in (run / folder).glob("*.csv")
                   if (run / folder).is_dir() and "partial" not in p.name)
    if not files:
        print("  no tables found"); return

    any_movement = False
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception:                                                # noqa: BLE001
            continue
        if "fact_id" not in frame.columns:
            continue
        frame["fact_id"] = frame.fact_id.astype(str)
        present = errors & set(frame.fact_id)
        if not present:
            continue
        keys = [c for c in GROUP_KEYS if c in frame.columns and frame[c].nunique() <= 12]
        values = [c for c in VALUE_KEYS
                  if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
        if not values:
            continue
        kept = frame[~frame.fact_id.isin(present)]
        if keys:
            before = frame.groupby(keys)[values].mean()
            after = kept.groupby(keys)[values].mean()
        else:
            before = frame[values].mean().to_frame().T
            after = kept[values].mean().to_frame().T
        comparison = pd.concat(
            [before.stack().rename("before"), after.stack().rename("after")], axis=1
        ).dropna()
        if comparison.empty:
            continue
        comparison["change"] = comparison.after - comparison.before
        comparison = comparison.reindex(
            comparison.change.abs().sort_values(ascending=False).index)
        worst = float(comparison.change.abs().max())
        print(f"\n  {path.relative_to(run).as_posix()}")
        print(f"    {len(present)} affected items, {frame.fact_id.nunique()} facts, "
              f"largest change {worst:+.4f}")
        if worst >= 0.005:
            any_movement = True
            for index, row in comparison.head(top).iterrows():
                parts = index if isinstance(index, tuple) else (index,)
                label = ", ".join(f"{k}={v}" for k, v in zip(keys, parts[:-1])) or "overall"
                print(f"      {label} | {parts[-1]}: {row.before:+.4f} -> "
                      f"{row.after:+.4f} ({row.change:+.4f})")
    if not any_movement:
        print("\n  No cell in any table moves by 0.005 or more. The mismatched items can")
        print("  be dropped from the corpus without restating a single reported number,")
        print("  which is the cleanest way to close this out.")
    else:
        print("\n  At least one cell moves materially. Those numbers need restating from")
        print("  the corrected corpus rather than annotating.")


# ------------------------------------ 3. are the errors inside the causal set?
def errors_in_causal_set(run: Path) -> None:
    rule("3. Do the mismatched items fall inside the patched set?")
    cand = load(run, "patching/causal_candidates.csv")
    if cand is None:
        print("  no causal candidates"); return
    sel = cand[cand.stable == 1]
    hit = sorted(set(CORPUS_ERRORS) & set(sel.fact_id.astype(str)))
    print(f"  stable items: {len(sel)} | mismatched items among them: {len(hit)}")
    if hit:
        print(f"  {', '.join(hit)}")
        print("  These contribute to the reported patching effect and should be excluded")
        print("  before the numbers are final.")
    else:
        print("  None. The patching results are unaffected by the corpus errors.")


# ------------------------------- 4. do a model's failures pass its own gate?
def failures_versus_gate(run: Path, name: str) -> None:
    rule(f"4. {name}: do the analysed failures involve facts the model actually knows?")
    gate = load(run, "tables/knowledge_gate.csv")
    ef = load(run, "tables/exact_failure_probe_summary.csv")
    if gate is None or ef is None:
        print("  missing knowledge gate or exact-failure table"); return
    for ctx in sorted(ef.context.unique()):
        col = f"{ctx}_known_any"
        if col not in gate.columns:
            continue
        known = int(gate[col].sum())
        n_fail = int(ef[ef.context == ctx].n_exact_failures.max())
        print(f"  {ctx}: {n_fail} failures analysed, {known} of {len(gate)} items known in some script")
    print("  A model whose failures are mostly on items it never learned will show a")
    print("  probe recall near chance, which is a different result from knowledge that")
    print("  is present but unreachable.")


# --------------------------------------- 5. attention vs MLP, resolved by layer
def components_by_layer(run: Path) -> None:
    rule("5. Attention against MLP per layer, rather than averaged over the band")
    c = load(run, "patching/component_patching_summary.csv")
    if c is None:
        print("  no component patching"); return
    c = c[c.truth == 1]
    for ctx in sorted(c.context.unique()):
        # Weighted by the number of facts behind each cell. Done with sums rather than
        # groupby.apply so it does not depend on the pandas version on the cluster,
        # where include_groups is not always available.
        sub = c[c.context == ctx].assign(_weighted=lambda f: f.mean_delta * f.n)
        totals = sub.groupby(["layer", "component"])[["_weighted", "n"]].sum()
        piv = (totals["_weighted"] / totals["n"].replace(0, np.nan)).unstack()
        print(f"\n  {ctx}")
        print("   " + piv.round(3).to_string().replace("\n", "\n   "))
        if {"attention", "mlp"} <= set(piv.columns):
            ratio = (piv["attention"] / piv["mlp"].replace(0, np.nan)).round(2)
            best = piv["attention"].idxmax()
            print(f"   attention peaks at layer {best}; attention/MLP ratio by layer:")
            print("   " + ratio.to_string().replace("\n", "\n   "))


# ------------------------ 6. band-corrected exact-failure recall for deep models
def exact_failure_band(run: Path, name: str, reference_layers: int = 28) -> None:
    rule(f"6. {name}: exact-failure recall over a depth-matched band")
    ef = load(run, "tables/exact_failure_probe_summary.csv")
    if ef is None:
        print("  no exact-failure table"); return
    n_layers = int(ef.layer.max()) + 1
    lo = int(round(10 * n_layers / reference_layers))
    hi = int(round(18 * n_layers / reference_layers))
    print(f"  model depth {n_layers}; the 10-18 band on a {reference_layers}-layer model "
          f"corresponds to {lo}-{hi} here")
    for ctx in sorted(ef.context.unique()):
        s = ef[ef.context == ctx]
        a = s[s.layer.between(10, 18)].groupby("position").probe_same_recall_exact_failures.mean()
        b = s[s.layer.between(lo, hi)].groupby("position").probe_same_recall_exact_failures.mean()
        for pos in sorted(set(a.index) | set(b.index)):
            print(f"  {ctx} {pos}: layers 10-18 {a.get(pos, np.nan):.3f} "
                  f"| depth-matched {lo}-{hi} {b.get(pos, np.nan):.3f}")


# ---------------------------- 7. extend single-model analyses to the other models
def extend_to_all(runs: dict) -> None:
    rule("7. Analyses currently reported for one model, extended to all")

    print("\n  Isolated-name retrieval, top-1 at layer 8")
    for name, run in runs.items():
        d = load(run, "tables/isolated_retrieval_summary.csv")
        if d is None:
            print(f"    {name}: absent"); continue
        s = d[d.layer == 8].set_index("role").top1
        print(f"    {name}: first {s.get('a', np.nan):.3f}  second {s.get('b', np.nan):.3f}")

    print("\n  Exact-failure recall, layers 10-18")
    for name, run in runs.items():
        d = load(run, "tables/exact_failure_probe_summary.csv")
        if d is None:
            print(f"    {name}: absent"); continue
        s = d[d.layer.between(10, 18)].groupby(["context", "position"]).\
            probe_same_recall_exact_failures.mean()
        print(f"    {name}: " + "  ".join(f"{c}/{p} {v:.3f}" for (c, p), v in s.items()))

    print("\n  Context-by-script interaction, mean margin points")
    for name, run in runs.items():
        d = load(run, "tables/context_script_interaction_per_fact.csv")
        if d is None:
            print(f"    {name}: absent"); continue
        col = [c for c in d.columns if "interaction" in c]
        if col:
            print(f"    {name}: {d[col[0]].mean():+.3f}  "
                  f"(positive means the gap is larger under English prompts)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=path pairs, e.g. qwen=../../archive/299_fact/qwen")
    ap.add_argument("--primary", default=None,
                    help="which run the corpus-level checks use; defaults to the first")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec}")
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"skipping {name}: {p} does not exist"); continue
        runs[name] = p
    if not runs:
        raise SystemExit("no run directories found")

    primary = args.primary or next(iter(runs))
    print(f"runs: {', '.join(runs)} | primary: {primary}")

    auto_both_on_clean(runs[primary])
    without_corpus_errors(runs[primary])
    for name, run in runs.items():
        print(f"\n[{name}]")
        all_aggregates_without_errors(run)
    errors_in_causal_set(runs[primary])
    for name, run in runs.items():
        failures_versus_gate(run, name)
    components_by_layer(runs[primary])
    for name, run in runs.items():
        ef = load(run, "tables/exact_failure_probe_summary.csv")
        if ef is not None and int(ef.layer.max()) + 1 != 28:
            exact_failure_band(run, name)
    extend_to_all(runs)
    print()


if __name__ == "__main__":
    main()

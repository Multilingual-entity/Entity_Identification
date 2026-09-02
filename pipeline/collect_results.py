"""Pull the headline number out of every finished analysis, for every model, into one place.

The archive keeps the files. This keeps the findings. Forty-five tables per run across
three models is not something anyone reads, and the numbers the paper actually rests on are
scattered across them: an accuracy here, a peak layer there, a patching delta three
directories away. When the corpus is rescaled and the tables are regenerated, the question
that matters is which of those headline numbers moved, and that cannot be answered by
diffing forty-five files.

Every row of the output names its own source file, so any number can be traced back and
recomputed. Nothing here is a new analysis; it is a reading of analyses already done.

    python collect_results.py --runs qwen=../../archive/299_fact/qwen \
                                     gemma=../../archive/299_fact/gemma \
                                     llama=../../archive/299_fact/llama \
                              --out ../../archive/299_fact/results

Writes results_collected.csv, one row per number, and RESULTS.md, the same content grouped
for reading.

Two things it deliberately does not do. It does not fill in a number a run does not have:
Gemma has no causal patching because its accuracy gate selected nothing, and that absence is
a finding rather than a gap to be papered over. And it does not average across models, since
they differ in depth, in bias, and in whether they can do the task in Marathi at all.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import dprime

REFERENCE_LAYERS = 28
BAND = (10, 18)          # the preregistered band, on a 28-layer model


def load(run: Path, rel: str):
    path = run / rel
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:                                                    # noqa: BLE001
        return None


class Collector:
    def __init__(self, name: str, run: Path):
        self.name, self.run, self.rows = name, run, []

    def add(self, section: str, metric: str, value, source: str, **keys) -> None:
        record = {"model": self.name, "section": section, "metric": metric}
        record.update(keys)
        record["value"] = None if value is None else float(value)
        record["source"] = source
        self.rows.append(record)

    # ---------------------------------------------------------------- behaviour
    def behaviour(self) -> None:
        beh = load(self.run, "tables/behavior.csv")
        if beh is None:
            return
        for (context, condition), sub in beh.groupby(["context", "condition"]):
            same, diff = sub[sub.truth == 1], sub[sub.truth == 0]
            if same.empty or diff.empty:
                continue
            hit = float((same.pred_semantic == "yes").mean())
            fa = float((diff.pred_semantic == "yes").mean())
            d, criterion = dprime(hit, fa, len(same), len(diff))
            common = dict(context=context, condition=condition)
            src = "tables/behavior.csv"
            self.add("behaviour", "accuracy", sub.correct.mean(), src, **common)
            self.add("behaviour", "hit_rate", hit, src, **common)
            self.add("behaviour", "false_alarm_rate", fa, src, **common)
            self.add("behaviour", "d_prime", d, src, **common)
            self.add("behaviour", "criterion", criterion, src, **common)
            self.add("behaviour", "mean_margin", sub.yes_minus_no_margin.mean(), src, **common)
        # The script gap is the paper's opening claim, so it is stated directly rather
        # than left to be subtracted from two rows.
        for context, sub in beh[beh.condition.isin(["DEVDEV", "LATLAT"])].groupby("context"):
            acc = sub.groupby("condition").correct.mean()
            if {"DEVDEV", "LATLAT"} <= set(acc.index):
                self.add("behaviour", "script_gap_accuracy",
                         acc["LATLAT"] - acc["DEVDEV"], "tables/behavior.csv", context=context)

    def knowledge_gate(self) -> None:
        gate = load(self.run, "tables/knowledge_gate.csv")
        if gate is None:
            return
        for column in [c for c in gate.columns if c.endswith("_known_any")]:
            self.add("knowledge gate", "items_known_in_some_script", gate[column].sum(),
                     "tables/knowledge_gate.csv", context=column[: -len("_known_any")])
        self.add("knowledge gate", "items_total", len(gate), "tables/knowledge_gate.csv")

    def tokenizer(self) -> None:
        tok = load(self.run, "tables/tokenization.csv")
        if tok is None:
            return
        for script, sub in tok.groupby("script"):
            self.add("tokenizer", "mean_tokens_per_name", sub.n_tokens.mean(),
                     "tables/tokenization.csv", script=script)
            self.add("tokenizer", "mean_tokens_per_char",
                     (sub.n_tokens / sub.n_chars.clip(lower=1)).mean(),
                     "tables/tokenization.csv", script=script)

    # ------------------------------------------------------------------ probes
    def isolated_retrieval(self) -> None:
        iso = load(self.run, "tables/isolated_retrieval_summary.csv")
        if iso is None:
            return
        for role, sub in iso.groupby("role"):
            best = sub.loc[sub.top1.idxmax()]
            self.add("isolated retrieval", "top1_peak", best.top1,
                     "tables/isolated_retrieval_summary.csv", role=role, layer=int(best.layer))

    def in_prompt_retrieval(self) -> None:
        pair = load(self.run, "tables/pair_retrieval_summary.csv")
        if pair is None:
            return
        # The claim is about reading the two names out of the prompt, so it is read at the
        # entity positions with both names in the native script.
        sub = pair[(pair.position.isin(["E1", "E2"]))] if "position" in pair else pair
        for keys, group in sub.groupby([c for c in ("context", "role", "position")
                                        if c in sub.columns]):
            best = group.loc[group.top1.idxmax()]
            labels = dict(zip([c for c in ("context", "role", "position")
                               if c in sub.columns],
                              keys if isinstance(keys, tuple) else (keys,)))
            self.add("in-prompt retrieval", "top1_peak", best.top1,
                     "tables/pair_retrieval_summary.csv", layer=int(best.layer), **labels)

    def relation_probe(self) -> None:
        for rel, section in (("tables/relation_probe_summary.csv", "relation probe"),
                             ("tables/relation_probe_heldout_summary.csv",
                              "relation probe, held out")):
            frame = load(self.run, rel)
            if frame is None:
                continue
            keys = [c for c in ("source_context", "target_context", "position",
                                "evaluation_split") if c in frame.columns]
            for values, group in frame.groupby(keys):
                best = group.loc[group.balanced_accuracy.idxmax()]
                labels = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
                self.add(section, "balanced_accuracy_peak", best.balanced_accuracy, rel,
                         layer=int(best.layer), **labels)

    def exact_failure(self) -> None:
        ef = load(self.run, "tables/exact_failure_probe_summary.csv")
        if ef is None:
            return
        n_layers = int(ef.layer.max()) + 1
        lo = int(round(BAND[0] * n_layers / REFERENCE_LAYERS))
        hi = int(round(BAND[1] * n_layers / REFERENCE_LAYERS))
        for (context, position), sub in ef.groupby(["context", "position"]):
            fixed = sub[sub.layer.between(*BAND)].probe_same_recall_exact_failures.mean()
            matched = sub[sub.layer.between(lo, hi)].probe_same_recall_exact_failures.mean()
            common = dict(context=context, position=position)
            src = "tables/exact_failure_probe_summary.csv"
            self.add("exact-failure probe", f"recall_layers_{BAND[0]}_{BAND[1]}", fixed,
                     src, **common)
            # For a model of another depth the fixed band points somewhere else, so the
            # depth-matched value is the one that compares across models.
            self.add("exact-failure probe", "recall_depth_matched", matched, src,
                     depth_matched_band=f"{lo}-{hi}", **common)
            self.add("exact-failure probe", "n_exact_failures",
                     sub.n_exact_failures.max(), src, **common)

    def extras(self) -> None:
        frag = load(self.run, "tables/fragmentation_correlation.csv")
        if frag is not None and len(frag):
            self.add("fragmentation", "spearman_rho", frag.rho.iloc[0],
                     "tables/fragmentation_correlation.csv")
            self.add("fragmentation", "p_value", frag.p_value.iloc[0],
                     "tables/fragmentation_correlation.csv")
        auc = load(self.run, "tables/failure_prediction_oof_auc.csv")
        if auc is not None:
            for row in auc.itertuples():
                self.add("failure prediction", "oof_auc", row.oof_auc,
                         "tables/failure_prediction_oof_auc.csv",
                         context=row.context, feature_set=row.feature_set)
        inter = load(self.run, "tables/context_script_interaction_per_fact.csv")
        if inter is not None:
            column = [c for c in inter.columns if "interaction" in c]
            if column:
                self.add("context by script", "mean_interaction", inter[column[0]].mean(),
                         "tables/context_script_interaction_per_fact.csv")

    # ---------------------------------------------------------------- patching
    def causal(self) -> None:
        candidates = load(self.run, "patching/causal_candidates.csv")
        if candidates is not None:
            stable = candidates[candidates.stable == 1]
            src = "patching/causal_candidates.csv"
            # Two different counts, and they are easy to confuse. A fact can pass the gate
            # in English and in Marathi, so the per-context counts sum to more than the
            # number of distinct facts. Patching is run per context, so the per-context
            # count is the one that limits each causal claim.
            self.add("causal gate", "distinct_facts_passing_any_context",
                     stable.fact_id.nunique(), src)
            self.add("causal gate", "context_by_fact_pairs_passing", len(stable), src)
            for context, sub in stable.groupby("context"):
                self.add("causal gate", "facts_passing_in_this_context",
                         sub.fact_id.nunique(), src, context=context)

        summary = load(self.run, "patching/causal_patching_summary.csv")
        if summary is None:
            return
        true_pairs = summary[summary.truth == 1]
        src = "patching/causal_patching_summary.csv"
        for (context, experiment, site), group in true_pairs.groupby(
                ["context", "experiment", "site"]):
            best = group.loc[group.mean_delta.idxmax()]
            common = dict(context=context, experiment=experiment, site=site,
                          layer_or_window=str(best.layer_or_window))
            self.add("causal patching", "mean_delta_best", best.mean_delta, src, **common)
            if "flip_rate" in group.columns:
                self.add("causal patching", "flip_rate_at_best", best.flip_rate, src, **common)

    def components(self) -> None:
        comp = load(self.run, "patching/component_patching_summary.csv")
        if comp is None:
            return
        true_pairs = comp[comp.truth == 1]
        src = "patching/component_patching_summary.csv"
        for (context, component), group in true_pairs.groupby(["context", "component"]):
            weighted = ((group.mean_delta * group.n).sum() / group.n.sum()
                        if group.n.sum() else np.nan)
            self.add("component patching", "mean_delta_over_band", weighted, src,
                     context=context, component=component)
            best = group.loc[group.mean_delta.idxmax()]
            self.add("component patching", "mean_delta_best_layer", best.mean_delta, src,
                     context=context, component=component, layer=int(best.layer))

    def heads(self) -> None:
        head = load(self.run, "patching/head_confirm_summary.csv")
        if head is None:
            return
        for row in head[head.truth == 1].itertuples():
            self.add("head patching", "mean_delta", row.mean_delta,
                     "patching/head_confirm_summary.csv",
                     context=row.context, experiment=row.experiment, split=row.split)

    # ---------------------------------------------------------------- recovery
    def recovery(self) -> None:
        anchors = load(self.run, "recovery/prompt_anchor_summary.csv")
        if anchors is not None:
            test = anchors[anchors.split == "test"] if "split" in anchors else anchors
            for (context, anchor), group in test.groupby(["context", "anchor"]):
                self.add("prompt anchors", "accuracy_test",
                         (group.accuracy * group.n).sum() / group.n.sum()
                         if group.n.sum() else np.nan,
                         "recovery/prompt_anchor_summary.csv",
                         context=context, anchor=anchor)
        vectors = load(self.run, "recovery/vector_recovery_test_summary.csv")
        if vectors is not None:
            for (context, kind), group in vectors.groupby(["context", "vector_kind"]):
                self.add("steering vectors", "accuracy_test", group.accuracy.mean(),
                         "recovery/vector_recovery_test_summary.csv",
                         context=context, vector_kind=kind)
        chosen = load(self.run, "recovery/vector_hyperparameters_selected.csv")
        if chosen is not None:
            for row in chosen.itertuples():
                self.add("steering vectors", "selected_layer", getattr(row, "layer", None),
                         "recovery/vector_hyperparameters_selected.csv", context=row.context)
                self.add("steering vectors", "selected_alpha", getattr(row, "alpha", None),
                         "recovery/vector_hyperparameters_selected.csv", context=row.context)

    def run_all(self) -> pd.DataFrame:
        for step in (self.behaviour, self.knowledge_gate, self.tokenizer,
                     self.isolated_retrieval, self.in_prompt_retrieval, self.relation_probe,
                     self.exact_failure, self.extras, self.causal, self.components,
                     self.heads, self.recovery):
            try:
                step()
            except Exception as exc:                                     # noqa: BLE001
                print(f"  {self.name}: {step.__name__} failed ({exc})")
        return pd.DataFrame(self.rows)


def as_markdown_table(pivot: pd.DataFrame) -> str:
    """Render a table without pandas.to_markdown, which needs an optional dependency the
    cluster image does not carry."""
    index_names = [n or "" for n in pivot.index.names]
    header = index_names + [str(c) for c in pivot.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for key, row in pivot.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        cells = ["" if pd.isna(v) else f"{v:g}" for v in row]
        lines.append("| " + " | ".join([str(k) if not pd.isna(k) else ""
                                        for k in key] + cells) + " |")
    return "\n".join(lines)


def markdown(frame: pd.DataFrame, models: list) -> str:
    """One table per section, models side by side, so a difference is visible by eye."""
    out = ["# Collected results", "",
           "Every number below is read from a finished run; nothing is recomputed. The",
           "`source` column of `results_collected.csv` gives the file each came from.",
           "", f"Models: {', '.join(models)}", ""]
    key_columns = [c for c in frame.columns
                   if c not in ("model", "section", "metric", "value", "source")]
    for section in frame.section.unique():
        sub = frame[frame.section == section]
        used = [c for c in key_columns if sub[c].notna().any()]
        out += [f"## {section}", ""]
        pivot = sub.pivot_table(index=["metric"] + used, columns="model", values="value",
                                aggfunc="first")
        pivot = pivot.reindex(columns=[m for m in models if m in pivot.columns])
        out += [as_markdown_table(pivot.round(4)), ""]
        missing = [m for m in models if m not in pivot.columns or pivot[m].isna().all()]
        if missing:
            out += [f"*Absent for {', '.join(missing)}. An empty cell means the analysis "
                    f"was not run or produced nothing, which is itself a result rather "
                    f"than a gap to fill.*", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", type=Path, required=True, help="directory for the outputs")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec}")
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"skipping {name}: {p} does not exist")
            continue
        runs[name] = p
    if not runs:
        raise SystemExit("no run directories found")

    frames = []
    for name, run in runs.items():
        print(f"reading {name} from {run}")
        frames.append(Collector(name, run).run_all())
    frame = pd.concat(frames, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ordered = ["model", "section", "metric"] + \
              [c for c in frame.columns
               if c not in ("model", "section", "metric", "value", "source")] + \
              ["value", "source"]
    frame = frame[ordered]
    frame.to_csv(args.out / "results_collected.csv", index=False)
    (args.out / "RESULTS.md").write_text(markdown(frame, list(runs)), encoding="utf-8")

    print()
    print(f"{len(frame)} numbers across {frame.section.nunique()} sections")
    print(frame.groupby(["section", "model"]).size().unstack(fill_value=0).to_string())
    print()
    print(f"written: {args.out / 'results_collected.csv'}")
    print(f"written: {args.out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

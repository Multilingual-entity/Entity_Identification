"""Run the full Marathi/English × entity-script behavioral factorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    contexts_in,
    contexts_in_gate,
    check_contexts,
    CONDITIONS,
    add_common_args,
    bootstrap_mean_by_fact,
    build_prompt_frame,
    cleanup_model,
    ensure_run_dirs,
    load_model_and_tokenizer,
    load_prepared_data,
    score_prompt_frame,
    seed_everything,
    write_json,
)


def build_gate(canonical: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    positive = canonical.loc[canonical["truth"] == 1]
    per_fact = (
        positive.groupby(["fact_id", "context", "condition"], as_index=False)
        .agg(accuracy=("correct", "mean"), margin=("yes_minus_no_margin", "mean"))
    )
    pieces = []
    for context, sub in per_fact.groupby("context"):
        accuracy = sub.pivot(index="fact_id", columns="condition", values="accuracy")
        margin = sub.pivot(index="fact_id", columns="condition", values="margin")
        accuracy.columns = [f"{context}_{col}" for col in accuracy.columns]
        margin.columns = [f"{context}_{col}_margin" for col in margin.columns]
        pieces.extend([accuracy, margin])
    gate = pd.concat(pieces, axis=1).reset_index()
    for context in contexts_in(canonical):
        cols = [f"{context}_{condition}" for condition in CONDITIONS]
        gate[f"{context}_known_any"] = (gate[cols].max(axis=1) >= 1.0).astype(np.int8)
        gate[f"{context}_LL_success_DD_fail"] = (
            (gate[f"{context}_LATLAT"] >= 1.0) & (gate[f"{context}_DEVDEV"] <= 0.0)
        ).astype(np.int8)
        gate[f"{context}_DD_success_LL_fail"] = (
            (gate[f"{context}_DEVDEV"] >= 1.0) & (gate[f"{context}_LATLAT"] <= 0.0)
        ).astype(np.int8)
        gate[f"{context}_robust_all"] = (gate[cols].min(axis=1) >= 1.0).astype(np.int8)
    return gate.merge(
        corpus[["fact_id", "split", "qid", "relation_tier", "generated_lat", "dev_lang"]],
        on="fact_id",
        how="left",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"],
                        help="prompt languages; must have templates in question_templates.json")
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    final_path = paths["tables"] / "behavior.csv"
    if final_path.exists() and not args.force:
        print(f"Already complete: {final_path}. Use --force to rerun.")
        return
    if args.force:
        for path in (final_path, paths["tables"] / "behavior.partial.csv"):
            if path.exists():
                path.unlink()

    corpus, negatives = load_prepared_data(args.run_dir)
    paraphrases = (0,) if args.canonical_only else (0, 1, 2)
    prompts = build_prompt_frame(
        corpus,
        negatives,
        contexts=tuple(args.contexts),
        conditions=tuple(CONDITIONS),
        truths=(1, 0),
        paraphrases=paraphrases,
        yes_letters=("A", "B"),
    )
    prompts.to_csv(paths["data"] / "behavior_prompts.csv", index=False)
    model, tokenizer, device, batch_size, metadata = load_model_and_tokenizer(
        args.model, args.batch_size
    )
    results = score_prompt_frame(
        model,
        tokenizer,
        device,
        prompts,
        batch_size,
        checkpoint_path=paths["tables"] / "behavior.partial.csv",
    )
    results.to_csv(final_path, index=False)

    summary = (
        results.groupby(["context", "condition", "truth", "paraphrase_id"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_margin=("yes_minus_no_margin", "mean"),
            n=("correct", "size"),
        )
    )
    summary.to_csv(paths["tables"] / "behavior_summary.csv", index=False)
    canonical = results.loc[results["paraphrase_id"] == 0].copy()
    gate = build_gate(canonical, corpus)
    gate.to_csv(paths["tables"] / "knowledge_gate.csv", index=False)
    bootstrap = bootstrap_mean_by_fact(
        canonical,
        "correct",
        ["context", "condition", "truth"],
        seed=args.seed,
    )
    bootstrap.to_csv(paths["tables"] / "behavior_bootstrap.csv", index=False)

    # Fact-paired context interaction in positive margins.
    paired = (
        canonical.loc[canonical["truth"] == 1]
        .groupby(["fact_id", "context", "condition"])["yes_minus_no_margin"]
        .mean()
        .unstack(["context", "condition"])
    )
    # Columns are named from the contexts actually present rather than from a fixed
    # pair, so a corpus in any language produces its own <lang>_script_gap column.
    contexts_present = list(dict.fromkeys(c for c, _ in paired.columns))
    interaction = pd.DataFrame(
        {
            "fact_id": paired.index,
            **{
                f"{c}_script_gap": paired[(c, "LATLAT")] - paired[(c, "DEVDEV")]
                for c in contexts_present
            },
        }
    )
    # The interaction is signed English minus the native-script prompt language, which
    # is the convention every downstream reader and the paper assume.
    native = next((c for c in contexts_present if c != "en"), None)
    if native is not None and "en" in contexts_present:
        interaction["context_by_script_interaction"] = (
            interaction["en_script_gap"] - interaction[f"{native}_script_gap"]
        )
    interaction.to_csv(paths["tables"] / "context_script_interaction_per_fact.csv", index=False)
    metadata.update(
        seed=args.seed,
        n_facts=int(len(corpus)),
        n_prompts=int(len(results)),
        contexts=list(args.contexts),
        paraphrases=list(paraphrases),
    )
    write_json(paths["root"] / "model_run_metadata.json", metadata)
    print(summary.to_string(index=False))
    # Read the contexts off the gate rather than naming them. This is only a summary
    # print, but naming "mr" here failed the whole stage after every output had already
    # been written, which reads as a crash and blocks the runner from marking it done.
    print("\nCanonical positive knowledge gates:")
    gate_columns = [
        f"{context}{suffix}"
        for context in contexts_in_gate(gate)
        for suffix in ("_known_any", "_LL_success_DD_fail", "_robust_all")
    ]
    print(gate[gate_columns].sum().to_string())
    cleanup_model(model)


if __name__ == "__main__":
    main()

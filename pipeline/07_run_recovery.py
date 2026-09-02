"""Evaluate dual-script prompt anchors and held-out script-correction vectors."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from pipeline_common import (
    contexts_in,
    check_contexts,
    PromptBuilder,
    add_common_args,
    add_vector_margin,
    build_prompt_frame,
    ensure_run_dirs,
    load_model_and_tokenizer,
    load_prepared_data,
    scale_layer,
    score_prompt_frame,
    seed_everything,
    semantic_token_positions,
)


POSITIONS = ["E1", "E2", "Q", "READOUT"]
ANCHORS = ["latin_a", "latin_b", "latin_both", "auto_both"]


def learn_vectors(states: np.ndarray, meta: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    vectors = {}
    train = meta.loc[meta["split"] == "train"]
    for context in contexts_in(train):
        subset = train.loc[train["context"] == context]
        indexes = {
            condition: subset.loc[subset["condition"] == condition]
            .sort_values(["fact_id", "truth"])["state_row"]
            .to_numpy(dtype=int)
            for condition in ("DEVDEV", "LATDEV", "DEVLAT", "LATLAT")
        }
        full = states[indexes["LATLAT"]].astype(np.float32) - states[indexes["DEVDEV"]].astype(
            np.float32
        )
        first = states[indexes["LATDEV"]].astype(np.float32) - states[indexes["DEVDEV"]].astype(
            np.float32
        )
        second = states[indexes["DEVLAT"]].astype(np.float32) - states[indexes["DEVDEV"]].astype(
            np.float32
        )
        vectors[(context, "full")] = full.mean(axis=0)
        vectors[(context, "first_entity")] = first.mean(axis=0)
        vectors[(context, "second_entity")] = second.mean(axis=0)
        vectors[(context, "additive")] = vectors[(context, "first_entity")] + vectors[
            (context, "second_entity")
        ]
    return vectors


def unit_vectors(vectors: dict) -> dict:
    """Rescale every variant to unit length at each layer and position.

    Without this the comparison between variants is confounded with their length.
    ``additive`` is the sum of two vectors, so it is roughly twice as long as either
    component, and at a fixed scale it therefore delivers about twice the push. It beat
    ``full`` at test, 29.2 percent against 17.8, and on the un-normalised evidence there
    is no way to tell whether that is a better direction or merely a bigger step. The
    tuning curve was still rising at the top of the scale grid, which makes the second
    reading quite likely.

    Normalising moves all the magnitude into the scale parameter, so a sweep over scale
    compares directions and nothing else.
    """
    out = {}
    for key, arr in vectors.items():
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        out[key] = arr / np.where(norms == 0, 1.0, norms)
    return out


def semantic_margin(raw_a_minus_b: float, yes_letter: str) -> float:
    return raw_a_minus_b if yes_letter == "A" else -raw_a_minus_b


def correct_from_margin(margin: float, truth: int) -> int:
    return int(margin > 0) if int(truth) == 1 else int(margin < 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"],
                        help="prompt languages; must have templates in question_templates.json")
    parser.add_argument("--tuning-facts", type=int, default=30)
    parser.add_argument(
        "--scale-layers",
        action="store_true",
        help="Rescale the candidate layer grid by model depth. Use for models whose "
        "layer count differs from the 28-layer reference (e.g. llama).",
    )
    parser.add_argument("--normalise-vectors", action="store_true",
                        help="rescale every variant to unit length before the sweep, so "
                             "the comparison between variants is about direction rather "
                             "than magnitude. 'additive' is twice as long as its parts, "
                             "which confounds the un-normalised comparison")
    parser.add_argument("--tune-variant", default="full",
                        choices=["full", "additive", "first_entity", "second_entity"],
                        help="which variant the layer and scale search optimises. The "
                             "published run tuned on 'full' but reported 'additive' as "
                             "the winner, so 'additive' ran at settings chosen for a "
                             "different vector")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Scales to search for the correction vector.",
    )
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    corpus, negatives = load_prepared_data(args.run_dir)
    behavior = pd.read_csv(paths["tables"] / "behavior.csv")

    # A. Deployable prompt anchors. Existing DD and LL results are reused as baselines/ceilings.
    anchor_path = paths["recovery"] / "prompt_anchor_recovery.csv"
    if not anchor_path.exists() or args.force:
        partial = paths["recovery"] / "prompt_anchor_recovery.partial.csv"
        if args.force and partial.exists():
            partial.unlink()
        prompts = build_prompt_frame(
            corpus,
            negatives,
            contexts=args.contexts,
            conditions=("DEVDEV",),
            truths=(1, 0),
            paraphrases=(0, 1, 2),
            yes_letters=("A", "B"),
            anchors=ANCHORS,
        )
        model, tokenizer, device, batch_size, model_meta = load_model_and_tokenizer(
            args.model, args.batch_size
        )
        anchors = score_prompt_frame(
            model, tokenizer, device, prompts, batch_size, checkpoint_path=partial
        )
        anchors.to_csv(anchor_path, index=False)
    else:
        anchors = pd.read_csv(anchor_path)
        model, tokenizer, device, batch_size, model_meta = load_model_and_tokenizer(
            args.model, args.batch_size
        )

    baseline = behavior.loc[
        behavior["context"].isin(args.contexts) & behavior["condition"].isin(["DEVDEV", "LATLAT"])
    ].copy()
    baseline["anchor"] = baseline["condition"].map(
        {"DEVDEV": "none", "LATLAT": "latin_only_ceiling"}
    )
    prompt_combined = pd.concat(
        [
            baseline[
                [
                    "fact_id",
                    "split",
                    "context",
                    "condition",
                    "truth",
                    "paraphrase_id",
                    "yes_letter",
                    "anchor",
                    "yes_minus_no_margin",
                    "correct",
                ]
            ],
            anchors[
                [
                    "fact_id",
                    "split",
                    "context",
                    "condition",
                    "truth",
                    "paraphrase_id",
                    "yes_letter",
                    "anchor",
                    "yes_minus_no_margin",
                    "correct",
                ]
            ],
        ],
        ignore_index=True,
    )
    prompt_summary = (
        prompt_combined.groupby(["context", "split", "truth", "anchor"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_margin=("yes_minus_no_margin", "mean"),
            n_facts=("fact_id", "nunique"),
            n=("correct", "size"),
        )
    )
    prompt_summary.to_csv(paths["recovery"] / "prompt_anchor_summary.csv", index=False)

    # B. Fact-independent correction vectors with strict train/validation/test separation.
    pair_meta = pd.read_csv(paths["states"] / "pair_state_metadata.csv")
    pair_states = np.load(paths["states"] / "pair_states.npy", mmap_mode="r")
    vectors = learn_vectors(pair_states, pair_meta)
    if args.normalise_vectors:
        vectors = unit_vectors(vectors)
        print("vectors normalised to unit length; scale now carries all magnitude")
    vector_file = {
        f"{context}__{kind}": value.astype(np.float16)
        for (context, kind), value in vectors.items()
    }
    np.savez_compressed(paths["recovery"] / "script_correction_vectors.npz", **vector_file)
    builder = PromptBuilder(corpus, negatives)
    split_of = dict(zip(corpus["fact_id"].astype(str), corpus["split"]))
    validation_facts = [
        fid for fid in corpus.loc[corpus["split"] == "validation", "fact_id"].astype(str)
    ][: args.tuning_facts]
    n_layers = model_meta["n_layers"]
    reference_layers = (10, 12, 14, 16, 18)
    if args.scale_layers:
        candidate_layers = sorted(
            {scale_layer(layer, n_layers) for layer in reference_layers}
        )
    else:
        candidate_layers = [layer for layer in reference_layers if layer < n_layers]
    candidate_alphas = tuple(args.alphas)
    print(f"recovery grid: layers {candidate_layers} x alphas {list(candidate_alphas)} "
          f"| tuning on {args.tune_variant} "
          f"| {'normalised' if args.normalise_vectors else 'raw magnitudes'}")
    tuning_rows = []
    for context, layer, alpha in product(args.contexts, candidate_layers, candidate_alphas):
        vector = vectors[(context, args.tune_variant)][layer, POSITIONS.index("E2")]
        for fid, truth, yes_letter in product(validation_facts, (1, 0), ("A", "B")):
            record = builder.make(fid, context, "DEVDEV", truth, 0, yes_letter)
            position = semantic_token_positions(
                tokenizer, record["prompt"], record["a_text"], record["b_text"]
            )["E2"]
            raw = add_vector_margin(
                model,
                tokenizer,
                device,
                record["prompt"],
                [(layer, position, vector, alpha)],
            )
            margin = semantic_margin(raw, yes_letter)
            tuning_rows.append(
                {
                    "fact_id": fid,
                    "context": context,
                    "truth": truth,
                    "yes_letter": yes_letter,
                    "layer": layer,
                    "alpha": alpha,
                    "yes_minus_no_margin": margin,
                    "correct": correct_from_margin(margin, truth),
                }
            )
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(paths["recovery"] / "vector_tuning_validation.csv", index=False)
    tuning_summary = (
        tuning.groupby(["context", "layer", "alpha", "truth"], as_index=False)
        .agg(accuracy=("correct", "mean"), mean_margin=("yes_minus_no_margin", "mean"))
    )
    balanced = tuning_summary.pivot_table(
        index=["context", "layer", "alpha"], columns="truth", values="accuracy"
    ).reset_index()
    balanced["balanced_accuracy"] = (balanced[0] + balanced[1]) / 2
    balanced = balanced.rename(columns={0: "different_accuracy", 1: "same_accuracy"})
    best_rows = []
    for context, group in balanced.groupby("context"):
        best_rows.append(
            group.sort_values(
                ["balanced_accuracy", "different_accuracy", "alpha"],
                ascending=[False, False, True],
            ).iloc[0]
        )
    best = pd.DataFrame(best_rows)
    best.to_csv(paths["recovery"] / "vector_hyperparameters_selected.csv", index=False)

    test_facts = corpus.loc[corpus["split"] == "test", "fact_id"].astype(str).tolist()
    vector_test_rows = []
    for selected in best.itertuples():
        context, layer, alpha = selected.context, int(selected.layer), float(selected.alpha)
        for kind in ("full", "first_entity", "second_entity", "additive"):
            vector = vectors[(context, kind)][layer, POSITIONS.index("E2")]
            for fid, truth, paraphrase_id, yes_letter in product(
                test_facts, (1, 0), (0, 1, 2), ("A", "B")
            ):
                record = builder.make(fid, context, "DEVDEV", truth, paraphrase_id, yes_letter)
                position = semantic_token_positions(
                    tokenizer, record["prompt"], record["a_text"], record["b_text"]
                )["E2"]
                raw = add_vector_margin(
                    model,
                    tokenizer,
                    device,
                    record["prompt"],
                    [(layer, position, vector, alpha)],
                )
                margin = semantic_margin(raw, yes_letter)
                vector_test_rows.append(
                    {
                        "fact_id": fid,
                        "context": context,
                        "split": split_of[fid],
                        "truth": truth,
                        "paraphrase_id": paraphrase_id,
                        "yes_letter": yes_letter,
                        "vector_kind": kind,
                        "layer": layer,
                        "alpha": alpha,
                        "yes_minus_no_margin": margin,
                        "correct": correct_from_margin(margin, truth),
                    }
                )
        print(f"vector recovery complete for {context}: layer={layer}, alpha={alpha}")
    vector_test = pd.DataFrame(vector_test_rows)
    vector_test.to_csv(paths["recovery"] / "vector_recovery_test.csv", index=False)
    vector_summary = (
        vector_test.groupby(["context", "truth", "vector_kind"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_margin=("yes_minus_no_margin", "mean"),
            n_facts=("fact_id", "nunique"),
        )
    )
    vector_summary.to_csv(paths["recovery"] / "vector_recovery_test_summary.csv", index=False)
    print("Prompt recovery:")
    print(prompt_summary.loc[prompt_summary["split"] == "test"].to_string(index=False))
    print("\nVector recovery:")
    print(vector_summary.to_string(index=False))


if __name__ == "__main__":
    main()

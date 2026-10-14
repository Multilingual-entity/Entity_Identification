"""Evaluate dual-script prompt anchors and held-out script-correction vectors."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

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


def windows_for(reference_layers, n_layers: int, layer_window: int,
                scale: bool) -> tuple[list[int], dict[int, list[int]]]:
    """The grid layers and the band applied at each.

    Endpoints are scaled and the range between them taken, which is how
    04_run_causal_patching.py builds its windows. Scaling each layer individually would
    leave gaps and cover fewer layers than the window this is being compared against.
    """
    def to_model(ref: int) -> int:
        return scale_layer(ref, n_layers) if scale else ref

    candidate_layers, layer_bands = [], {}
    for ref in reference_layers:
        model_layer = to_model(ref)
        if model_layer >= n_layers:
            continue
        lo_ref = max(0, ref - layer_window + 1)
        lo, hi = to_model(lo_ref), model_layer
        band = [L for L in range(lo, hi + 1) if L < n_layers]
        if model_layer not in candidate_layers:
            candidate_layers.append(model_layer)
            layer_bands[model_layer] = band
    candidate_layers.sort()
    return candidate_layers, layer_bands


def jacobian_vectors(model, tokenizer, device, paths, corpus, negatives, contexts,
                     n_layers: int, d_model: int,
                     layers: list[int]) -> dict[tuple[str, str], np.ndarray]:
    """One direction per context and layer, taken from the output's local sensitivity.

    Built with 14_failure_mediator.py's own code rather than a second implementation, so
    the direction swept here is the direction that file reports. Only the E2 slot is
    filled, which is the only slot additions() reads.
    """
    import importlib.util

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("stage14", here / "14_failure_mediator.py")
    stage14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage14)

    candidates = pd.read_csv(paths["patching"] / "causal_candidates.csv")
    source = stage14.StateSource(model, tokenizer, device,
                                 PromptBuilder(corpus, negatives), paths)
    out = {}
    for context in contexts:
        gated = stage14.behavioural_groups(candidates, context, "train")["gated"][:60]
        if not gated:
            print(f"{context}: no gated train items, skipping the jacobian direction")
            continue
        # Only the layers the sweep applies the vector at. The array keeps the model's
        # full depth so additions() can index it by absolute layer; the rest stays zero
        # and is never read.
        v = np.zeros((n_layers, len(POSITIONS), d_model), dtype=np.float32)
        for layer in layers:
            v[layer, POSITIONS.index("E2")] = stage14.jacobian_direction(
                model, tokenizer, device, source, gated, context, layer, 0)
        out[(context, "jacobian")] = v
        print(f"{context}: jacobian direction over {len(gated)} gated train items, "
              f"layers {layers}")
    if not out:
        raise SystemExit("no jacobian directions could be built; check the gate")
    return out


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
        "--layers", type=int, nargs="+", default=None, metavar="L",
        help="candidate layers for the tuning grid, in reference-model space. "
             "Default (10, 12, 14, 16, 18). Pass 17 with --layer-window 6 to apply "
             "across exactly the 12--17 band the patching analysis uses.")
    parser.add_argument(
        "--layer-window", type=int, default=1, metavar="W",
        help="apply the vector at W consecutive layers ending at the grid layer, "
             "instead of at that layer alone. W=6 with grid layer 17 gives the 12--17 "
             "band the patching analysis uses. Each layer in the band contributes its "
             "own vector, matching how the patching window uses layer-specific donor "
             "states. W=1 reproduces the single-layer behaviour exactly.")
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
    parser.add_argument(
        "--vector-source", default="meandiff", choices=["meandiff", "jacobian"],
        help="where the correction direction comes from. 'meandiff' is the mean "
             "Lat--Lat minus Dev--Dev difference in four variants. 'jacobian' is the "
             "direction the output is locally sensitive to at the second entity, the "
             "one 14_failure_mediator.py builds, applied here as a single constant "
             "vector at recovery scales. It answers whether the repair needs that "
             "particular direction or only a large enough step")
    parser.add_argument("--tune-variant", default="full",
                        choices=["full", "additive", "first_entity", "second_entity",
                                 "jacobian"],
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
    parser.add_argument(
        "--apply-site", default="E2", choices=["E1", "E2", "Q", "READOUT"],
        help="the token position the correction is added at. The vector is always the "
             "one built at the second entity; this changes only where it is added. "
             "Adding it at the question mark or the readout is the control for whether "
             "the second-entity site is doing the work")
    parser.add_argument(
        "--apply-to", default="DEVDEV",
        choices=["DEVDEV", "LATLAT", "LATDEV", "DEVLAT"],
        help="the condition the correction is added to. The vector is always estimated "
             "from the training contrast; this changes only where it is applied. "
             "'LATLAT' is the control that separates a script repair from a general push "
             "toward SAME: if Lat--Lat positives also rise, the vector is not specific "
             "to the failing condition")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="stop after writing the selected layer and scale, without evaluating them "
             "on the test split. Use it for every run whose purpose is to choose a "
             "setting; the test evaluation is a separate, deliberate, single run. It "
             "does not make the run avoid test entirely: the prompt-anchor block above "
             "scores every split, and --force recomputes it. That block tunes nothing, "
             "so it spends nothing, but the honest claim is that no scale was scored "
             "on test, not that test was never read",
    )
    args = parser.parse_args()
    # Checked before anything loads: a flag conflict should not cost a model load.
    if args.tune_variant == "jacobian" and args.vector_source != "jacobian":
        raise SystemExit("--tune-variant jacobian needs --vector-source jacobian")
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
    if args.vector_source == "jacobian":
        _refs = tuple(args.layers) if args.layers else (10, 12, 14, 16, 18)
        _, _bands = windows_for(_refs, model_meta["n_layers"], args.layer_window,
                                args.scale_layers)
        _needed = sorted({L for band in _bands.values() for L in band})
        vectors = jacobian_vectors(model, tokenizer, device, paths, corpus, negatives,
                                   args.contexts, model_meta["n_layers"],
                                   pair_states.shape[-1], _needed)
        if args.tune_variant != "jacobian":
            print(f"--vector-source jacobian overrides --tune-variant "
                  f"{args.tune_variant!r}; there is only one variant to tune")
            args.tune_variant = "jacobian"
    else:
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
    reference_layers = tuple(args.layers) if args.layers else (10, 12, 14, 16, 18)

    def to_model(ref: int) -> int:
        return scale_layer(ref, n_layers) if args.scale_layers else ref

    # The window is a span of reference layers, so it is scaled as a span rather than
    # as a width. Taking W model layers below a scaled endpoint would give the wrong
    # band on any model whose depth differs from the reference.
    candidate_layers, layer_bands = windows_for(reference_layers, n_layers,
                                                args.layer_window, args.scale_layers)

    candidate_alphas = tuple(args.alphas)
    if args.apply_site != "E2":
        print(f"adding the correction at {args.apply_site}, not E2. The vector is still "
              f"the one estimated at the second entity; only the position it is added at "
              f"has changed, so this is the site control.")
    if args.apply_to != "DEVDEV":
        print(f"applying the correction to {args.apply_to}, not DEVDEV. This is the "
              f"control arm: the vector is still estimated from the training contrast, "
              f"and the baseline to compare against is unmodified {args.apply_to}.")
    print(f"recovery grid: layers {candidate_layers} x alphas {list(candidate_alphas)} "
          f"| tuning on {args.tune_variant} "
          f"| {'normalised' if args.normalise_vectors else 'raw magnitudes'}")

    def additions(kind: str, context: str, layer: int, position: int, alpha: float):
        """One (layer, position, vector, alpha) per layer in this grid layer's band."""
        return [(L, position, vectors[(context, kind)][L, POSITIONS.index("E2")], alpha)
                for L in layer_bands[layer]]

    if args.layer_window > 1:
        shown = ", ".join(f"{L}:{layer_bands[L]}" for L in candidate_layers)
        print(f"applying across {args.layer_window} reference layers per grid point; "
              f"bands are {shown}")

    tuning_rows = []
    for context, layer, alpha in product(args.contexts, candidate_layers, candidate_alphas):
        for fid, truth, yes_letter in product(validation_facts, (1, 0), ("A", "B")):
            record = builder.make(fid, context, args.apply_to, truth, 0, yes_letter)
            position = semantic_token_positions(
                tokenizer, record["prompt"], record["a_text"], record["b_text"]
            )[args.apply_site]
            raw = add_vector_margin(
                model,
                tokenizer,
                device,
                record["prompt"],
                additions(args.tune_variant, context, layer, position, alpha),
            )
            margin = semantic_margin(raw, yes_letter)
            tuning_rows.append(
                {
                    "fact_id": fid,
                    "context": context,
                    "truth": truth,
                    "yes_letter": yes_letter,
                    "layer": layer,
                    "layer_window": args.layer_window,
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

    if args.skip_test:
        # A selection run has no business evaluating on test. Without this the sweep
        # scores the held-out split on every invocation, so an exploratory grid search
        # silently produces a test number for each setting it tried.
        print("--skip-test: selection only, no test evaluation written")
        print(best.to_string(index=False))
        return

    test_facts = corpus.loc[corpus["split"] == "test", "fact_id"].astype(str).tolist()
    vector_test_rows = []
    for selected in best.itertuples():
        context, layer, alpha = selected.context, int(selected.layer), float(selected.alpha)
        kinds = sorted({k for (c, k) in vectors if c == context})
        for kind in kinds:
            for fid, truth, paraphrase_id, yes_letter in product(
                test_facts, (1, 0), (0, 1, 2), ("A", "B")
            ):
                record = builder.make(fid, context, args.apply_to, truth,
                                      paraphrase_id, yes_letter)
                position = semantic_token_positions(
                    tokenizer, record["prompt"], record["a_text"], record["b_text"]
                )[args.apply_site]
                raw = add_vector_margin(
                    model,
                    tokenizer,
                    device,
                    record["prompt"],
                    additions(kind, context, layer, position, alpha),
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
                        "layer_window": args.layer_window,
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

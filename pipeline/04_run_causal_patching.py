"""Factorial, windowed, joint-site, and controlled activation patching."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    contexts_in,
    check_contexts,
    PromptBuilder,
    add_common_args,
    append_records,
    ensure_run_dirs,
    load_model_and_tokenizer,
    load_prepared_data,
    patch_margin,
    read_json,
    scale_layer,
    seed_everything,
    semantic_token_positions,
    write_json,
)


POSITIONS = ["E1", "E2", "Q", "READOUT"]
SOURCE_CONDITIONS = ["LATDEV", "DEVLAT", "LATLAT"]
WINDOWS = {
    "9_11": (9, 11),
    "12_14": (12, 14),
    "15_17": (15, 17),
    "18_20": (18, 20),
    "12_17": (12, 17),
}
JOINT_SITES = {
    "E1+E2": ["E1", "E2"],
    "E2+Q": ["E2", "Q"],
    "E2+READOUT": ["E2", "READOUT"],
    "E1+E2+Q": ["E1", "E2", "Q"],
}


def stable_candidates(behavior: pd.DataFrame, corpus: pd.DataFrame,
                      gate: str = "accuracy", polarity: str = "positive",
                      margin_gap: float = 4.0) -> pd.DataFrame:
    """Select items on which the causal experiment is defined.

    Three ways to select, because one rule does not survive contact with models that
    have different response biases.

    ``accuracy`` is the original rule: the item must be answered correctly in Lat--Lat
    and incorrectly in Dev--Dev. It gave 115 items on Qwen, 13 on Llama and **zero** on
    Gemma. Gemma accepts nearly every positive pair, so it almost never fails one, so no
    item can qualify. That is a property of the measuring rule under a strong acceptance
    bias, not evidence that the effect is absent.

    ``margin`` selects on how far the model leans rather than on which letter it emits.
    Gemma's margin is lower under Devanagari; it simply is not low enough to flip the
    answer. A margin gate sees that where a correctness gate sees nothing. The threshold
    is in logit units, the same scale the patching deltas are reported in.

    ``polarity`` switches which question type the gate reads. Qwen fails positive pairs,
    so ``positive`` is right for it. Gemma fails *negative* pairs, since its bias makes
    it answer yes to those too, and its script gap is plainly visible there: 24.7 percent
    against 54.0 in English. For such a model ``negative`` is the only way in.
    """
    grouped = (
        behavior.groupby(["fact_id", "context", "condition", "truth"], as_index=False)
        .agg(accuracy=("correct", "mean"), margin=("yes_minus_no_margin", "mean"))
    )
    truth_of = {"positive": 1, "negative": 0}[polarity]
    rows = []
    split_of = dict(zip(corpus["fact_id"].astype(str), corpus["split"]))
    for context in contexts_in(grouped):
        subset = grouped.loc[grouped["context"] == context]
        positive = subset.loc[subset["truth"] == 1].pivot(
            index="fact_id", columns="condition", values="accuracy")
        negative = subset.loc[subset["truth"] == 0].pivot(
            index="fact_id", columns="condition", values="accuracy")
        joined = positive.add_prefix("pos_").join(negative.add_prefix("neg_"), how="inner")

        margins = subset.loc[subset["truth"] == truth_of].pivot(
            index="fact_id", columns="condition", values="margin")
        joined = joined.join(margins.add_prefix("margin_"), how="left")

        # A gap in the direction of the correct answer. For positive pairs the correct
        # answer is yes, so Lat--Lat should sit higher; for negative pairs it is no, so
        # Lat--Lat should sit lower. Taking the signed difference keeps one threshold.
        sign = 1.0 if truth_of == 1 else -1.0
        joined["margin_gap"] = sign * (joined.get("margin_LATLAT", np.nan)
                                       - joined.get("margin_DEVDEV", np.nan))

        if gate not in {"accuracy", "margin", "either"}:
            raise ValueError(f"unknown gate: {gate}")

        # Selection on the polarity the model actually fails.
        if truth_of == 1:
            fails_native, succeeds_latin = joined["pos_DEVDEV"], joined["pos_LATLAT"]
            other_ok = (joined["neg_LATLAT"] >= 0.70) & (joined["neg_DEVDEV"] >= 0.70)
        else:
            fails_native, succeeds_latin = joined["neg_DEVDEV"], joined["neg_LATLAT"]
            other_ok = (joined["pos_LATLAT"] >= 0.70) & (joined["pos_DEVDEV"] >= 0.70)

        # The item must be answered sensibly on the other polarity too, or it tells us
        # nothing about script: a model at chance on both is not failing for the reason
        # under study. This applies to every gate, so that a looser gate never admits an
        # item a stricter one would reject.
        by_accuracy = (succeeds_latin >= 0.80) & (fails_native <= 0.20)

        # A margin gap alone is not enough. An item already answered correctly in
        # Dev--Dev has nothing to repair, and admitting it would break the reported flip
        # rate, which equals "correct after patching" only while every selected item
        # starts wrong. On Qwen this excludes 16 of 332.
        by_margin = (joined["margin_gap"] >= margin_gap) & (fails_native < 0.80)

        ok = {"accuracy": by_accuracy,
              "margin": by_margin,
              "either": by_accuracy | by_margin}[gate] & other_ok

        joined["stable"] = ok.fillna(False).astype(np.int8)
        joined = joined.reset_index()
        joined["context"] = context
        joined["gate"] = gate
        joined["polarity"] = polarity
        joined["split"] = joined["fact_id"].astype(str).map(split_of)
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def correctness(margin: float, truth: int) -> int:
    return int(margin > 0) if int(truth) == 1 else int(margin < 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"],
                        help="prompt languages; must have templates in question_templates.json")
    parser.add_argument("--max-facts-per-context", type=int, default=60)
    parser.add_argument("--layer-low", type=int, default=8)
    parser.add_argument("--layer-high", type=int, default=20)
    parser.add_argument("--gate", choices=["accuracy", "margin", "either"],
                        default="accuracy",
                        help="how to select items. 'accuracy' is the original rule and "
                             "returns nothing on a model with a strong response bias; "
                             "'margin' selects on how far the model leans instead")
    parser.add_argument("--gate-polarity", choices=["positive", "negative"],
                        default="positive",
                        help="which question type the gate reads. Use 'negative' for a "
                             "model that accepts nearly every positive pair, such as "
                             "Gemma, whose failures are all on the negatives")
    parser.add_argument("--margin-gap", type=float, default=4.0,
                        help="minimum Lat--Lat minus Dev--Dev margin, in logit units, "
                             "when --gate is margin or either")
    parser.add_argument(
        "--scale-layers",
        action="store_true",
        help="Rescale the preregistered windows by model depth. Use for models whose "
        "layer count differs from the 28-layer reference (e.g. llama).",
    )
    args = parser.parse_args()
    check_contexts(args.contexts)
    # The cross-language factorial pairs each context with the other one, so the
    # experiment is defined only for exactly two.
    if len(args.contexts) != 2:
        raise SystemExit(
            f"--contexts must name exactly two prompt languages, got {args.contexts}; "
            "the cross-language factorial has no meaning otherwise."
        )
    partner_of = {args.contexts[0]: args.contexts[1],
                  args.contexts[1]: args.contexts[0]}
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    final_path = paths["patching"] / "causal_patching.csv"
    partial_path = paths["patching"] / "causal_patching.partial.csv"
    done_path = paths["patching"] / "causal_patching.done.json"
    if final_path.exists() and not args.force:
        print(f"Already complete: {final_path}. Use --force to rerun.")
        return
    if args.force:
        for path in (final_path, partial_path, done_path):
            if path.exists():
                path.unlink()

    corpus, negatives = load_prepared_data(args.run_dir)
    behavior = pd.read_csv(paths["tables"] / "behavior.csv")
    candidates = stable_candidates(behavior, corpus, gate=args.gate,
                                   polarity=args.gate_polarity,
                                   margin_gap=args.margin_gap)
    n_by_ctx = candidates.groupby("context").stable.sum().to_dict()
    print(f"gate={args.gate} polarity={args.gate_polarity} "
          f"margin_gap={args.margin_gap} -> stable {n_by_ctx}")
    candidates.to_csv(paths["patching"] / "causal_candidates.csv", index=False)
    selected = candidates.loc[
        (candidates["stable"] == 1) & candidates["context"].isin(args.contexts)
    ].copy()
    selected = (
        selected.sort_values(["context", "split", "fact_id"])
        .groupby("context", group_keys=False)
        .head(args.max_facts_per_context)
    )
    if selected.empty:
        print("No stable LATLAT-success/DEVDEV-failure facts. Causal patching skipped.")
        return

    pair_meta = pd.read_csv(paths["states"] / "pair_state_metadata.csv")
    pair_states = np.load(paths["states"] / "pair_states.npy", mmap_mode="r")
    state_index = {
        (str(row.fact_id), row.context, row.condition, int(row.truth)): int(row.state_row)
        for row in pair_meta.itertuples()
    }
    builder = PromptBuilder(corpus, negatives)
    wrong_of = dict(zip(negatives["fact_id"].astype(str), negatives["negative_fact_id"].astype(str)))
    model, tokenizer, device, _, model_meta = load_model_and_tokenizer(args.model, args.batch_size)
    n_layers = model_meta["n_layers"]
    layers = list(range(max(0, args.layer_low), min(n_layers, args.layer_high + 1)))
    if args.scale_layers:
        window_bounds = {
            name: (scale_layer(low, n_layers), scale_layer(high, n_layers))
            for name, (low, high) in WINDOWS.items()
        }
    else:
        window_bounds = dict(WINDOWS)
    windows = {
        name: [layer for layer in range(low, high + 1) if layer < n_layers]
        for name, (low, high) in window_bounds.items()
    }
    # Window keys keep their reference-model names so that downstream tables stay
    # comparable across models; the actual layers used are recorded alongside.
    write_json(
        paths["patching"] / "windows_used.json",
        {
            "model": args.model,
            "n_layers": n_layers,
            "scaled": bool(args.scale_layers),
            "single_layer_sweep": layers,
            "windows": {name: sorted(v) for name, v in windows.items()},
        },
    )
    window_desc = ", ".join(f"{k}={v[0]}-{v[-1]}" for k, v in windows.items() if v)
    print(f"layer sweep {layers[0]}-{layers[-1]} | windows {window_desc}")
    completed = set(read_json(done_path).get("completed", [])) if done_path.exists() else set()

    for candidate in selected.itertuples():
        key = f"{candidate.context}:{candidate.fact_id}"
        if key in completed:
            continue
        records = []
        fid = str(candidate.fact_id)
        context = candidate.context
        for truth in (1, 0):
            target = builder.make(fid, context, "DEVDEV", truth, 0, "A")
            target_positions = semantic_token_positions(
                tokenizer, target["prompt"], target["a_text"], target["b_text"]
            )
            target_row = state_index[(fid, context, "DEVDEV", truth)]
            baseline = float(pair_meta.loc[pair_meta["state_row"] == target_row, "yes_minus_no_margin"].iloc[0])

            # Single-layer factorial sources: only E1 Latin, only E2 Latin, or both Latin.
            for source_condition in SOURCE_CONDITIONS:
                source_row = state_index[(fid, context, source_condition, truth)]
                source_margin = float(
                    pair_meta.loc[pair_meta["state_row"] == source_row, "yes_minus_no_margin"].iloc[0]
                )
                for location in POSITIONS:
                    pos_index = POSITIONS.index(location)
                    for layer in layers:
                        patched = patch_margin(
                            model,
                            tokenizer,
                            device,
                            target["prompt"],
                            [(layer, target_positions[location], pair_states[source_row, layer, pos_index])],
                        )
                        records.append(
                            {
                                "fact_id": fid,
                                "context": context,
                                "split": candidate.split,
                                "truth": truth,
                                "experiment": "single_factorial",
                                "source_condition": source_condition,
                                "site": location,
                                "layer_or_window": str(layer),
                                "baseline_margin": baseline,
                                "source_margin": source_margin,
                                "patched_margin": patched,
                                "delta_margin": patched - baseline,
                                "baseline_correct": correctness(baseline, truth),
                                "patched_correct": correctness(patched, truth),
                            }
                        )

                # Repeated E2 patches reveal distributed/non-contiguous recovery.
                for window_name, window_layers in windows.items():
                    if not window_layers:
                        continue
                    patches = [
                        (
                            layer,
                            target_positions["E2"],
                            pair_states[source_row, layer, POSITIONS.index("E2")],
                        )
                        for layer in window_layers
                    ]
                    patched = patch_margin(model, tokenizer, device, target["prompt"], patches)
                    records.append(
                        {
                            "fact_id": fid,
                            "context": context,
                            "split": candidate.split,
                            "truth": truth,
                            "experiment": "window_factorial",
                            "source_condition": source_condition,
                            "site": "E2",
                            "layer_or_window": window_name,
                            "baseline_margin": baseline,
                            "source_margin": source_margin,
                            "patched_margin": patched,
                            "delta_margin": patched - baseline,
                            "baseline_correct": correctness(baseline, truth),
                            "patched_correct": correctness(patched, truth),
                        }
                    )

            # Context-language factorial: other-context DD changes only the surrounding
            # language; other-context LL changes context and both entity scripts.
            # The partner is read from --contexts rather than named, so that a corpus in
            # any language pairs with English. Naming "mr" here made every non-Marathi
            # run ask for states that were never cached.
            other_context = partner_of[context]
            for source_condition in ("DEVDEV", "LATLAT"):
                source_row = state_index[(fid, other_context, source_condition, truth)]
                source_margin = float(
                    pair_meta.loc[
                        pair_meta["state_row"] == source_row, "yes_minus_no_margin"
                    ].iloc[0]
                )
                for location in ("E2", "Q"):
                    position_index = POSITIONS.index(location)
                    for layer in layers:
                        patched = patch_margin(
                            model,
                            tokenizer,
                            device,
                            target["prompt"],
                            [
                                (
                                    layer,
                                    target_positions[location],
                                    pair_states[source_row, layer, position_index],
                                )
                            ],
                        )
                        records.append(
                            {
                                "fact_id": fid,
                                "context": context,
                                "source_context": other_context,
                                "split": candidate.split,
                                "truth": truth,
                                "experiment": "cross_context_factorial",
                                "source_condition": source_condition,
                                "site": location,
                                "layer_or_window": str(layer),
                                "baseline_margin": baseline,
                                "source_margin": source_margin,
                                "patched_margin": patched,
                                "delta_margin": patched - baseline,
                                "baseline_correct": correctness(baseline, truth),
                                "patched_correct": correctness(patched, truth),
                            }
                        )

            # Joint-location LL patches in the predeclared 12-17 window.
            ll_row = state_index[(fid, context, "LATLAT", truth)]
            joint_layers = windows["12_17"]
            for site_name, locations in JOINT_SITES.items():
                patches = [
                    (
                        layer,
                        target_positions[location],
                        pair_states[ll_row, layer, POSITIONS.index(location)],
                    )
                    for layer in joint_layers
                    for location in locations
                ]
                patched = patch_margin(model, tokenizer, device, target["prompt"], patches)
                records.append(
                    {
                        "fact_id": fid,
                        "context": context,
                        "split": candidate.split,
                        "truth": truth,
                        "experiment": "joint_12_17",
                        "source_condition": "LATLAT",
                        "site": site_name,
                        "layer_or_window": "12_17",
                        "baseline_margin": baseline,
                        "source_margin": float(
                            pair_meta.loc[pair_meta["state_row"] == ll_row, "yes_minus_no_margin"].iloc[0]
                        ),
                        "patched_margin": patched,
                        "delta_margin": patched - baseline,
                        "baseline_correct": correctness(baseline, truth),
                        "patched_correct": correctness(patched, truth),
                    }
                )

            # Wrong-entity and wrong-position controls at the same layers.
            control_fid = wrong_of[fid]
            control_row = state_index[(control_fid, context, "LATLAT", truth)]
            for layer in layers:
                wrong_entity = patch_margin(
                    model,
                    tokenizer,
                    device,
                    target["prompt"],
                    [(layer, target_positions["E2"], pair_states[control_row, layer, POSITIONS.index("E2")])],
                )
                wrong_position = patch_margin(
                    model,
                    tokenizer,
                    device,
                    target["prompt"],
                    [(layer, target_positions["E1"], pair_states[ll_row, layer, POSITIONS.index("E2")])],
                )
                for experiment, patched in (
                    ("wrong_entity_control", wrong_entity),
                    ("wrong_position_control", wrong_position),
                ):
                    records.append(
                        {
                            "fact_id": fid,
                            "context": context,
                            "split": candidate.split,
                            "truth": truth,
                            "experiment": experiment,
                            "source_condition": "LATLAT",
                            "site": "E2" if experiment == "wrong_entity_control" else "E2_to_E1",
                            "layer_or_window": str(layer),
                            "control_fact_id": control_fid if experiment == "wrong_entity_control" else "",
                            "baseline_margin": baseline,
                            "source_margin": np.nan,
                            "patched_margin": patched,
                            "delta_margin": patched - baseline,
                            "baseline_correct": correctness(baseline, truth),
                            "patched_correct": correctness(patched, truth),
                        }
                    )

            # Reverse direction verifies that the effect is not symmetric corruption.
            if truth == 1:
                ll_target = builder.make(fid, context, "LATLAT", 1, 0, "A")
                ll_positions = semantic_token_positions(
                    tokenizer, ll_target["prompt"], ll_target["a_text"], ll_target["b_text"]
                )
                ll_baseline = float(
                    pair_meta.loc[pair_meta["state_row"] == ll_row, "yes_minus_no_margin"].iloc[0]
                )
                for layer in layers:
                    reverse = patch_margin(
                        model,
                        tokenizer,
                        device,
                        ll_target["prompt"],
                        [(layer, ll_positions["E2"], pair_states[target_row, layer, POSITIONS.index("E2")])],
                    )
                    records.append(
                        {
                            "fact_id": fid,
                            "context": context,
                            "split": candidate.split,
                            "truth": 1,
                            "experiment": "reverse_DD_into_LL",
                            "source_condition": "DEVDEV",
                            "site": "E2",
                            "layer_or_window": str(layer),
                            "baseline_margin": ll_baseline,
                            "source_margin": baseline,
                            "patched_margin": reverse,
                            "delta_margin": reverse - ll_baseline,
                            "baseline_correct": correctness(ll_baseline, 1),
                            "patched_correct": correctness(reverse, 1),
                        }
                    )

        append_records(partial_path, records)
        completed.add(key)
        write_json(done_path, {"completed": sorted(completed)})
        print(f"completed {key}: {len(records)} interventions")

    partial_path.replace(final_path)
    results = pd.read_csv(final_path)
    summary = (
        results.groupby(
            ["context", "split", "truth", "experiment", "source_condition", "site", "layer_or_window"],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_delta=("delta_margin", "mean"),
            flip_rate=("patched_correct", "mean"),
            n_facts=("fact_id", "nunique"),
        )
    )
    summary.to_csv(paths["patching"] / "causal_patching_summary.csv", index=False)
    print("Saved:", final_path)


if __name__ == "__main__":
    main()

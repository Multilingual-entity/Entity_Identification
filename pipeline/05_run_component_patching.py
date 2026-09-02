"""Decompose the E2 causal effect into residual, attention, and MLP outputs."""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from pipeline_common import (
    check_contexts,
    PromptBuilder,
    add_common_args,
    append_records,
    ensure_run_dirs,
    fixed_yes_margin,
    get_layers,
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


def capture_components(model, tokenizer, device, prompt: str, source_position: int):
    import torch
    from pipeline_common import MAX_LENGTH, format_chat

    store = {"attention": {}, "mlp": {}}
    handles = []

    def make_hook(kind: str, layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store[kind][layer] = hidden[0, source_position].detach().to(torch.float16).cpu().numpy()

        return hook

    for layer, block in enumerate(get_layers(model)):
        handles.append(block.self_attn.register_forward_hook(make_hook("attention", layer)))
        handles.append(block.mlp.register_forward_hook(make_hook("mlp", layer)))
    encoded = tokenizer(
        format_chat(tokenizer, prompt),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    del encoded
    return store


def component_patch_margin(
    model,
    tokenizer,
    device,
    prompt: str,
    target_position: int,
    layer: int,
    vector,
    component: str,
) -> float:
    import torch

    block = get_layers(model)[layer]
    module = block.self_attn if component == "attention" else block.mlp

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        hidden[:, target_position, :] = torch.as_tensor(
            vector, device=hidden.device, dtype=hidden.dtype
        )
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = module.register_forward_hook(hook)
    try:
        return fixed_yes_margin(model, tokenizer, device, prompt)
    finally:
        handle.remove()


def cumulative_component_margin(
    model,
    tokenizer,
    device,
    prompt: str,
    target_position: int,
    layers,
    vectors_by_layer: dict,
    component: str,
) -> float:
    """Patch one component's output at every layer up to and including the last.

    The single-layer version is not a fair comparison against the residual patch, and
    the paper says so: a residual patch imports everything accumulated from layer 0,
    while a component patch imports only what that one layer wrote. The gap between
    100 percent and 22 percent is therefore partly built into the design rather than
    being a finding about distribution.

    Patching a component cumulatively gives it the same accumulated history the
    residual patch enjoys, so the comparison becomes like for like. If attention alone
    then reproduces most of the residual effect, the effect is carried by attention; if
    it still does not, the shortfall is real rather than an artefact of the method.
    """
    import torch

    blocks = get_layers(model)
    handles = []

    def make_hook(vector):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = (output[0] if is_tuple else output).clone()
            hidden[:, target_position, :] = torch.as_tensor(
                vector, device=hidden.device, dtype=hidden.dtype
            )
            return (hidden, *output[1:]) if is_tuple else hidden
        return hook

    try:
        for layer in layers:
            if layer not in vectors_by_layer:
                continue
            block = blocks[layer]
            module = block.self_attn if component == "attention" else block.mlp
            handles.append(module.register_forward_hook(make_hook(vectors_by_layer[layer])))
        return fixed_yes_margin(model, tokenizer, device, prompt)
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"],
                        help="prompt languages; must have templates in question_templates.json")
    parser.add_argument("--max-facts-per-context", type=int, default=30)
    parser.add_argument("--cumulative", action="store_true",
                        help="also patch each component at every layer up to the current "
                             "one, giving it the same accumulated history the residual "
                             "patch has. Without this the residual comparison is unfair "
                             "by construction and bounds rather than measures each "
                             "component's share")
    parser.add_argument("--layer-low", type=int, default=8)
    parser.add_argument("--layer-high", type=int, default=20)
    parser.add_argument("--scale-layers", action="store_true",
                        help="rescale the layer band by model depth. The band was fixed "
                             "on a 28-layer model, so on a 32-layer model the same "
                             "absolute numbers point at a shallower part of the network "
                             "and the result is not comparable")
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    final_path = paths["patching"] / "component_patching.csv"
    partial_path = paths["patching"] / "component_patching.partial.csv"
    done_path = paths["patching"] / "component_patching.done.json"
    if final_path.exists() and not args.force:
        print(f"Already complete: {final_path}. Use --force to rerun.")
        return
    if args.force:
        for path in (final_path, partial_path, done_path):
            if path.exists():
                path.unlink()

    corpus, negatives = load_prepared_data(args.run_dir)
    candidates = pd.read_csv(paths["patching"] / "causal_candidates.csv")
    selected = candidates.loc[
        (candidates["stable"] == 1) & candidates["context"].isin(args.contexts)
    ]
    selected = (
        selected.sort_values(["context", "split", "fact_id"])
        .groupby("context", group_keys=False)
        .head(args.max_facts_per_context)
    )
    if selected.empty:
        print("No stable causal facts; component patching skipped.")
        return
    pair_meta = pd.read_csv(paths["states"] / "pair_state_metadata.csv")
    pair_states = np.load(paths["states"] / "pair_states.npy", mmap_mode="r")
    state_index = {
        (str(row.fact_id), row.context, row.condition, int(row.truth)): int(row.state_row)
        for row in pair_meta.itertuples()
    }
    builder = PromptBuilder(corpus, negatives)
    model, tokenizer, device, _, model_meta = load_model_and_tokenizer(args.model, args.batch_size)
    n_layers = model_meta["n_layers"]
    if args.scale_layers:
        low, high = scale_layer(args.layer_low, n_layers), scale_layer(args.layer_high, n_layers)
    else:
        low, high = args.layer_low, args.layer_high
    layers = range(max(0, low), min(n_layers, high + 1))
    print(f"layer band {list(layers)}"
          f"{' (rescaled from %d-%d)' % (args.layer_low, args.layer_high) if args.scale_layers else ''}")
    completed = set(read_json(done_path).get("completed", [])) if done_path.exists() else set()

    for candidate in selected.itertuples():
        key = f"{candidate.context}:{candidate.fact_id}"
        if key in completed:
            continue
        records = []
        fid, context = str(candidate.fact_id), candidate.context
        for truth in (1, 0):
            target = builder.make(fid, context, "DEVDEV", truth, 0, "A")
            source = builder.make(fid, context, "LATLAT", truth, 0, "A")
            target_position = semantic_token_positions(
                tokenizer, target["prompt"], target["a_text"], target["b_text"]
            )["E2"]
            source_position = semantic_token_positions(
                tokenizer, source["prompt"], source["a_text"], source["b_text"]
            )["E2"]
            source_row = state_index[(fid, context, "LATLAT", truth)]
            target_row = state_index[(fid, context, "DEVDEV", truth)]
            baseline = float(
                pair_meta.loc[pair_meta["state_row"] == target_row, "yes_minus_no_margin"].iloc[0]
            )
            components = capture_components(
                model, tokenizer, device, source["prompt"], source_position
            )
            for layer in layers:
                residual = patch_margin(
                    model,
                    tokenizer,
                    device,
                    target["prompt"],
                    [(layer, target_position, pair_states[source_row, layer, POSITIONS.index("E2")])],
                )
                values = {"residual": residual}
                for component in ("attention", "mlp"):
                    values[component] = component_patch_margin(
                        model,
                        tokenizer,
                        device,
                        target["prompt"],
                        target_position,
                        layer,
                        components[component][layer],
                        component,
                    )
                if args.cumulative:
                    upto = [l for l in layers if l <= layer]
                    for component in ("attention", "mlp"):
                        values[f"{component}_cumulative"] = cumulative_component_margin(
                            model,
                            tokenizer,
                            device,
                            target["prompt"],
                            target_position,
                            upto,
                            components[component],
                            component,
                        )
                for component, patched in values.items():
                    records.append(
                        {
                            "fact_id": fid,
                            "context": context,
                            "split": candidate.split,
                            "truth": truth,
                            "layer": layer,
                            "component": component,
                            "baseline_margin": baseline,
                            "patched_margin": patched,
                            "delta_margin": patched - baseline,
                            "baseline_correct": int(baseline > 0) if truth == 1 else int(baseline < 0),
                            "patched_correct": int(patched > 0) if truth == 1 else int(patched < 0),
                        }
                    )
            del components
            gc.collect()
        append_records(partial_path, records)
        completed.add(key)
        write_json(done_path, {"completed": sorted(completed)})
        print(f"completed {key}: {len(records)} component interventions")
    partial_path.replace(final_path)
    results = pd.read_csv(final_path)
    summary = (
        results.groupby(["context", "split", "truth", "layer", "component"], as_index=False)
        .agg(mean_delta=("delta_margin", "mean"), flip_rate=("patched_correct", "mean"), n=("fact_id", "nunique"))
    )
    summary.to_csv(paths["patching"] / "component_patching_summary.csv", index=False)
    print("Saved:", final_path)


if __name__ == "__main__":
    main()

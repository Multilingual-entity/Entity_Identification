"""Screen exact attention-head output patches, then confirm top sites held out."""

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
    scale_layer,
    seed_everything,
    semantic_token_positions,
)


def capture_head_inputs(model, tokenizer, device, prompt: str, position: int):
    import torch
    from pipeline_common import MAX_LENGTH, format_chat

    store = {}
    handles = []

    def make_hook(layer):
        def hook(_module, inputs):
            store[layer] = inputs[0][0, position].detach().to(torch.float16).cpu().numpy()

        return hook

    for layer, block in enumerate(get_layers(model)):
        handles.append(block.self_attn.o_proj.register_forward_pre_hook(make_hook(layer)))
    encoded = tokenizer(
        format_chat(tokenizer, prompt), return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    ).to(device)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    del encoded
    return store


def patch_heads_margin(
    model,
    tokenizer,
    device,
    prompt: str,
    target_position: int,
    source_by_layer: dict[int, np.ndarray],
    sites: list[tuple[int, int]],
    head_dim: int,
) -> float:
    import torch

    grouped: dict[int, list[int]] = {}
    for layer, head in sites:
        grouped.setdefault(int(layer), []).append(int(head))
    handles = []

    def make_hook(layer: int, heads: list[int]):
        def hook(_module, inputs):
            hidden = inputs[0].clone()
            source = torch.as_tensor(source_by_layer[layer], device=hidden.device, dtype=hidden.dtype)
            for head in heads:
                start, end = head * head_dim, (head + 1) * head_dim
                hidden[:, target_position, start:end] = source[start:end]
            return (hidden, *inputs[1:])

        return hook

    for layer, heads in grouped.items():
        module = get_layers(model)[layer].self_attn.o_proj
        handles.append(module.register_forward_pre_hook(make_hook(layer, heads)))
    try:
        return fixed_yes_margin(model, tokenizer, device, prompt)
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"],
                        help="prompt languages; must have templates in question_templates.json")
    parser.add_argument("--screen-facts", type=int, default=12)
    parser.add_argument("--top-sites", type=int, default=12)
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
    screen_path = paths["patching"] / "head_screen_train.csv"
    test_path = paths["patching"] / "head_confirm_heldout.csv"
    if test_path.exists() and not args.force:
        print(f"Already complete: {test_path}. Use --force to rerun.")
        return
    if args.force:
        for path in (screen_path, test_path, paths["patching"] / "head_sites_selected.csv"):
            if path.exists():
                path.unlink()

    corpus, negatives = load_prepared_data(args.run_dir)
    candidates = pd.read_csv(paths["patching"] / "causal_candidates.csv")
    stable = candidates.loc[
        (candidates["stable"] == 1) & candidates["context"].isin(args.contexts)
    ].copy()
    discovery = stable.loc[stable["split"] == "train"].groupby("context", group_keys=False).head(
        args.screen_facts
    )
    heldout = stable.loc[stable["split"].isin(["validation", "test"])]
    if discovery.empty or heldout.empty:
        print("Insufficient train/held-out stable facts; head patching skipped.")
        return
    builder = PromptBuilder(corpus, negatives)
    model, tokenizer, device, _, model_meta = load_model_and_tokenizer(args.model, args.batch_size)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // n_heads))
    n_layers = model_meta["n_layers"]
    if args.scale_layers:
        low, high = scale_layer(args.layer_low, n_layers), scale_layer(args.layer_high, n_layers)
    else:
        low, high = args.layer_low, args.layer_high
    layers = range(max(0, low), min(n_layers, high + 1))
    print(f"layer band {list(layers)}"
          f"{' (rescaled from %d-%d)' % (args.layer_low, args.layer_high) if args.scale_layers else ''}")

    screen_records = []
    for candidate in discovery.itertuples():
        fid, context = str(candidate.fact_id), candidate.context
        source = builder.make(fid, context, "LATLAT", 1, 0, "A")
        target = builder.make(fid, context, "DEVDEV", 1, 0, "A")
        source_position = semantic_token_positions(
            tokenizer, source["prompt"], source["a_text"], source["b_text"]
        )["E2"]
        target_position = semantic_token_positions(
            tokenizer, target["prompt"], target["a_text"], target["b_text"]
        )["E2"]
        source_heads = capture_head_inputs(model, tokenizer, device, source["prompt"], source_position)
        baseline = fixed_yes_margin(model, tokenizer, device, target["prompt"])
        for layer in layers:
            for head in range(n_heads):
                patched = patch_heads_margin(
                    model,
                    tokenizer,
                    device,
                    target["prompt"],
                    target_position,
                    source_heads,
                    [(layer, head)],
                    head_dim,
                )
                screen_records.append(
                    {
                        "fact_id": fid,
                        "context": context,
                        "layer": layer,
                        "head": head,
                        "baseline_margin": baseline,
                        "patched_margin": patched,
                        "delta_margin": patched - baseline,
                    }
                )
        print(f"screened {context}:{fid}")
        gc.collect()
    pd.DataFrame(screen_records).to_csv(screen_path, index=False)
    ranking = (
        pd.DataFrame(screen_records)
        .groupby(["context", "layer", "head"], as_index=False)
        .agg(mean_delta=("delta_margin", "mean"), n_facts=("fact_id", "nunique"))
        .sort_values(["context", "mean_delta"], ascending=[True, False])
    )
    selected_sites = ranking.groupby("context", group_keys=False).head(args.top_sites).copy()
    selected_sites["rank"] = selected_sites.groupby("context")["mean_delta"].rank(
        method="first", ascending=False
    ).astype(int)
    selected_sites.to_csv(paths["patching"] / "head_sites_selected.csv", index=False)

    test_records = []
    for candidate in heldout.itertuples():
        fid, context = str(candidate.fact_id), candidate.context
        context_sites = selected_sites.loc[selected_sites["context"] == context].sort_values("rank")
        if context_sites.empty:
            continue
        for truth in (1, 0):
            source = builder.make(fid, context, "LATLAT", truth, 0, "A")
            target = builder.make(fid, context, "DEVDEV", truth, 0, "A")
            source_position = semantic_token_positions(
                tokenizer, source["prompt"], source["a_text"], source["b_text"]
            )["E2"]
            target_position = semantic_token_positions(
                tokenizer, target["prompt"], target["a_text"], target["b_text"]
            )["E2"]
            source_heads = capture_head_inputs(model, tokenizer, device, source["prompt"], source_position)
            baseline = fixed_yes_margin(model, tokenizer, device, target["prompt"])
            for row in context_sites.itertuples():
                patched = patch_heads_margin(
                    model,
                    tokenizer,
                    device,
                    target["prompt"],
                    target_position,
                    source_heads,
                    [(int(row.layer), int(row.head))],
                    head_dim,
                )
                test_records.append(
                    {
                        "fact_id": fid,
                        "context": context,
                        "split": candidate.split,
                        "truth": truth,
                        "experiment": "single_head",
                        "rank": int(row.rank),
                        "layer": int(row.layer),
                        "head": int(row.head),
                        "baseline_margin": baseline,
                        "patched_margin": patched,
                        "delta_margin": patched - baseline,
                        "baseline_correct": int(baseline > 0) if truth == 1 else int(baseline < 0),
                        "patched_correct": int(patched > 0) if truth == 1 else int(patched < 0),
                    }
                )
            for top_k in (1, 3, 5, 10):
                subset = context_sites.head(min(top_k, len(context_sites)))
                sites = [(int(row.layer), int(row.head)) for row in subset.itertuples()]
                patched = patch_heads_margin(
                    model,
                    tokenizer,
                    device,
                    target["prompt"],
                    target_position,
                    source_heads,
                    sites,
                    head_dim,
                )
                test_records.append(
                    {
                        "fact_id": fid,
                        "context": context,
                        "split": candidate.split,
                        "truth": truth,
                        "experiment": f"joint_top_{top_k}",
                        "rank": np.nan,
                        "layer": np.nan,
                        "head": np.nan,
                        "baseline_margin": baseline,
                        "patched_margin": patched,
                        "delta_margin": patched - baseline,
                        "baseline_correct": int(baseline > 0) if truth == 1 else int(baseline < 0),
                        "patched_correct": int(patched > 0) if truth == 1 else int(patched < 0),
                    }
                )
        print(f"confirmed {context}:{fid}")
    pd.DataFrame(test_records).to_csv(test_path, index=False)
    summary = (
        pd.DataFrame(test_records)
        .groupby(["context", "split", "truth", "experiment"], as_index=False)
        .agg(mean_delta=("delta_margin", "mean"), flip_rate=("patched_correct", "mean"), n=("fact_id", "nunique"))
    )
    summary.to_csv(paths["patching"] / "head_confirm_summary.csv", index=False)
    print("Saved:", test_path)


if __name__ == "__main__":
    main()

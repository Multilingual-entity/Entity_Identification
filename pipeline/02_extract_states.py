"""Extract isolated-entity and actual pair-prompt hidden states, resumably."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    CONDITIONS,
    MAX_LENGTH,
    add_common_args,
    ensure_run_dirs,
    format_chat,
    load_model_and_tokenizer,
    load_prepared_data,
    read_json,
    seed_everything,
    semantic_char_positions,
    token_covering_char,
    write_json,
)


POSITION_NAMES = ["E1", "E2", "Q", "READOUT"]


def extract_pair_states(
    model,
    tokenizer,
    device,
    metadata: pd.DataFrame,
    output_path: Path,
    progress_path: Path,
    batch_size: int,
    n_layers: int,
    d_model: int,
    force: bool,
) -> None:
    import torch
    from tqdm.auto import tqdm

    shape = (len(metadata), n_layers, len(POSITION_NAMES), d_model)
    if force:
        for path in (output_path, progress_path):
            if path.exists():
                path.unlink()
    start_row = 0
    if output_path.exists() and progress_path.exists():
        states = np.lib.format.open_memmap(output_path, mode="r+")
        if states.shape != shape:
            raise RuntimeError(f"Existing state shape {states.shape} != expected {shape}; use --force.")
        start_row = int(read_json(progress_path)["next_row"])
    else:
        states = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=shape)
        write_json(progress_path, {"next_row": 0, "shape": shape})

    for start in tqdm(range(start_row, len(metadata), batch_size), desc="Pair-prompt states"):
        chunk = metadata.iloc[start : start + batch_size]
        formatted = [format_chat(tokenizer, prompt) for prompt in chunk["prompt"]]
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        batch_positions = []
        sequence_length = int(encoded["input_ids"].shape[1])
        for text, offsets, (_, row) in zip(formatted, offsets_batch, chunk.iterrows()):
            chars = semantic_char_positions(text, str(row["a_text"]), str(row["b_text"]))
            positions = {name: token_covering_char(offsets, index) for name, index in chars.items()}
            positions["READOUT"] = sequence_length - 1
            batch_positions.append(positions)
        encoded = encoded.to(device)
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for layer, hidden in enumerate(output.hidden_states[1:]):
            vectors = torch.stack(
                [
                    torch.stack([hidden[i, batch_positions[i][name]] for name in POSITION_NAMES])
                    for i in range(len(chunk))
                ]
            )
            states[start : start + len(chunk), layer] = vectors.detach().to(torch.float16).cpu().numpy()
        states.flush()
        write_json(progress_path, {"next_row": start + len(chunk), "shape": shape})
        del encoded, output
        gc.collect()


def extract_isolated_states(
    model,
    tokenizer,
    device,
    metadata: pd.DataFrame,
    output_path: Path,
    progress_path: Path,
    batch_size: int,
    n_layers: int,
    d_model: int,
    force: bool,
) -> None:
    import torch
    from tqdm.auto import tqdm

    shape = (len(metadata), n_layers, d_model)
    if force:
        for path in (output_path, progress_path):
            if path.exists():
                path.unlink()
    start_row = 0
    if output_path.exists() and progress_path.exists():
        states = np.lib.format.open_memmap(output_path, mode="r+")
        if states.shape != shape:
            raise RuntimeError(f"Existing isolated state shape {states.shape} != {shape}; use --force.")
        start_row = int(read_json(progress_path)["next_row"])
    else:
        states = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=shape)
        write_json(progress_path, {"next_row": 0, "shape": shape})

    for start in tqdm(range(start_row, len(metadata), batch_size), desc="Isolated entity states"):
        chunk = metadata.iloc[start : start + batch_size]
        raw = [f"Entity: [{text}]" for text in chunk["text"]]
        formatted = [format_chat(tokenizer, prompt) for prompt in raw]
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        positions = []
        for text, offsets, entity in zip(formatted, offsets_batch, chunk["text"]):
            entity_start = text.find(str(entity))
            close = text.find("]", entity_start + len(str(entity)))
            positions.append(token_covering_char(offsets, close))
        encoded = encoded.to(device)
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for layer, hidden in enumerate(output.hidden_states[1:]):
            vectors = torch.stack([hidden[i, positions[i]] for i in range(len(chunk))])
            states[start : start + len(chunk), layer] = vectors.detach().to(torch.float16).cpu().numpy()
        states.flush()
        write_json(progress_path, {"next_row": start + len(chunk), "shape": shape})
        del encoded, output
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    corpus, _ = load_prepared_data(args.run_dir)
    behavior_path = paths["tables"] / "behavior.csv"
    if not behavior_path.exists():
        raise FileNotFoundError("Run 01_run_behavior.py before state extraction.")
    behavior = pd.read_csv(behavior_path)
    pair_meta = behavior.loc[
        (behavior["paraphrase_id"] == 0) & (behavior["yes_letter"] == "A"),
        [
            "fact_id",
            "split",
            "context",
            "condition",
            "truth",
            "prompt",
            "a_text",
            "b_text",
            "correct",
            "yes_minus_no_margin",
        ],
    ].drop_duplicates(["fact_id", "context", "condition", "truth"])
    pair_meta = pair_meta.sort_values(["context", "condition", "truth", "fact_id"]).reset_index(drop=True)
    pair_meta.insert(0, "state_row", np.arange(len(pair_meta), dtype=np.int64))
    pair_meta.to_csv(paths["states"] / "pair_state_metadata.csv", index=False)

    isolated_records = []
    for _, row in corpus.iterrows():
        for role in ("a", "b"):
            for script in ("dev", "lat"):
                isolated_records.append(
                    {
                        "fact_id": row["fact_id"],
                        "split": row["split"],
                        "role": role,
                        "script": script,
                        "text": row[f"name_{role}_{script}"],
                    }
                )
    isolated_meta = pd.DataFrame(isolated_records)
    isolated_meta.insert(0, "state_row", np.arange(len(isolated_meta), dtype=np.int64))
    isolated_meta.to_csv(paths["states"] / "isolated_state_metadata.csv", index=False)

    model, tokenizer, device, batch_size, model_meta = load_model_and_tokenizer(
        args.model, args.batch_size
    )
    extract_isolated_states(
        model,
        tokenizer,
        device,
        isolated_meta,
        paths["states"] / "isolated_states.npy",
        paths["states"] / "isolated_states.progress.json",
        batch_size,
        model_meta["n_layers"],
        model_meta["d_model"],
        args.force,
    )
    extract_pair_states(
        model,
        tokenizer,
        device,
        pair_meta,
        paths["states"] / "pair_states.npy",
        paths["states"] / "pair_states.progress.json",
        max(1, batch_size // 2),
        model_meta["n_layers"],
        model_meta["d_model"],
        args.force,
    )
    write_json(
        paths["states"] / "state_manifest.json",
        {
            **model_meta,
            "position_order": POSITION_NAMES,
            "n_pair_rows": int(len(pair_meta)),
            "n_isolated_rows": int(len(isolated_meta)),
            "layer_convention": "layer 0 is output of transformer block 0",
        },
    )
    print("State extraction complete:", paths["states"])


if __name__ == "__main__":
    main()

r"""Does the model read the same direction in Devanagari as it does in Latin?

Stage 09 takes the Jacobian in the Devanagari run and projects both states onto it. That
answers "how much does the Devanagari state carry along the direction the Devanagari run
reads", and it finds less. It cannot separate the two ways that can happen:

  magnitude   the two runs read the same direction, and the Devanagari state simply sits
              nearer the threshold along it. A push along that direction should then help.

  direction   the two runs read different directions, and the Devanagari state is not
              deficient so much as pointed elsewhere. A push along the Latin direction
              would then do little, which is what the failed correction vector looked
              like.

The paper currently asserts the first reading. Distinguishing them needs the Jacobian
from *both* runs, which stage 09 computes and discards: it takes the gradient in the
Devanagari run and keeps only the state from the Latin one.

This script keeps both, and reports:

  cos(g_dev, g_lat)      do the two runs read the same direction at all
  ||g_dev||, ||g_lat||   is the model less sensitive under Devanagari, or equally so
  own-frame projections  each state onto its own run's direction
  cross-frame projection the Devanagari state onto the Latin run's direction, which is
                         what a correction vector learned from Latin runs would move
  random baseline        cosine between g_dev and a random direction, because in a
                         4096-dimensional space a small cosine is not by itself evidence
                         of anything and the reference point has to be measured
  predicted vs actual    g . dv against the margin change from actually adding dv, which
                         is what tells you whether the linear picture holds at all

    python 13_jacobian_script_pair.py --run-dir results/qwen_mr --model qwen
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    PromptBuilder,
    add_common_args,
    check_contexts,
    ensure_run_dirs,
    format_chat,
    get_layers,
    label_token_ids,
    load_model_and_tokenizer,
    load_prepared_data,
    scale_layer,
    seed_everything,
    semantic_token_positions,
)

HERE = Path(__file__).resolve().parent
POSITIONS = ["E1", "E2", "Q", "READOUT"]


def _stage09():
    """Reuse stage 09's Jacobian rather than reimplementing it.

    The hook has to detach the block output, mark it as requiring grad and let the rest
    of the forward run from that tensor, or the gradient reaching it is not the quantity
    wanted. Copying that by hand is how the two stages would drift apart.
    """
    spec = importlib.util.spec_from_file_location("jl", HERE / "09_jacobian_lens.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["jl"] = module
    spec.loader.exec_module(module)
    return module.margin_jacobian, module.cosine


def add_and_measure(model, tokenizer, device, prompt, layer, position, yes_letter,
                    delta: np.ndarray):
    """Margin after adding `delta` to the hidden state, for the linearity check."""
    a_id, b_id = label_token_ids(tokenizer)
    yes_id, no_id = (a_id, b_id) if yes_letter == "A" else (b_id, a_id)
    import torch

    shift = torch.tensor(delta, dtype=torch.float32, device=device)

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[0, position] = hidden[0, position] + shift.to(hidden.dtype)
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            encoded = tokenizer(format_chat(tokenizer, prompt), return_tensors="pt",
                                truncation=True, max_length=384).to(device)
            logits = model(**encoded, use_cache=False).logits[0, -1].float()
            return float((logits[yes_id] - logits[no_id]).item())
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"])
    parser.add_argument("--site", default="E2", choices=POSITIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--scale-layers", action="store_true")
    parser.add_argument("--max-facts", type=int, default=60)
    parser.add_argument("--linearity-facts", type=int, default=20,
                        help="items on which to check the first-order prediction against "
                             "an actual intervention. Each costs one extra forward pass")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="step size for that check, as a fraction of the state norm. "
                             "Small on purpose: the Jacobian is a local approximation and "
                             "a large step tests something else")
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)

    out_path = paths["patching"] / "jacobian_script_pair.csv"
    if out_path.exists() and not args.force:
        print(f"Already complete: {out_path}. Use --force to rerun.")
        return

    margin_jacobian, cosine = _stage09()
    corpus, negatives = load_prepared_data(args.run_dir)
    model, tokenizer, device, _, meta = load_model_and_tokenizer(args.model, args.batch_size)
    n_layers = meta["n_layers"]

    if args.layers:
        layers = [l for l in args.layers if l < n_layers]
    else:
        band = range(12, 18)
        layers = sorted({scale_layer(l, n_layers) for l in band}) if args.scale_layers \
            else [l for l in band if l < n_layers]
    print(f"site {args.site} | layers {layers} | {args.max_facts} facts")

    builder = PromptBuilder(corpus, negatives)
    fact_ids = corpus["fact_id"].astype(str).tolist()[: args.max_facts]
    rng = np.random.default_rng(args.seed)

    rows = []
    checked_alignment = False
    for context in args.contexts:
        for layer in layers:
            for n, fid in enumerate(fact_ids):
                dev = builder.make(fid, context, "DEVDEV", 1, 0, "A")
                lat = builder.make(fid, context, "LATLAT", 1, 0, "A")
                dpos = semantic_token_positions(
                    tokenizer, dev["prompt"], dev["a_text"], dev["b_text"])[args.site]
                lpos = semantic_token_positions(
                    tokenizer, lat["prompt"], lat["a_text"], lat["b_text"])[args.site]

                g_dev, s_dev, m_dev = margin_jacobian(
                    model, tokenizer, device, dev["prompt"], layer, dpos, "A")
                g_lat, s_lat, m_lat = margin_jacobian(
                    model, tokenizer, device, lat["prompt"], layer, lpos, "A")

                n_dev = np.linalg.norm(g_dev) or 1.0
                n_lat = np.linalg.norm(g_lat) or 1.0
                # A cosine near zero means little on its own in a space this wide, so the
                # same quantity against a random direction is reported alongside it.
                random_dir = rng.normal(size=g_dev.shape).astype(g_dev.dtype)

                row = {
                    "fact_id": fid, "context": context, "layer": layer, "site": args.site,
                    "margin_dev": m_dev, "margin_lat": m_lat,
                    "grad_norm_dev": float(n_dev), "grad_norm_lat": float(n_lat),
                    "cos_gdev_glat": cosine(g_dev, g_lat),
                    "cos_gdev_random": cosine(g_dev, random_dir),
                    # Each state read in the frame its own run defines.
                    "proj_dev_own": float(s_dev @ g_dev / n_dev),
                    "proj_lat_own": float(s_lat @ g_lat / n_lat),
                    # The Devanagari state read in the Latin run's frame: what a
                    # correction vector fitted on Latin runs would actually move.
                    "proj_dev_in_lat_frame": float(s_dev @ g_lat / n_lat),
                    "proj_lat_in_dev_frame": float(s_lat @ g_dev / n_dev),
                }

                # First-order prediction against an actual small intervention, on a
                # subset because it costs an extra forward pass each.
                if n < args.linearity_facts:
                    delta = (s_lat - s_dev) * args.epsilon
                    row["predicted_delta"] = float(g_dev @ delta)
                    actual = add_and_measure(model, tokenizer, device, dev["prompt"],
                                             layer, dpos, "A", delta)
                    row["actual_delta"] = actual - m_dev
                    # A zero shift must reproduce the margin the gradient pass measured.
                    # If it does not, the two are not reading the same prompt, position
                    # or layer, and every number in this row is meaningless. Checked once
                    # rather than every item, since one failure condemns the run.
                    if not checked_alignment:
                        null = add_and_measure(model, tokenizer, device, dev["prompt"],
                                               layer, dpos, "A", np.zeros_like(delta))
                        if abs(null - m_dev) > 1e-2:
                            raise SystemExit(
                                f"alignment check failed: a zero-shift intervention gives "
                                f"{null:.4f} where the gradient pass measured {m_dev:.4f}. "
                                f"The two passes are not reading the same state.")
                        print(f"  alignment check passed: zero shift reproduces the "
                              f"margin to {abs(null - m_dev):.1e}")
                        checked_alignment = True
                rows.append(row)

            print(f"  {context} layer {layer}: {len(fact_ids)} facts")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)

    agg = frame.groupby(["context", "layer"]).agg(
        cos_gdev_glat=("cos_gdev_glat", "mean"),
        cos_gdev_random=("cos_gdev_random", "mean"),
        grad_norm_dev=("grad_norm_dev", "mean"),
        grad_norm_lat=("grad_norm_lat", "mean"),
        proj_dev_own=("proj_dev_own", "mean"),
        proj_lat_own=("proj_lat_own", "mean"),
        proj_dev_in_lat_frame=("proj_dev_in_lat_frame", "mean"),
        n=("fact_id", "count"),
    ).reset_index()
    # The quantity the paper needs: how much of the deficit survives when each state is
    # read in its own frame. If it survives, the two runs disagree about magnitude; if it
    # vanishes, they disagree about direction.
    agg["deficit_own_frame"] = agg.proj_lat_own - agg.proj_dev_own
    agg["deficit_lat_frame"] = agg.proj_lat_own - agg.proj_dev_in_lat_frame
    agg.to_csv(paths["patching"] / "jacobian_script_pair_summary.csv", index=False)

    lin = frame.dropna(subset=["predicted_delta"])
    print()
    print(agg.round(4).to_string(index=False))
    if len(lin) > 2:
        r = float(np.corrcoef(lin.predicted_delta, lin.actual_delta)[0, 1])
        print(f"\nfirst-order check on {len(lin)} item-layers at epsilon {args.epsilon}: "
              f"predicted against actual margin change, r = {r:.3f}")
        print("A high r means the linear picture holds at this step size; a low one means")
        print("the Jacobian describes the neighbourhood and not the intervention.")
    print()
    print("cos_gdev_glat against cos_gdev_random is the comparison that matters: a cosine")
    print("is only small relative to what a random direction would give.")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()

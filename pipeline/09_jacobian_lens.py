"""Which direction at the second entity does the output actually depend on?

The probe shows the answer is present in the representation. Patching shows that
changing the representation changes the answer. Neither says what the downstream
computation *reads*, and the head search found no sparse circuit to point at.

The Jacobian answers that directly. Differentiating the yes-minus-no margin with respect
to the hidden state at one position gives a single vector: the direction the output is
locally sensitive to. Everything orthogonal to it is invisible downstream, however much
information it carries.

Three things follow, none of which the current results can settle.

  Alignment. Does the learned correction vector point along the Jacobian? The paper
  implies the vector works by pushing the representation the way the output reads it,
  and has no evidence for that. A high cosine confirms it; a low one means the vector
  works some other way and the account needs revising.

  Direction or magnitude. Project the Devanagari and Latin states onto the Jacobian. If
  Devanagari carries less along the same direction, the deficit is one of magnitude and
  the threshold reading is right. If it points elsewhere, it is a misalignment. The
  paper cannot currently tell these apart.

  Prediction. If the projection separates items the model gets right from those it gets
  wrong, the deficit has a scalar measure computed from one backward pass. That works on
  models whose causal gate never fires, so it sidesteps the problem that has cost the
  paper two of its three models.

The Jacobian is a local linear approximation. It describes what the output is sensitive
to near the current activation, and does not predict the effect of a full state swap,
which is a large move outside that neighbourhood. It complements patching rather than
replacing it, and should be reported as the direction of local sensitivity rather than
as a concept space.
"""
from __future__ import annotations

import argparse

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

POSITIONS = ["E1", "E2", "Q", "READOUT"]


def margin_jacobian(model, tokenizer, device, prompt: str, layer: int, position: int,
                    yes_letter: str):
    """d(margin) / d(hidden state at `position`, output of `layer`).

    One backward pass. The hook captures the block output, marks it as requiring grad,
    and lets the rest of the forward run from it, so the gradient reaching that tensor is
    exactly the sensitivity of the margin to that state.
    """
    a_id, b_id = label_token_ids(tokenizer)
    yes_id, no_id = (a_id, b_id) if yes_letter == "A" else (b_id, a_id)
    layers = get_layers(model)
    captured = {}

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.detach().clone().requires_grad_(True)
        captured["h"] = hidden
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = layers[layer].register_forward_hook(hook)
    try:
        encoded = tokenizer(format_chat(tokenizer, prompt), return_tensors="pt",
                            truncation=True, max_length=384).to(device)
        logits = model(**encoded, use_cache=False).logits[0, -1].float()
        margin = logits[yes_id] - logits[no_id]
        model.zero_grad(set_to_none=True)
        margin.backward()
        grad = captured["h"].grad[0, position].detach().float().cpu().numpy()
        state = captured["h"][0, position].detach().float().cpu().numpy()
        return grad, state, float(margin.item())
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"])
    parser.add_argument("--site", default="E2", choices=POSITIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="layers to probe; defaults to the preregistered band")
    parser.add_argument("--scale-layers", action="store_true",
                        help="rescale the default band by model depth")
    parser.add_argument("--max-facts", type=int, default=80)
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)

    out_path = paths["patching"] / "jacobian_lens.csv"
    if out_path.exists() and not args.force:
        print(f"Already complete: {out_path}. Use --force to rerun.")
        return

    corpus, negatives = load_prepared_data(args.run_dir)
    behavior = pd.read_csv(paths["tables"] / "behavior.csv")
    model, tokenizer, device, _, meta = load_model_and_tokenizer(args.model, args.batch_size)
    n_layers = meta["n_layers"]

    if args.layers:
        layers = [l for l in args.layers if l < n_layers]
    else:
        band = range(12, 18)
        layers = sorted({scale_layer(l, n_layers) for l in band}) if args.scale_layers \
            else [l for l in band if l < n_layers]
    print(f"site {args.site} | layers {layers}")

    # The correction vector, if stage 07 has produced one. Its alignment with the
    # Jacobian is the point of the whole exercise.
    vec_path = paths["recovery"] / "script_correction_vectors.npz"
    vectors = np.load(vec_path) if vec_path.exists() else None
    if vectors is None:
        print("no correction vectors found; alignment will be skipped. Run 07 first.")

    builder = PromptBuilder(corpus, negatives)
    fact_ids = corpus["fact_id"].astype(str).tolist()[: args.max_facts]
    correct_of = {(str(r.fact_id), r.context): r.correct
                  for r in behavior[(behavior.truth == 1)
                                    & (behavior.condition == "DEVDEV")
                                    & (behavior.paraphrase_id == 0)
                                    & (behavior.yes_letter == "A")].itertuples()}

    rows = []
    for context in args.contexts:
        for layer in layers:
            for n, fid in enumerate(fact_ids):
                record = builder.make(fid, context, "DEVDEV", 1, 0, "A")
                pos = semantic_token_positions(
                    tokenizer, record["prompt"], record["a_text"], record["b_text"])[args.site]
                grad, dev_state, margin = margin_jacobian(
                    model, tokenizer, device, record["prompt"], layer, pos, "A")

                latin = builder.make(fid, context, "LATLAT", 1, 0, "A")
                lpos = semantic_token_positions(
                    tokenizer, latin["prompt"], latin["a_text"], latin["b_text"])[args.site]
                _, lat_state, lat_margin = margin_jacobian(
                    model, tokenizer, device, latin["prompt"], layer, lpos, "A")

                row = {
                    "fact_id": fid, "context": context, "layer": layer, "site": args.site,
                    "margin_devdev": margin, "margin_latlat": lat_margin,
                    "grad_norm": float(np.linalg.norm(grad)),
                    # How much of each state lies along the direction the output reads.
                    # If Devanagari carries less of it, the deficit is one of magnitude.
                    "proj_devdev": float(dev_state @ grad / (np.linalg.norm(grad) or 1)),
                    "proj_latlat": float(lat_state @ grad / (np.linalg.norm(grad) or 1)),
                    "cos_states": cosine(dev_state, lat_state),
                    "correct_devdev": correct_of.get((fid, context), np.nan),
                }
                # Does the difference the patch would inject point where the output reads?
                row["cos_grad_statediff"] = cosine(grad, lat_state - dev_state)
                if vectors is not None:
                    for kind in ("full", "additive", "first_entity", "second_entity"):
                        key = f"{context}__{kind}"
                        if key in vectors:
                            v = vectors[key][layer, POSITIONS.index(args.site)].astype(np.float32)
                            row[f"cos_grad_{kind}"] = cosine(grad, v)
                rows.append(row)
                if (n + 1) % 20 == 0:
                    print(f"  {context} layer {layer}: {n + 1}/{len(fact_ids)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)

    agg = frame.groupby(["context", "layer"]).agg(
        grad_norm=("grad_norm", "mean"),
        proj_devdev=("proj_devdev", "mean"),
        proj_latlat=("proj_latlat", "mean"),
        cos_grad_statediff=("cos_grad_statediff", "mean"),
        n=("fact_id", "nunique"))
    agg["proj_deficit"] = agg.proj_latlat - agg.proj_devdev
    for kind in ("full", "additive"):
        col = f"cos_grad_{kind}"
        if col in frame.columns:
            agg[col] = frame.groupby(["context", "layer"])[col].mean()
    agg.to_csv(paths["patching"] / "jacobian_lens_summary.csv")

    print()
    print(agg.round(4).to_string())
    print()
    print("proj_deficit is how much more of the output-sensitive direction the Latin")
    print("state carries. Positive means the failure is a shortfall along the direction")
    print("that matters, rather than a representation pointing somewhere else.")

    if "cos_grad_additive" in frame.columns:
        m = frame.cos_grad_additive.mean()
        print(f"\nmean cosine between the correction vector and the Jacobian: {m:+.3f}")
        print("  High means the vector works by pushing along the direction the output")
        print("  reads, which is what the paper implies but has not shown.")

    ok = frame.dropna(subset=["correct_devdev"])
    if len(ok) and ok.correct_devdev.nunique() > 1:
        print("\nprojection by outcome, to see whether it separates success from failure:")
        print(ok.groupby(["context", "layer", "correct_devdev"]).proj_devdev.mean()
              .unstack().round(3).to_string())

    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()

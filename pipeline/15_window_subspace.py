r"""The rank-one subspace applied across a layer window, as the reference arm.

Experiment 0 asks whether a fact-independent direction repairs the gap when applied
across layers 12--17 rather than at one layer. If that fails, the result is ambiguous: it
could mean no fixed direction works at the window, or it could mean the window needs
each item's own donor coordinate rather than one shared amount. Without separating those,
a null is unreadable.

This is the upper bound that separates them. It applies the rank-one sensitivity subspace
at every layer in the window, using that layer's own direction and that item's own donor
state, which is the most informed intervention short of replacing the state outright. If
even this flips nothing across the window, a constant vector was never going to, and
section 7's "unresolved" becomes "resolved negative at this site".

Stage 14 patches one layer, because SubspacePatch hooks a single block. Rather than
thread a window through that file's folds, arms and caches, this composes one
SubspacePatch per layer and drives them together. Everything else is imported from stage
14, so the subspace construction, the gating and the margin convention cannot drift apart
from it.

    python 15_window_subspace.py --run-dir results/qwen_mr --model qwen \
        --contexts mr en --layers 12 13 14 15 16 17 --rank 1 --split test
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("stage14", HERE / "14_failure_mediator.py")
stage14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage14)

from pipeline_common import (  # noqa: E402
    PromptBuilder, add_common_args, ensure_run_dirs, load_model_and_tokenizer,
    load_prepared_data, seed_everything,
)


class WindowPatch:
    """One SubspacePatch per layer, set and cleared together.

    Composing the reviewed single-layer hook keeps the algebra identical to stage 14's;
    a second implementation of the same intervention would be free to disagree with it.
    """

    def __init__(self, model, layers: list[int]):
        self.layers = list(layers)
        self.patches = {L: stage14.SubspacePatch(model, L) for L in self.layers}

    def off(self) -> None:
        for p in self.patches.values():
            p.off()

    def subspace(self, position: int, donors: dict, subspaces: dict) -> None:
        """donors and subspaces are keyed by layer, so each layer uses its own."""
        for L, p in self.patches.items():
            p.subspace(position, donors[L], subspaces[L])

    def whole(self, position: int, donors: dict) -> None:
        for L, p in self.patches.items():
            p.whole(position, donors[L])

    def remove(self) -> None:
        for p in self.patches.values():
            p.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--contexts", nargs="+", default=["mr", "en"])
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="the window, e.g. 12 13 14 15 16 17")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--method", default="jacobian",
                    choices=["jacobian", "meandiff", "pca", "probe", "random"])
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--allow-test", action="store_true",
                    help="required for --split test, so a held-out evaluation is "
                         "deliberate rather than a default")
    ap.add_argument("--train-paraphrases", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--eval-paraphrases", type=int, nargs="+", default=[2])
    ap.add_argument("--max-train-facts", type=int, default=60)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test; the test split is spent once")

    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    corpus, negatives = load_prepared_data(args.run_dir)
    # The same file stage 14 reads, so the gate cannot differ between them.
    candidates = pd.read_csv(paths["patching"] / "causal_candidates.csv")

    import torch
    model, tokenizer, device, _, _ = load_model_and_tokenizer(args.model, args.batch_size)
    builder = PromptBuilder(corpus, negatives)
    source = stage14.StateSource(model, tokenizer, device, builder, paths)

    fold = stage14.fold_tag(args.train_paraphrases, args.eval_paraphrases)
    print(f"window {args.layers}, rank {args.rank}, {args.method}, "
          f"fold {fold}, split {args.split}")

    rows = []
    for context in args.contexts:
        fit = stage14.behavioural_groups(candidates, context, "train")
        ev = stage14.behavioural_groups(candidates, context, args.split)
        gated_fit = fit["gated"][: args.max_train_facts]
        if not gated_fit or not ev["gated"]:
            print(f"{context}: no gated items in one of the splits, skipping")
            continue

        # One subspace per layer, each built exactly as stage 14 builds it.
        subspaces, seed = {}, stage14.stable_seed(args.seed, context, args.method,
                                                  args.rank, fold)
        for L in args.layers:
            diffs = np.stack([
                source.e2_state(f, context, "LATLAT", t, p, L)
                - source.e2_state(f, context, "DEVDEV", t, p, L)
                for p in args.train_paraphrases for f, t in gated_fit
            ])
            probe_dir = (stage14.probe_direction(source, corpus, context, L)
                         if args.method == "probe" else None)
            jac_dir = (stage14.jacobian_direction(model, tokenizer, device, source,
                                                  gated_fit, context, L,
                                                  args.train_paraphrases[0])
                       if args.method == "jacobian" else None)
            subspaces[L] = torch.as_tensor(
                stage14.derived_subspace(args.method, args.rank, diffs,
                                         np.random.default_rng(seed + L),
                                         probe_dir, jac_dir), device=device)

        patch = WindowPatch(model, args.layers)
        try:
            for group in ("gated", "negatives"):
                for fid, truth in ev[group]:
                    for p in args.eval_paraphrases:
                        rec = source.prompt(fid, context, "DEVDEV", truth, p)
                        pos = source.e2_position(rec)
                        donors = {
                            L: torch.as_tensor(
                                source.e2_state(fid, context, "LATLAT", truth, p, L),
                                device=device)
                            for L in args.layers}

                        patch.off()
                        base = float(stage14.margin_with_grad(
                            model, tokenizer, device, rec["prompt"], False))
                        patch.whole(pos, donors)
                        whole = float(stage14.margin_with_grad(
                            model, tokenizer, device, rec["prompt"], False))
                        patch.subspace(pos, donors, subspaces)
                        sub = float(stage14.margin_with_grad(
                            model, tokenizer, device, rec["prompt"], False))
                        patch.off()

                        tc = stage14.toward_correct
                        rows.append(dict(
                            context=context, split=args.split, fold=fold,
                            window="_".join(map(str, args.layers)), rank=args.rank,
                            method=args.method, group=group, fact_id=fid, truth=truth,
                            paraphrase_id=p, baseline_margin=base,
                            whole_margin=whole, subspace_margin=sub,
                            delta_whole=tc(whole - base, truth),
                            delta_subspace=tc(sub - base, truth),
                            baseline_correct=int(tc(base, truth) > 0),
                            whole_correct=int(tc(whole, truth) > 0),
                            subspace_correct=int(tc(sub, truth) > 0)))
        finally:
            patch.remove()

    if not rows:
        raise SystemExit("no rows produced; check the gate and the splits")

    items = pd.DataFrame(rows)
    out = paths["patching"] / "window_subspace_items.csv"
    items.to_csv(out, index=False)

    print()
    print(f"{'ctx':5}{'group':11}{'n':>4}{'whole Δ':>10}{'sub Δ':>9}"
          f"{'share':>8}{'whole flip':>12}{'sub flip':>10}")
    summary = []
    for (ctx, grp), g in items.groupby(["context", "group"]):
        per = g.groupby("fact_id")
        w, s = per.delta_whole.mean().mean(), per.delta_subspace.mean().mean()
        wf, sf = per.whole_correct.max().mean(), per.subspace_correct.max().mean()
        share = s / w if abs(w) > 1e-9 else float("nan")
        print(f"{ctx:5}{grp:11}{per.ngroups:>4}{w:>10.2f}{s:>9.2f}{share:>8.2f}"
              f"{wf * 100:>11.0f}%{sf * 100:>9.0f}%")
        summary.append(dict(context=ctx, group=grp, n_items=per.ngroups,
                            delta_whole=w, delta_subspace=s, share=share,
                            whole_flip=wf, subspace_flip=sf))
    pd.DataFrame(summary).to_csv(
        paths["patching"] / "window_subspace_summary.csv", index=False)
    print(f"\nSaved: {out}")
    print("\nRead the gated row against section 7. If the whole patch flips a large share")
    print("and the rank-one subspace flips little, the window needs more than one")
    print("direction. If neither flips, no additive intervention at this site was going")
    print("to work and the repair result is negative rather than unresolved.")


if __name__ == "__main__":
    main()

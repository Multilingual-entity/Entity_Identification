"""Does the second-slot penalty follow the position, or the name that sits in it?

In-prompt retrieval is strong at the first entity and about chance at the second. The
paper reads that as a positional effect: whatever sits second is retrieved worse. But the
corpus never separates the two possibilities, because the first slot always holds the
canonical Wikidata label and the second always holds the alias. The alias is the rarer
string, so an equally good explanation is that the second slot looks worse only because
the rarer name is always in it.

Swapping the order separates them. Build the same question with the alias first and the
canonical name second, and score both orders.

  If the penalty stays with the slot, it is positional, and the retrieval account holds:
  the model reads the first name well and the second poorly whatever they are.

  If the penalty follows the alias into the first slot, the effect is about the name and
  not the position, and the paper's positional claim has to be withdrawn.

  If it does both, the two contribute separately and the design gives the size of each.

The script contrast is preserved inside every order, so the Devanagari deficit can be read
off within order as well, which is a free replication on a prompt set nothing was tuned on.

Scripts stay attached to names rather than to slots. In the swapped order under LATDEV,
the name written in Latin in the canonical order is still written in Latin. The recorded
slot1_script and slot2_script columns say what actually appeared where, so the analysis
can be done either way.
"""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from pipeline_common import (
    CONDITIONS,
    QUESTION_TEMPLATES,
    PromptBuilder,
    add_common_args,
    capture_prompt_states,
    check_contexts,
    dprime,
    ensure_run_dirs,
    fixed_yes_margin,
    load_model_and_tokenizer,
    load_prepared_data,
    patch_margin,
    scale_layer,
    score_prompt_frame,
    seed_everything,
    semantic_token_positions,
)

POSITIONS = ["E1", "E2", "Q", "READOUT"]


class OrderedPromptBuilder(PromptBuilder):
    """PromptBuilder that can put the second name first.

    The canonical order reproduces exactly what the rest of the pipeline produces, so the
    canonical arm of this run is a replication of the main behaviour run through a
    different code path. Any disagreement between them is a bug, and main() checks for it.
    """

    def make_ordered(self, fact_id, context, condition, truth, paraphrase_id,
                     yes_letter, order):
        a_script, b_script = CONDITIONS[condition]
        name_a, name_b = self.names(fact_id, condition, truth)
        if order == "canonical":
            slot1, slot2 = name_a, name_b
            slot1_role, slot2_role = "a", "b"
            slot1_script, slot2_script = a_script, b_script
        elif order == "swapped":
            slot1, slot2 = name_b, name_a
            slot1_role, slot2_role = "b", "a"
            slot1_script, slot2_script = b_script, a_script
        else:
            raise ValueError(f"unknown order: {order}")

        question = QUESTION_TEMPLATES[context][paraphrase_id].format(a=slot1, b=slot2)
        instruction, no_letter = self.answer_instruction(yes_letter)
        return {
            "fact_id": str(fact_id),
            "context": context,
            "condition": condition,
            "truth": int(truth),
            "paraphrase_id": int(paraphrase_id),
            "yes_letter": yes_letter,
            "no_letter": no_letter,
            "order": order,
            "slot1_role": slot1_role,
            "slot2_role": slot2_role,
            "slot1_script": slot1_script,
            "slot2_script": slot2_script,
            "correct_semantic": "yes" if int(truth) == 1 else "no",
            "a_text": slot1,
            "b_text": slot2,
            "anchor": "none",
            "question": question,
            "prompt": question + "\n\n" + instruction,
        }


def build_frame(corpus, negatives, contexts, conditions, truths, paraphrases,
                yes_letters, orders):
    builder = OrderedPromptBuilder(corpus, negatives)
    split_of = dict(zip(corpus["fact_id"].astype(str), corpus["split"].astype(str)))
    records = []
    for fid in corpus["fact_id"].astype(str):
        for context, condition, truth, paraphrase_id, yes_letter, order in product(
            contexts, conditions, truths, paraphrases, yes_letters, orders
        ):
            record = builder.make_ordered(fid, context, condition, truth,
                                          paraphrase_id, yes_letter, order)
            record["split"] = split_of[fid]
            records.append(record)
    frame = pd.DataFrame(records)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def sensitivity_table(scored: pd.DataFrame, keys) -> pd.DataFrame:
    """Hit rate, false alarm rate, sensitivity and bias for each cell."""
    rows = []
    for key, sub in scored.groupby(list(keys)):
        key = key if isinstance(key, tuple) else (key,)
        same = sub[sub.truth == 1]
        diff = sub[sub.truth == 0]
        if same.empty or diff.empty:
            continue
        hit = float((same.pred_semantic == "yes").mean())
        fa = float((diff.pred_semantic == "yes").mean())
        d, criterion = dprime(hit, fa, len(same), len(diff))
        record = dict(zip(keys, key))
        record.update(
            accuracy=float(sub.correct.mean()),
            hit_rate=hit,
            false_alarm_rate=fa,
            say_yes_rate=float((sub.pred_semantic == "yes").mean()),
            d_prime=d,
            criterion=criterion,
            mean_margin=float(sub.yes_minus_no_margin.mean()),
            n_facts=int(sub.fact_id.nunique()),
            n=int(len(sub)),
        )
        rows.append(record)
    return pd.DataFrame(rows)


def select_patch_facts(scored: pd.DataFrame, context: str, order: str, gate: str,
                       margin_gap: float, max_facts: int) -> list:
    """Facts this order gets right in Latin and wrong in Devanagari.

    Selection is redone inside each order rather than inherited from stage 04, because the
    whole question is whether the same items still qualify once the names change places.
    A fact that passes the gate in one order and not the other is itself a result, so the
    counts are reported alongside the patching effect.
    """
    sub = scored[(scored.context == context) & (scored.order == order) & (scored.truth == 1)]
    if sub.empty:
        return []
    per = (sub.groupby(["fact_id", "condition"])
           .agg(correct=("correct", "mean"), margin=("yes_minus_no_margin", "mean"))
           .unstack())
    for condition in ("DEVDEV", "LATLAT"):
        for measure in ("correct", "margin"):
            if (measure, condition) not in per.columns:
                per[(measure, condition)] = np.nan
    gap = per[("margin", "LATLAT")] - per[("margin", "DEVDEV")]
    by_accuracy = (per[("correct", "LATLAT")] >= 0.80) & (per[("correct", "DEVDEV")] <= 0.20)
    by_margin = (gap >= margin_gap) & (per[("correct", "DEVDEV")] < 0.80)
    keep = {"accuracy": by_accuracy, "margin": by_margin,
            "either": by_accuracy | by_margin}[gate]
    ordered = gap[keep.fillna(False)].sort_values(ascending=False)
    return [str(fid) for fid in ordered.index[:max_facts]]


def run_patching(model, tokenizer, device, builder, scored, contexts, layers,
                 sites, gate, margin_gap, max_facts, paraphrase_id):
    """Patch the second-entity residual from the Latin prompt into the native one.

    Run inside both orders. If the causal effect is about the second position, it should
    survive the swap and stay at E2. If it follows the alias name into the first slot, it
    should move to E1 when the names do.
    """
    rows = []
    gate_counts = []
    for context in contexts:
        for order in ("canonical", "swapped"):
            facts = select_patch_facts(scored, context, order, gate, margin_gap, max_facts)
            gate_counts.append({"context": context, "order": order, "n_selected": len(facts)})
            print(f"  {context} {order}: {len(facts)} facts pass the gate")
            for n, fid in enumerate(facts):
                for truth in (1, 0):
                    target = builder.make_ordered(fid, context, "DEVDEV", truth,
                                                  paraphrase_id, "A", order)
                    source = builder.make_ordered(fid, context, "LATLAT", truth,
                                                  paraphrase_id, "A", order)
                    source_states, _ = capture_prompt_states(model, tokenizer, device, source)
                    target_positions = semantic_token_positions(
                        tokenizer, target["prompt"], target["a_text"], target["b_text"])
                    baseline = fixed_yes_margin(model, tokenizer, device, target["prompt"])
                    for site in sites:
                        index = POSITIONS.index(site)
                        for layer in layers:
                            patched = patch_margin(
                                model, tokenizer, device, target["prompt"],
                                [(layer, target_positions[site], source_states[layer, index])])
                            rows.append({
                                "fact_id": fid, "context": context, "order": order,
                                "truth": truth, "site": site, "layer": layer,
                                "site_holds_role": (target["slot1_role"] if site == "E1"
                                                    else target["slot2_role"]),
                                "baseline_margin": baseline,
                                "patched_margin": patched,
                                "delta_margin": patched - baseline,
                                "baseline_correct": int(baseline > 0) if truth == 1
                                else int(baseline < 0),
                                "patched_correct": int(patched > 0) if truth == 1
                                else int(patched < 0),
                            })
                    del source_states
                if (n + 1) % 5 == 0:
                    print(f"    {context} {order}: {n + 1}/{len(facts)} facts", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(gate_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"])
    parser.add_argument("--conditions", nargs="+", default=["DEVDEV", "LATLAT"],
                        choices=sorted(CONDITIONS),
                        help="the two matched-script conditions are enough to separate "
                             "position from name role; add the mixed ones for the full "
                             "factorial at twice the cost")
    parser.add_argument("--paraphrases", type=int, nargs="+", default=[0])
    parser.add_argument("--yes-letters", nargs="+", default=["A", "B"],
                        help="both letters, so the answer key is counterbalanced and a "
                             "letter preference cannot be read as a name effect")
    parser.add_argument("--patch", action="store_true",
                        help="after scoring, patch the Latin state into the native prompt "
                             "inside both orders. Answers whether the causal effect is "
                             "positional or belongs to the alias name")
    parser.add_argument("--patch-sites", nargs="+", default=["E1", "E2"],
                        choices=POSITIONS,
                        help="both entity positions, since the point is to see whether "
                             "the effect moves when the names do")
    parser.add_argument("--patch-gate", default="either",
                        choices=["accuracy", "margin", "either"])
    parser.add_argument("--patch-margin-gap", type=float, default=4.0)
    parser.add_argument("--patch-max-facts", type=int, default=20)
    parser.add_argument("--layer-low", type=int, default=8)
    parser.add_argument("--layer-high", type=int, default=20)
    parser.add_argument("--scale-layers", action="store_true",
                        help="rescale the layer band by model depth")
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)

    out_path = paths["tables"] / "name_order_swap.csv"
    partial_path = paths["tables"] / "name_order_swap.partial.csv"
    if out_path.exists() and not args.force:
        print(f"Already complete: {out_path}. Use --force to rerun.")
        return
    if args.force and partial_path.exists():
        partial_path.unlink()

    corpus, negatives = load_prepared_data(args.run_dir)
    frame = build_frame(corpus, negatives, args.contexts, args.conditions, (1, 0),
                        args.paraphrases, args.yes_letters, ("canonical", "swapped"))
    print(f"{len(frame)} prompts, {len(frame) // 2} per order")

    model, tokenizer, device, batch_size, model_meta = load_model_and_tokenizer(
        args.model, args.batch_size)
    scored = score_prompt_frame(model, tokenizer, device, frame, batch_size,
                                checkpoint_path=partial_path)
    scored.to_csv(out_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    by_order = sensitivity_table(scored, ("context", "condition", "order"))
    by_order.to_csv(paths["tables"] / "name_order_swap_summary.csv", index=False)
    print()
    print("Accuracy and sensitivity by order:")
    print(by_order.round(4).to_string(index=False))

    # Which name held the first slot, and how the item then scored.
    role_slot = (scored.groupby(["context", "condition", "order", "slot1_role"],
                                as_index=False)
                 .agg(accuracy=("correct", "mean"),
                      mean_margin=("yes_minus_no_margin", "mean"),
                      n_facts=("fact_id", "nunique")))
    role_slot.to_csv(paths["tables"] / "name_order_role_by_slot.csv", index=False)
    print()
    print("First-slot occupant against outcome:")
    print(role_slot.round(4).to_string(index=False))

    # The two effects, stated as single numbers per language and condition.
    effects = []
    for (context, condition), sub in by_order.groupby(["context", "condition"]):
        piv = sub.set_index("order")
        if not {"canonical", "swapped"} <= set(piv.index):
            continue
        effects.append({
            "context": context,
            "condition": condition,
            "accuracy_canonical": piv.loc["canonical", "accuracy"],
            "accuracy_swapped": piv.loc["swapped", "accuracy"],
            "accuracy_change": piv.loc["swapped", "accuracy"] - piv.loc["canonical", "accuracy"],
            "d_prime_canonical": piv.loc["canonical", "d_prime"],
            "d_prime_swapped": piv.loc["swapped", "d_prime"],
            "d_prime_change": piv.loc["swapped", "d_prime"] - piv.loc["canonical", "d_prime"],
        })
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(paths["tables"] / "name_order_swap_effects.csv", index=False)
    print()
    print("Effect of swapping:")
    print(effect_frame.round(4).to_string(index=False))
    print()
    print("A change near zero means the question is answered on the pair as a whole and")
    print("the second-slot deficit is positional. A large drop when the alias moves to")
    print("the front means the deficit belongs to the name rather than to the slot.")

    # Per-item paired difference, which is what a cluster bootstrap would resample.
    paired = (scored.groupby(["fact_id", "context", "condition", "order"], as_index=False)
              .agg(correct=("correct", "mean"), margin=("yes_minus_no_margin", "mean")))
    wide = paired.pivot_table(index=["fact_id", "context", "condition"],
                              columns="order", values=["correct", "margin"]).reset_index()
    wide.columns = ["_".join(c).rstrip("_") for c in wide.columns]
    if "correct_canonical" in wide and "correct_swapped" in wide:
        wide["correct_change"] = wide.correct_swapped - wide.correct_canonical
        wide["margin_change"] = wide.margin_swapped - wide.margin_canonical
    wide.to_csv(paths["tables"] / "name_order_swap_per_fact.csv", index=False)

    # Consistency check: the main run produced the canonical order by a different path.
    behavior_path = paths["tables"] / "behavior.csv"
    if behavior_path.exists():
        behavior = pd.read_csv(behavior_path)
        ref = behavior[behavior.context.isin(args.contexts)
                       & behavior.condition.isin(args.conditions)
                       & behavior.paraphrase_id.isin(args.paraphrases)
                       & behavior.yes_letter.isin(args.yes_letters)]
        mine = scored[scored.order == "canonical"]
        if len(ref):
            a = ref.groupby(["context", "condition"]).correct.mean()
            b = mine.groupby(["context", "condition"]).correct.mean()
            joined = pd.concat([a.rename("main_run"), b.rename("this_run")], axis=1).dropna()
            joined["difference"] = joined.this_run - joined.main_run
            print()
            print("Canonical arm against the main behaviour run (should be identical):")
            print(joined.round(4).to_string())
            if len(joined) and joined.difference.abs().max() > 1e-9:
                print("  MISMATCH: the canonical arm disagrees with the main run. One of")
                print("  the two prompt paths has changed, and the swap result is not")
                print("  safe to read until that is resolved.")

    if args.patch:
        n_layers = model_meta["n_layers"]
        if args.scale_layers:
            low = scale_layer(args.layer_low, n_layers)
            high = scale_layer(args.layer_high, n_layers)
        else:
            low, high = args.layer_low, args.layer_high
        layers = list(range(max(0, low), min(n_layers, high + 1)))
        print()
        print(f"patching layers {layers} at {args.patch_sites}, gate={args.patch_gate}")
        builder = OrderedPromptBuilder(corpus, negatives)
        patches, gate_counts = run_patching(
            model, tokenizer, device, builder, scored, args.contexts, layers,
            args.patch_sites, args.patch_gate, args.patch_margin_gap,
            args.patch_max_facts, args.paraphrases[0])
        gate_counts.to_csv(paths["patching"] / "name_order_patch_gate.csv", index=False)
        if patches.empty:
            print("  no fact passed the gate in either order; patching produced nothing.")
            print("  That is itself reportable: it means the swapped prompts do not")
            print("  reproduce the Latin-succeeds, native-fails pattern the design needs.")
        else:
            patches.to_csv(paths["patching"] / "name_order_patching.csv", index=False)
            summary = (patches.groupby(["context", "order", "truth", "site", "layer"],
                                       as_index=False)
                       .agg(mean_delta=("delta_margin", "mean"),
                            recovery_rate=("patched_correct", "mean"),
                            n_facts=("fact_id", "nunique")))
            summary.to_csv(paths["patching"] / "name_order_patching_summary.csv", index=False)
            print()
            print("Patching effect by order and site, true pairs, best layer per cell:")
            true_pairs = summary[summary.truth == 1]
            best = (true_pairs.sort_values("mean_delta", ascending=False)
                    .groupby(["context", "order", "site"], as_index=False).first())
            print(best.round(4).to_string(index=False))
            print()
            print("If the effect stays at E2 in both orders, it is positional. If it")
            print("moves to E1 when the alias moves there, it belongs to the name.")
            print("The site_holds_role column in the row-level file says which name")
            print("occupied each patched position.")

    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()

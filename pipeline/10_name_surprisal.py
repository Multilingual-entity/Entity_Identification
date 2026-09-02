"""Is the Devanagari deficit explained by how unfamiliar the name string itself is?

The paper argues the failure is retrieval, not storage. The strongest surviving rival
account is simpler: the Devanagari string is rarer in pretraining, so the model is worse
at every stage of handling it, and nothing about entity identity is involved. Tokenizer
fertility was tested and did not explain the gap, but fertility measures how a string is
chopped, not how expected it is. A name can be chopped identically in both scripts and
still be far less predictable in one of them.

Surprisal measures that directly. For each name in each script, score the name's own
tokens under the language model given a neutral carrier, and record the total negative
log likelihood, the value per token, and the value per character. Then ask three things.

  Does the surprisal gap track the behavioural gap? Correlate the per-item Devanagari
  minus Latin surprisal difference against the per-item Latin minus Devanagari margin
  difference. A strong positive correlation would say the deficit is familiarity.

  Does it survive fertility? Fertility and surprisal are correlated by construction, so
  the partial correlation controlling fertility is the number that matters. If surprisal
  adds nothing beyond fertility, this is the fertility result again rather than a new one.

  Does it separate right from wrong? Compare mean surprisal on the items the model gets
  right in Devanagari against the items it gets wrong. If the wrong items are no more
  surprising, familiarity is not what decides individual cases.

A null result here is worth as much as a positive one: it closes the rival account the
paper currently answers only with fertility.

Note on tokenization. The carrier and the name are tokenized separately and their ids
concatenated, so the boundary token may differ from what the tokenizer would produce for
the joined string. The same procedure is applied to both scripts, so the comparison
between them is unaffected; the absolute values are approximate.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline_common import (
    add_common_args,
    check_contexts,
    ensure_run_dirs,
    load_model_and_tokenizer,
    load_prepared_data,
    seed_everything,
)

# A neutral frame, so the name is scored in a sentence rather than from a bare start of
# sequence. Only reviewed languages get a hand-written carrier; anything else scores the
# name unconditionally, which is still comparable across scripts within that language.
CARRIERS = {
    "en": "The person's name is ",
    "mr": "व्यक्तीचे नाव आहे ",
}


def rankdata(values) -> np.ndarray:
    """Average ranks, ties shared. Avoids a scipy dependency on the cluster image."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b, n_perm: int = 5000, seed: int = 0) -> tuple[float, float, int]:
    """Rank correlation with a permutation p-value. Returns (rho, p, n)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan"), float("nan"), int(len(a))
    ra, rb = rankdata(a), rankdata(b)
    rho = pearson(ra, rb)
    rng = np.random.default_rng(seed)
    null = np.array([pearson(ra, rng.permutation(rb)) for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(rho)).mean())
    return float(rho), p, int(len(a))


def partial_spearman(x, y, z, seed: int = 0) -> float:
    """Rank correlation of x and y with z held constant.

    Surprisal and fertility are correlated by construction: a string cut into more pieces
    tends to accumulate more negative log likelihood. Holding fertility constant asks
    whether surprisal carries anything fertility did not already say.
    """
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 4:
        return float("nan")
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    rxy, rxz, ryz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    denom = np.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


def name_surprisal(model, tokenizer, device, carrier: str, name: str) -> dict:
    """Total and per-unit negative log likelihood of `name`'s tokens after `carrier`."""
    import torch

    name = str(name)
    prefix_ids = tokenizer(carrier, add_special_tokens=True).input_ids if carrier else \
        tokenizer("", add_special_tokens=True).input_ids
    # A language with no hand-written carrier scores the name unconditionally, and for a
    # tokenizer that emits no BOS -- Qwen2 is one -- that leaves the prefix empty. The
    # first token would then have nothing to be predicted from, and the logit slice comes
    # out empty. Fall back to a single sentinel so every token is still scored, which is
    # the usual convention for unconditional scoring.
    if not prefix_ids:
        sentinel = getattr(tokenizer, "bos_token_id", None)
        if sentinel is None:
            sentinel = getattr(tokenizer, "eos_token_id", None)
        if sentinel is None:
            raise SystemExit(
                "the tokenizer emits neither a BOS nor an EOS token, so a name cannot be "
                "scored without a carrier phrase; add one for this language in CARRIERS."
            )
        prefix_ids = [int(sentinel)]
    name_ids = tokenizer(name, add_special_tokens=False).input_ids
    if not name_ids:
        return {"n_tokens": 0, "n_chars": len(name), "nll_total": float("nan"),
                "nll_per_token": float("nan"), "nll_per_char": float("nan"),
                "fertility": float("nan")}
    ids = torch.tensor([prefix_ids + name_ids], device=device)
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits[0].float()
    # logits[i] predicts token i+1, so the name's first token is predicted at index len-1.
    start = len(prefix_ids) - 1
    logprobs = torch.log_softmax(logits[start : start + len(name_ids)], dim=-1)
    targets = torch.tensor(name_ids, device=logprobs.device)
    nll = -logprobs.gather(1, targets[:, None]).squeeze(1)
    total = float(nll.sum().item())
    n_chars = max(1, len(name))
    del ids, logits, logprobs, nll
    return {
        "n_tokens": int(len(name_ids)),
        "n_chars": int(len(name)),
        "nll_total": total,
        "nll_per_token": total / len(name_ids),
        "nll_per_char": total / n_chars,
        "fertility": len(name_ids) / n_chars,
    }


def behavioural_gap(behavior: pd.DataFrame) -> pd.DataFrame:
    """Per fact and prompt language: how much better Latin is than native, in margin
    points and in accuracy, on the true pairs."""
    b = behavior[(behavior.truth == 1) & behavior.condition.isin(["DEVDEV", "LATLAT"])].copy()
    b["fact_id"] = b.fact_id.astype(str)
    agg = (b.groupby(["fact_id", "context", "condition"], as_index=False)
             .agg(margin=("yes_minus_no_margin", "mean"), correct=("correct", "mean")))
    # Built column by column rather than by renaming a pivot, because a run missing one
    # condition would otherwise silently shift the labels onto the wrong columns.
    out = None
    for measure in ("margin", "correct"):
        piv = agg.pivot_table(index=["fact_id", "context"], columns="condition",
                              values=measure)
        for condition in ("DEVDEV", "LATLAT"):
            if condition not in piv.columns:
                piv[condition] = np.nan
        piv = piv[["DEVDEV", "LATLAT"]].rename(
            columns={"DEVDEV": f"{measure}_dev", "LATLAT": f"{measure}_lat"})
        out = piv if out is None else out.join(piv)
    out = out.reset_index()
    out["margin_gap"] = out.margin_lat - out.margin_dev
    out["accuracy_gap"] = out.correct_lat - out.correct_dev
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"],
                        help="prompt languages, used only to pick the carrier sentence "
                             "and to join against the behavioural gap")
    parser.add_argument("--no-carrier", action="store_true",
                        help="score each name from the start of the sequence instead of "
                             "inside a neutral sentence")
    parser.add_argument("--max-facts", type=int, default=0,
                        help="0 means the whole corpus")
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)

    out_path = paths["tables"] / "name_surprisal.csv"
    if out_path.exists() and not args.force:
        print(f"Already complete: {out_path}. Use --force to rerun.")
        return

    corpus, _ = load_prepared_data(args.run_dir)
    if args.max_facts:
        corpus = corpus.head(args.max_facts)
    behavior_path = paths["tables"] / "behavior.csv"
    behavior = pd.read_csv(behavior_path) if behavior_path.exists() else None
    if behavior is None:
        print("no behaviour table; surprisal will be recorded but not correlated")

    model, tokenizer, device, _, _ = load_model_and_tokenizer(args.model, args.batch_size)

    rows = []
    for n, row in enumerate(corpus.itertuples()):
        for context in args.contexts:
            carrier = "" if args.no_carrier else CARRIERS.get(context, "")
            for role in ("a", "b"):
                for script in ("dev", "lat"):
                    name = str(getattr(row, f"name_{role}_{script}"))
                    if not name.strip():
                        continue
                    stats = name_surprisal(model, tokenizer, device, carrier, name)
                    stats.update(fact_id=str(row.fact_id), context=context, role=role,
                                 script=script, name=name, split=getattr(row, "split", ""),
                                 carrier=bool(carrier))
                    rows.append(stats)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(corpus)} facts", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)

    # Per fact and language: the gap summed over the two names, which is what the prompt
    # actually contains.
    per_name = frame.pivot_table(index=["fact_id", "context", "role"], columns="script",
                                 values=["nll_per_char", "nll_total", "fertility",
                                         "nll_per_token"]).reset_index()
    per_name.columns = ["_".join(c).rstrip("_") for c in per_name.columns]
    for base in ("nll_per_char", "nll_total", "fertility", "nll_per_token"):
        if f"{base}_dev" in per_name and f"{base}_lat" in per_name:
            per_name[f"{base}_gap"] = per_name[f"{base}_dev"] - per_name[f"{base}_lat"]
    gap_cols = [c for c in per_name.columns if c.endswith("_gap")]
    if not gap_cols:
        raise SystemExit("no name was scored in both scripts; check the corpus fields")
    per_fact = (per_name.groupby(["fact_id", "context"], as_index=False)
                .agg({c: "sum" for c in gap_cols}))
    per_fact.to_csv(paths["tables"] / "name_surprisal_per_fact.csv", index=False)

    summary = (frame.groupby(["context", "script"], as_index=False)
               .agg(nll_per_token=("nll_per_token", "mean"),
                    nll_per_char=("nll_per_char", "mean"),
                    fertility=("fertility", "mean"),
                    n_tokens=("n_tokens", "mean"),
                    n=("name", "size")))
    summary.to_csv(paths["tables"] / "name_surprisal_summary.csv", index=False)
    print()
    print("Surprisal by script (higher means less expected to the model):")
    print(summary.round(4).to_string(index=False))

    if behavior is None:
        print(f"\nwritten: {out_path}")
        return

    gaps = behavioural_gap(behavior)
    joined = per_fact.merge(gaps, on=["fact_id", "context"], how="inner")
    joined.to_csv(paths["tables"] / "name_surprisal_vs_behaviour.csv", index=False)

    print()
    print("Does the surprisal gap predict the behavioural gap?")
    print("  positive rho = names the model finds less expected in Devanagari are also")
    print("  the names it handles worse in Devanagari")
    corr_rows = []
    for context, sub in joined.groupby("context"):
        for measure in ("nll_per_char_gap", "nll_total_gap", "nll_per_token_gap"):
            if measure not in sub:
                continue
            rho, p, n = spearman(sub[measure], sub["margin_gap"], seed=args.seed)
            partial = partial_spearman(sub[measure], sub["margin_gap"],
                                       sub.get("fertility_gap", pd.Series(np.nan, sub.index)))
            corr_rows.append({"context": context, "measure": measure, "outcome": "margin_gap",
                              "spearman_rho": rho, "p_permutation": p,
                              "partial_rho_given_fertility": partial, "n_facts": n})
        if "fertility_gap" in sub:
            rho, p, n = spearman(sub["fertility_gap"], sub["margin_gap"], seed=args.seed)
            corr_rows.append({"context": context, "measure": "fertility_gap",
                              "outcome": "margin_gap", "spearman_rho": rho,
                              "p_permutation": p,
                              "partial_rho_given_fertility": float("nan"), "n_facts": n})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(paths["tables"] / "name_surprisal_correlations.csv", index=False)
    print(corr.round(4).to_string(index=False))
    print()
    print("  partial_rho_given_fertility is the number that matters. If it collapses")
    print("  towards zero, surprisal is restating the fertility result rather than")
    print("  adding a second, independent familiarity effect.")

    # Right against wrong, on the items the model was actually asked about.
    dev = behavior[(behavior.truth == 1) & (behavior.condition == "DEVDEV")].copy()
    dev["fact_id"] = dev.fact_id.astype(str)
    outcome = (dev.groupby(["fact_id", "context"], as_index=False)
               .agg(dev_accuracy=("correct", "mean")))
    outcome["outcome"] = np.where(outcome.dev_accuracy >= 0.5, "mostly right", "mostly wrong")
    dev_names = frame[frame.script == "dev"].groupby(["fact_id", "context"], as_index=False).agg(
        nll_per_char=("nll_per_char", "mean"), fertility=("fertility", "mean"))
    split = dev_names.merge(outcome, on=["fact_id", "context"], how="inner")
    if split.outcome.nunique() > 1:
        table = (split.groupby(["context", "outcome"], as_index=False)
                 .agg(nll_per_char=("nll_per_char", "mean"),
                      fertility=("fertility", "mean"), n_facts=("fact_id", "nunique")))
        table.to_csv(paths["tables"] / "name_surprisal_by_outcome.csv", index=False)
        print("\nDevanagari surprisal, items answered correctly against incorrectly:")
        print(table.round(4).to_string(index=False))
        print("  Similar values on both rows mean familiarity does not decide individual")
        print("  cases, which is what the retrieval account predicts.")

    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()

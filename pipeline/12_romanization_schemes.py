"""Automatic romanization as a deployable fix, done properly and scored on held-out data.

The paper reports one automatic-romanization anchor, auto_both, produced by ITRANS. Three
things are wrong with that result as it stands.

  The romanizer leaks. ITRANS returns Devanagari characters for names containing certain
  vowel signs, so about a fifth of the corpus received a mixed-script hybrid rather than a
  romanization. Those items cannot test what the anchor claims to test.

  One scheme is not the method. ITRANS, IAST, and Harvard-Kyoto disagree about how to
  write the same sound, and a reader-facing deployment would strip diacritics anyway. If
  the anchor works under one scheme and not another, the finding is about that scheme.

  Nothing was held out. The anchor was reported over the whole corpus, so there is no
  separation between choosing a scheme and reporting its effect.

This script fixes all three. It scores every scheme on the same prompts, measures how
faithfully each reproduces the corpus's own Latin form, selects a single winner on the
validation split alone, and reports that winner's effect on the test split it never saw.
The human Latin field is included as a ceiling: it is what a perfect romanizer would
produce, so the gap between it and the best automatic scheme is the cost of automating.

Every table is also reported restricted to the items the scheme romanizes cleanly, since
a scheme that fails on a fifth of its input is being scored partly on its own failures.
"""
from __future__ import annotations

import argparse
import unicodedata
from itertools import product

import numpy as np
import pandas as pd

from pipeline_common import (
    QUESTION_TEMPLATES,
    PromptBuilder,
    add_common_args,
    check_contexts,
    ensure_run_dirs,
    load_model_and_tokenizer,
    load_prepared_data,
    load_tokenizer_only,
    score_prompt_frame,
    seed_everything,
)

# gold_lat is the corpus's own Latin field: the ceiling, not a scheme.
SCHEMES = ["gold_lat", "itrans", "iast", "iast_ascii", "hk", "translit_ascii"]

_INDIC_TARGETS = {"itrans": "ITRANS", "iast": "IAST", "iast_ascii": "IAST", "hk": "HK"}


def strip_diacritics(text: str) -> str:
    """Decompose, drop combining marks, recompose. Turns IAST into plain ASCII.

    This is what a search box or a subtitle track would do with a diacritic-bearing form,
    so it is the realistic deployment target rather than a degraded variant.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    kept = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", kept)


def make_romanizer(scheme: str):
    """Return (function, backend_name). The function maps a native string to Latin.

    Falls back rather than failing: a missing optional dependency makes the affected
    scheme return its input unchanged, which the cleanliness table then reports as a
    zero clean rate rather than a crash halfway through a GPU run.
    """
    if scheme == "gold_lat":
        return None, "corpus"

    if scheme in _INDIC_TARGETS:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
        except ImportError:
            return (lambda t: str(t)), "unavailable"
        target = getattr(sanscript, _INDIC_TARGETS[scheme])

        def convert(text: str) -> str:
            try:
                out = transliterate(str(text), sanscript.DEVANAGARI, target)
            except Exception:                                            # noqa: BLE001
                return str(text)
            return strip_diacritics(out) if scheme == "iast_ascii" else out

        return convert, "indic_transliteration"

    if scheme == "translit_ascii":
        # Script-agnostic, so this is the only scheme that survives the move to Tamil,
        # Cyrillic, or Han in the multilingual corpora.
        try:
            from anyascii import anyascii
            return (lambda t: anyascii(str(t))), "anyascii"
        except ImportError:
            pass
        try:
            from unidecode import unidecode
            return (lambda t: unidecode(str(t))), "unidecode"
        except ImportError:
            return (lambda t: strip_diacritics(t)), "nfkd_only"

    raise ValueError(f"unknown scheme: {scheme}")


def is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in str(text))


def edit_distance(a: str, b: str) -> int:
    a, b = str(a), str(b)
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def fold(text: str) -> str:
    """Case and diacritic insensitive form, for comparing a romanization to the gold."""
    return strip_diacritics(str(text)).lower().replace(" ", "")


def scheme_quality(corpus: pd.DataFrame, romanizers: dict, tokenizer=None) -> pd.DataFrame:
    """How faithful each scheme is, computed without touching the GPU.

    Reported before the behavioural run so a scheme that produces nonsense can be seen to
    produce nonsense, rather than being diagnosed from an accuracy number afterwards.

    When a tokenizer is supplied, each romanization is also counted in tokens. This is the
    mechanism the anchor is supposed to work by: the native name is chopped into many
    pieces and the Latin form into few, so appending a romanization is meant to hand the
    model a cheaper handle on the same entity. A scheme whose output is chopped as badly
    as the native form is not providing that handle, whatever its accuracy comes out at,
    and a scheme that is faithful but fragmented tells a different story from one that is
    compact but wrong.
    """
    def count(text: str) -> int:
        if tokenizer is None:
            return -1
        return len(tokenizer.encode(str(text), add_special_tokens=False))

    rows = []
    for scheme, (fn, backend) in romanizers.items():
        for row in corpus.itertuples():
            for role in ("a", "b"):
                native = str(getattr(row, f"name_{role}_dev"))
                gold = str(getattr(row, f"name_{role}_lat"))
                produced = gold if fn is None else fn(native)
                distance = edit_distance(fold(produced), fold(gold))
                n_native, n_gold, n_produced = count(native), count(gold), count(produced)
                rows.append({
                    "scheme": scheme,
                    "backend": backend,
                    "fact_id": str(row.fact_id),
                    "role": role,
                    "native": native,
                    "gold": gold,
                    "produced": produced,
                    "clean_ascii": int(is_ascii(produced)),
                    "exact_match_folded": int(fold(produced) == fold(gold)),
                    "edit_distance": distance,
                    "normalised_edit_distance": distance / max(1, len(fold(gold))),
                    "n_tokens_native": n_native,
                    "n_tokens_gold": n_gold,
                    "n_tokens_produced": n_produced,
                    # Below 1.0 means the romanization is cheaper to represent than the
                    # native name, which is the whole premise of the anchor.
                    "token_ratio_to_native": (n_produced / n_native) if n_native > 0 else float("nan"),
                    "fertility_produced": (n_produced / max(1, len(produced))) if n_produced >= 0
                    else float("nan"),
                    # The anchored prompt contains both forms, so this is what the model
                    # actually pays for the name.
                    "n_tokens_anchored": (n_native + n_produced + 2) if n_native > 0 else -1,
                })
    return pd.DataFrame(rows)


def clean_fact_ids(quality: pd.DataFrame, scheme: str) -> set:
    """Facts where both names romanize to pure ASCII under this scheme."""
    sub = quality[quality.scheme == scheme]
    per_fact = sub.groupby("fact_id").clean_ascii.min()
    return set(per_fact[per_fact == 1].index)


def build_frame(corpus, negatives, contexts, romanizers, paraphrases, yes_letters,
                arm="anchored"):
    """One DEVDEV prompt set per scheme.

    Two arms, because they test different claims and the paper needs both.

    ``anchored`` writes the romanization in brackets after the native name, which is the
    deployable fix: the reader keeps the Devanagari and the model gets a Latin handle.

    ``replaced`` substitutes the romanization for the native name entirely. That is the
    fragmentation test. If the deficit were caused by Devanagari being segmented into
    many pieces, then removing the Devanagari while keeping the name should recover the
    Latin-script performance. The earlier exploratory run found that it does not, and
    that result is what the fragmentation argument rests on, so it has to be measurable
    here rather than cited from a corpus this paper no longer uses.
    """
    builder = PromptBuilder(corpus, negatives)
    split_of = dict(zip(corpus["fact_id"].astype(str), corpus["split"].astype(str)))
    records = []
    for fid in corpus["fact_id"].astype(str):
        for scheme, (fn, _backend) in romanizers.items():
            for context, truth, paraphrase_id, yes_letter in product(
                contexts, (1, 0), paraphrases, yes_letters
            ):
                base = builder.make(fid, context, "DEVDEV", truth, paraphrase_id, yes_letter)
                # The Latin forms belong to whichever rows supplied the two names, and for
                # a false pair the second name comes from the negative row rather than from
                # this fact. names() already resolved that, so recover it the same way.
                latin_a, latin_b = builder.names(fid, "LATLAT", truth)
                if fn is None:
                    roman_a, roman_b = latin_a, latin_b
                else:
                    roman_a, roman_b = fn(base["a_text"]), fn(base["b_text"])
                if arm == "replaced":
                    a_text, b_text = roman_a, roman_b
                else:
                    a_text = f"{base['a_text']} ({roman_a})"
                    b_text = f"{base['b_text']} ({roman_b})"
                question = QUESTION_TEMPLATES[context][paraphrase_id].format(a=a_text, b=b_text)
                instruction, no_letter = PromptBuilder.answer_instruction(yes_letter)
                records.append({
                    "fact_id": fid,
                    "context": context,
                    "condition": "DEVDEV",
                    "truth": int(truth),
                    "paraphrase_id": int(paraphrase_id),
                    "yes_letter": yes_letter,
                    "no_letter": no_letter,
                    "scheme": scheme,
                    "anchor": f"roman_{scheme}",
                    "arm": arm,
                    "correct_semantic": "yes" if int(truth) == 1 else "no",
                    "a_text": a_text,
                    "b_text": b_text,
                    "roman_a": roman_a,
                    "roman_b": roman_b,
                    "split": split_of[fid],
                    "question": question,
                    "prompt": question + "\n\n" + instruction,
                })
    frame = pd.DataFrame(records)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def summarise(scored, clean_by_scheme, label):
    rows = []
    for (context, split, scheme), sub in scored.groupby(["context", "split", "scheme"]):
        clean = clean_by_scheme.get(scheme, set())
        restricted = sub[sub.fact_id.astype(str).isin(clean)]
        rows.append({
            "context": context, "split": split, "scheme": scheme, "arm": label,
            "accuracy": float(sub.correct.mean()),
            "accuracy_clean_only": float(restricted.correct.mean()) if len(restricted)
            else float("nan"),
            "mean_margin": float(sub.yes_minus_no_margin.mean()),
            "say_yes_rate": float((sub.pred_semantic == "yes").mean()),
            "n_facts": int(sub.fact_id.nunique()),
            "n_facts_clean": int(restricted.fact_id.nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--contexts", nargs="+", default=["mr", "en"])
    parser.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES)
    parser.add_argument("--paraphrases", type=int, nargs="+", default=[0])
    parser.add_argument("--yes-letters", nargs="+", default=["A", "B"])
    parser.add_argument("--arms", nargs="+", default=["anchored", "replaced"],
                        choices=["anchored", "replaced"],
                        help="anchored writes the romanization beside the native name, "
                             "which is the deployable fix; replaced substitutes it for "
                             "the native name, which is the fragmentation test. Running "
                             "both doubles the prompt count")
    parser.add_argument("--quality-only", action="store_true",
                        help="write the faithfulness tables and stop, without loading a "
                             "model. Runs anywhere and is worth doing before booking a GPU")
    args = parser.parse_args()
    check_contexts(args.contexts)
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)

    out_path = paths["recovery"] / "romanization_schemes.csv"
    partial_path = paths["recovery"] / "romanization_schemes.partial.csv"
    if out_path.exists() and not args.force and not args.quality_only:
        print(f"Already complete: {out_path}. Use --force to rerun.")
        return
    if args.force and partial_path.exists():
        partial_path.unlink()

    corpus, negatives = load_prepared_data(args.run_dir)
    romanizers = {scheme: make_romanizer(scheme) for scheme in args.schemes}
    for scheme, (_fn, backend) in romanizers.items():
        if backend in {"unavailable", "nfkd_only"}:
            print(f"WARNING: {scheme} has no working backend ({backend}); its rows will "
                  f"show a low clean rate and should not be read as a result about the "
                  f"scheme")

    # The tokenizer alone, so token counts are available even in --quality-only mode.
    try:
        tokenizer_only = load_tokenizer_only(args.model)
    except Exception as exc:                                             # noqa: BLE001
        print(f"tokenizer unavailable ({exc}); token counts will be recorded as -1")
        tokenizer_only = None

    quality = scheme_quality(corpus, romanizers, tokenizer_only)
    quality.to_csv(paths["recovery"] / "romanization_quality.csv", index=False)
    quality_summary = (quality.groupby(["scheme", "backend"], as_index=False)
                       .agg(clean_ascii_rate=("clean_ascii", "mean"),
                            exact_match_rate=("exact_match_folded", "mean"),
                            mean_normalised_edit=("normalised_edit_distance", "mean"),
                            tokens_native=("n_tokens_native", "mean"),
                            tokens_produced=("n_tokens_produced", "mean"),
                            token_ratio_to_native=("token_ratio_to_native", "mean"),
                            tokens_anchored=("n_tokens_anchored", "mean"),
                            n_names=("native", "size")))
    quality_summary.to_csv(paths["recovery"] / "romanization_quality_summary.csv", index=False)
    print()
    print("Faithfulness and token cost of each scheme, against the corpus's own Latin form:")
    print(quality_summary.round(4).to_string(index=False))
    print("  clean_ascii_rate below 1.0 means the scheme returned native characters and")
    print("  those items received a mixed-script hybrid rather than a romanization.")
    print("  token_ratio_to_native below 1.0 means the romanization is cheaper to")
    print("  represent than the native name, which is the premise the anchor rests on.")

    clean_by_scheme = {scheme: clean_fact_ids(quality, scheme) for scheme in args.schemes}
    if args.quality_only:
        print("\n--quality-only: stopping before the model run.")
        return

    frames = [build_frame(corpus, negatives, args.contexts, romanizers,
                          args.paraphrases, args.yes_letters, arm=arm)
              for arm in args.arms]
    frame = pd.concat(frames, ignore_index=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    print(f"\n{len(frame)} prompts across {len(args.schemes)} schemes "
          f"and {len(args.arms)} arms ({', '.join(args.arms)})")

    model, tokenizer, device, batch_size, _ = load_model_and_tokenizer(
        args.model, args.batch_size)
    scored = score_prompt_frame(model, tokenizer, device, frame, batch_size,
                                checkpoint_path=partial_path)
    scored.to_csv(out_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    # Summarised per arm and concatenated, so the anchored and replaced results sit in
    # one table and can be read against each other and against the Dev--Dev baseline.
    summary = pd.concat(
        [summarise(scored[scored.arm == arm], clean_by_scheme, arm) for arm in args.arms],
        ignore_index=True)
    summary.to_csv(paths["recovery"] / "romanization_scheme_summary.csv", index=False)
    print()
    print("Accuracy by scheme and split:")
    print(summary.round(4).to_string(index=False))

    # Selection on validation only, reporting on test only.
    picks = []
    # Scheme selection reads the anchored arm only, so the deployable-fix result and the
    # fragmentation test do not select for each other.
    anchored = summary[summary.arm == "anchored"]
    for context, sub in anchored[anchored.split == "validation"].groupby("context"):
        automatic = sub[sub.scheme != "gold_lat"]
        if automatic.empty:
            continue
        best = automatic.sort_values("accuracy", ascending=False).iloc[0]
        test = anchored[(anchored.context == context) & (anchored.split == "test")]
        chosen = test[test.scheme == best.scheme]
        ceiling = test[test.scheme == "gold_lat"]
        picks.append({
            "context": context,
            "selected_scheme": best.scheme,
            "validation_accuracy": float(best.accuracy),
            "test_accuracy": float(chosen.accuracy.iloc[0]) if len(chosen) else float("nan"),
            "test_accuracy_clean_only":
                float(chosen.accuracy_clean_only.iloc[0]) if len(chosen) else float("nan"),
            "test_accuracy_human_latin_ceiling":
                float(ceiling.accuracy.iloc[0]) if len(ceiling) else float("nan"),
        })
    selection = pd.DataFrame(picks)
    selection.to_csv(paths["recovery"] / "romanization_selected.csv", index=False)

    # Against the unanchored Devanagari baseline and the all-Latin ceiling.
    behavior_path = paths["tables"] / "behavior.csv"
    if behavior_path.exists():
        behavior = pd.read_csv(behavior_path)
        base = behavior[behavior.context.isin(args.contexts)
                        & behavior.condition.isin(["DEVDEV", "LATLAT"])
                        & behavior.paraphrase_id.isin(args.paraphrases)
                        & behavior.yes_letter.isin(args.yes_letters)]
        reference = (base.groupby(["context", "split", "condition"], as_index=False)
                     .agg(accuracy=("correct", "mean"),
                          mean_margin=("yes_minus_no_margin", "mean"),
                          n_facts=("fact_id", "nunique")))
        reference.to_csv(paths["recovery"] / "romanization_reference.csv", index=False)
        print()
        print("Unanchored reference points:")
        print(reference[reference.split == "test"].round(4).to_string(index=False))

    print()
    print("Scheme chosen on validation, effect reported on test:")
    print(selection.round(4).to_string(index=False) if len(selection) else "  none")
    print()
    print("The gap between test_accuracy and test_accuracy_human_latin_ceiling is the")
    print("cost of automating the romanization. The gap between test_accuracy and the")
    print("clean-only column is how much the scheme's own failures cost it.")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()

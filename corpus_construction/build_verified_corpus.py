r"""Turn a completed review sheet into a corpus the pipeline can run.

Three filters, applied in order and reported separately, because they remove items for
different reasons and a later reader needs to be able to tell them apart.

  1. The reviewer's own verdict. Items marked keep=no are gone; that judgement is not
     second-guessed here.

  2. Named failures. Items found defective on a second reading -- a Devanagari field
     machine-mapped from the Latin, a parenthetical disambiguator carried into the name,
     the two fields crossed. These are listed explicitly rather than detected, so the
     reason for each is on the record.

  3. Items answerable without knowing the entity. If one name contains the other, or the
     two native names differ only in spelling, a model can answer by string overlap alone.
     These are not errors and the reviewer was right not to reject them, but they measure
     something other than entity knowledge, and a fifth of the corpus behaving that way
     dilutes the effect toward zero.

Hard negatives are rebuilt over the survivors rather than filtered. A negative pointing at
a removed item would either break the map or silently drop a third item, and re-pairing
keeps every survivor matched to the best available distractor.

    python build_verified_corpus.py --review ../../review_hi_verified_286.csv \
        --corpus out/corpora/hi/corpus_selected.csv --out out/verified/hi
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import sys
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# Found defective on a second reading. Each is here with its reason so the decision is
# auditable rather than a bare identifier.
NAMED_FAILURES = {
    "hi_q470256": "Devanagari is a character mapping of the Latin, not a Hindi rendering",
    "hi_q553955": "Johan Bojer is not Johan Kristopher Hansen; Latin also lowercase",
    "hi_q278015": "कार्तिक and Karthika are different names in Devanagari",
    "hi_q126188": "(इंग्लैंड) is a disambiguator, not part of the name",
    "hi_q158533": "the Devanagari of one name matches the Latin of the other",
    "hi_q115547": "Bill Gold is the graphic designer, a different person",
}


def fold_latin(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c.lower() for c in stripped if c.isalnum())


def fold_native(text: str) -> str:
    """Combining marks kept: Indic vowel signs are combining characters."""
    return "".join(c.lower() for c in unicodedata.normalize("NFC", str(text)) if c.isalnum())


def load_hard_negatives():
    """Reuse the assembler's own pairing so the negatives are scored the same way."""
    spec = importlib.util.spec_from_file_location("assemble", HERE / "02_assemble.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["assemble"] = module
    spec.loader.exec_module(module)
    return module.hard_negatives


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-overlap", action="store_true",
                    help="retain items where one name contains the other. They are valid "
                         "pairs; they are removed by default because a model can answer "
                         "them by string overlap without knowing the entity")
    ap.add_argument("--native-threshold", type=float, default=0.80)
    ap.add_argument("--exclude-file", type=Path, default=None,
                    help="one fact_id per line to drop, for a later review pass")
    ap.add_argument("--allow-stale", action="store_true",
                    help="carry a verdict forward even when the item's names have changed "
                         "since the sheet was written. Off by default: a verdict recorded "
                         "against different names is not a verdict about this item")
    args = ap.parse_args()

    review = pd.read_csv(args.review)
    review["fact_id"] = review.fact_id.astype(str)
    review["keep"] = review.keep.astype(str).str.strip().str.lower()
    corpus = pd.read_csv(args.corpus)
    corpus["fact_id"] = corpus.fact_id.astype(str)

    print(f"review : {len(review)} rows")
    print(f"corpus : {len(corpus)} items\n")

    kept = set(review.loc[review.keep == "yes", "fact_id"])
    print(f"1. reviewer kept                        {len(kept)}")

    # A verdict belongs to the names it was given against. The corpus is rebuilt whenever
    # the alias selection changes, and matching on fact_id alone would silently carry an
    # approval onto a different pair: one Hindi item kept as "Susan Wojcicki" became
    # "Ben aissa hedi", a different person entirely, and stayed marked keep.
    fields = ["name_a_dev", "name_a_lat", "name_b_dev", "name_b_lat"]
    joined = review.merge(corpus, on="fact_id", suffixes=("_reviewed", "_now"))
    changed = {
        r.fact_id for r in joined.itertuples()
        if any(str(getattr(r, f + "_reviewed")) != str(getattr(r, f + "_now"))
               for f in fields)
    } & kept
    if changed and not args.allow_stale:
        kept -= changed
        print(f"   stale verdicts dropped               {len(changed)}  -> {len(kept)}")
        print(f"     these items' names changed after the sheet was written, so the")
        print(f"     recorded verdict is about a different pair. Re-review them.")
    elif changed:
        print(f"   {len(changed)} stale verdicts carried forward (--allow-stale)")

    if args.exclude_file:
        later = {line.strip() for line in
                 args.exclude_file.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")}
        dropped = kept & later
        kept -= dropped
        print(f"   later review pass rejected           {len(dropped)}  -> {len(kept)}")

    named = kept & set(NAMED_FAILURES)
    kept -= named
    print(f"2. named failures removed               {len(named)}  -> {len(kept)}")
    for fid in sorted(named):
        print(f"     {fid}: {NAMED_FAILURES[fid]}")

    working = corpus[corpus.fact_id.isin(kept)].copy()
    contained = [
        (fold_latin(a) in fold_latin(b)) or (fold_latin(b) in fold_latin(a))
        for a, b in zip(working.name_a_lat, working.name_b_lat)
    ]
    near = [
        difflib.SequenceMatcher(None, fold_native(a), fold_native(b)).ratio()
        >= args.native_threshold
        for a, b in zip(working.name_a_dev, working.name_b_dev)
    ]
    trivial = [c or n for c, n in zip(contained, near)]
    print(f"\n3. answerable by string overlap         {sum(trivial)}")
    print(f"     one name contains the other          {sum(contained)}")
    print(f"     native names {args.native_threshold:.0%}+ alike            {sum(near)}")

    if args.keep_overlap:
        final = working
        print("     retained (--keep-overlap)")
    else:
        final = working[[not t for t in trivial]]
        print(f"     removed                              -> {len(final)}")

    args.out.mkdir(parents=True, exist_ok=True)
    final = final.drop(columns=[c for c in ("review_flags",) if c in final.columns])
    final.to_csv(args.out / "corpus_selected.csv", index=False, encoding="utf-8")

    # Rebuilt, not filtered: a negative pointing at a removed item would cascade.
    negatives = load_hard_negatives()(final.reset_index(drop=True))
    negatives.to_csv(args.out / "hard_negative_map.csv", index=False, encoding="utf-8")

    print(f"\nfinal corpus: {len(final)} items")
    print(f"  native  {int((final.origin == 'native').sum())}"
          f"  foreign {int((final.origin == 'foreign').sum())}")
    print(f"  negatives rebuilt over the survivors: {len(negatives)} rows, "
          f"{int((negatives.fact_id == negatives.negative_fact_id).sum())} self-paired")
    print(f"\nwritten: {args.out / 'corpus_selected.csv'}")
    print(f"written: {args.out / 'hard_negative_map.csv'}")


if __name__ == "__main__":
    main()

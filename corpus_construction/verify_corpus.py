"""Check that a corpus asks the question it claims to ask.

Every item is a pair of names in two scripts, and the whole design rests on one property:
the Devanagari field and the Latin field must be the SAME name written two ways. If they
are different names, Dev--Dev and Lat--Lat ask about different aliases and the condition
varies the name as well as the script, measuring neither cleanly.

That property is not enforced anywhere in the fetch. It has to be checked.

    python verify_corpus.py ../archive/299_fact/qwen/data/corpus.csv
    python verify_corpus.py out/corpora/mr/corpus_selected.csv --show 20

Reports six things, each of which breaks the design in a different way:

    name pair mismatch   the two script fields hold different names
    script purity        a Devanagari field containing Latin, or the reverse
    no contrast          the two fields are identical, so the item tests nothing
    self negative        an item paired with itself
    duplicate names      the same name appearing as both members of a pair
    duplicate items      the same entity twice

The similarity threshold is a heuristic on romanized text and cannot be exact. It is
deliberately reported as a distribution rather than a verdict, with the worst cases printed
so a reader can judge them.
"""
from __future__ import annotations

import argparse
import difflib
import unicodedata
from pathlib import Path

import pandas as pd

LATIN_RANGES = ((0x0000, 0x024F), (0x0250, 0x02AF), (0x1E00, 0x1EFF))


def romanize(text: str) -> str:
    try:
        from anyascii import anyascii
        return anyascii(str(text))
    except ImportError:
        pass
    try:
        from unidecode import unidecode
        return unidecode(str(text))
    except ImportError:
        return str(text)


def fold(text: str) -> str:
    """Case, spacing and punctuation removed. For comparing romanized Latin strings."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c.lower() for c in stripped if c.isalnum())


def fold_native(text: str) -> str:
    """The same, but WITHOUT dropping combining marks.

    In Devanagari and most Indic scripts the vowel signs are combining characters, so
    stripping them deletes the vowels and makes unrelated words collapse together. An
    earlier version of this file did exactly that and reported nineteen Marathi items as
    having two identical names when they were only spelling variants.
    """
    normalized = unicodedata.normalize("NFC", str(text))
    return "".join(c.lower() for c in normalized if c.isalnum())


def similarity(native: str, latin: str) -> float:
    return difflib.SequenceMatcher(None, fold(romanize(native)), fold(latin)).ratio()


def has_non_latin_letter(text: str) -> bool:
    for ch in str(text):
        if not unicodedata.category(ch).startswith("L"):
            continue
        if not any(lo <= ord(ch) <= hi for lo, hi in LATIN_RANGES):
            return True
    return False


def has_latin_letter(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in str(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--negatives", type=Path, default=None)
    ap.add_argument("--show", type=int, default=12, help="worst mismatches to print")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="below this the two script fields are treated as different names")
    args = ap.parse_args()

    corpus = pd.read_csv(args.corpus)
    n = len(corpus)
    print(f"{args.corpus}")
    print(f"{n} items\n")

    rows = []
    for r in corpus.itertuples():
        rows.append({
            "fact_id": str(r.fact_id),
            "sim_a": similarity(r.name_a_dev, r.name_a_lat),
            "sim_b": similarity(r.name_b_dev, r.name_b_lat),
            "a_dev": str(r.name_a_dev), "a_lat": str(r.name_a_lat),
            "b_dev": str(r.name_b_dev), "b_lat": str(r.name_b_lat),
        })
    check = pd.DataFrame(rows)

    print("1. do the two script fields hold the same NAME?")
    for role in ("a", "b"):
        s = check[f"sim_{role}"]
        bad = int((s < args.threshold).sum())
        print(f"   name_{role}: mean similarity {s.mean():.2f} | "
              f"below {args.threshold}: {bad} ({bad / n:.0%}) | "
              f"below 0.3: {int((s < 0.3).sum())}")
    worst = check.sort_values("sim_b").head(args.show)
    print(f"\n   worst {len(worst)} on the second name:")
    for w in worst.itertuples():
        print(f"     {w.sim_b:.2f}  {w.b_dev}   ||   {w.b_lat}")

    print("\n2. script purity")
    for field, want_native in (("name_a_dev", True), ("name_b_dev", True),
                               ("name_a_lat", False), ("name_b_lat", False)):
        if field not in corpus.columns:
            continue
        values = corpus[field].astype(str)
        if want_native:
            wrong = int((~values.map(has_non_latin_letter)).sum())
            leak = int(values.map(has_latin_letter).sum())
            print(f"   {field}: {wrong} with no native character, {leak} containing Latin")
        else:
            leak = int(values.map(has_non_latin_letter).sum())
            print(f"   {field}: {leak} containing a non-Latin character")

    print("\n3. is there a contrast at all?")
    for role in ("a", "b"):
        same = int((corpus[f"name_{role}_dev"].astype(str).str.strip()
                    == corpus[f"name_{role}_lat"].astype(str).str.strip()).sum())
        print(f"   name_{role}: {same} items where the two fields are identical")

    print("\n4. within-item duplication and near-duplication")
    exact = int((corpus.name_a_dev.astype(str).map(fold_native)
                 == corpus.name_b_dev.astype(str).map(fold_native)).sum())
    near_rows = []
    for r in corpus.itertuples():
        a, b = fold_native(r.name_a_dev), fold_native(r.name_b_dev)
        if a == b:
            continue
        score = difflib.SequenceMatcher(None, a, b).ratio()
        if score >= 0.85:
            near_rows.append((score, str(r.name_a_dev), str(r.name_b_dev),
                              str(r.name_a_lat), str(r.name_b_lat)))
    print(f"   {exact} items whose two native names are identical")
    print(f"   {len(near_rows)} items whose two native names are 85%+ similar")
    if near_rows:
        print("     A spelling variant in the native script paired with a genuinely")
        print("     different name form in Latin makes the two conditions ask questions of")
        print("     different difficulty, which is a confound in the opposite direction to")
        print("     the one the paper documents:")
        for score, ad, bd, al, bl in sorted(near_rows, reverse=True)[:8]:
            print(f"       {score:.2f}  {ad} / {bd}   (Latin: {al} / {bl})")

    print("\n5. duplicate items")
    print(f"   duplicate fact_id: {int(corpus.fact_id.duplicated().sum())}")
    if "qid" in corpus.columns:
        print(f"   duplicate qid    : {int(corpus.qid.duplicated().sum())}")
    print(f"   duplicate name_a : {int(corpus.name_a_dev.astype(str).map(fold).duplicated().sum())}")

    negatives_path = args.negatives or args.corpus.parent / "hard_negative_map.csv"
    if negatives_path.exists():
        print("\n6. negatives")
        neg = pd.read_csv(negatives_path)
        neg["fact_id"] = neg.fact_id.astype(str)
        neg["negative_fact_id"] = neg.negative_fact_id.astype(str)
        ids = set(corpus.fact_id.astype(str))
        print(f"   rows {len(neg)} | self-paired: {int((neg.fact_id == neg.negative_fact_id).sum())}")
        print(f"   pointing outside the corpus: {int((~neg.negative_fact_id.isin(ids)).sum())}")
        print(f"   items with no negative     : {len(ids - set(neg.fact_id))}")
    else:
        print("\n6. negatives: no hard_negative_map.csv beside the corpus")

    bad_b = int((check.sim_b < args.threshold).sum())
    print("\n" + "=" * 70)
    print(f"{bad_b} of {n} items ({bad_b / n:.0%}) have a second name whose two script")
    print("fields do not appear to be the same name. Those items vary the name as well as")
    print("the script, so they cannot measure a script effect cleanly.")
    print("The threshold is a heuristic; read the worst cases above before acting on it.")


if __name__ == "__main__":
    main()

r"""Turn a corpus into a sheet a person can work through.

The automated checks cannot answer the question that matters -- are these two names the
same person, and is each field a name at all -- so the point of this file is to put the
evidence in front of a reader and get out of the way. Machine columns are advisory and
named as such; the verdict columns are left blank.

The flags are worth reading in the order they appear. A mismatched first name breaks the
item outright, since that field is the canonical name both conditions are built from. A
near-identical native pair is subtler: it makes the native condition answerable by string
similarity while the Latin condition still requires knowing the entity, so the two are no
longer the same question asked twice.

    python make_review_sheet.py out/corpora/mr/corpus_selected.csv
    python make_review_sheet.py out/corpora/ta/corpus_selected.csv --out ta_review.csv
"""
from __future__ import annotations

import argparse
import difflib
import unicodedata
from pathlib import Path

import pandas as pd


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


def fold_latin(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c.lower() for c in stripped if c.isalnum())


def fold_native(text: str) -> str:
    """Combining marks kept: in Indic scripts the vowel signs are combining characters,
    and stripping them collapses unrelated words together."""
    return "".join(c.lower() for c in unicodedata.normalize("NFC", str(text)) if c.isalnum())


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.60)
    args = ap.parse_args()

    corpus = pd.read_csv(args.corpus)
    out = args.out or args.corpus.parent / f"review_{args.corpus.parent.name}.csv"

    rows = []
    for r in corpus.itertuples():
        sim_a = ratio(fold_latin(romanize(r.name_a_dev)), fold_latin(r.name_a_lat))
        sim_b = ratio(fold_latin(romanize(r.name_b_dev)), fold_latin(r.name_b_lat))
        native_pair = ratio(fold_native(r.name_a_dev), fold_native(r.name_b_dev))
        lat_a, lat_b = fold_latin(r.name_a_lat), fold_latin(r.name_b_lat)

        notes = []
        if sim_a < args.threshold:
            notes.append("first name may differ between the two scripts")
        if sim_b < args.threshold:
            notes.append("second name may differ between the two scripts")
        if native_pair >= 0.85:
            notes.append(f"the two native names are {native_pair:.0%} alike, so the native "
                         f"condition is near-trivial while the Latin one is not")
        if lat_a and lat_b and (lat_a in lat_b or lat_b in lat_a):
            notes.append("one Latin name contains the other")
        if fold_native(r.name_a_dev) == fold_native(r.name_b_dev):
            notes.append("the two native names are identical")

        rows.append({
            "fact_id": str(r.fact_id),
            "qid": getattr(r, "qid", ""),
            "name_a_dev": r.name_a_dev, "name_a_lat": r.name_a_lat,
            "name_b_dev": r.name_b_dev, "name_b_lat": r.name_b_lat,
            "origin": getattr(r, "origin", ""),
            "sim_first_name": round(sim_a, 2),
            "sim_second_name": round(sim_b, 2),
            "native_pair_similarity": round(native_pair, 2),
            "machine_notes": "; ".join(notes),
            "same_person": "",
            "both_are_real_names": "",
            "two_scripts_same_name_a": "",
            "two_scripts_same_name_b": "",
            "keep": "",
            "reviewer_notes": "",
        })

    sheet = pd.DataFrame(rows).sort_values(
        ["machine_notes", "sim_second_name"], ascending=[False, True])
    # utf-8-sig so Excel renders the native script instead of mojibake.
    sheet.to_csv(out, index=False, encoding="utf-8-sig")

    flagged = int((sheet.machine_notes != "").sum())
    print(f"written: {out}")
    print(f"{len(sheet)} items | {flagged} carry a flag | {len(sheet) - flagged} clean")
    print()
    for label in ("first name may differ", "second name may differ",
                  "near-trivial", "contains the other", "identical"):
        n = int(sheet.machine_notes.str.contains(label, regex=False).sum())
        print(f"  {label:<40}{n:>5}")
    print()
    print("Flagged rows sort to the top. The blank columns are yours:")
    print("  same_person / both_are_real_names / two_scripts_same_name_a")
    print("  two_scripts_same_name_b / keep / reviewer_notes")


if __name__ == "__main__":
    main()

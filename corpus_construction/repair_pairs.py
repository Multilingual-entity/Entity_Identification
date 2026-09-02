# DO NOT RUN THIS AGAINST A VERIFIED CORPUS.
#
# This script repairs mismatched name pairs by choosing the alias pair with the highest
# string similarity. That objective is wrong. Maximising similarity replaced
# "harivansh rai shrivastav / Harivansh Rai Shrivastav" with "bachchan / Bachchan":
# both are names of the same person, but the pair no longer tests what the corpus is
# for, and the damage is invisible to any automatic check because the result looks
# well-formed. The corpus it was run on had to be restored from backup.
#
# The shipped corpora were repaired by hand against Wikidata labels instead. Kept here
# because the approach is a reasonable thing to try and the reason it fails is not
# obvious until you look at what it produces.

r"""Repair items whose two script fields hold different names, instead of dropping them.

Most such items are not broken data. Wikidata lists several aliases per entity in each
language, and the build picks one from each list independently: it chose नवाब राय on the
Hindi side and "Dhanpat Rai" on the English side, which are both Premchand's names but not
the same name. If any Hindi alias matches any English alias, the item is fine and only the
choice was wrong.

So: reconsider every pairing rather than the one that was made. For each entity, score all
Hindi aliases against all English aliases and take the best-matching pair. An item is only
dropped when no pairing works, which is the case where the data really is unusable.

Three smaller repairs are applied first, because they are unambiguous:

    a field holding several aliases    "दामो या दारोमा" -> the one that matches
    a trailing token the other lacks   "रघुपति सहाय फिराक" -> "रघुपति सहाय"
    a field identical to the label     the alias slot was filled with the label again

Nothing is invented. Every replacement is an alias Wikidata already lists for that entity.

    python repair_pairs.py --corpus out/verified/hi/corpus_selected.csv --lang hi
"""
from __future__ import annotations

import argparse
import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

from wikidata import labels

HERE = Path(__file__).resolve().parent


def romanize(text: str) -> str:
    try:
        from anyascii import anyascii
        return anyascii(str(text))
    except ImportError:
        from unidecode import unidecode
        return unidecode(str(text))


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c.lower() for c in stripped if c.isalnum())


def match(native: str, latin: str) -> float:
    return difflib.SequenceMatcher(None, fold(romanize(native)), fold(latin)).ratio()


def split_multi(text: str) -> list:
    """A field holding several aliases, joined by a comma or the Hindi/Marathi 'or'."""
    parts = re.split(r"\s*(?:,|;|\bया\b|\bकिंवा\b|\bअथवा\b)\s*", str(text))
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--lang", required=True, help="Wikidata language code, e.g. hi")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="below this the two fields are treated as different names")
    args = ap.parse_args()

    corpus = pd.read_csv(args.corpus)
    out = args.out or args.corpus
    print(f"{len(corpus)} items | repairing pairs below {args.threshold:.2f}\n")

    broken = [
        i for i, r in corpus.iterrows()
        if match(r.name_b_dev, r.name_b_lat) < args.threshold
        or match(r.name_a_dev, r.name_a_lat) < args.threshold
    ]
    print(f"{len(broken)} items have a field pair that does not match")
    if not broken:
        return

    # Cached from the original fetch, so this costs nothing and works offline.
    qids = [str(corpus.loc[i, "qid"]) for i in broken]
    print("reading the alias lists for those entities")
    entries = labels(qids, [args.lang, "en"])

    repaired, dropped, unchanged = [], [], 0
    for i in broken:
        row = corpus.loc[i]
        entry = entries.get(str(row.qid), {})
        native_options = list(entry.get("aliases", {}).get(args.lang, []))
        english_options = list(entry.get("aliases", {}).get("en", []))
        native_label = entry.get("labels", {}).get(args.lang, row.name_a_dev)
        english_label = entry.get("labels", {}).get("en", row.name_a_lat)

        # The multi-alias and trailing-token cases, folded in as extra candidates.
        native_options += split_multi(row.name_b_dev)
        english_options += split_multi(row.name_b_lat)
        native_options = [n for n in dict.fromkeys(native_options)
                          if fold(n) != fold(native_label)]
        english_options = [e for e in dict.fromkeys(english_options)
                           if fold(e) != fold(english_label)]

        best, score = None, match(row.name_b_dev, row.name_b_lat)
        for native in native_options:
            for english in english_options:
                s = match(native, english)
                if s > score:
                    best, score = (native, english), s

        if best and score >= args.threshold:
            repaired.append((i, row.name_b_dev, row.name_b_lat, best[0], best[1], score))
            corpus.at[i, "name_b_dev"] = best[0]
            corpus.at[i, "name_b_lat"] = best[1]
        elif best:
            unchanged += 1
        else:
            dropped.append((i, row.name_a_lat, row.name_b_dev, row.name_b_lat))

    print(f"\nrepaired {len(repaired)}:")
    for _, od, ol, nd, nl, s in repaired:
        print(f"  {od}  /  {ol}")
        print(f"    -> {nd}  /  {nl}    ({s:.2f})")

    if dropped:
        print(f"\nno usable pairing, drop these {len(dropped)}:")
        for _, a, bd, bl in dropped:
            print(f"  {a}: {bd} / {bl}")
    if unchanged:
        print(f"\n{unchanged} improved but still below the threshold; left as they were")

    keep = [i for i in corpus.index if i not in {d[0] for d in dropped}]
    corpus.loc[keep].to_csv(out, index=False, encoding="utf-8")
    print(f"\n{len(keep)} items written to {out}")


if __name__ == "__main__":
    main()

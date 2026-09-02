"""Resolve every country QID in languages.py and print what it actually is.

The country lists decide which entities count as native to a language, and a native entity
is half the design: for a foreign entity the Latin form is canonical and the native script
should hurt, while for a native entity the native script IS canonical and the prediction
reverses. Testing only foreign entities measures one half of the account and cannot
falsify it.

Those lists are hand-written QIDs, which is exactly the kind of thing that is wrong
silently. A transposed digit produces a valid but different country, the origin split
comes out lopsided, and nothing downstream says why. This resolves each one against
Wikidata and prints its English label and description, so a mistake is visible as a
country nobody meant.

One real error found this way: Swahili was listed with Q1029, which is Mozambique.

    python verify_country_qids.py
    python verify_country_qids.py --languages ru sr ar zh
"""
from __future__ import annotations

import argparse

from languages import LANGUAGES
from wikidata import labels

# Anything whose label does not read like a state is worth a second look. Kept as a hint
# rather than a rule, because historical entities are described inconsistently.
SUSPECT_WORDS = ("city", "province", "region", "language", "surname", "given name",
                 "district", "village", "genus", "film", "album", "terminal",
                 "revolution", "flask", "university")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=None)
    args = ap.parse_args()

    keys = args.languages or list(LANGUAGES)
    wanted = sorted({q for k in keys if k in LANGUAGES
                     for q in LANGUAGES[k].country_qids})
    print(f"resolving {len(wanted)} distinct QIDs across {len(keys)} languages\n")

    resolved = labels(wanted, ["en"])
    english = {}
    for qid in wanted:
        entry = resolved.get(qid, {})
        english[qid] = entry.get("labels", {}).get("en", "")

    missing = [q for q, name in english.items() if not name]
    suspect = [(q, name) for q, name in english.items()
               if name and any(w in name.lower() for w in SUSPECT_WORDS)]

    for key in keys:
        if key not in LANGUAGES:
            print(f"unknown language: {key}")
            continue
        lang = LANGUAGES[key]
        print(f"[{key}] {lang.name}")
        for qid in lang.country_qids:
            name = english.get(qid) or "*** DID NOT RESOLVE ***"
            print(f"    {qid:<12} {name}")
        print()

    if missing:
        print(f"{len(missing)} QIDs did not resolve: {', '.join(missing)}")
        print("  These match no entity, so they can never mark an entity as native.")
    if suspect:
        print(f"{len(suspect)} QIDs resolve to something that may not be a state:")
        for qid, name in suspect:
            print(f"    {qid}  {name}")
    if not missing and not suspect:
        print("Every QID resolves to a plausible state.")
        print()
        print("This checks that the identifiers are real, not that the list is complete.")
        print("Completeness shows up in the run itself: 01_entity_pool.py now prints how")
        print("many entities carry a home country before the English-alias filter, which")
        print("separates a short country list from a design limit.")


if __name__ == "__main__":
    main()

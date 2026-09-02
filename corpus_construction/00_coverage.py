"""How many of the existing 299 entities have an attested name in each candidate script?

This decides the language list. A language whose coverage is too low cannot support a
shared-core design and must either be dropped or run with its own entity set.

Run first. Nothing else should be built until this has been read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from languages import LANGUAGES
from wikidata import labels

HERE = Path(__file__).resolve().parent
# The 299-fact Qwen run was moved into the archive when the corpus was rescaled, so
# the paper's original corpus now lives there rather than at the top level.
DEFAULT_CORPUS = HERE.parent / "archive" / "299_fact" / "qwen" / "data" / "corpus.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "coverage.csv")
    ap.add_argument("--min-usable", type=int, default=150,
                    help="languages below this are flagged as not viable for a shared core")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    qids = corpus["qid"].astype(str).tolist()
    print(f"checking {len(qids)} entities against {len(LANGUAGES)} languages")

    langs = sorted({l.wikidata_lang for l in LANGUAGES.values()} | {"en"})
    data = labels(qids, langs)
    print(f"fetched {len(data)} entities from Wikidata\n")

    rows = []
    per_lang_qids: dict = {}
    for lang in LANGUAGES.values():
        covered, pure, aliased = [], 0, 0
        for qid in qids:
            ent = data.get(qid)
            if not ent:
                continue
            name = ent["labels"].get(lang.wikidata_lang)
            if not name:
                continue
            covered.append(qid)
            if lang.script_purity(name) and (lang.contrast == "diacritic" or lang.in_script(name)):
                pure += 1
            if ent["aliases"].get(lang.wikidata_lang):
                aliased += 1
        per_lang_qids[lang.key] = covered
        rows.append({
            "key": lang.key, "name": lang.name, "script": lang.script,
            "contrast": lang.contrast, "family": lang.family,
            "has_label": len(covered),
            "script_pure": pure,
            # An identity pair needs BOTH names in the native script. The second name
            # comes from an alias in that language, which is far rarer than a label, so
            # this is the number that decides viability, not has_label.
            "usable_pairs": aliased,
            "pct_label": round(100 * len(covered) / len(qids), 1),
            "pct_pairs": round(100 * aliased / len(qids), 1),
            "viable_on_existing_corpus": int(aliased >= args.min_usable),
        })

    out = pd.DataFrame(rows).sort_values("usable_pairs", ascending=False)
    out.to_csv(args.out, index=False)

    print(out.to_string(index=False))
    print()

    viable = out[out.viable_on_existing_corpus == 1].key.tolist()
    thin = out[out.viable_on_existing_corpus == 0].key.tolist()

    print(f"viable on the existing {len(qids)} entities ({len(viable)}): "
          f"{', '.join(viable) if viable else 'none'}")
    if thin:
        print(f"too thin on the existing entities ({len(thin)}): {', '.join(thin)}")
        print("  These are not unusable languages. The existing corpus was selected for")
        print("  Marathi, so it happens to hold few entities with aliases in these")
        print("  languages. Build them their own pool with 01_entity_pool.py instead.")

    if viable:
        core = set(qids)
        for k in viable:
            core &= set(per_lang_qids[k])
        print(f"\nstrict shared core across all viable languages: {len(core)} entities")
        if len(core) < args.min_usable:
            print("  -> too small. Run 00b_core_tiers.py to find where the intersection")
            print("     falls off, then use hub-and-spoke for the rest: every language")
            print("     shares entities with English, not with the other languages.")
        pd.DataFrame({"qid": sorted(core)}).to_csv(
            args.out.parent / "shared_core_qids.csv", index=False)

    print(f"\nwritten: {args.out}")
    print("Note: viability is judged on usable_pairs, not has_label. A label alone does")
    print("not make an identity pair; the second name must also exist in that script.")


if __name__ == "__main__":
    main()

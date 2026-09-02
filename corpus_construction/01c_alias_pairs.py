r"""Fetch alias pairs for review, and build a corpus from the reviewed sheet.

Aliases are the only source that carries the names this corpus needs. The statement route
was tried and returned 29 usable entities from 4945: birth name and native-language name
are recorded in both languages for about one percent of Hindi entities. The 207 verified
rows are all aliases, and only one of them was reachable from a statement. Nicknames,
honorific forms and expanded names -- Bapu, Netaji, Pranab Kumar Mukherjee -- live in the
alias list and nowhere else.

The old build romanized the Hindi alias with anyascii and used that to choose the English
one. That is a transliterator deciding corpus membership, and stage 12 exists to measure
transliterators, so this file removes it. The two selections are made differently:

  name_a   the Wikidata label in each language. One per language, so there is no choice to
           make and nothing to bias. This was never contaminated and does not change.

  name_b   the native side is the alias least like the native label, comparing Devanagari
           to Devanagari, which needs no romanization. The English side is not chosen here
           at all. Every English alias is written to the sheet and the reviewer picks the
           one that is the same name, which is the judgement no metric reproduced above
           75 percent agreement in earlier testing.

So the machine narrows and orders; the person decides. Two steps:

    python 01c_alias_pairs.py fetch --lang hi --target 400 --exclude done.csv
    ... fill in name_b_lat and keep in the sheet ...
    python 01c_alias_pairs.py build --sheet out/corpora/hi/review_alias.csv --lang hi
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from languages import LANGUAGES
from wikidata import labels

HERE = Path(__file__).resolve().parent

PARTICLES = {"do", "da", "de", "del", "della", "di", "du", "dos", "van", "von", "der",
             "den", "bin", "ibn", "al", "el", "la", "le", "ter", "ten", "of", "the", "y"}


def load_pool_module():
    """01_entity_pool.py holds the banded fetch and the country lookup. Its name starts
    with a digit, so it cannot be imported by name."""
    spec = importlib.util.spec_from_file_location("pool", HERE / "01_entity_pool.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pool"] = module
    spec.loader.exec_module(module)
    return module


def fold_native(text: str) -> str:
    """Combining marks kept: Indic vowel signs are combining characters, and dropping them
    collapses unrelated words together."""
    return "".join(c.lower() for c in unicodedata.normalize("NFC", str(text)) if c.isalnum())


def defects(text: str, native: bool, in_script) -> list:
    """Faults visible in one field alone. Nothing here compares the two scripts, so no
    romanization can reach the corpus through it. Advisory: it orders the sheet and
    removes nothing, having been 59 percent precise against a hand review of 418 rows."""
    text = str(text)
    found = []
    if re.search(r"[(\[]", text):
        found.append("parenthetical")
    if "|" in text or "\\" in text:
        found.append("stray separator")
    if "," in text or ";" in text or {"या", "किंवा", "अथवा"} & set(text.split()):
        found.append("several names in one field")
    if native and not in_script(text):
        found.append("not in the native script")
    if native and re.search(r"[A-Za-z]", text):
        found.append("Latin letters in the native field")
    words = [w for w in text.split() if len(w) > 1 and w.lower() not in PARTICLES]
    if not native and len(words) > 1 and any(w[0].isupper() for w in words) \
            and any(w[0].islower() and w[0].isalpha() for w in words):
        found.append("inconsistently capitalised")
    return found


def furthest_alias(label: str, aliases: list) -> str | None:
    """The native alias least like the native label.

    Devanagari against Devanagari, so no romanization is involved. The point is to avoid a
    second name that is a respelling of the first: if both names are nearly the same
    string, the native condition can be answered by looking at the two strings, while the
    Latin condition still requires knowing the entity, and the two are no longer the same
    question asked twice.
    """
    folded_label = fold_native(label)
    scored = [
        (1.0 - difflib.SequenceMatcher(None, folded_label, fold_native(a)).ratio(), a)
        for a in dict.fromkeys(aliases) if fold_native(a) and fold_native(a) != folded_label
    ]
    if not scored:
        return None
    distance, alias = max(scored)
    return alias if distance >= 0.15 else None


# Same shape as NAMES_QUERY so fetch_band's truncate-and-split logic works unchanged, with
# one line added: the person must hold a home nationality.
#
# This is a targeted fetch, not a bigger one. The plain query asks for anyone with a Hindi
# alias, and most of them turn out to be foreign -- 302 of 418 in the last pool -- so the
# row limit is spent on the arm that is already full. Entering through P27 instead is also
# cheaper, because a country is indexed and there are far fewer Indians than humans.
#
# Over-fetching this arm is safe. The design matches native against foreign on fame, and
# balance() trims the surplus afterwards; what it cannot do is invent native entities that
# were never fetched.
NATIVE_NAMES_QUERY = """
SELECT ?person ?nativeLabel ?nativeAlias ?enLabel ?sitelinks WHERE {{
  VALUES ?home {{ {homes} }}
  ?person wdt:P31 wd:Q5 ;
          wdt:P27 ?home ;
          wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {lo} && ?sitelinks < {hi})
  ?person rdfs:label    ?nativeLabel . FILTER(LANG(?nativeLabel) = "{lang}")
  ?person skos:altLabel ?nativeAlias . FILTER(LANG(?nativeAlias) = "{lang}")
  ?person rdfs:label    ?enLabel     . FILTER(LANG(?enLabel) = "en")
}}
LIMIT {limit}
"""


def fetch(args) -> None:
    lang = LANGUAGES[args.lang]
    module = load_pool_module()

    if args.native_only:
        homes = " ".join(f"wd:{q}" for q in lang.country_qids)
        # fetch_band reads NAMES_QUERY from its own module, so swapping it there is what
        # lets the band splitting and limit reduction be reused rather than reimplemented.
        module.NAMES_QUERY = NATIVE_NAMES_QUERY.replace("{homes}", homes)
        print(f"native only: {', '.join(lang.country_qids)}\n")

    def in_script(text):
        return any(lo <= ord(c) <= hi for c in str(text) for lo, hi in lang.script_ranges)

    out_dir = args.out or HERE / "out" / "corpora" / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    already = set()
    if args.exclude and args.exclude.exists():
        already = set(pd.read_csv(args.exclude).fact_id.astype(str))
        print(f"excluding {len(already)} entities already reviewed\n")

    frames = []
    for lo, hi in module.bands_for(args.min_sitelinks):
        collected = sum(len(f) for f in frames)
        if collected >= args.target * args.oversample:
            break
        frame, ok = module.fetch_band(lang, lo, hi, args.limit, f"{lo}-{hi}")
        if frame is not None and not frame.empty:
            frames.append(frame)
        print(f"  band {lo:>5}-{hi:<6} {0 if frame is None else len(frame):>6} rows"
              f"{'' if ok else '   (incomplete)'}")
    if not frames:
        sys.exit("no rows returned")

    raw = pd.concat(frames, ignore_index=True)
    raw["qid"] = raw["person"].map(module.qid_from_uri)
    # Every alias arrives as its own row, so an entity with six aliases appears six times.
    # The whole list is wanted here, not one row of it.
    grouped = raw.groupby("qid").agg(
        nativeLabel=("nativeLabel", "first"),
        enLabel=("enLabel", "first"),
        sitelinks=("sitelinks", "first"),
        nativeAliases=("nativeAlias", lambda s: list(dict.fromkeys(s.dropna()))),
    ).reset_index()
    grouped = grouped[~grouped.qid.map(lambda q: f"{args.lang}_{q.lower()}").isin(already)]
    # Saved before any filtering. The statement run discarded 4945 rows to keep 29 and left
    # nothing to try a second approach against, which cost a whole fetch.
    grouped.to_csv(out_dir / "pool_raw.csv", index=False, encoding="utf-8")
    print(f"\n{len(grouped)} entities, raw pool saved to {out_dir / 'pool_raw.csv'}")

    grouped["name_b_dev"] = [
        furthest_alias(r.nativeLabel, r.nativeAliases) for r in grouped.itertuples()
    ]
    grouped = grouped[grouped.name_b_dev.notna()]
    print(f"{len(grouped)} have a native alias that is a genuinely different name")

    print("\nfetching English aliases (all of them: the reviewer chooses)")
    english = labels(grouped.qid.tolist(), ["en"])

    rows = []
    for r in grouped.itertuples():
        options = list(dict.fromkeys(
            english.get(r.qid, {}).get("aliases", {}).get("en", [])))
        options = [o for o in options if o.strip().lower() != str(r.enLabel).strip().lower()]
        if not options:
            continue
        d = (defects(r.nativeLabel, True, in_script) + defects(r.enLabel, False, in_script)
             + defects(r.name_b_dev, True, in_script))
        rows.append({
            "fact_id": f"{args.lang}_{r.qid.lower()}",
            "qid": r.qid,
            "name_a_dev": r.nativeLabel,
            "name_a_lat": r.enLabel,
            "name_b_dev": r.name_b_dev,
            "name_b_lat": "",
            "en_alias_options": " | ".join(options),
            "other_native_aliases": " | ".join(
                a for a in r.nativeAliases if a != r.name_b_dev),
            "keep": "",
            "reviewer_notes": "",
            "sitelinks": int(float(r.sitelinks)) if str(r.sitelinks).strip() else 0,
            "pair_defects": "; ".join(dict.fromkeys(d)),
        })
    sheet = pd.DataFrame(rows).sort_values(
        ["pair_defects", "sitelinks"], ascending=[False, False])
    path = out_dir / "review_alias.csv"
    # utf-8-sig so Excel shows Devanagari instead of mojibake.
    sheet.to_csv(path, index=False, encoding="utf-8-sig")

    flagged = int((sheet.pair_defects != "").sum())
    print(f"\n{len(sheet)} rows for review ({flagged} carry a defect and sort first)")
    print(f"written: {path}\n")
    print("For each row: read en_alias_options, copy the one that is the same name as")
    print("name_b_dev into name_b_lat, and put yes in keep. If none of them is that name,")
    print("leave keep as no. other_native_aliases is there if a different native alias")
    print("pairs better; replace name_b_dev with it if so.")
    print(f"\nthen: python 01c_alias_pairs.py build --sheet {path} --lang {args.lang}")


def build(args) -> None:
    lang = LANGUAGES[args.lang]
    module = load_pool_module()
    out_dir = args.out or HERE / "out" / "corpora" / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = pd.read_csv(args.sheet)
    sheet["keep"] = sheet.keep.astype(str).str.strip().str.lower()
    kept = sheet[sheet.keep == "yes"].copy()
    print(f"{len(sheet)} reviewed, {len(kept)} kept")

    blank = kept[kept.name_b_lat.astype(str).str.strip().isin(["", "nan"])]
    if len(blank):
        print(f"  {len(blank)} kept rows have no name_b_lat filled in; dropping them")
        for fid in blank.fact_id.head(5):
            print(f"    {fid}")
        kept = kept.drop(blank.index)

    # The English name must be one the reviewer was actually offered, so a typo becomes an
    # error here rather than a silent corpus entry.
    bad = [
        r.fact_id for r in kept.itertuples()
        if str(r.name_b_lat).strip().lower() not in
        {o.strip().lower() for o in str(r.en_alias_options).split("|")}
    ]
    if bad:
        print(f"  {len(bad)} rows have a name_b_lat that is not among the options offered:")
        for fid in bad[:8]:
            print(f"    {fid}")
        print("  fix them in the sheet, or re-run with --allow-free-text")
        if not args.allow_free_text:
            sys.exit(1)

    print("\nfetching country and occupation")
    extra = module.enrich(kept.qid.tolist()).drop_duplicates("qid")
    before = len(kept)
    kept = kept.merge(extra, on="qid", how="left")
    assert len(kept) == before, "the country join duplicated rows"

    home = set(lang.country_qids)
    kept["country_qids"] = kept.get("countries", pd.Series("", index=kept.index)).fillna("")
    kept["occupation_qids"] = kept.get("occupations", pd.Series("", index=kept.index)).fillna("")
    kept["origin"] = kept.country_qids.map(
        lambda c: "native" if home & set(str(c).split("|")) else "foreign")
    kept["dev_lang"] = lang.wikidata_lang
    kept["name_b_dev_source"] = "alias"
    kept["name_b_lat_source"] = "alias, chosen by a reader"
    kept["relation_tier"] = "alias_observed"
    kept["relation_tier_num"] = 1
    kept["generated_lat"] = False

    columns = ["fact_id", "qid", "name_a_dev", "name_a_lat", "name_b_dev", "name_b_lat",
               "dev_lang", "origin", "name_b_dev_source", "name_b_lat_source",
               "relation_tier", "relation_tier_num", "occupation_qids", "country_qids",
               "sitelinks", "generated_lat"]
    corpus = kept[[c for c in columns if c in kept.columns]].reset_index(drop=True)

    if args.merge and args.merge.exists():
        existing = pd.read_csv(args.merge)
        existing = existing[[c for c in columns if c in existing.columns]]
        overlap = set(existing.fact_id) & set(corpus.fact_id)
        corpus = pd.concat([existing, corpus[~corpus.fact_id.isin(overlap)]],
                           ignore_index=True)
        print(f"\nmerged with {args.merge.name}: {len(existing)} + "
              f"{len(corpus) - len(existing)} new = {len(corpus)}")

    corpus.to_csv(out_dir / "corpus_selected.csv", index=False, encoding="utf-8")

    # Rebuilt over the survivors rather than filtered: a negative pointing at a row that is
    # not here would either break the map or silently drop a third entity.
    spec = importlib.util.spec_from_file_location("assemble", HERE / "02_assemble.py")
    assemble = importlib.util.module_from_spec(spec)
    sys.modules["assemble"] = assemble
    spec.loader.exec_module(assemble)
    negatives = assemble.hard_negatives(corpus)
    negatives.to_csv(out_dir / "hard_negative_map.csv", index=False, encoding="utf-8")

    print(f"\ncorpus: {len(corpus)} items")
    print(f"  native {int((corpus.origin == 'native').sum())}"
          f"   foreign {int((corpus.origin == 'foreign').sum())}")
    print(f"  negatives: {len(negatives)} rows")
    print(f"\nwritten: {out_dir / 'corpus_selected.csv'}")
    print(f"written: {out_dir / 'hard_negative_map.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="build a review sheet")
    f.add_argument("--lang", required=True)
    f.add_argument("--target", type=int, default=400, help="entities wanted after review")
    f.add_argument("--oversample", type=float, default=3.0,
                   help="rows to fetch per row wanted. Roughly half survive review")
    f.add_argument("--limit", type=int, default=1200, help="rows per sitelink band")
    f.add_argument("--min-sitelinks", type=int, default=12)
    f.add_argument("--native-only", action="store_true",
                   help="only people holding a home nationality. The plain query spends "
                        "its row limit mostly on foreign entities, so this is how the "
                        "native arm gets filled")
    f.add_argument("--exclude", type=Path, default=None,
                   help="CSV of fact_ids already reviewed, so they are not shown again")
    f.add_argument("--out", type=Path, default=None)
    f.set_defaults(func=fetch)

    b = sub.add_parser("build", help="turn a filled sheet into a corpus")
    b.add_argument("--sheet", type=Path, required=True)
    b.add_argument("--lang", required=True)
    b.add_argument("--merge", type=Path, default=None,
                   help="an existing corpus to append to, e.g. the 207 verified rows")
    b.add_argument("--allow-free-text", action="store_true",
                   help="permit a name_b_lat that was not among the offered aliases")
    b.add_argument("--out", type=Path, default=None)
    b.set_defaults(func=build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

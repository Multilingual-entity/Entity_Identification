r"""Build the entity pool from names humans wrote in both scripts. No transliteration.

The previous selector romanized the native name to decide whether it matched the Latin
one, which cannot be used here. Stage 12 scores each romanization scheme on how faithfully
it reproduces the corpus's own Latin field and reports the human field as the ceiling a
perfect romanizer would reach. If ITRANS and IAST also decide which pairs enter the corpus,
the corpus is filtered down to the pairs those schemes already handle, their measured
fidelity rises for that reason alone, and the gap stage 12 exists to report shrinks toward
zero. The measurement would then be partly of its own selection.

So every pair here is linked by attestation instead, and the linkage comes from the data:

  name_a   the Hindi Wikipedia article title against the English Wikipedia article title.
           These are the same article in two languages, connected by the interlanguage
           link, so the correspondence is asserted by editors and not inferred by us.

  name_b   a Wikidata statement carrying a value in both languages. P1477 (birth name) with
           a Hindi value and an English value is one claim about one name, written twice.
           Item-valued P735 and P734 are stronger still: one item, two labels.

Wikidata's alias lists cannot do this. The Hindi list and the English list are independent
sets with nothing tying an entry in one to an entry in the other, which is exactly why the
old build paired नवाब राय with Dhanpat Rai -- both true of Premchand, neither the same name
as the other. Pairing across unlinked lists needs a similarity judgement, and a similarity
judgement across scripts needs a transliterator. Statements remove the need for both.

The structural screen that remains is single-field only: brackets, joined lists, stray
escapes, script purity. None of it compares the two scripts, so none of it can leak a
transliterator's opinion into the corpus. Its output orders the review sheet and drops
nothing; precision against a hand review of 418 rows was 59 percent, which is far too low
to filter on and quite good enough to sort by.

    python 01b_attested_pairs.py --lang hi --target 800
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

from languages import LANGUAGES
from wikidata import SparqlTooLarge, entities, labels, qid_from_uri, sparql

HERE = Path(__file__).resolve().parent

# Statements whose value is one name recorded in more than one language. Order is priority:
# a birth name is a better second name than a decomposed given-plus-family pair, because it
# is a name someone was actually called rather than one assembled here.
MONOLINGUAL_PROPS = [
    ("P1477", "birth name"),
    ("P742",  "pseudonym"),
    ("P1449", "nickname"),
    ("P1448", "official name"),
    ("P1559", "name in native language"),
]
# Item-valued, so the two renderings are two labels on one item and cannot disagree about
# which name they refer to. Composed in this order into a single second name.
ITEM_NAME_PROPS = [("P735", "given name"), ("P734", "family name")]

PARTICLES = {"do", "da", "de", "del", "della", "di", "du", "dos", "van", "von", "der",
             "den", "bin", "ibn", "al", "el", "la", "le", "ter", "ten", "of", "the", "y"}

# One row per entity that has an article in both Wikipedias. Entering through the smaller
# Wikipedia's sitelink is the indexed path; filtering eleven million humans by label
# language is not, and times out. See the note in 01_entity_pool.py.
PAIR_QUERY = """
SELECT ?person ?nativeTitle ?enTitle ?sitelinks WHERE {{
  ?nativeArticle schema:isPartOf <https://{wiki}.wikipedia.org/> ;
                 schema:about ?person ;
                 schema:name ?nativeTitle .
  ?person wdt:P31 wd:Q5 ;
          wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {lo} && ?sitelinks < {hi})
  ?enArticle schema:isPartOf <https://en.wikipedia.org/> ;
             schema:about ?person ;
             schema:name ?enTitle .
}}
LIMIT {limit}
"""


def structural_defects(text: str, native: bool, in_script) -> list:
    """Defects visible in one field on its own. Nothing here crosses the two scripts."""
    text = str(text)
    found = []
    if re.search(r"[(\[]", text):
        found.append("parenthetical disambiguator")
    if "|" in text or "\\" in text:
        found.append("stray separator or escape")
    if "," in text or ";" in text or {"या", "किंवा", "अथवा", "or"} & set(text.split()):
        found.append("several names in one field")
    if native and not in_script(text):
        found.append("native field is not in the native script")
    if native and re.search(r"[A-Za-z]", text):
        found.append("Latin letters in the native field")
    # Particles are lowercase by convention, not by carelessness: "Edson Arantes do
    # Nascimento" is correctly written, "Johan kristopher hansen" is not.
    words = [w for w in text.split() if len(w) > 1 and w.lower() not in PARTICLES]
    if not native and len(words) > 1 and any(w[0].isupper() for w in words) \
            and any(w[0].islower() and w[0].isalpha() for w in words):
        found.append("inconsistently capitalised")
    return found


def second_name(entity: dict, wikidata_lang: str, item_labels: dict) -> tuple:
    """A second name for this entity attested in both languages, or (None, None, None).

    Returns the native form, the English form, and the property it came from, so the
    provenance of every pair is on the record rather than implied.
    """
    claims = entity.get("claims", {})

    for prop, description in MONOLINGUAL_PROPS:
        native_values, english_values = [], []
        for statement in claims.get(prop, []):
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            if not isinstance(value, dict):
                continue
            text, language = value.get("text"), value.get("language")
            if not text:
                continue
            if language == wikidata_lang:
                native_values.append(text)
            elif language == "en":
                english_values.append(text)
        # Exactly one each, or fall through. With two values per language there are four
        # possible pairings and nothing in the data says which is which -- choosing would
        # be the same guess across unlinked lists that this whole module exists to avoid.
        if len(native_values) == 1 and len(english_values) == 1:
            return native_values[0], english_values[0], description

    native_parts, english_parts = [], []
    for prop, _ in ITEM_NAME_PROPS:
        for statement in claims.get(prop, []):
            qid = statement.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            qid = qid.get("id") if isinstance(qid, dict) else None
            entry = item_labels.get(qid, {}).get("labels", {}) if qid else {}
            if entry.get(wikidata_lang) and entry.get("en"):
                native_parts.append(entry[wikidata_lang])
                english_parts.append(entry["en"])
                break
    if len(native_parts) == len(ITEM_NAME_PROPS):
        return " ".join(native_parts), " ".join(english_parts), "given and family name"
    return None, None, None


def name_item_qids(blocks: dict) -> list:
    """Every P735/P734 value, so their labels can be fetched in one batch."""
    wanted = set()
    for entity in blocks.values():
        for prop, _ in ITEM_NAME_PROPS:
            for statement in entity.get("claims", {}).get(prop, []):
                value = statement.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and value.get("id"):
                    wanted.add(value["id"])
    return sorted(wanted)


def fetch_pairs(lang, target: int) -> pd.DataFrame:
    """Walk down the fame bands until enough entities have an article in both languages."""
    frames, seen = [], set()
    # target here is already the oversampled figure: how many to fetch, not how many are
    # wanted at the end.
    bands = [(200, 10000), (120, 200), (80, 120), (55, 80), (40, 55),
             (30, 40), (22, 30), (16, 22), (12, 16), (8, 12), (5, 8)]
    for lo, hi in bands:
        collected = sum(len(f) for f in frames)
        if collected >= target:
            break
        limit = min(2000, max(300, target - collected))
        try:
            rows = sparql(PAIR_QUERY.format(wiki=lang.wikidata_lang, lo=lo, hi=hi, limit=limit))
        except (SparqlTooLarge, RuntimeError) as exc:
            print(f"  band {lo}-{hi}: skipped ({type(exc).__name__})")
            continue
        if not rows:
            print(f"  band {lo}-{hi}: nothing returned")
            continue
        frame = pd.DataFrame(rows)
        frame["qid"] = frame["person"].map(qid_from_uri)
        frame = frame[~frame.qid.isin(seen)]
        seen |= set(frame.qid)
        frames.append(frame)
        print(f"  band {lo:>4}-{hi:<5} {len(frame):>5} entities with an article in both "
              f"({sum(len(f) for f in frames)} so far)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("qid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="key in languages.py, e.g. hi")
    ap.add_argument("--target", type=int, default=800,
                    help="entities wanted in the finished pool")
    ap.add_argument("--oversample", type=float, default=6.0,
                    help="entities to fetch per entity wanted. Most of the loss is at the "
                         "statement step, where an entity is dropped for having no second "
                         "name recorded in both languages, and that rate is not known "
                         "until it has run. Raise this if the run comes up short")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lang = LANGUAGES[args.lang]
    wikidata_lang = lang.wikidata_lang
    # script_ranges are codepoint bounds, the same definition the rest of the pipeline
    # uses, so the "is this actually in the native script" check cannot drift from it.
    def in_native_script(text: str) -> bool:
        return any(lo <= ord(c) <= hi for c in str(text)
                   for lo, hi in lang.script_ranges)
    out_dir = args.out or HERE / "out" / "corpora" / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{lang.name}: entities with an article in both Wikipedias")
    pool = fetch_pairs(lang, int(args.target * args.oversample))
    if pool.empty:
        sys.exit(f"no entities returned; is there a {lang.wikidata_lang}.wikipedia.org?")
    print(f"\n{len(pool)} entities carry a linked article pair\n")

    print("reading statements for the second name")
    blocks = entities(pool.qid.tolist())
    item_qids = name_item_qids(blocks)
    print(f"  resolving {len(item_qids)} given-name and family-name items")
    item_labels = labels(item_qids, [wikidata_lang, "en"]) if item_qids else {}

    rows, sources = [], {}
    for r in pool.itertuples():
        native_b, english_b, source = second_name(
            blocks.get(r.qid, {}), wikidata_lang, item_labels)
        if not native_b or not english_b:
            continue
        # A second name identical to the first asks whether a name is itself.
        if native_b.strip() == str(r.nativeTitle).strip() \
                or english_b.strip().lower() == str(r.enTitle).strip().lower():
            continue
        sources[source] = sources.get(source, 0) + 1
        rows.append({
            "fact_id": f"{args.lang}_{r.qid.lower()}",
            "qid": r.qid,
            "name_a_dev": str(r.nativeTitle).strip(),
            "name_a_lat": str(r.enTitle).strip(),
            "name_b_dev": native_b.strip(),
            "name_b_lat": english_b.strip(),
            "dev_lang": wikidata_lang,
            "name_b_dev_source": source,
            "name_b_lat_source": source,
            "sitelinks": int(float(r.sitelinks)) if str(r.sitelinks).strip()
                         not in ("", "nan", "None") else 0,
        })
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} have a second name attested in both languages")
    for source, n in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"    {source:<24}{n:>6}")
    if df.empty:
        sys.exit("no attested second names; widen the property list")

    defects = []
    for r in df.itertuples():
        d = (structural_defects(r.name_a_dev, True, in_native_script)
             + structural_defects(r.name_a_lat, False, in_native_script)
             + structural_defects(r.name_b_dev, True, in_native_script)
             + structural_defects(r.name_b_lat, False, in_native_script))
        defects.append("; ".join(dict.fromkeys(d)))
    df["pair_defects"] = defects
    flagged = int((df.pair_defects != "").sum())
    print(f"\n{flagged} carry a structural defect, {len(df) - flagged} are clean")
    print("  nothing is dropped for this: the checks are 59% precise against a hand")
    print("  review, so they order the review sheet and the reader decides")

    # Country and occupation, reused from the banded fetcher so origin is computed the
    # same way in both paths.
    spec = importlib.util.spec_from_file_location("pool", HERE / "01_entity_pool.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pool"] = module
    spec.loader.exec_module(module)
    print("\nfetching country and occupation")
    extra = module.enrich(df.qid.tolist()).drop_duplicates("qid")
    before_merge = len(df)
    df = df.merge(extra, on="qid", how="left")
    assert len(df) == before_merge, "the country join duplicated rows"
    home = set(lang.country_qids)
    df["country_qids"] = df.get("countries", pd.Series("", index=df.index)).fillna("")
    df["occupation_qids"] = df.get("occupations", pd.Series("", index=df.index)).fillna("")
    df["origin"] = df.country_qids.map(
        lambda c: "native" if home & set(str(c).split("|")) else "foreign")
    df["relation_tier"] = "statement_attested"
    df["relation_tier_num"] = 1
    df["generated_lat"] = False

    df = df.sort_values(["pair_defects", "sitelinks"], ascending=[True, False])
    columns = ["fact_id", "qid", "name_a_dev", "name_a_lat", "name_b_dev", "name_b_lat",
               "dev_lang", "origin", "name_b_dev_source", "name_b_lat_source",
               "relation_tier", "relation_tier_num", "occupation_qids", "country_qids",
               "sitelinks", "generated_lat", "pair_defects"]
    df = df[[c for c in columns if c in df.columns]]
    path = out_dir / "pool_attested.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    print(f"\n{len(df)} entities written")
    print(f"  native {int((df.origin == 'native').sum())}"
          f"   foreign {int((df.origin == 'foreign').sum())}")
    print(f"  clean {len(df) - flagged}   flagged {flagged}")
    # Said here rather than left to be noticed later. The loss is almost entirely at the
    # statement step, and how large it is depends on how well the language is covered on
    # Wikidata, which is not knowable before the run.
    if len(df) < args.target:
        no_second_name = len(pool) - len(rows)
        print(f"\n  SHORT: {len(df)} of the {args.target} wanted.")
        print(f"  {len(pool)} entities were fetched; {no_second_name} of them "
              f"({no_second_name / len(pool):.0%}) had no second name recorded in both "
              f"languages.")
        print(f"  Re-run with --oversample {args.oversample * args.target / max(len(df), 1):.0f}"
              f", or add properties to MONOLINGUAL_PROPS.")
    print(f"\nwritten: {path}")
    print(f"\nnext: python make_review_sheet.py {path}")


if __name__ == "__main__":
    main()

"""Build a per-language entity pool, both native and foreign, directly from Wikidata.

Replaces the earlier native-only script. Two things forced the change.

The existing 299 entities were selected for Marathi. Tamil holds only 84 of them with a
usable identity pair, Urdu 37, Punjabi 31. That is a property of the selection, not of
those languages: Wikidata has plenty of entities with Tamil labels and Tamil aliases,
they simply are not these entities. Querying per language recovers them.

And every language needs both origins. For a foreign entity the canonical form is Latin,
so the native script should hurt. For an entity native to the language the native script
IS canonical, so the prediction reverses. Testing only foreign entities measures one half
of the account and cannot falsify it.

An identity pair needs two name forms in the native script, so the query requires a label
and an alias, not a label alone.
"""
from __future__ import annotations

import argparse
import difflib
import unicodedata
from pathlib import Path

import pandas as pd

from languages import LANGUAGES, THIN
from wikidata import SparqlTooLarge, labels, qid_from_uri, sparql

HERE = Path(__file__).resolve().parent


from pair_selection import choose_pair  # noqa: E402


def romanize(text: str) -> str:
    """A rough Latin rendering of any script, for comparing names across scripts.

    Only ever used to decide which of several English aliases best matches the native one.
    It does not need to be a good transliteration, only a consistent one.
    """
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


def best_matching_alias(native_alias: str, candidates: list) -> str | None:
    """Pick the English alias that is the same NAME as the native one.

    This is the difference between a corpus that tests a script contrast and one that does
    not. Wikidata lists several aliases per entity in each language, and taking the first
    of each independently pairs a native alias with an unrelated English one: the Devanagari
    field reads "Agnes Gonxha Bojaxhiu" while the English field reads "Saint Teresa of
    Calcutta", or the Devanagari reads "Hema Malini" and the English reads "Dream Girl".
    Dev--Dev then asks about one alias and Lat--Lat about a different one, so the condition
    varies the name as well as the script and measures neither cleanly.

    Roughly thirty percent of the Marathi corpus was affected this way, and over half of
    the Tamil. The information to fix it was already fetched -- every English alias comes
    back in the same API call -- and was being discarded by taking element zero.
    """
    import difflib

    if not candidates:
        return None
    target = fold_for_match(romanize(native_alias))
    best, best_score = None, -1.0
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, target, fold_for_match(candidate)).ratio()
        if score > best_score:
            best, best_score = candidate, score
    return best


def fold_for_match(text: str) -> str:
    """Case, spacing and punctuation removed, so only the letters are compared."""
    kept = [c.lower() for c in strip_diacritics(text) if c.isalnum()]
    return "".join(kept)


def strip_diacritics(text: str) -> str:
    """Decompose, drop combining marks, recompose. The comparison form for a
    diacritic-contrast language, and the same operation stage 12 uses."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    kept = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", kept)

# Two queries, deliberately. Folding country and occupation into the name query with
# OPTIONAL and GROUP_CONCAT makes it time out silently on the public endpoint. Fetching
# names first and enriching a known QID list afterwards is far cheaper and never hangs.
NAMES_QUERY = """
SELECT ?person ?nativeLabel ?nativeAlias ?enLabel ?sitelinks WHERE {{
  ?person wdt:P31 wd:Q5 ;
          wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {lo} && ?sitelinks < {hi})
  ?person rdfs:label    ?nativeLabel . FILTER(LANG(?nativeLabel) = "{lang}")
  ?person skos:altLabel ?nativeAlias . FILTER(LANG(?nativeAlias) = "{lang}")
  ?person rdfs:label    ?enLabel     . FILTER(LANG(?enLabel) = "en")
}}
LIMIT {limit}
"""

ENRICH_QUERY = """
SELECT ?person
       (GROUP_CONCAT(DISTINCT ?countryQ;   separator="|") AS ?countries)
       (GROUP_CONCAT(DISTINCT ?occQ;       separator="|") AS ?occupations)
WHERE {{
  VALUES ?person {{ {values} }}
  OPTIONAL {{ ?person wdt:P27  ?country . BIND(STRAFTER(STR(?country), "entity/") AS ?countryQ) }}
  OPTIONAL {{ ?person wdt:P106 ?occ .     BIND(STRAFTER(STR(?occ),     "entity/") AS ?occQ) }}
}}
GROUP BY ?person
"""

# For a language with very few speakers on Wikidata, NAMES_QUERY is the wrong shape and no
# amount of narrowing rescues it. It filters eleven million humans by sitelink count and
# then joins three separate label patterns, and when almost none of them carry a label in
# the target language the LIMIT never fills, so the engine scans everything and hits the
# sixty-second ceiling. Sanskrit fails this way at every band and every row limit tried.
#
# Asking for the alias first does not help either, and it is worth saying why, because the
# reason rules out a whole family of fixes. FILTER(LANG(?alias) = "sa") is not an index
# lookup. The engine has to walk every alias of every human, hundreds of millions of
# triples, and test each one's language tag. The language being rare makes that worse, not
# better: nothing matches, so nothing lets it stop early.
#
# The way in is a sitelink, which is indexed. Every entity with an article on that
# language's own Wikipedia is one query away, and for a small Wikipedia that set is small.
# Those are also precisely the entities likely to carry a label and alias in the language,
# so little is lost by entering this way. Names, aliases and sitelink counts are then
# filled in for the resulting bounded QID list, the same way country and occupation are.
THIN_SITELINK_QUERY = """
SELECT DISTINCT ?person WHERE {{
  ?article schema:about ?person ;
           schema:isPartOf <https://{lang}.wikipedia.org/> .
  ?person wdt:P31 wd:Q5 .
}}
LIMIT {limit}
"""

SITELINKS_QUERY = """
SELECT ?person ?sitelinks WHERE {{
  VALUES ?person {{ {values} }}
  ?person wikibase:sitelinks ?sitelinks .
}}
"""


def enrich(qids: list, chunk: int = 200) -> pd.DataFrame:
    """Country and occupation for a known QID list. Cheap because the set is bounded."""
    frames = []
    for i in range(0, len(qids), chunk):
        values = " ".join(f"wd:{q}" for q in qids[i:i + chunk])
        try:
            rows = sparql(ENRICH_QUERY.format(values=values))
        except RuntimeError as exc:
            print(f"    enrich chunk {i // chunk}: failed ({str(exc)[:60]})")
            continue
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["qid", "countries", "occupations"])
    df = pd.concat(frames, ignore_index=True)
    df["qid"] = df["person"].map(qid_from_uri)
    return df[["qid", "countries", "occupations"]]

# Sitelink bands keep each query small enough for the public endpoint and give a handle
# on fame, which must be matched across languages or a gap difference could just be a
# difference in how well known each language's entities are.
BANDS = ((100, 10000), (50, 100), (30, 50), (20, 30), (12, 20))


def bands_for(min_sitelinks: int) -> tuple:
    """The band list, optionally extended below the usual floor of twelve sitelinks.

    Twelve was chosen so every entity is at least modestly documented. Going lower admits
    less famous people and is the only way to raise the ceiling for a language that has
    simply run out of well-linked entities, which is what limits Serbian and Chinese here.

    The cost is real and has to be stated wherever it is used: fame is matched across
    languages by band, so a language fetched down to five sitelinks is not comparable with
    one fetched to twelve unless both are cut back to the same floor before analysis. The
    band label is recorded per row, so that cut is always possible after the fact.
    """
    bands = list(BANDS)
    if min_sitelinks < 12:
        bands.append((min_sitelinks, 12))
    return tuple(bands)


def fetch_band(lang, lo: int, hi: int, limit: int, label: str, depth: int = 0,
               max_depth: int = 6, limit_floor: int = 50):
    """One sitelink band, split in half and retried when the endpoint truncates.

    A truncated response means the result set was too big to stream inside the time limit,
    so the fix is a narrower band rather than another attempt. Splitting on sitelinks is
    order-free and deterministic, unlike LIMIT with OFFSET, which needs an ORDER BY that
    would itself make the query slower.

    The split point is geometric rather than arithmetic because sitelink counts follow a
    power law: halving 100-10000 at 5050 leaves almost everything on one side, whereas
    splitting at 1000 actually divides the work.

    Returns (frame, ok) where ok is False if any sub-band was abandoned. The caller needs
    that flag: a band that quietly returns nothing removes a whole fame range from one
    language's pool while leaving it in another's, and the fame matching the design rests
    on is then broken in a way no downstream table would reveal.
    """
    query = NAMES_QUERY.format(lang=lang.wikidata_lang, lo=lo, hi=hi, limit=limit)
    try:
        rows = sparql(query)
    except SparqlTooLarge:
        # Reduce the row limit before splitting on a narrow band. Splitting multiplies the
        # number of requests, and the low bands are both the narrowest and the densest, so
        # 12-20 fans out into a dozen sub-queries that each still truncate. Against an
        # endpoint that is already rate-limiting, that turns one failure into several and
        # provokes the 502s and 504s that follow. Asking for fewer rows costs one request
        # and leaves the band whole.
        if hi - lo <= 10 and limit > limit_floor:
            reduced = max(limit_floor, limit // 2)
            print(f"    band {lo}-{hi}: truncated, narrow band so reducing the limit to "
                  f"{reduced} rather than splitting")
            return fetch_band(lang, lo, hi, reduced, label, depth, max_depth, limit_floor)
        if depth < max_depth and hi - lo > 1:
            mid = max(lo + 1, min(hi - 1, int(round((lo * hi) ** 0.5))))
            print(f"    band {lo}-{hi}: truncated, splitting at {mid}")
            left, ok_left = fetch_band(lang, lo, mid, limit, label, depth + 1,
                                       max_depth, limit_floor)
            right, ok_right = fetch_band(lang, mid, hi, limit, label, depth + 1,
                                         max_depth, limit_floor)
            parts = [p for p in (left, right) if not p.empty]
            combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            return combined, (ok_left and ok_right)
        # A narrow band cannot be split any further, and the low bands are the narrow
        # ones: 12-20 runs out of room after three halvings. Asking for fewer rows is the
        # remaining lever, and it costs only coverage within a band whose fame range is
        # already tight, rather than losing the band altogether.
        if limit > limit_floor:
            reduced = max(limit_floor, limit // 2)
            print(f"    band {lo}-{hi}: truncated at limit {limit}, retrying at {reduced}")
            return fetch_band(lang, lo, hi, reduced, label, depth, max_depth, limit_floor)
        print(f"    band {lo}-{hi}: ABANDONED at limit {limit} after splitting to the limit")
        return pd.DataFrame(), False
    except RuntimeError as exc:
        print(f"    band {lo}-{hi}: failed ({str(exc)[:70]})")
        return pd.DataFrame(), False
    if not rows:
        print(f"    band {lo}-{hi}: 0")
        return pd.DataFrame(), True
    frame = pd.DataFrame(rows)
    frame["band"] = label
    indent = "  " * depth
    print(f"    {indent}band {lo}-{hi}: {len(rows)}")
    return frame, True


def sitelinks_for(qids: list, chunk: int = 200) -> dict:
    """Sitelink counts for a known QID list. Bounded, so it never hangs."""
    out: dict = {}
    for i in range(0, len(qids), chunk):
        values = " ".join(f"wd:{q}" for q in qids[i:i + chunk])
        try:
            rows = sparql(SITELINKS_QUERY.format(values=values))
        except RuntimeError as exc:                                      # noqa: BLE001
            print(f"    sitelinks chunk {i // chunk}: failed ({str(exc)[:60]})")
            continue
        for row in rows:
            out[qid_from_uri(row["person"])] = int(float(row["sitelinks"]))
    return out


def fetch_thin(lang_key: str, limit: int = 4000) -> tuple:
    """Alias-first fetch for a language too sparse for the banded query.

    Returns the same frame and coverage shape as fetch(), so everything downstream is
    unchanged. Bands are assigned after the fact from the sitelink counts rather than
    queried for, which is possible here precisely because the whole set is small.
    """
    lang = LANGUAGES[lang_key]
    print(f"    thin mode: entities with an article on {lang.wikidata_lang}.wikipedia.org")
    try:
        rows = sparql(THIN_SITELINK_QUERY.format(lang=lang.wikidata_lang, limit=limit))
    except RuntimeError as exc:                                          # noqa: BLE001
        print(f"    thin fetch failed ({str(exc)[:70]})")
        return pd.DataFrame(), pd.DataFrame(
            [{"lang": lang_key, "band": "thin", "complete": 0, "n_rows": 0}])
    qids = [qid_from_uri(r["person"]) for r in rows]
    print(f"    {len(qids)} people have an article on that Wikipedia")
    if not qids:
        return pd.DataFrame(), pd.DataFrame(
            [{"lang": lang_key, "band": "thin", "complete": 1, "n_rows": 0}])

    print("    fetching names for that list")
    fetched = labels(qids, [lang.wikidata_lang, "en"])
    counts = sitelinks_for(qids)

    records = []
    for qid in qids:
        entry = fetched.get(qid, {})
        native_label = entry.get("labels", {}).get(lang.wikidata_lang)
        native_aliases = entry.get("aliases", {}).get(lang.wikidata_lang, [])
        english_label = entry.get("labels", {}).get("en")
        # The three requirements the banded query enforces in SPARQL, applied here instead.
        if not native_label or not native_aliases or not english_label:
            continue
        # Same choice as the banded path: the alias least like the label, not the first.
        chosen = max(native_aliases,
                     key=lambda a: 1.0 - difflib.SequenceMatcher(
                         None, fold_for_match(native_label), fold_for_match(a)).ratio())
        if difflib.SequenceMatcher(None, fold_for_match(native_label),
                                   fold_for_match(chosen)).ratio() > 0.85:
            continue
        records.append({
            "person": f"http://www.wikidata.org/entity/{qid}",
            "nativeLabel": native_label,
            "nativeAlias": chosen,
            "enLabel": english_label,
            "sitelinks": counts.get(qid, 0),
            "band": "thin",
        })
    frame = pd.DataFrame(records)
    print(f"    {len(frame)} have a native label, a native alias and an English label")
    coverage = pd.DataFrame([{"lang": lang_key, "band": "thin", "complete": 1,
                              "n_rows": int(len(frame))}])
    return frame, coverage


def fetch(lang_key: str, per_band: int, thin: bool = False,
          min_sitelinks: int = 12) -> tuple:
    """Returns (frame, coverage) where coverage records what each band actually yielded.

    Both fetch paths converge here, so the filtering, enrichment and origin labelling that
    follow are identical whichever way the entities were found.
    """
    lang = LANGUAGES[lang_key]
    if thin:
        df, coverage_frame = fetch_thin(lang_key)
        if df.empty:
            return df, coverage_frame
    else:
        frames, coverage = [], []
        for lo, hi in bands_for(min_sitelinks):
            label = f"{lo}-{hi}"
            frame, ok = fetch_band(lang, lo, hi, per_band, label)
            coverage.append({"lang": lang_key, "band": label, "complete": int(ok),
                             "n_rows": int(len(frame))})
            if not frame.empty:
                frames.append(frame)
        coverage_frame = pd.DataFrame(coverage)
        if not frames:
            return pd.DataFrame(), coverage_frame
        df = pd.concat(frames, ignore_index=True)
    coverage = coverage_frame
    df["qid"] = df["person"].map(qid_from_uri)

    # The query returns one row per alias, so an entity with six aliases arrives six times.
    # Keeping whichever came first picked a spelling variant far too often: तेनझिंग नोर्गे
    # against तेनझिंग नॉर्गे, one vowel sign apart, or झुल्फिकार अली भुट्टो against
    # झुल्फिकारअली भुट्टो, one space apart. A hundred and thirty-six Marathi items ended up
    # with two native names identical once spacing is ignored, and three hundred and
    # twenty-three at least eighty-five percent alike.
    #
    # That breaks the design in a way that is easy to miss, because the item still looks
    # well formed. The native condition asks whether two spellings denote one person, which
    # string similarity answers on its own; the Latin condition asks whether Hema Malini is
    # Dream Girl, which needs the entity. The two conditions stop being the same question
    # asked twice, and the comparison between them measures the difference in difficulty as
    # much as the difference in script.
    #
    # So choose rather than take: keep, per entity, the alias LEAST like the label, then
    # drop anything still too close to be a second name at all. This is the same correction
    # as best_matching_alias makes on the English side, in the opposite direction: there the
    # goal is the alias closest to the native one, here it is the alias furthest from the
    # label.
    import difflib

    df["_alias_distance"] = [
        1.0 - difflib.SequenceMatcher(None, fold_for_match(label),
                                      fold_for_match(alias)).ratio()
        for label, alias in zip(df["nativeLabel"], df["nativeAlias"])
    ]
    before_choice = df["qid"].nunique()
    df = (df.sort_values("_alias_distance", ascending=False)
            .drop_duplicates(subset=["qid"], keep="first"))
    too_similar = int((df["_alias_distance"] < 0.15).sum())
    df = df[df["_alias_distance"] >= 0.15]
    df = df.drop(columns=["_alias_distance"])
    print(f"    {len(df)} of {before_choice} entities have an alias that is a genuinely "
          f"different name; {too_similar} dropped as spelling variants of the label")

    if lang.contrast == "script":
        for col in ("nativeLabel", "nativeAlias"):
            df = df[df[col].map(lang.in_script) & df[col].map(lang.script_purity)]
    else:
        # A diacritic-contrast language needs names that actually carry diacritics. Without
        # this the two conditions are the same string and the item measures nothing: it is
        # the exact counterpart of the script-purity filter above, which is applied and was
        # missing here.
        #
        # It matters more than it sounds. Seventy-seven percent of the Vietnamese pool, and
        # ninety percent of the Indonesian, were plain ASCII names of international
        # celebrities. They survived the fetch, filled the foreign quota, and were then
        # thrown out one stage later as having no contrast, which is why Vietnamese built
        # thirty-five items from a pool of three hundred. Filtering here instead means the
        # quota is filled with items that can be tested.
        before = len(df)
        for col in ("nativeLabel", "nativeAlias"):
            df = df[df[col].map(lambda t: strip_diacritics(t) != str(t))]
        print(f"    {len(df)} of {before} carry diacritics in both names, so the "
              f"stripped form is a genuine second surface form")

    if df.empty:
        return df, coverage
    print(f"    enriching {len(df)} entities with country and occupation")
    df = df.merge(enrich(df["qid"].tolist()), on="qid", how="left")
    home = set(lang.country_qids)
    is_home = df.get("countries", pd.Series("", index=df.index)).fillna("").map(
        lambda c: bool(home & set(str(c).split("|"))))
    with_country = df.get("countries", pd.Series("", index=df.index)).fillna("").ne("").sum()
    print(f"    {int(with_country)} have a country of citizenship, "
          f"{int(is_home.sum())} of them in this language's own countries")

    # An English alias is required, not optional. The Latin condition needs a Latin form
    # of BOTH names; using the English label for both would make Lat--Lat ask whether
    # "Lady Gaga" and "Lady Gaga" are the same person, which is trivially yes.
    print(f"    fetching English aliases for the second name")
    en = labels(df["qid"].tolist(), ["en"])
    # Choose the pairing rather than the English alias alone. best_matching_alias scored
    # every candidate against the native one but returned its best unconditionally, so an
    # entity whose aliases do not correspond was kept anyway with its least-bad pair. That
    # produced most of the breakage in the Hindi corpus. choose_pair reconsiders both sides
    # together and ranks by whether the components correspond, so a full name is not lost
    # to a surname that happens to score higher.
    chosen = [
        choose_pair(
            [alias] if isinstance(alias, str) and alias.strip() else [],
            en.get(qid, {}).get("aliases", {}).get("en", []),
            label, en.get(qid, {}).get("labels", {}).get("en", ""),
        )
        for qid, alias, label in zip(df["qid"], df["nativeAlias"], df["nativeLabel"])
    ]
    df["enAlias"] = [c["english"] if c else None for c in chosen]
    # Recorded, never acted on. Precision against a hand review of 418 Hindi rows is 59%
    # for the structural checks and 68% for the component ones, so using either as a filter
    # would discard roughly one good entity for every two bad ones. They are worth exactly
    # what they cost here: an ordering for the review sheet, so the reader meets the likely
    # breakage first. 217 of those 418 rows carry a defect and hold 140 of the 158 genuine
    # failures, which makes the reading far more productive without deciding anything.
    df["pair_defects"] = ["; ".join(c["defects"]) if c else "" for c in chosen]
    df["pair_score"] = [round(c["score"], 3) if c else 0.0 for c in chosen]
    flagged = int((df["pair_defects"] != "").sum())
    print(f"    {len(df) - flagged} of {len(df)} pairs are structurally clean; "
          f"{flagged} carry a defect and sort to the top of the review sheet")
    before, home_before = len(df), int(is_home.sum())
    df = df[df.enAlias.notna()]
    df = df[df.enAlias.str.strip().str.lower() != df.enLabel.str.strip().str.lower()]
    print(f"    {len(df)} of {before} have a distinct English alias")

    df["origin"] = df.get("countries", pd.Series("", index=df.index)).fillna("").map(
        lambda c: "native" if home & set(str(c).split("|")) else "foreign")
    home_after = int((df.origin == "native").sum())
    # Which filter is costing the native entities matters, because the two causes need
    # opposite fixes. If the count was already low before the English-alias filter, the
    # country list in languages.py is incomplete, usually because a historical state such
    # as the Soviet Union or the Ottoman Empire is missing and its people are recorded
    # under it rather than under the modern country. If the count collapses at the alias
    # filter instead, the requirement for a second English name is what excludes them,
    # which is a limitation of the design rather than a fixable list.
    if home_before:
        print(f"    native entities: {home_before} before the English-alias filter, "
              f"{home_after} after ({home_after / home_before:.0%} survive)")
    else:
        print("    native entities: none even before the English-alias filter, so the "
              "country list for this language is the thing to fix")
    df["sitelinks"] = pd.to_numeric(df["sitelinks"], errors="coerce").fillna(0).astype(int)
    return df, coverage


def quarantine_stale(out_dir: Path, key: str) -> None:
    """Move aside a pool file left by an earlier run of a language now being skipped.

    Skipping writes nothing, so without this an earlier file survives, and the next stage
    discovers pools by globbing `pool_*.csv` and would pick up the very file the skip
    exists to reject. Renamed rather than deleted, and renamed to a prefix the glob does
    not match, so the data is recoverable but cannot be used by accident.
    """
    stale = out_dir / f"pool_{key}.csv"
    if not stale.exists():
        return
    target = out_dir / f"stale_pool_{key}.csv"
    if target.exists():
        target.unlink()
    stale.rename(target)
    print(f"    moved the earlier pool aside as {target.name}; it would otherwise be "
          f"picked up by 02_assemble.py")


def balance(df: pd.DataFrame, target_native: int, target_foreign: int) -> pd.DataFrame:
    """Take the most-linked entities of each origin, so fame is comparable across runs."""
    parts = []
    for origin, target in (("native", target_native), ("foreign", target_foreign)):
        part = df[df.origin == origin].sort_values("sitelinks", ascending=False).head(target)
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--native", type=int, default=150, help="native entities per language")
    ap.add_argument("--foreign", type=int, default=300, help="foreign entities per language")
    ap.add_argument("--per-band", type=int, default=400, help="rows requested per sitelink band")
    ap.add_argument("--min-sitelinks", type=int, default=12,
                    help="lowest sitelink count to admit. Below 12 this adds an extra band "
                         "and raises the ceiling for languages that have run out of "
                         "well-linked entities, at the cost of fame comparability unless "
                         "every language is later cut back to the same floor")
    ap.add_argument("--thin", action="store_true",
                    help="force the alias-first fetch for every language named. Use for a "
                         "language so sparse that the banded query never fills its row "
                         "limit and times out instead")
    ap.add_argument("--no-thin", action="store_true",
                    help="force the banded fetch even for languages marked thin")
    ap.add_argument("--allow-missing-bands", action="store_true",
                    help="write a pool even when a sitelink band could not be retrieved. "
                         "Off by default: a language with a hole in its fame distribution "
                         "is not comparable with one without, and nothing downstream would "
                         "reveal the difference")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "pool")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys = args.languages or list(LANGUAGES)
    summary, all_coverage = [], []
    for k in keys:
        if k not in LANGUAGES:
            print(f"unknown language: {k}")
            continue
        lang = LANGUAGES[k]
        print(f"[{k}] {lang.name} ({lang.script})")
        use_thin = args.thin or (k in THIN and not args.no_thin)
        raw, coverage = fetch(k, args.per_band, thin=use_thin,
                              min_sitelinks=args.min_sitelinks)
        all_coverage.append(coverage)
        missing = coverage.loc[coverage.complete == 0, "band"].tolist() if len(coverage) else []
        if raw.empty:
            print("    nothing returned")
            quarantine_stale(args.out, k)
            summary.append({"key": k, "raw": 0, "native": 0, "foreign": 0, "kept": 0,
                            "bands_missing": len(missing)})
            continue
        if missing and not args.allow_missing_bands:
            print(f"    SKIPPED: bands {', '.join(missing)} could not be retrieved. This "
                  f"language would enter the corpus with a different fame range from the "
                  f"others, which is exactly the confound the banding exists to remove. "
                  f"Rerun this language, or pass --allow-missing-bands to accept it.")
            quarantine_stale(args.out, k)
            summary.append({"key": k, "raw": len(raw), "native": 0, "foreign": 0,
                            "kept": 0, "bands_missing": len(missing)})
            continue
        kept = balance(raw, args.native, args.foreign)
        cols = ["qid", "nativeLabel", "nativeAlias", "enLabel", "enAlias", "origin",
                "sitelinks", "countries", "occupations", "band"]
        kept = kept[[c for c in cols if c in kept.columns]]
        kept.insert(0, "lang", k)
        kept.to_csv(args.out / f"pool_{k}.csv", index=False, encoding="utf-8")
        n_nat = int((kept.origin == "native").sum())
        n_for = int((kept.origin == "foreign").sum())
        print(f"    kept {len(kept)}  (native {n_nat}, foreign {n_for})")
        summary.append({"key": k, "raw": len(raw), "native": n_nat,
                        "foreign": n_for, "kept": len(kept),
                        "bands_missing": len(missing)})

    s = pd.DataFrame(summary)
    s.to_csv(args.out / "pool_summary.csv", index=False)
    if all_coverage:
        coverage = pd.concat(all_coverage, ignore_index=True)
        coverage.to_csv(args.out / "pool_band_coverage.csv", index=False)
    print()
    print(s.to_string(index=False))

    incomplete = s[s.get("bands_missing", 0) > 0].key.tolist() if len(s) else []
    if incomplete:
        print(f"\nincomplete band coverage: {', '.join(incomplete)}")
        print("  See pool_band_coverage.csv for which bands. Fame is matched across")
        print("  languages by band, so a language missing a band is not comparable with")
        print("  one that has it, and the gap difference between them would be partly a")
        print("  difference in how well known their entities are.")

    thin = s[s.kept < 150].key.tolist()
    if thin:
        print(f"\nunder 150 after balancing: {', '.join(thin)}")
        print("  Try --per-band 800, or extend BANDS downward to admit less-linked entities.")
    no_native = s[s.native < 30].key.tolist()
    if no_native:
        print(f"\nfew native entities: {', '.join(no_native)}")
        print("  The origin contrast will be weak for these; check the country QIDs in")
        print("  languages.py, since a language spoken across several countries needs all")
        print("  of them listed.")


if __name__ == "__main__":
    main()

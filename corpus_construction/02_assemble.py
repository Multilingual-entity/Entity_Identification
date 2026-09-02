"""Assemble a per-language corpus from the entity pool, audit it, and pair negatives.

Reads `out/pool/pool_<key>.csv` from 01_entity_pool.py and emits one directory per
language in the column layout the paperB pipeline consumes. The pipeline's columns are
named `_dev` for historical reasons; here `_dev` means "native form", whatever the
script, and `_lat` means "canonical Latin form". For Latin-script languages `_dev` holds
the diacritic-bearing form and `_lat` the stripped one, so the same four conditions
express the canonical-form contrast with script held constant.

Two audits run, not one.

  Script purity, as before: the native field must be in the native script and free of
  Latin, the Latin field free of native script.

  Name consistency, which the original corpus lacked. The two fields must be the same
  name in two scripts, not two different names of one person. Both of
  q485697 (Kepler Laveran Lima Ferreira against Pepe) and q855252 (Rama IX against
  Bhumibol Adulyadej) pass a purity check and fail this one. Ten such items reached the
  published corpus.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import unicodedata
from pathlib import Path

import pandas as pd

from languages import LANGUAGES

HERE = Path(__file__).resolve().parent
SEED = 20260813

# Script-aware romanisers, best first. Each returns None if it cannot handle the input.
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _indic
except ImportError:
    _indic = None
try:
    from unidecode import unidecode as _unidecode
except ImportError:
    _unidecode = None

INDIC_SCHEMES = {
    "Devanagari": "DEVANAGARI", "Bengali": "BENGALI", "Gujarati": "GUJARATI",
    "Gurmukhi": "GURMUKHI", "Tamil": "TAMIL", "Telugu": "TELUGU",
    "Kannada": "KANNADA", "Malayalam": "MALAYALAM",
}

# Scripts where romanisation is reliable enough for an edit-distance check. Han, Kana and
# Hangul romanise to something unrelated to the Latin name (卡卡 against Gaga), so
# similarity there is meaningless and the check is skipped rather than trusted.
COMPARABLE = {"Devanagari", "Bengali", "Gujarati", "Gurmukhi", "Tamil", "Telugu",
              "Kannada", "Malayalam", "Cyrillic", "Greek", "Latin"}


def strip_diacritics(text: str) -> str:
    out = "".join(ch for ch in unicodedata.normalize("NFD", str(text))
                  if unicodedata.category(ch) != "Mn")
    table = {"ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i", "İ": "I",
             "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "æ": "ae", "Æ": "Ae",
             "đ": "d", "Đ": "D", "œ": "oe", "Œ": "Oe", "ß": "ss",
             "ŋ": "n", "Ŋ": "N", "ə": "e", "Ə": "E"}
    return unicodedata.normalize("NFC", "".join(table.get(c, c) for c in out))


def romanise(text: str, script: str) -> str | None:
    if script in INDIC_SCHEMES and _indic is not None:
        try:
            raw = _indic(str(text), getattr(sanscript, INDIC_SCHEMES[script]), sanscript.IAST)
            return strip_diacritics(raw).lower()
        except Exception:                                           # noqa: BLE001
            pass
    if _unidecode is not None:
        return _unidecode(str(text)).lower().strip()
    return None


def name_similarity(native: str, latin: str, script: str) -> float | None:
    """1.0 means the two fields hold the same name; low means different names."""
    if script not in COMPARABLE:
        return None
    rom = romanise(native, script)
    if not rom:
        return None
    return difflib.SequenceMatcher(None, rom, strip_diacritics(latin).lower()).ratio()


def tiebreak(a: str, b: str) -> float:
    h = hashlib.sha256(f"{SEED}:{a}:{b}".encode()).hexdigest()
    return int(h[:6], 16) / 16 ** 6 * 0.01


def hard_negatives(corpus: pd.DataFrame) -> pd.DataFrame:
    """Same rule as the original: occupation, country, relation tier, origin."""
    rows, recs = [], corpus.to_dict("records")
    split = lambda v: set(str(v or "").split("|")) - {""}
    for i, row in enumerate(recs):
        occ_i, cty_i = split(row.get("occupation_qids")), split(row.get("country_qids"))
        best_j, best = None, -1e9
        for j, other in enumerate(recs):
            if i == j:
                continue
            score = 0.0
            if occ_i & split(other.get("occupation_qids")):
                score += 100
            if cty_i & split(other.get("country_qids")):
                score += 20
            if other.get("relation_tier_num") == row.get("relation_tier_num"):
                score += 5
            if other.get("origin") == row.get("origin"):
                score += 2
            score += tiebreak(row["fact_id"], other["fact_id"])
            if score > best:
                best, best_j = score, j
        rows.append({"fact_id": row["fact_id"],
                     "negative_fact_id": recs[best_j]["fact_id"],
                     "negative_match_score": best})
    return pd.DataFrame(rows)


def audit(corpus: pd.DataFrame, lang, min_similarity: float,
          hard_floor: float = 0.30) -> pd.DataFrame:
    rows = []
    for _, r in corpus.iterrows():
        issues, flags, sims = [], [], {}
        for role in ("a", "b"):
            nat, lat = str(r[f"name_{role}_dev"]), str(r[f"name_{role}_lat"])
            if not nat.strip() or not lat.strip():
                issues.append(f"missing_name_{role}")
                continue
            if lang.contrast == "script":
                if not lang.in_script(nat):
                    issues.append(f"no_native_script_{role}")
                if not lang.script_purity(nat):
                    issues.append(f"latin_in_native_{role}")
                if lang.in_script(lat):
                    issues.append(f"native_in_latin_{role}")
            elif nat == lat:
                issues.append(f"no_diacritic_contrast_{role}")
            sim = name_similarity(nat, lat, lang.script)
            sims[f"similarity_{role}"] = None if sim is None else round(sim, 3)
            # Two thresholds, because a single one is either too loose or too costly.
            # On the published corpus, 17 items fell below 0.45 and only 10 were real
            # errors; auto-dropping at that level discards good data. Below the hard
            # floor the two fields are almost certainly different names, so drop. In
            # between, keep the item and send it to the manual sheet.
            if sim is not None:
                if sim < hard_floor:
                    issues.append(f"name_mismatch_{role}")
                elif sim < min_similarity:
                    flags.append(f"low_similarity_{role}")
        rows.append({"fact_id": r["fact_id"], "audit_pass": int(not issues),
                     "issues": ";".join(issues), "review_flags": ";".join(flags), **sims})
    return pd.DataFrame(rows)


def build(key: str, pool_dir: Path, out_dir: Path, n_native: int, n_foreign: int,
          min_similarity: float, hard_floor: float, exclude_qids=()) -> dict:
    lang = LANGUAGES[key]
    path = pool_dir / f"pool_{key}.csv"
    if not path.exists():
        return {"key": key, "built": 0, "note": "no pool file"}

    pool = pd.read_csv(path)
    # Entities already known to be bad must not come back through a rebuild. Ten items in
    # the original corpus were found to pair two different people, three of them inside the
    # set the causal result is computed on. A larger corpus built from the same source will
    # re-select them unless they are named here, and the second discovery of a known error
    # is a far worse look than the first.
    n_excluded = 0
    if len(exclude_qids):
        wanted = {str(q).strip().upper() for q in exclude_qids}
        mask = pool["qid"].astype(str).str.upper().isin(wanted)
        n_excluded = int(mask.sum())
        if n_excluded:
            print(f"    excluded {n_excluded} known-bad entities: "
                  f"{', '.join(sorted(pool.loc[mask, 'qid'].astype(str)))}")
        pool = pool[~mask]
    parts = [pool[pool.origin == o].head(n) for o, n in
             (("native", n_native), ("foreign", n_foreign))]
    pool = pd.concat(parts, ignore_index=True)

    rows = []
    for _, r in pool.iterrows():
        if lang.contrast == "script":
            nat_a, nat_b = str(r["nativeLabel"]), str(r["nativeAlias"])
            lat_a, lat_b = str(r["enLabel"]), str(r.get("enAlias", "") or "")
            if not lat_b.strip() or lat_a.strip().lower() == lat_b.strip().lower():
                continue        # no distinct Latin form for the second name
        else:
            nat_a, nat_b = str(r["nativeLabel"]), str(r["nativeAlias"])
            lat_a, lat_b = strip_diacritics(nat_a), strip_diacritics(nat_b)
            if nat_a == lat_a and nat_b == lat_b:
                continue
        rows.append({
            "fact_id": f"{key}_{str(r['qid']).lower()}", "qid": r["qid"],
            "name_a_dev": nat_a, "name_a_lat": lat_a,
            "name_b_dev": nat_b, "name_b_lat": lat_b,
            "dev_lang": key, "origin": r["origin"],
            # provenance, as the original corpus recorded it: both second-name
            # forms come from Wikidata aliases rather than a separate property
            "name_b_dev_source": "alias", "name_b_lat_source": "alias",
            "relation_tier": "alias_observed", "relation_tier_num": 1,
            "occupation_qids": r.get("occupations", ""),
            "country_qids": r.get("countries", ""),
            "sitelinks": r.get("sitelinks", 0),
            "generated_lat": False,
        })

    corpus = pd.DataFrame(rows)
    if corpus.empty:
        return {"key": key, "built": 0, "note": "nothing survived construction"}

    a = audit(corpus, lang, min_similarity, hard_floor)
    merged = corpus.merge(a, on="fact_id")
    excluded = merged[merged.audit_pass == 0].copy()
    kept_ids = set(merged.loc[merged.audit_pass == 1, "fact_id"])
    flag_of = dict(zip(merged.fact_id, merged.review_flags.fillna("")))
    corpus = corpus[corpus.fact_id.isin(kept_ids)].reset_index(drop=True)
    corpus["review_flags"] = corpus.fact_id.map(flag_of).fillna("")
    negs = hard_negatives(corpus) if len(corpus) > 1 else pd.DataFrame()

    d = out_dir / key
    d.mkdir(parents=True, exist_ok=True)
    corpus.drop(columns=["review_flags"]).to_csv(
        d / "corpus_selected.csv", index=False, encoding="utf-8")
    negs.to_csv(d / "hard_negative_map.csv", index=False, encoding="utf-8")
    a.to_csv(d / "data_audit.csv", index=False, encoding="utf-8")
    excluded.to_csv(d / "excluded_rows.csv", index=False, encoding="utf-8")

    # A sheet for the manual pass. The automated checks cannot tell a nickname from a
    # description, and for scripts nobody on the team reads, Wikidata is being trusted
    # completely.
    flagged = corpus[corpus.review_flags.fillna("").astype(str).str.len() > 0]
    sample = corpus.head(30)
    review = pd.concat([flagged, sample]).drop_duplicates(subset=["fact_id"])[
        ["fact_id", "name_a_dev", "name_a_lat", "name_b_dev", "name_b_lat",
         "origin", "review_flags"]].copy()
    for col in ("same_person", "both_are_names", "script_correct", "notes"):
        review[col] = ""
    review.to_csv(d / "manual_review_sheet.csv", index=False, encoding="utf-8")

    mism = int(excluded.issues.str.contains("name_mismatch", na=False).sum()) if len(excluded) else 0
    n_flag = int((corpus.review_flags.fillna("").astype(str).str.len() > 0).sum())
    return {"key": key, "built": len(corpus), "flagged_for_review": n_flag,
            "native": int((corpus.origin == "native").sum()),
            "foreign": int((corpus.origin == "foreign").sum()),
            "excluded": len(excluded), "name_mismatch": mism,
            "similarity_checked": int(lang.script in COMPARABLE)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--pool-dir", type=Path, default=HERE / "out" / "pool")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "corpora")
    ap.add_argument("--native", type=int, default=150)
    ap.add_argument("--foreign", type=int, default=300)
    ap.add_argument("--min-similarity", type=float, default=0.50,
                    help="below this the item is flagged for manual review")
    ap.add_argument("--hard-floor", type=float, default=0.30,
                    help="below this the two fields are different names and the item is dropped")
    ap.add_argument("--exclude-qids", nargs="+", default=None,
                    help="QIDs to keep out of the corpus, e.g. items already found to "
                         "pair two different people")
    ap.add_argument("--exclude-file", type=Path, default=None,
                    help="file with one QID per line; # starts a comment")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys = args.languages or [p.stem.replace("pool_", "") for p in
                              sorted(args.pool_dir.glob("pool_*.csv"))]
    exclude = list(args.exclude_qids or [])
    if args.exclude_file:
        exclude += [line.strip() for line in
                    args.exclude_file.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
    if exclude:
        print(f"excluding {len(exclude)} entities by QID")
    rows = [build(k, args.pool_dir, args.out, args.native, args.foreign,
                  args.min_similarity, args.hard_floor, exclude)
            for k in keys if k in LANGUAGES]
    s = pd.DataFrame(rows)
    s.to_csv(args.out / "corpus_summary.csv", index=False)
    print(s.to_string(index=False))

    if "similarity_checked" in s.columns:
        skipped = s[s.similarity_checked == 0].key.tolist()
        if skipped:
            print(f"\nname-consistency check skipped for: {', '.join(skipped)}")
            print("  Romanisation of these scripts does not resemble the Latin name, so an")
            print("  edit-distance check would be meaningless. Rely on the manual sheet.")
    thin = s[s.get("built", 0) < 100].key.tolist() if "built" in s.columns else []
    if thin:
        print(f"\nunder 100 items: {', '.join(thin)}")


if __name__ == "__main__":
    main()

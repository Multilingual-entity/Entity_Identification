"""Audit the fixed corpus and create deterministic train/validation/test splits."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import add_common_args, ensure_run_dirs, seed_everything, sha256_file, write_json


LAT_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")

# The native field is not always Devanagari. Non-Latin corpora carry Tamil, Cyrillic,
# Han and so on, and the Latin-script corpora carry a diacritic-bearing form whose script
# is Latin by design. So the test is "contains a letter from a non-Latin script" rather
# than a check for one particular script.
#
# Two things must not count as non-Latin, or legitimate names get rejected. Latin
# Extended Additional holds Vietnamese letters such as U+1ED3 in "H\u1ED3 Ch\u00ED Minh", and
# punctuation such as the curly apostrophe in "O\u2019Flahertie" is not a script at all. So
# the check is restricted to characters whose Unicode category is a letter, and Latin
# letters are recognised across every Latin block rather than only the first two.
_LATIN_RANGES = ((0x0000, 0x024F),     # ASCII, Latin-1, Latin Extended-A and -B
                 (0x0250, 0x02AF),     # IPA extensions
                 (0x1E00, 0x1EFF))     # Latin Extended Additional, incl. Vietnamese


def has_non_latin_letter(text: str) -> bool:
    for ch in str(text):
        if not unicodedata.category(ch).startswith("L"):
            continue                    # skip marks, punctuation, digits, spaces
        if not any(lo <= ord(ch) <= hi for lo, hi in _LATIN_RANGES):
            return True
    return False


def native_is_latin_script(corpus) -> bool:
    """True when the corpus varies diacritics rather than script.

    Detected from the data: if no native field in the corpus contains a non-Latin
    character, the contrast cannot be a script contrast.
    """
    for col in ("name_a_dev", "name_b_dev"):
        if col in corpus.columns and corpus[col].astype(str).map(has_non_latin_letter).any():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, include_model=False)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("csv_qwen7b_instruct/corpus_selected.csv"),
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=Path("csv_qwen7b_instruct/hard_negative_map.csv"),
    )
    parser.add_argument("--include-flagged", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)

    corpus = pd.read_csv(args.corpus)
    negatives = pd.read_csv(args.negatives)
    required = {"fact_id", "name_a_dev", "name_a_lat", "name_b_dev", "name_b_lat"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Corpus is missing columns: {sorted(missing)}")
    corpus["fact_id"] = corpus["fact_id"].astype(str)
    negatives["fact_id"] = negatives["fact_id"].astype(str)
    negatives["negative_fact_id"] = negatives["negative_fact_id"].astype(str)

    audit = []
    corpus_ids = set(corpus["fact_id"])
    neg_map = dict(zip(negatives["fact_id"], negatives["negative_fact_id"]))
    diacritic_corpus = native_is_latin_script(corpus)
    if diacritic_corpus:
        print("Native fields are Latin script: treating this as a diacritic contrast "
              "and skipping the script-purity test.")
    for _, row in corpus.iterrows():
        issues: list[str] = []
        for role in ("a", "b"):
            dev = str(row.get(f"name_{role}_dev", "") or "")
            lat = str(row.get(f"name_{role}_lat", "") or "")
            if not dev.strip() or not lat.strip():
                issues.append(f"missing_name_{role}")
            if diacritic_corpus:
                if dev == lat:
                    issues.append(f"no_contrast_{role}")
                continue
            if not has_non_latin_letter(dev):
                issues.append(f"no_native_script_{role}")
            if LAT_RE.search(dev):
                issues.append(f"latin_in_native_{role}")
            if has_non_latin_letter(lat):
                issues.append(f"native_script_in_latin_{role}")
        fid = str(row["fact_id"])
        neg = neg_map.get(fid)
        if neg is None:
            issues.append("missing_negative")
        elif neg not in corpus_ids:
            issues.append("negative_not_in_corpus")
        elif neg == fid:
            issues.append("self_negative")
        audit.append(
            {
                "fact_id": fid,
                "audit_pass": int(not issues),
                "issues": ";".join(issues),
            }
        )
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(paths["data"] / "data_audit.csv", index=False)
    # A corpus that arrives carrying its own audit_pass and issues columns, as one does
    # when it is exported from a review sheet, would otherwise collide with the audit
    # computed here: pandas renames both sides to _x and _y and the next line fails with
    # a KeyError. The audit that matters is the one this stage just ran against these
    # negatives, so any inherited columns are dropped rather than merged around.
    inherited = [c for c in ("audit_pass", "issues") if c in corpus.columns]
    if inherited:
        print(f"  corpus already had {inherited}; replacing with this run's audit")
        corpus = corpus.drop(columns=inherited)
    corpus = corpus.merge(audit_df, on="fact_id", how="left")
    excluded = corpus.loc[corpus["audit_pass"] == 0].copy()
    excluded.to_csv(paths["data"] / "excluded_rows.csv", index=False)
    if not args.include_flagged:
        corpus = corpus.loc[corpus["audit_pass"] == 1].copy()

    retained = set(corpus["fact_id"])
    negatives = negatives.loc[
        negatives["fact_id"].isin(retained) & negatives["negative_fact_id"].isin(retained)
    ].copy()
    # If excluding a row removes someone else's negative, exclude that source fact as well.
    valid_negative_sources = set(negatives["fact_id"])
    dropped_for_negative = retained - valid_negative_sources
    if dropped_for_negative:
        corpus = corpus.loc[~corpus["fact_id"].isin(dropped_for_negative)].copy()
        retained = set(corpus["fact_id"])
        negatives = negatives.loc[
            negatives["fact_id"].isin(retained) & negatives["negative_fact_id"].isin(retained)
        ].copy()

    rng = np.random.default_rng(args.seed)
    ids = corpus["fact_id"].astype(str).to_numpy()
    shuffled = ids[rng.permutation(len(ids))]
    n_train = int(round(0.60 * len(ids)))
    n_val = int(round(0.20 * len(ids)))
    split_of = {fid: "train" for fid in shuffled[:n_train]}
    split_of.update({fid: "validation" for fid in shuffled[n_train : n_train + n_val]})
    split_of.update({fid: "test" for fid in shuffled[n_train + n_val :]})
    corpus["split"] = corpus["fact_id"].map(split_of)
    corpus = corpus.sort_values("fact_id").reset_index(drop=True)
    negatives = negatives.sort_values("fact_id").reset_index(drop=True)

    corpus.to_csv(paths["data"] / "corpus.csv", index=False)
    negatives.to_csv(paths["data"] / "hard_negative_map.csv", index=False)
    corpus[["fact_id", "split"]].to_csv(paths["data"] / "splits.csv", index=False)
    review_path = paths["data"] / "manual_review_sheet.csv"
    if not review_path.exists() or args.force:
        review = corpus[
            [
                "fact_id",
                "qid",
                "name_a_dev",
                "name_a_lat",
                "name_b_dev",
                "name_b_lat",
                "relation_tier",
                "name_b_dev_source",
                "name_b_lat_source",
            ]
        ].merge(negatives, on="fact_id", how="left")
        negative_names = corpus[["fact_id", "name_b_dev", "name_b_lat"]].rename(
            columns={
                "fact_id": "negative_fact_id",
                "name_b_dev": "negative_b_dev",
                "name_b_lat": "negative_b_lat",
            }
        )
        review = review.merge(negative_names, on="negative_fact_id", how="left")
        review["manual_positive_valid"] = ""
        review["manual_negative_valid"] = ""
        review["manual_script_valid"] = ""
        review["manual_notes"] = ""
        review.to_csv(review_path, index=False)
    manifest = {
        "source_corpus": str(args.corpus.resolve()),
        "source_negatives": str(args.negatives.resolve()),
        "source_corpus_sha256": sha256_file(args.corpus),
        "source_negatives_sha256": sha256_file(args.negatives),
        "seed": args.seed,
        "include_flagged": bool(args.include_flagged),
        "n_source": int(len(audit_df)),
        "n_audit_failed": int((audit_df["audit_pass"] == 0).sum()),
        "n_prepared": int(len(corpus)),
        "n_train": int((corpus["split"] == "train").sum()),
        "n_validation": int((corpus["split"] == "validation").sum()),
        "n_test": int((corpus["split"] == "test").sum()),
    }
    write_json(paths["data"] / "manifest.json", manifest)
    print(manifest)
    if len(excluded):
        print("Flagged rows (excluded by default):")
        print(excluded[["fact_id", "issues"]].to_string(index=False))


if __name__ == "__main__":
    main()

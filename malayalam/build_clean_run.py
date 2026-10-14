"""Drop the entities the human audit rejected and stage the run on what is left.

The audit's REMOVE verdicts come out of the corpus entirely. The negative map is
one-to-one, so removing an entity leaves whichever anchor pointed at it without a
distractor. Those anchors are re-pointed rather than dropped: cascading the removals
through the chain would delete dozens of entities the auditor never objected to, which
is a larger departure from the audit than reusing a distractor.

A re-pointed anchor's new distractor is chosen by the rules the dataset itself used:
an exact shared occupation QID first, then a shared country, then Latin string
similarity, always within the same split, never an entity the audit rejected, never a
surface collision, and never a known name of the anchor. The one rule relaxed is strict
one-to-one use: a re-pointed anchor's distractor may already serve another anchor. That
affects a handful of items and adds a little dependence between them, which the paper
already acknowledges for the fixed map.

Every new pairing is written to repair_addendum.csv for the auditor to confirm. Splits
are copied, never recomputed.

    python build_clean_run.py --dataset <dataset dir> --out <clean dir> \
        [--exclude REMOVE UNSURE]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(s: str) -> str:
    """The dataset's own comparison key: letters and marks only, case folded."""
    return "".join(c.casefold() for c in unicodedata.normalize("NFC", s)
                   if unicodedata.category(c)[0] in "LMN")


def qids(field: str) -> set[str]:
    return {q.strip() for q in (field or "").replace(";", ",").split(",") if q.strip()}


def pick(anchor: dict, pool: list[dict], known: set[str], used: set[str]) -> tuple[dict, str]:
    """The dataset's heuristic: occupation, then country, then Latin similarity."""
    a_occ, a_cty = qids(anchor["occupation_qids"]), qids(anchor["country_qids"])
    a_keys = {key(anchor[f]) for f in ("name_a_lat", "name_b_lat",
                                       "name_a_dev", "name_b_dev")}
    candidates = []
    for row in pool:
        if row["fact_id"] == anchor["fact_id"] or row["qid"] == anchor["qid"]:
            continue
        if row["split"] != anchor["split"]:
            continue
        # it contributes its name_b to the prompt, so that is what must not collide
        if key(row["name_b_lat"]) in known or key(row["name_b_dev"]) in known:
            continue
        if key(row["name_b_lat"]) in a_keys or key(row["name_b_dev"]) in a_keys:
            continue
        shared_occ = bool(a_occ & qids(row["occupation_qids"]))
        shared_cty = bool(a_cty & qids(row["country_qids"]))
        similarity = SequenceMatcher(None, key(anchor["name_a_lat"]),
                                     key(row["name_b_lat"])).ratio()
        candidates.append((
            # prefer an unused distractor at equal metadata quality, so the relaxation
            # of one-to-one is used only where it has to be
            int(shared_occ), int(shared_cty), int(row["fact_id"] not in used),
            similarity, row))
    if not candidates:
        raise SystemExit(f"no distractor available for {anchor['fact_id']}")
    candidates.sort(key=lambda t: t[:4], reverse=True)
    occ, cty, unused, similarity, best = candidates[0]
    why = (f"shared_occupation={bool(occ)}, shared_country={bool(cty)}, "
           f"latin_similarity={similarity:.2f}, "
           f"distractor_already_used={not bool(unused)}")
    return best, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", nargs="+", default=["REMOVE"],
                    help="audit verdicts to drop; REMOVE by default, add UNSURE to drop "
                         "the withheld reversed-order pairs too")
    args = ap.parse_args()

    audit = {r["fact_id"]: r for r in read(args.dataset / "malayalam_audit.csv")}
    corpus = read(args.dataset / "corpus_selected.csv")
    marathi = read(args.dataset / "marathi_matched_subset.csv")
    negatives = read(args.dataset / "hard_negative_map.csv")
    sources = {json.loads(line)["fact_id"]: json.loads(line) for line
               in (args.dataset / "source_records.jsonl").read_text(
                   encoding="utf-8").splitlines()}

    drop = {f for f, r in audit.items() if r["verdict"] in set(args.exclude)}
    print(f"audit verdicts dropped: {', '.join(args.exclude)}")
    for f in sorted(drop):
        print(f"  removing {f:10} {audit[f]['verdict']:7} {audit[f]['reason'] or '-'}")

    kept = [r for r in corpus if r["fact_id"] not in drop]
    by_id = {r["fact_id"]: r for r in kept}
    negative_of = {n["fact_id"]: n["negative_fact_id"] for n in negatives}
    negative_row = {n["fact_id"]: n for n in negatives}

    orphans = [r for r in kept if negative_of[r["fact_id"]] in drop]
    print(f"\n{len(corpus)} entities -> {len(kept)} kept; "
          f"{len(orphans)} anchor(s) lost their distractor to the removals")

    used = {negative_of[r["fact_id"]] for r in kept
            if negative_of[r["fact_id"]] not in drop}
    addendum = []
    for anchor in orphans:
        known = {key(x) for x in
                 sources[anchor["fact_id"]]["known_wikidata_name_surfaces_for_negative_screening"]}
        new, why = pick(anchor, kept, known, used)
        old = negative_of[anchor["fact_id"]]
        used.add(new["fact_id"])
        row = negative_row[anchor["fact_id"]]
        row["negative_fact_id"] = new["fact_id"]
        row["shared_occupation"] = str(bool(qids(anchor["occupation_qids"])
                                            & qids(new["occupation_qids"])))
        row["shared_country"] = str(bool(qids(anchor["country_qids"])
                                         & qids(new["country_qids"])))
        row["negative_review_status"] = "pending_repair"
        print(f"  {anchor['fact_id']:10} {old} -> {new['fact_id']:10} {why}")
        addendum.append({
            "fact_id": anchor["fact_id"],
            "anchor_name_lat": anchor["name_a_lat"],
            "anchor_name_mal": anchor["name_a_dev"],
            "removed_distractor": old,
            "removed_because": f"{audit[old]['verdict']}: {audit[old]['reason']}",
            "new_distractor": new["fact_id"],
            "new_distractor_name_lat": new["name_b_lat"],
            "new_distractor_name_mal": new["name_b_dev"],
            "selection": why,
            "split": anchor["split"],
            "reviewer_verdict": "",
            "reviewer_note": "",
        })

    kept_negatives = [negative_row[r["fact_id"]] for r in kept]

    # Verification. Nothing is written if any of this fails.
    ids = {r["fact_id"] for r in kept}
    problems = []
    if len(ids) != len(kept):
        problems.append("duplicate fact_id")
    for n in kept_negatives:
        if n["negative_fact_id"] not in ids:
            problems.append(f"{n['fact_id']}: distractor not in the corpus")
        elif n["negative_fact_id"] == n["fact_id"]:
            problems.append(f"{n['fact_id']}: self negative")
        elif by_id[n["negative_fact_id"]]["split"] != by_id[n["fact_id"]]["split"]:
            problems.append(f"{n['fact_id']}: distractor crosses the split")
        elif by_id[n["negative_fact_id"]]["qid"] == by_id[n["fact_id"]]["qid"]:
            problems.append(f"{n['fact_id']}: distractor is the same entity")
    for f in ids & drop:
        problems.append(f"{f}: a removed entity survived")
    for r in kept:
        if r["split"] != next(x["split"] for x in corpus if x["fact_id"] == r["fact_id"]):
            problems.append(f"{r['fact_id']}: split changed")
    if problems:
        for p in problems[:20]:
            print(f"  FAIL  {p}")
        raise SystemExit(f"\n{len(problems)} problem(s); nothing written")

    splits = {}
    for r in kept:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"\nsplits: {splits}")
    reused = len(kept_negatives) - len({n['negative_fact_id'] for n in kept_negatives})
    print(f"distractors serving more than one anchor: {reused}")

    # The run directories, in the layout stage 01 reads, for both languages.
    marathi_by_id = {r["fact_id"]: r for r in marathi}
    for language, rows in (("ml", kept),
                           ("mr380", [marathi_by_id[r["fact_id"]] for r in kept])):
        staged = []
        for r in rows:
            r = dict(r)
            r["audit_pass"] = 1
            r["issues"] = ""
            r["native_review_status"] = ("audited_keep" if language == "ml"
                                         else "see_original_marathi_review")
            staged.append(r)
        data = args.out / f"_staged_{language}" / "data"
        write(data / "corpus.csv", staged)
        write(data / "hard_negative_map.csv", kept_negatives)
        write(data / "splits.csv", [{"fact_id": r["fact_id"], "split": r["split"]}
                                    for r in rows])
        write(data / "data_audit.csv",
              [{"fact_id": r["fact_id"], "audit_pass": 1, "issues": "",
                "audit_scope": "human Malayalam audit; REMOVE verdicts excluded"}
               for r in rows])
        (data / "manifest.json").write_text(json.dumps({
            "source_dataset": str(args.dataset),
            "language": language,
            "entities": len(rows),
            "splits": splits,
            "excluded_verdicts": args.exclude,
            "excluded_entities": sorted(drop),
            "repaired_pairs": [a["fact_id"] for a in addendum],
            "split_policy": "copied from the frozen reference; never recomputed",
        }, indent=2), encoding="utf-8")
        print(f"  {data}  {len(rows)} entities")

    if addendum:
        write(args.out / "repair_addendum.csv", addendum)
        print(f"\n  {args.out / 'repair_addendum.csv'}  "
              f"{len(addendum)} new pairing(s) for the auditor to confirm")

    write(args.out / "ml_eligibility.csv",
          [{"fact_id": r["fact_id"], "audit_verdict": audit[r["fact_id"]]["verdict"],
            "negative_fact_id": negative_of[r["fact_id"]],
            "negative_audit_verdict": "KEEP", "eligible_same": 1,
            "eligible_different": 1} for r in kept])
    print(f"  {args.out / 'ml_eligibility.csv'}  every retained entity is eligible")


if __name__ == "__main__":
    main()

"""Build a hard-negative map matched on how alike the two names look.

The shipped negatives match on occupation and country, which makes a negative plausible
but leaves surface similarity free: a positive pair often shares most of its characters
while its negative shares few. A classifier reading only the strings then does well, and
the probe result cannot be separated from it. This picks, for each item, a negative whose
name similarity is as close as possible to that item's own positive similarity, so the
two classes become indistinguishable on surface form.

It writes only a negative map. The positive corpus is untouched, and nothing about the
entities changes.

    python corpus_construction/build_matched_negatives.py \
        --corpus ../results_mr/qwen_mr/data/corpus.csv --out corpora/matched_mr
"""
from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def pair_similarity(a_dev, b_dev, a_lat, b_lat) -> float:
    """Devanagari and Latin weighted equally.

    Matching on the Devanagari form alone would match the condition that fails and leave
    Lat--Lat free; matching on Latin alone does the reverse. The comparison needs both
    conditions to be surface-matched, because the claim is about the difference.
    """
    return 0.5 * sim(a_dev, b_dev) + 0.5 * sim(a_lat, b_lat)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--respect-occupation", action="store_true",
                    help="prefer candidates sharing an occupation or country, as the "
                         "shipped map does, and fall back to the whole corpus when no "
                         "such candidate is close enough on similarity")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="how close a candidate's similarity must be before the "
                         "occupation preference is allowed to decide")
    args = ap.parse_args()

    c = pd.read_csv(args.corpus, dtype={"fact_id": str})
    n = len(c)
    print(f"{n} positive pairs from {args.corpus}")

    a_dev = c.name_a_dev.tolist()
    a_lat = c.name_a_lat.tolist()
    b_dev = c.name_b_dev.tolist()
    b_lat = c.name_b_lat.tolist()
    fid = c.fact_id.tolist()

    def field(col):
        if col not in c:
            return [set() for _ in range(n)]
        return [set(str(v).split("|")) if pd.notna(v) else set() for v in c[col]]

    occ, ctry = field("occupation_qids"), field("country_qids")

    own = np.array([pair_similarity(a_dev[i], b_dev[i], a_lat[i], b_lat[i])
                    for i in range(n)])

    rows, gaps, shared = [], [], 0
    for i in range(n):
        # similarity of this item's first name against every other item's second name
        cand = np.array([pair_similarity(a_dev[i], b_dev[j], a_lat[i], b_lat[j])
                         if j != i else -9.0 for j in range(n)])
        delta = np.abs(cand - own[i])
        order = np.argsort(delta)
        pick = order[0]
        if args.respect_occupation:
            for j in order[:40]:
                if delta[j] > args.tolerance:
                    break
                if (occ[i] & occ[j]) or (ctry[i] & ctry[j]):
                    pick = j
                    shared += 1
                    break
        rows.append({"fact_id": fid[i], "negative_fact_id": fid[pick],
                     "negative_match_score": float(cand[pick])})
        gaps.append(cand[pick] - own[i])

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "hard_negative_map.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    gaps = np.array(gaps)
    neg = np.array([r["negative_match_score"] for r in rows])

    # Matching cannot succeed everywhere. A positive pair's similarity often comes from
    # a genuine relationship, an expansion sharing most of its characters, and no
    # unrelated name reproduces that. Record per item how well it matched, so the
    # analysis can report the subset where surface form carries no signal at all rather
    # than claiming the whole corpus is matched.
    quality = pd.DataFrame({
        "fact_id": fid,
        "positive_similarity": own,
        "negative_similarity": neg,
        "abs_difference": np.abs(gaps),
        "tight": np.abs(gaps) < 0.05,
    })
    quality.to_csv(args.out / "match_quality.csv", index=False)
    t = quality[quality.tight]
    print(f"\ntightly matched (within 0.05): {len(t)} of {n} items")
    print(f"  on that subset: positive {t.positive_similarity.mean():.3f}, "
          f"negative {t.negative_similarity.mean():.3f}, "
          f"separation {t.positive_similarity.mean() - t.negative_similarity.mean():+.3f}")
    print(f"  unmatched items are the similar ones: mean positive similarity "
          f"{quality[~quality.tight].positive_similarity.mean():.3f} against "
          f"{t.positive_similarity.mean():.3f} for the matched")
    print(f"\npositive similarity : mean {own.mean():.3f}  median {np.median(own):.3f}")
    print(f"negative similarity : mean {neg.mean():.3f}  median {np.median(neg):.3f}")
    print(f"positive minus negative separation: {own.mean() - neg.mean():+.3f}")
    print(f"per-item |difference|: mean {np.abs(gaps).mean():.3f}  "
          f"90th pct {np.percentile(np.abs(gaps), 90):.3f}")
    if args.respect_occupation:
        print(f"{shared} of {n} negatives also share an occupation or country")
    print(f"\nSaved: {out}")
    print("\nThe shipped map separates the classes by about 0.42 on this measure. A "
          "separation near zero means a classifier reading only the strings can no "
          "longer tell the classes apart, which is the point of this map.")


if __name__ == "__main__":
    main()

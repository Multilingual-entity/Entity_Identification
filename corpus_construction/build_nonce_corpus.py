r"""Build a string-comparison task with no entity knowledge in it.

The rank-one result could mean the sensitivity direction is an identity-processing
direction, or a general string-comparison direction. Nothing in the existing corpus can
tell those apart, because every item there is both a string comparison and an entity
lookup.

This makes a corpus where only the first is true. Nonce people are assembled from name
parts attested in the real corpus, recombined so the person does not exist: the parts are
real Devanagari and Latin forms that correspond to each other, so the strings look and
tokenize like the real thing, but no amount of entity knowledge helps. A positive pair is
two forms of the same nonce name related by a rule the real corpus also contains --
dropping a middle part, initialising, adding an honorific. A negative pair applies the
same rule to a different nonce name.

Answering therefore requires comparing strings and nothing else. Run the same analysis on
it as on the real corpus: if the sensitivity direction transfers comparably, it is a
string-comparison direction and the identity framing has to go. If it transfers much
less, that is evidence for identity specificity that the matched-negative corpus cannot
give on its own.

Emits the column layout 00_prepare_data.py consumes, so the pipeline runs unchanged.

    python corpus_construction/build_nonce_corpus.py \
        --source ../results_mr/qwen_mr/data/corpus.csv --out out/nonce_mr --n 600
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

HONORIFICS = [("श्री", "Shri"), ("डॉ", "Dr"), ("प्रा", "Prof")]


def parts(dev: str, lat: str) -> list[tuple[str, str]] | None:
    """Positionally paired (Devanagari, Latin) name parts, or None if they disagree."""
    d, l = str(dev).split(), str(lat).split()
    return list(zip(d, l)) if len(d) == len(l) >= 2 else None


def harvest(corpus: pd.DataFrame) -> tuple[list, list]:
    """Attested part pairs, split into leading and trailing positions.

    Keeping the position preserves the shape of a real name: parts that appear first in
    attested names stay first, so the nonce names are not obviously scrambled.
    """
    lead, trail = [], []
    for _, row in corpus.iterrows():
        for dev_col, lat_col in (("name_a_dev", "name_a_lat"),
                                 ("name_b_dev", "name_b_lat")):
            p = parts(row[dev_col], row[lat_col])
            if not p:
                continue
            lead.append(p[0])
            trail.append(p[-1])
    dedup = lambda xs: sorted({x for x in xs})
    return dedup(lead), dedup(trail)


def applies(rule: str, n_parts: int) -> bool:
    """Whether a rule can actually change a name of this length.

    Without this check variant() falls through to reversal, and the item is stored
    under the rule that was asked for rather than the one that ran.
    """
    if rule == "drop_middle":
        return n_parts >= 3
    if rule == "reorder":
        return n_parts >= 2
    return True


def variant(name: list[tuple[str, str]], rule: str) -> list[tuple[str, str]]:
    """A second form of the same nonce name, by a rule the real corpus also contains."""
    if rule == "drop_middle" and len(name) >= 3:
        return [name[0], name[-1]]
    if rule == "initial":
        first = (name[0][0][0], name[0][1][0] + ".")
        return [first] + name[1:]
    if rule == "honorific":
        h = HONORIFICS[len(name) % len(HONORIFICS)]
        return [h] + name
    return name[::-1] if len(name) > 1 else name


def render(p: list[tuple[str, str]]) -> tuple[str, str]:
    return " ".join(d for d, _ in p), " ".join(l for _, l in p)


def overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="an existing corpus.csv, used only as a source of name parts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--dev-lang", default="mr")
    ap.add_argument("--allow-initial-rule", action="store_true",
                    help="include the character-level initialising rule. Off by "
                         "default: parts come from attested pairs and some are loose "
                         "transliterations, so the first character of the Devanagari "
                         "need not correspond to the first character of the Latin. "
                         "The corpus pairs अनास्तासिया with Nastia, which initialises "
                         "to अ against N and makes the two scripts different tasks.")
    ap.add_argument("--negatives", default="random", choices=["random", "matched"],
                    help="how to pick the different-person partner. 'random' mirrors "
                         "the real corpus, which matches on occupation and country but "
                         "not on name similarity, and keeps the string separation "
                         "comparable so behavioural results can be compared. 'matched' "
                         "picks a partner whose overlap equals the positive's, which "
                         "makes the task much harder on the string axis and is only "
                         "appropriate for the direction test.")
    ap.add_argument("--surplus", type=float, default=3.0,
                    help="generate this multiple of --n, then subsample so the positive "
                         "overlap distribution matches the source corpus. Matching it "
                         "matters: if the nonce task were easier, a direction that "
                         "transfers less would be ambiguous between an easier task and "
                         "a direction specific to identity.")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    source = pd.read_csv(args.source)
    lead, trail = harvest(source)
    print(f"harvested {len(lead)} leading and {len(trail)} trailing attested part pairs")
    if len(lead) < 20 or len(trail) < 20:
        raise SystemExit("too few parts; is --source a corpus.csv?")

    rng = np.random.default_rng(args.seed)
    rules = ["drop_middle", "honorific", "reorder"]
    if args.allow_initial_rule:
        rules.append("initial")
    want = int(args.n * max(1.0, args.surplus))
    rows, seen = [], set()
    attempts = 0
    while len(rows) < want and attempts < want * 60:
        attempts += 1
        n_parts = int(rng.choice([2, 3], p=[0.7, 0.3]))
        name = [lead[rng.integers(len(lead))]]
        while len(name) < n_parts:
            name.append(trail[rng.integers(len(trail))])
        a_dev, a_lat = render(name)
        # A recombination that reproduces a real name is not nonce; drop it.
        if (a_dev, a_lat) in seen or ((source.name_a_dev == a_dev).any()
                                      or (source.name_b_dev == a_dev).any()):
            continue
        usable = [r for r in rules if applies(r, len(name))]
        if not usable:
            continue
        rule = usable[int(rng.integers(len(usable)))]
        v = variant(name, rule)
        b_dev, b_lat = render(v)
        if b_dev == a_dev:
            continue
        seen.add((a_dev, a_lat))
        fid = "n" + hashlib.sha256(f"{a_dev}|{b_dev}".encode()).hexdigest()[:10]
        rows.append(dict(fact_id=fid, qid=fid, name_a_dev=a_dev, name_a_lat=a_lat,
                         name_b_dev=b_dev, name_b_lat=b_lat, dev_lang=args.dev_lang,
                         relation_tier=rule, relation_tier_num=rules.index(rule) + 1,
                         name_b_dev_source="nonce", name_b_lat_source="nonce",
                         generated_lat=0, occupation_qids="[]", country_qids="[]",
                         review_method="generated", corpus_batch="nonce",
                         origin="nonce", audit_pass=1, issues="",
                         overlap_dev=round(overlap(a_dev, b_dev), 4),
                         overlap_lat=round(overlap(a_lat, b_lat), 4)))

    corpus = pd.DataFrame(rows)
    print(f"generated {len(corpus)} candidates")

    # Subsample so the positive-overlap distribution follows the source corpus.
    src_overlap = np.array([overlap(str(a), str(b)) for a, b
                            in zip(source.name_a_dev, source.name_b_dev)])
    if len(corpus) > args.n:
        edges = np.quantile(src_overlap, np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        want_per_bin = np.diff(np.round(np.linspace(0, args.n, 11))).astype(int)
        picked = []
        for k in range(10):
            in_bin = corpus.index[(corpus.overlap_dev > edges[k])
                                  & (corpus.overlap_dev <= edges[k + 1])].to_numpy()
            take = min(len(in_bin), want_per_bin[k])
            if take:
                picked += list(rng.choice(in_bin, take, replace=False))
        short = args.n - len(picked)
        if short > 0:
            rest = corpus.index.difference(picked).to_numpy()
            picked += list(rng.choice(rest, min(short, len(rest)), replace=False))
        corpus = corpus.loc[picked].reset_index(drop=True)
    if len(corpus) < args.n:
        print(f"warning: produced {len(corpus)} of {args.n} requested")
    print(f"source positive overlap: mean {src_overlap.mean():.3f}, "
          f"median {np.median(src_overlap):.3f}")

    # Splits, deterministic and in the same proportions as the real corpus.
    order = rng.permutation(len(corpus))
    n_train, n_val = int(len(corpus) * 0.6), int(len(corpus) * 0.2)
    split = np.empty(len(corpus), dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_val]] = "validation"
    split[order[n_train + n_val:]] = "test"
    corpus["split"] = split

    # Negatives matched on how alike the pair is, so the negative is as hard as the
    # positive on exactly the cue the task leaves available.
    target = corpus.overlap_dev.to_numpy()
    names_b = corpus.name_b_dev.to_numpy()
    neg_rows = []
    for i, row in enumerate(corpus.itertuples()):
        cand = np.array([j for j in range(len(corpus)) if j != i])
        if args.negatives == "matched":
            sims = np.array([overlap(row.name_a_dev, names_b[j]) for j in cand])
            best = int(np.argmin(np.abs(sims - target[i])))
            pick, score = cand[best], float(abs(sims[best] - target[i]))
        else:
            pick = int(rng.choice(cand))
            score = float(abs(overlap(row.name_a_dev, names_b[pick]) - target[i]))
        neg_rows.append(dict(fact_id=row.fact_id,
                             negative_fact_id=corpus.fact_id.iloc[pick],
                             negative_match_score=round(score, 4)))
    negatives = pd.DataFrame(neg_rows)

    # The separation a string classifier can exploit, which has to be comparable with
    # the real corpus for a behavioural comparison to mean anything.
    by = corpus.set_index("fact_id")
    neg_overlap = np.array([overlap(by.loc[r.fact_id].name_a_dev,
                                    by.loc[r.negative_fact_id].name_b_dev)
                            for r in negatives.itertuples()])
    separation = float(corpus.overlap_dev.mean() - neg_overlap.mean())

    args.out.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.out / "corpus_selected.csv", index=False)
    negatives.to_csv(args.out / "hard_negative_map.csv", index=False)

    print(f"\n{len(corpus)} nonce items -> {args.out}")
    print("  splits:", corpus.split.value_counts().to_dict())
    print("  rules :", corpus.relation_tier.value_counts().to_dict())
    print(f"  positive overlap, Devanagari: mean {corpus.overlap_dev.mean():.3f}, "
          f"median {corpus.overlap_dev.median():.3f}")
    print(f"  negative mode: {args.negatives}, "
          f"mean negative overlap {neg_overlap.mean():.3f}")
    print(f"  positive minus negative separation: {separation:.3f}")
    print("  the real Marathi corpus separates by 0.423; a much smaller separation "
          "means\n  a harder string task, which would confound a behavioural "
          "comparison")
    gap = abs(corpus.overlap_dev.mean() - src_overlap.mean())
    print(f"\noverlap gap against the source corpus: {gap:.3f}")
    if gap > 0.03:
        print("That is large enough to confound the transfer comparison with difficulty.")
        print("Raise --surplus so the subsampler has more candidates to choose from.")


if __name__ == "__main__":
    main()

"""How large is the shared core if you do not insist on all languages at once?

00_coverage.py reports the intersection across every viable language, which is the
strictest possible reading and came back at 56. Adding languages shrinks an intersection
monotonically, so the useful question is where the curve falls off. This walks languages
in coverage order and reports the intersection at each step, for both criteria:

  label      the entity has a name in that script
  pair       the entity has a name AND a second attested form in that script,
             which is what an identity pair actually requires

The pair criterion is the binding one and is much stricter. Runs entirely from the
Wikidata cache written by 00_coverage.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from languages import LANGUAGES
from wikidata import labels

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=HERE.parent / "archive" / "299_fact" / "qwen" / "data" / "corpus.csv")
    ap.add_argument("--coverage", type=Path, default=HERE / "out" / "coverage.csv")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()

    corpus = pd.read_csv(args.corpus)
    qids = corpus["qid"].astype(str).tolist()
    cov = pd.read_csv(args.coverage).sort_values("script_pure", ascending=False)

    langs = sorted({l.wikidata_lang for l in LANGUAGES.values()} | {"en"})
    data = labels(qids, langs)                      # served from cache

    has_label, has_pair = {}, {}
    for lang in LANGUAGES.values():
        lab, pair = set(), set()
        for qid in qids:
            ent = data.get(qid)
            if not ent:
                continue
            name = ent["labels"].get(lang.wikidata_lang)
            if not name:
                continue
            if lang.contrast == "script" and not (
                    lang.in_script(name) and lang.script_purity(name)):
                continue
            lab.add(qid)
            if ent["aliases"].get(lang.wikidata_lang):
                pair.add(qid)
        has_label[lang.key], has_pair[lang.key] = lab, pair

    order = [k for k in cov.key.tolist() if k in has_label]
    rows, running_lab, running_pair = [], None, None
    for i, k in enumerate(order, 1):
        running_lab = has_label[k] if running_lab is None else running_lab & has_label[k]
        running_pair = has_pair[k] if running_pair is None else running_pair & has_pair[k]
        rows.append({"n_languages": i, "added": k,
                     "own_labels": len(has_label[k]), "own_pairs": len(has_pair[k]),
                     "core_labels": len(running_lab), "core_pairs": len(running_pair)})
    tiers = pd.DataFrame(rows)
    tiers.to_csv(args.out / "core_tiers.csv", index=False)
    print(tiers.to_string(index=False))

    print("\nA usable core needs enough items to survive the knowledge gate.")
    print("On the Marathi run 299 items yielded 125 script-sensitive and 39 patchable.")
    for threshold, label in ((200, "comfortable"), (150, "workable"), (100, "minimum")):
        ok = tiers[tiers.core_pairs >= threshold]
        if len(ok):
            row = ok.iloc[-1]
            names = ", ".join(order[: int(row.n_languages)])
            print(f"\n  {label} (>= {threshold} pairs): {int(row.n_languages)} languages, "
                  f"{int(row.core_pairs)} items")
            print(f"    {names}")
        else:
            print(f"\n  {label} (>= {threshold} pairs): not reachable by any subset")

    best = tiers[tiers.core_pairs >= 150]
    if len(best):
        n = int(best.iloc[-1].n_languages)
        keep = order[:n]
        core = None
        for k in keep:
            core = has_pair[k] if core is None else core & has_pair[k]
        pd.DataFrame({"qid": sorted(core)}).to_csv(args.out / "core_pairs_qids.csv", index=False)
        pd.DataFrame({"lang": keep}).to_csv(args.out / "core_languages.csv", index=False)
        print(f"\nwritten: core_pairs_qids.csv ({len(core)} entities), core_languages.csv")

    thin = [k for k in order if len(has_pair[k]) < 100]
    if thin:
        print(f"\nunder 100 usable pairs even alone, needs its own native set: {', '.join(thin)}")


if __name__ == "__main__":
    main()

"""Freeze everything behind the current results into one self-contained folder.

The corpus is about to grow from 299 facts to roughly 800, which reselects the splits, the
causal gate and every tuned hyperparameter. If the larger run is not finished, the 299-fact
results are what the paper stands on, and they have to remain complete and readable without
anyone reconstructing which corpus, which model revision and which code produced them.

Copies rather than moves, deliberately. The paper and the notebooks refer to the existing
paths, and moving the directories would break them for no gain: the whole archive is about
200 MB, which is not worth risking a broken reference over.

    python archive_run.py --to ../../archive/299_fact
    python archive_run.py --to ../../archive/299_fact --dry-run

What it gathers:
    the three run directories, renamed by model
    the exact corpus and negative files that fed them
    a provenance record per run: corpus hashes, seed, splits, model revision, table shapes
    the paper source as it stands at the moment of freezing
    a README naming what the archive backs and what it cannot do

What it cannot gather: the hidden-state arrays. Those are several GB per model and live on
the machine the runs happened on, not in these directories. Without them the archived run
can be re-analysed at the level of its tables but no new probe or patch can be computed
from it. If the 299-fact results might need extending later, keep the state arrays on the
run machine until the paper is finished.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# name in the archive -> where it lives now
RUNS = {
    "qwen": ROOT / "shareable_results",
    "gemma": ROOT / "gemma_a100_results",
    "llama": ROOT / "llama_a100_results",
}
# The earlier notebook run, which several tables and about fourteen passages in the paper
# are drawn from and label as such. It is a separate run from the pipeline results, with its
# own file naming (h0 to h4), and the paper cannot be reconstructed without it. Leaving it
# out was the gap this archive originally had.
EARLIER_RUN = {
    "qwen": HERE / "csv_qwen7b_instruct",
    "gemma": HERE / "csv_gemma2b_it",
}

# The executed notebooks and state arrays behind that earlier run. my_runs/ holds the
# outputs of two different papers side by side; only the entity study belongs here, so the
# representation caches for the other paper are deliberately left where they are.
#
# The notebooks are the provenance of the earlier numbers: each carries its outputs inline,
# so a figure quoted in the paper can be traced to the cell that produced it. The two zips
# are the only copy of that run's hidden states, which makes them the one part of this
# archive that could not be regenerated from anything else.
MY_RUNS = HERE.parent.parent / "my_runs"
EARLIER_NOTEBOOKS = ["run_entity.ipynb", "run_gemma_entity.ipynb", "run_addenda.ipynb",
                     "run_bridge.ipynb", "run_attn_mlp.ipynb"]
EARLIER_STATES = ["entity_results_archive.zip", "entity_results_gemma2bit_archive.zip"]

CORPUS_FILES = [
    HERE / "csv_qwen7b_instruct" / "corpus_selected.csv",
    HERE / "csv_qwen7b_instruct" / "hard_negative_map.csv",
]
# The tex loads \bibliography{references_v3}, not references.bib, so archiving the obvious
# name was not enough: the document would not have built. Every bibliography and every
# style file in the paper folder is taken instead of a chosen list, because they are small
# and guessing which one the document depends on is exactly the mistake that was made.
PAPER_FILES = ["paperB_neurips.tex"]
PAPER_PATTERNS = ["*.bib", "*.sty", "*.bst", "*.cls"]


README = """# {label}

Frozen on {stamp}.

This folder holds everything behind the {n_facts}-fact results: the run directories, the
corpus and negatives that fed them, a provenance record for each, and the paper source as
it stood when the archive was made.

## Why it exists

The corpus was rescaled after this point. That changes the train/validation/test splits,
which items pass the causal gate, and the tuned steering layer and scale, so results from
the larger corpus are not row-comparable with these. Both sets are valid; they are answers
about different corpora. This archive keeps the earlier one complete and interpretable on
its own.

## What each run is

{runs}

## Two different runs, and the paper uses both

`qwen/`, `gemma/` and `llama/` are the pipeline runs. `earlier_notebook_run/` is an earlier,
separate analysis of the same corpus, with its own file naming (h0 to h4). The paper draws
several tables and about fourteen passages from it and marks them as such. Both are needed
to reconstruct the paper.

## Re-reading it

The provenance files record the corpus hash, the seed, the split sizes, the model
identifier and revision, and the row count and checksum of every table. To compare against
a later run:

    python provenance.py --compare <this>/qwen <newer run dir>

## What is missing, and it matters

The pipeline runs' hidden-state arrays are not here. They are several GB per model and live
on the machine the runs happened on, so those three runs can be re-read and re-aggregated
at the level of their tables, but nothing new can be probed or patched from them without
re-extracting states, which means re-running the model over the corpus.

The earlier notebook run does have its states, in
`earlier_notebook_run/states/`: the two zips hold the same CSVs as
`earlier_notebook_run/qwen` and `/gemma` plus `h1_entity_states_float16.npy` and
`h2_relation_states_float16.npy`. That makes the earlier run the only one in this archive
that can still be extended rather than merely re-read.

Not archived, and belonging to the other paper: the representation caches in `my_runs/`
(`reps_*.npy`, about 3.5 GB) are the Paper A representation study, not this work.

The manual review of the corpus was never completed. Ten items are known to pair two
different people, listed in multilingual/known_bad_qids.txt, and three of them fall inside
the set the causal result is computed on. That is a known limitation of these numbers.
"""


def copy_tree(source: Path, target: Path, dry: bool) -> int:
    if not source.exists():
        print(f"  missing, skipped: {source}")
        return 0
    size = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    print(f"  {source.name:<28} -> {target.name:<10} {size / 1e6:8.1f} MB")
    if not dry:
        if target.exists():
            print(f"    already present, left alone")
            return size
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    return size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=Path, required=True, help="archive directory to create")
    ap.add_argument("--label", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--update", action="store_true",
                    help="add anything missing to an archive that already exists, and "
                         "rewrite its README, instead of refusing because the directory "
                         "is there")
    ap.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    args = ap.parse_args()

    target = args.to
    label = args.label or f"{target.name} archive"
    dry = args.dry_run
    if dry:
        print("DRY RUN: nothing will be written\n")
    elif target.exists() and not args.update:
        raise SystemExit(f"{target} already exists; pass --update to add what is missing, "
                         f"or choose another --to")
    if not dry:
        target.mkdir(parents=True, exist_ok=True)

    total = 0
    print("run directories")
    for name, source in RUNS.items():
        total += copy_tree(source, target / name, dry)

    print("\nearlier notebook run")
    for name, source in EARLIER_RUN.items():
        total += copy_tree(source, target / "earlier_notebook_run" / name, dry)
    for folder, names in (("notebooks", EARLIER_NOTEBOOKS), ("states", EARLIER_STATES)):
        for name in names:
            source = MY_RUNS / name
            if not source.exists():
                print(f"  missing, skipped: {source}")
                continue
            destination = target / "earlier_notebook_run" / folder / name
            print(f"  {name:<40} {source.stat().st_size / 1e6:8.1f} MB")
            total += source.stat().st_size
            if not dry:
                if destination.exists():
                    print("    already present, left alone")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    print("\ncorpus")
    if not dry:
        (target / "corpus").mkdir(exist_ok=True)
    for path in CORPUS_FILES:
        if path.exists():
            print(f"  {path.name}")
            if not dry:
                shutil.copy2(path, target / "corpus" / path.name)
        else:
            print(f"  missing, skipped: {path}")

    print("\npaper")
    if not dry:
        (target / "paper").mkdir(exist_ok=True)
    wanted = [args.paper_dir / name for name in PAPER_FILES]
    for pattern in PAPER_PATTERNS:
        wanted += sorted(args.paper_dir.glob(pattern))
    seen = set()
    for path in wanted:
        if not path.exists() or path.name in seen:
            continue
        seen.add(path.name)
        destination = target / "paper" / path.name
        print(f"  {path.name}" + ("  (already present)" if destination.exists() else ""))
        if not dry and not destination.exists():
            shutil.copy2(path, destination)

    # Provenance last, so it describes the copies rather than the originals.
    print("\nprovenance")
    n_facts = "299"
    summaries = []
    if not dry:
        import provenance as prov

        (target / "provenance").mkdir(exist_ok=True)
        for name in RUNS:
            run = target / name
            if not run.exists():
                continue
            record = prov.snapshot(run, name)
            manifest = record.get("manifest") or {}
            model = record.get("model") or {}
            own = str(manifest.get("n_prepared", n_facts))
            record["label"] = f"{name} {own}-fact"
            n_facts = own
            (target / "provenance" / f"{name}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            summaries.append(
                f"- **{name}** — {model.get('model_id', 'model not recorded')}"
                f"{' @ ' + model.get('model_revision') if model.get('model_revision') else ''}, "
                f"{model.get('n_layers', '?')} layers, "
                f"{manifest.get('n_prepared', '?')} facts, seed {manifest.get('seed', '?')}")
            print(f"  {name}.json")

    if not dry:
        (target / "README.md").write_text(
            README.format(label=label,
                          stamp=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                          n_facts=n_facts,
                          runs="\n".join(summaries) or "- (none found)"),
            encoding="utf-8")
        print("  README.md")

    print(f"\ntotal {total / 1e6:.0f} MB"
          + ("" if dry else f" written to {target.resolve()}"))
    if dry:
        print("rerun without --dry-run to write it")
    else:
        print("\nThe originals are untouched. Nothing that refers to them has moved.")


if __name__ == "__main__":
    main()

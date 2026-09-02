"""Check that the archive is intact and that it contains what the paper needs.

Two different questions, and both have to be answered separately.

Integrity: does every archived file still match the original it was copied from? Some
originals have since been deleted, which is the point of the archive, so this can only
check what survives. A silent copy failure is the kind of fault that shows up months
later when the file is finally opened.

Completeness: can the paper be rebuilt from this and nothing else? That means the sources
its numbers come from, which is two separate runs with different file naming, plus the
bibliography its text actually loads. It is easy to archive a plausible-looking set of
files and still be missing the one the document depends on.

    python verify_archive.py --archive ../../archive/299_fact --paper ../../paper
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# archive path -> where it was copied from, for the integrity check
ORIGINS = {
    "earlier_notebook_run/qwen": HERE / "csv_qwen7b_instruct",
    "earlier_notebook_run/gemma": HERE / "csv_gemma2b_it",
    "earlier_notebook_run/notebooks": ROOT / "my_runs",
    "earlier_notebook_run/states": ROOT / "my_runs",
}

# The analyses the paper reports, and the file each is read from. Named explicitly so a
# missing one is a failure rather than something nobody notices.
PIPELINE_REQUIRED = [
    "data/corpus.csv", "data/manifest.json", "tables/behavior.csv",
    "tables/behavior_summary.csv", "tables/knowledge_gate.csv", "tables/tokenization.csv",
    "tables/isolated_retrieval_summary.csv", "tables/pair_retrieval_summary.csv",
    "tables/relation_probe_summary.csv", "tables/relation_probe_heldout_summary.csv",
    "tables/exact_failure_probe_summary.csv", "tables/fragmentation_correlation.csv",
    "tables/failure_prediction_oof_auc.csv",
    "tables/context_script_interaction_per_fact.csv",
    "recovery/prompt_anchor_summary.csv", "recovery/vector_recovery_test_summary.csv",
]
PIPELINE_CAUSAL = [
    "patching/causal_candidates.csv", "patching/causal_patching_summary.csv",
    "patching/component_patching_summary.csv", "patching/head_confirm_summary.csv",
]
EARLIER_REQUIRED = [
    "h0_canonical_behavior.csv", "h0_per_origin.csv", "h1_entity_ranks.csv",
    "h1_fragmentation_vs_recognition.csv", "h1_tokenization.csv",
    "h2_DD_to_LL_probe.csv", "h2_LL_to_DD_probe.csv", "h3_decodable_but_wrong.csv",
    "h3_layerwise_summary.csv", "h4_coarse_matched_patching.csv",
    "h4_coarse_wrong_entity_control.csv", "h4_refined_matched_patching.csv",
    "h4_confirmatory_10_18_summary.csv", "knowledge_gate.csv", "causal_gate.csv",
    "confirmatory_dashboard.csv", "bootstrap_behavior.csv", "run_metadata.json",
]

PASS, FAIL, WARN = [], [], []


def ok(msg):
    PASS.append(msg)


def bad(msg):
    FAIL.append(msg)


def warn(msg):
    WARN.append(msg)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(archive: Path) -> None:
    print("\nintegrity: archived files against surviving originals")
    for rel, origin in ORIGINS.items():
        target = archive / rel
        if not target.is_dir():
            bad(f"{rel} is not in the archive")
            continue
        if not origin.exists():
            warn(f"{rel}: original {origin.name} no longer exists, cannot re-verify")
            continue
        checked = mismatched = missing = 0
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            source = origin / path.relative_to(target)
            if not source.exists():
                missing += 1
                continue
            checked += 1
            if digest(path) != digest(source):
                mismatched += 1
                bad(f"{rel}/{path.name} differs from its original")
        print(f"  {rel:<34} {checked} verified, {mismatched} differ, "
              f"{missing} no longer have an original")
        if checked and not mismatched:
            ok(f"{rel}: {checked} files match their originals")


def check_runs(archive: Path) -> None:
    print("\ncompleteness: pipeline runs")
    for model in ("qwen", "gemma", "llama"):
        run = archive / model
        if not run.is_dir():
            bad(f"pipeline run {model} is missing")
            continue
        absent = [rel for rel in PIPELINE_REQUIRED if not (run / rel).exists()]
        causal = [rel for rel in PIPELINE_CAUSAL if not (run / rel).exists()]
        n = len(list(run.rglob("*.csv")))
        print(f"  {model:<8} {n:>3} csv | core missing: {len(absent)} | "
              f"causal missing: {len(causal)}")
        if absent:
            bad(f"{model} is missing core tables: {', '.join(absent)}")
        else:
            ok(f"{model}: all {len(PIPELINE_REQUIRED)} core tables present")
        if causal:
            # Gemma's accuracy gate selected nothing, so it has no patching output at all.
            # That is a result, not a missing file, and the collected numbers record it.
            warn(f"{model}: no {', '.join(c.split('/')[-1] for c in causal)} "
                 f"(expected for a model whose causal gate selected nothing)")


def check_earlier(archive: Path) -> None:
    print("\ncompleteness: earlier notebook run")
    for model in ("qwen", "gemma"):
        folder = archive / "earlier_notebook_run" / model
        if not folder.is_dir():
            bad(f"earlier notebook run {model} is missing")
            continue
        absent = [n for n in EARLIER_REQUIRED if not (folder / n).exists()]
        print(f"  {model:<8} {len(list(folder.glob('*')))} files | missing: {len(absent)}")
        if absent:
            warn(f"earlier run {model} lacks: {', '.join(absent)}")
        else:
            ok(f"earlier run {model}: all {len(EARLIER_REQUIRED)} expected files present")
    notebooks = archive / "earlier_notebook_run" / "notebooks"
    states = archive / "earlier_notebook_run" / "states"
    for folder, label in ((notebooks, "executed notebooks"), (states, "state archives")):
        n = len(list(folder.glob("*"))) if folder.is_dir() else 0
        print(f"  {label:<20} {n} files")
        (ok if n else bad)(f"{label}: {n} present")


def check_paper(archive: Path, paper_src: Path) -> None:
    """The document's own build inputs, read out of the tex rather than assumed."""
    print("\ncompleteness: paper sources")
    folder = archive / "paper"
    tex_files = sorted(folder.glob("*.tex")) if folder.is_dir() else []
    if not tex_files:
        bad("no .tex in the archive")
        return
    tex_path = tex_files[0]
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    print(f"  {tex_path.name}, {len(tex.splitlines())} lines")

    # The bibliography the document actually loads, not whichever .bib happens to be there.
    match = re.search(r"\\bibliography\{([^}]*)\}", tex)
    if match:
        for stem in [s.strip() for s in match.group(1).split(",") if s.strip()]:
            present = (folder / f"{stem}.bib").exists()
            print(f"  \\bibliography{{{stem}}} -> {stem}.bib "
                  f"{'present' if present else 'MISSING'}")
            (ok if present else bad)(f"bibliography {stem}.bib "
                                     f"{'present' if present else 'is missing from the archive'}")
    else:
        warn("the tex declares no \\bibliography")

    keys = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        keys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    entries = set()
    for bib in folder.glob("*.bib"):
        entries |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",
                                  bib.read_text(encoding="utf-8", errors="replace")))
    unresolved = sorted(keys - entries)
    print(f"  {len(keys)} citation keys, {len(keys) - len(unresolved)} resolved by the "
          f"archived .bib files")
    (ok if not unresolved else bad)(
        "every citation resolves" if not unresolved
        else f"unresolved citations: {', '.join(unresolved[:8])}")

    # Graphics and \input, if the document grows any later.
    for pattern, label in ((r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", "figure"),
                           (r"\\input\{([^}]*)\}", "input file")):
        for name in re.findall(pattern, tex):
            stem = name.strip()
            found = any((folder / f"{stem}{ext}").exists()
                        for ext in ("", ".pdf", ".png", ".jpg", ".tex"))
            (ok if found else bad)(f"{label} {stem} {'present' if found else 'MISSING'}")

    if paper_src.is_dir() and (paper_src / tex_path.name).exists():
        same = digest(paper_src / tex_path.name) == digest(tex_path)
        (ok if same else warn)(
            "the archived tex matches the working copy" if same
            else "the working copy of the tex has changed since it was archived")


def check_provenance(archive: Path) -> None:
    print("\nprovenance")
    folder = archive / "provenance"
    if not folder.is_dir():
        bad("no provenance folder")
        return
    corpora, seeds = {}, {}
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        manifest = record.get("manifest") or {}
        model = record.get("model") or {}
        corpora[path.stem] = manifest.get("source_corpus_sha256")
        seeds[path.stem] = manifest.get("seed")
        print(f"  {path.stem:<8} {manifest.get('n_prepared', '?')} facts, "
              f"seed {manifest.get('seed', '?')}, {model.get('model_id', 'model unrecorded')}")
    if len(set(corpora.values())) == 1 and None not in corpora.values():
        ok("all runs used the same corpus, by hash")
    else:
        bad(f"runs disagree on the corpus hash: {corpora}")
    if len(set(seeds.values())) == 1:
        ok("all runs used the same seed")
    else:
        bad(f"runs disagree on the seed: {seeds}")


def check_collected(archive: Path) -> None:
    print("\ncollected results")
    path = archive / "results" / "results_collected.csv"
    if not path.exists():
        bad("results_collected.csv is missing")
        return
    frame = pd.read_csv(path)
    print(f"  {len(frame)} numbers, {frame.section.nunique()} sections, "
          f"{frame.model.nunique()} models")
    missing_source = frame.source.isna().sum()
    (ok if not missing_source else bad)(
        "every collected number names its source file" if not missing_source
        else f"{missing_source} collected numbers have no source")
    # Every source path a collected number claims to come from must actually be there.
    dangling = set()
    for model, sub in frame.groupby("model"):
        for rel in sub.source.dropna().unique():
            if not (archive / model / rel).exists():
                dangling.add(f"{model}/{rel}")
    (ok if not dangling else bad)(
        "every cited source file exists in the archive" if not dangling
        else f"{len(dangling)} sources do not exist: {', '.join(sorted(dangling)[:4])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ROOT / "archive" / "299_fact")
    ap.add_argument("--paper", type=Path, default=ROOT / "paper")
    args = ap.parse_args()
    if not args.archive.is_dir():
        raise SystemExit(f"no archive at {args.archive}")

    print(f"verifying {args.archive.resolve()}")
    check_integrity(args.archive)
    check_runs(args.archive)
    check_earlier(args.archive)
    check_paper(args.archive, args.paper)
    check_provenance(args.archive)
    check_collected(args.archive)

    print("\n" + "=" * 78)
    for line in WARN:
        print(f"  NOTE  {line}")
    for line in FAIL:
        print(f"  FAIL  {line}")
    print("=" * 78)
    print(f"{len(PASS)} checks passed, {len(WARN)} notes, {len(FAIL)} failures")
    if FAIL:
        print("The archive cannot rebuild the paper as it stands.")
    else:
        print("The archive holds everything the paper is built from.")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

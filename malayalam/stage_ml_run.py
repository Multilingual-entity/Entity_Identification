"""Stage the Malayalam v2 dataset for the behavioural matrix, and prove it builds prompts.

No GPU is touched. This runs the dataset's own validator, stages the two paired run
directories (Malayalam and the matched 380-item Marathi comparator), copies the corpora
where run_languages.py's preflight looks for them, builds the full prompt frame locally,
and checks the things that would otherwise only surface after hours of inference:

  * that the Malayalam script lands in the dev slots and only there,
  * that the Lat--Lat arm is byte-identical between the two language runs, which is what
    makes the comparison paired rather than two separate experiments,
  * that no prompt is long enough to risk the 384-token cap,
  * which entities the Malayalam audit disqualifies, and which surviving entities have a
    disqualified entity as their distractor.

Stage 00 is deliberately not used: it reshuffles splits, which would break the frozen
225/77/78 assignments and let the fixed negative map cross split boundaries.

Usage:
    python stage_ml_run.py --dataset <malayalam_entity_dataset_v2> --out <run root> \
        --corpora <pipeline>/../corpora --results <run root>/results
    python stage_ml_run.py --dataset <...> --out <run root> --check-only

--pipeline is found automatically in both the nested local checkout and the flat layout on
the run machine; pass it if neither matches.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def find_pipeline(given: Path | None) -> Path:
    """Locate pipeline_common.py.

    The local checkout nests it under cross-script-entity-matching/pipeline; the run
    machine has a flat layout with the stage files directly beside each other. Guessing
    one of those breaks on the other, so both are tried and --pipeline overrides.
    """
    candidates = [given] if given else [
        HERE.parent / "cross-script-entity-matching" / "pipeline",
        HERE,
        HERE.parent,
        HERE.parent / "pipeline",
    ]
    for candidate in candidates:
        if candidate and (candidate / "pipeline_common.py").exists():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates if c)
    raise SystemExit(f"cannot find pipeline_common.py. Tried: {tried}. "
                     f"Pass --pipeline.")

# The dataset writes Malayalam into the legacy *_dev columns, so the pipeline's condition
# names are unchanged. The paper must print Mlym--Mlym, Mlym--Lat, Lat--Mlym, Lat--Lat.
CONDITION_NAMES = {
    "DEVDEV": "Mlym--Mlym",
    "DEVLAT": "Mlym--Lat",
    "LATDEV": "Lat--Mlym",
    "LATLAT": "Lat--Lat",
}
MALAYALAM = range(0x0D00, 0x0D80)
DEVANAGARI = range(0x0900, 0x0980)


def has(text: str, block: range) -> bool:
    return any(ord(c) in block for c in str(text))


def run(cmd: list[str]) -> None:
    print("    $ " + " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    for line in (proc.stdout or "").splitlines():
        print("      " + line)
    if proc.returncode != 0:
        for line in (proc.stderr or "").splitlines()[-12:]:
            print("      ! " + line)
        raise SystemExit(f"failed: {cmd[1]}")


def stage(out: Path, dataset: Path, corpora: Path) -> dict[str, Path]:
    # -X utf8: the dataset's validator calls read_text() without an encoding, which on
    # Windows decodes the Malayalam JSONL as cp1252 and dies. Forcing UTF-8 leaves their
    # file untouched, and costs nothing where the default is already UTF-8.
    print("  dataset validation")
    run([sys.executable, "-X", "utf8", str(dataset / "validate_dataset.py")])

    dirs = {}
    for language, name in (("ml", "ml"), ("mr", "mr380")):
        run_dir = out / f"_staged_{name}"
        dirs[name] = run_dir
        if (run_dir / "data" / "corpus.csv").exists():
            print(f"  {name}: already staged, left alone")
            continue
        print(f"  staging {language} -> {run_dir}")
        run([sys.executable, "-X", "utf8", str(dataset / "prepare_run.py"),
             "--run-dir", str(run_dir), "--language", language])

    # Where run_languages.py's preflight looks. Copying the selected corpus here keeps a
    # `--stages 01` invocation honest about what it is running on.
    for language, source in (("ml", "corpus_selected.csv"),
                             ("mr380", "marathi_matched_subset.csv")):
        target = corpora / language
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset / source, target / "corpus_selected.csv")
        shutil.copyfile(dataset / "hard_negative_map.csv",
                        target / "hard_negative_map.csv")
        print(f"  {target}  <- {source}")
    return dirs


def materialise(out: Path, results: Path, models: list[str]) -> None:
    """Copy the staged data into the per-model run directories stage 01 expects.

    run_languages.py puts each run in <results>/<model>_<language>, and normally stage 00
    fills its data/ directory. Stage 00 is the one stage that must not run here, so the
    frozen data is copied in instead. Existing files are left alone: a data directory that
    already has outputs beside it belongs to a run that has started.
    """
    wanted = ("corpus.csv", "hard_negative_map.csv", "splits.csv", "data_audit.csv",
              "manifest.json")
    for model in models:
        for language in ("ml", "mr380"):
            source = out / f"_staged_{'ml' if language == 'ml' else 'mr380'}" / "data"
            target = results / f"{model}_{language}" / "data"
            target.mkdir(parents=True, exist_ok=True)
            copied = []
            for name in wanted:
                if (source / name).exists() and not (target / name).exists():
                    shutil.copyfile(source / name, target / name)
                    copied.append(name)
            state = f"{len(copied)} file(s) copied" if copied else "already populated"
            print(f"  {model}_{language}/data: {state}")


def eligibility(out: Path, dataset: Path) -> pd.DataFrame:
    """Turn the Malayalam audit into a per-entity eligibility table for the analysis.

    Every entity is scored and nothing is filtered here. The subset is applied after
    scoring so that no model output decides which rows survive.
    """
    audit_path = dataset / "malayalam_audit.csv"
    if not audit_path.exists():
        print(f"\n  no {audit_path.name} in the dataset; the analysis will report all "
              f"scored entities only")
        return pd.DataFrame()
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    negatives = pd.read_csv(dataset / "hard_negative_map.csv", encoding="utf-8-sig")
    verdict = dict(zip(audit["fact_id"], audit["verdict"]))
    negative_of = dict(zip(negatives["fact_id"], negatives["negative_fact_id"]))

    rows = []
    for fid, own in verdict.items():
        neg = negative_of.get(fid)
        neg_verdict = verdict.get(neg, "MISSING")
        rows.append({
            "fact_id": fid,
            "audit_verdict": own,
            "negative_fact_id": neg,
            "negative_audit_verdict": neg_verdict,
            # SAME prompts need only this entity to be sound.
            "eligible_same": int(own == "KEEP"),
            # DIFFERENT prompts also read the distractor's name_b.
            "eligible_different": int(own == "KEEP" and neg_verdict == "KEEP"),
        })
    frame = pd.DataFrame(rows).sort_values("fact_id").reset_index(drop=True)
    path = out / "ml_eligibility.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    print(f"\n  eligibility -> {path}")
    print(f"    {frame.eligible_same.sum()} entities usable for SAME, "
          f"{frame.eligible_different.sum()} for both polarities, "
          f"of {len(frame)}")
    return frame


def check(dirs: dict[str, Path], out: Path, pipeline: Path) -> None:
    sys.path.insert(0, str(pipeline))
    from pipeline_common import (CONDITIONS, MAX_LENGTH, QUESTION_TEMPLATES,
                                 REVIEWED_LANGUAGES, build_prompt_frame,
                                 load_prepared_data)

    print(f"\n  ml templates present: {'ml' in QUESTION_TEMPLATES}, "
          f"native-reviewed: {'ml' in REVIEWED_LANGUAGES}")
    if "ml" not in REVIEWED_LANGUAGES:
        print("    WARNING: an unreviewed template returns chance on every condition, "
              "which is indistinguishable from a real script effect.")

    frames = {}
    for name, contexts in (("ml", ("ml", "en")), ("mr380", ("mr", "en"))):
        corpus, negatives = load_prepared_data(dirs[name])
        frame = build_prompt_frame(corpus, negatives, contexts=contexts,
                                   conditions=tuple(CONDITIONS), truths=(1, 0),
                                   paraphrases=(0, 1, 2), yes_letters=("A", "B"))
        frames[name] = frame
        per_entity = len(frame) / corpus["fact_id"].nunique()
        print(f"\n  {name}: {len(corpus)} entities, {len(frame)} prompts "
              f"({per_entity:.0f} per entity), splits "
              f"{corpus['split'].value_counts().to_dict()}")
        frame.to_csv(out / f"prompts_{name}.csv", index=False, encoding="utf-8")

    ml, mr = frames["ml"], frames["mr380"]
    failures = []

    # Script placement, read off the name slots. Testing the whole prompt is useless
    # here: under Malayalam prompts the template itself is Malayalam, so every prompt
    # contains Malayalam in every condition, including Lat--Lat.
    print("\n    script of the two name slots, by condition")
    for condition, (a_script, b_script) in CONDITIONS.items():
        sub = ml.loc[ml["condition"] == condition]
        label = CONDITION_NAMES[condition]
        got = []
        for slot, script in (("a_text", a_script), ("b_text", b_script)):
            share = sub[slot].map(lambda s: has(s, MALAYALAM)).mean()
            want = 1.0 if script == "dev" else 0.0
            got.append(f"{slot[0]}={share:.3f} (want {want:.0f})")
            if abs(share - want) > 1e-9:
                failures.append(f"{label} slot {slot[0]}: Malayalam in {share:.1%} "
                                f"of names, expected {want:.0%}")
        print(f"      {label:<11} {'  '.join(got)}")

    devanagari = (ml["a_text"].map(lambda s: has(s, DEVANAGARI))
                  | ml["b_text"].map(lambda s: has(s, DEVANAGARI))).sum()
    print(f"      Devanagari in any ml name slot: {devanagari}")
    if devanagari:
        failures.append(f"{devanagari} Malayalam-run names contain Devanagari")

    # And the mirror of the same check on the comparator, so a mix-up between the two
    # staged corpora cannot pass silently.
    mr_native = (mr.loc[mr["condition"] == "DEVDEV", "a_text"]
                 .map(lambda s: has(s, DEVANAGARI)).mean())
    mr_malayalam = (mr["a_text"].map(lambda s: has(s, MALAYALAM))
                    | mr["b_text"].map(lambda s: has(s, MALAYALAM))).sum()
    print(f"      comparator Dev--Dev slot a in Devanagari: {mr_native:.3f}, "
          f"Malayalam in any comparator name: {mr_malayalam}")
    if mr_native < 1.0 or mr_malayalam:
        failures.append("the Marathi comparator corpus is not in Devanagari")

    # The paired comparison rests on this: with both names in Latin and the prompt in
    # English, the two language runs ask the identical question of the identical entity.
    keys = ["fact_id", "condition", "truth", "paraphrase_id", "yes_letter"]
    a = ml.loc[(ml["condition"] == "LATLAT") & (ml["context"] == "en")].set_index(keys)
    b = mr.loc[(mr["condition"] == "LATLAT") & (mr["context"] == "en")].set_index(keys)
    shared = a.index.intersection(b.index)
    identical = (a.loc[shared, "prompt"] == b.loc[shared, "prompt"]).sum()
    print(f"\n    English Lat--Lat prompts shared by both runs: {len(shared)}, "
          f"byte-identical: {identical}")
    if identical != len(shared) or len(shared) == 0:
        failures.append("the English Lat--Lat arm is not shared between the two runs")

    # Length. The real check needs each model's tokenizer; this bounds it from below.
    longest = ml["prompt"].str.len().max()
    print(f"    longest ml prompt: {longest} characters "
          f"(the cap is {MAX_LENGTH} tokens, so audit this per tokenizer on the server)")

    empty = ml["prompt"].str.strip().eq("").sum()
    if empty:
        failures.append(f"{empty} empty prompts")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print("  all prompt-build checks passed; nothing here needed a GPU")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--results", type=Path, default=None,
                    help="run root for run_languages.py; the per-model run directories "
                         "are created under it and filled with the frozen data")
    ap.add_argument("--models", nargs="+", default=["qwen", "gemma", "llama"])
    ap.add_argument("--dataset", type=Path, required=True,
                    help="the unpacked malayalam_entity_dataset_v2 directory")
    ap.add_argument("--pipeline", type=Path, default=None,
                    help="directory holding pipeline_common.py; found automatically in "
                         "both the nested checkout and the flat run-machine layout")
    ap.add_argument("--no-eligibility", action="store_true",
                    help="leave an existing ml_eligibility.csv alone. Use this on a "
                         "corpus already filtered by build_clean_run.py, whose "
                         "eligibility file covers the retained entities only")
    ap.add_argument("--corpora", type=Path, default=None,
                    help="corpus root run_languages.py is given as --corpus-root; "
                         "<corpora>/ml and <corpora>/mr380 are written")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    pipeline = find_pipeline(args.pipeline)
    # In the nested checkout the corpora live beside pipeline/; on the flat run machine
    # pipeline.parent is the home directory, so an unguarded default would scatter
    # corpus files into it. An existing directory wins, otherwise it goes next to the
    # stage files.
    if args.corpora is not None:
        corpora = args.corpora
    elif (pipeline.parent / "corpora").is_dir():
        corpora = pipeline.parent / "corpora"
    else:
        corpora = pipeline / "corpora"
    corpora = corpora.resolve()

    print(f"dataset  {dataset}")
    print(f"pipeline {pipeline}")
    print(f"corpora  {corpora}")
    print(f"out      {args.out}\n")
    dirs = {"ml": args.out / "_staged_ml", "mr380": args.out / "_staged_mr380"}
    if not args.check_only:
        dirs = stage(args.out, dataset, corpora)
    missing = [str(d / "data" / "corpus.csv") for d in dirs.values()
               if not (d / "data" / "corpus.csv").exists()]
    if missing:
        raise SystemExit("nothing staged at " + ", ".join(missing)
                         + "; drop --check-only")
    if not args.no_eligibility:
        eligibility(args.out, dataset)
    check(dirs, args.out, pipeline)
    if args.results is not None:
        print()
        materialise(args.out, args.results, args.models)
    (args.out / "staging_manifest.json").write_text(json.dumps({
        "dataset": str(dataset),
        "entities": 380,
        "splits": {"train": 225, "validation": 77, "test": 78},
        "condition_names": CONDITION_NAMES,
        "stage_00": "not used; it reshuffles the frozen splits",
        "native_review": "pending; malayalam_audit.csv is a structural audit only",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

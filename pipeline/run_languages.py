"""Run the pipeline over several languages, so that one failing language costs only itself.

A multilingual sweep is a long unattended job, and the failure that matters is not a crash
but a quiet one: a language whose question template is wrong returns fifty percent on every
condition, which is indistinguishable from a real script effect until someone reads the
template. Llama's Marathi arm failed exactly that way and cost a full run.

So this does two things a shell loop does not.

It preflights before spending anything. Every language is checked for a corpus, a negative
map and three question templates, and warned about if no speaker has reviewed those
templates. Nothing that fails the first two is launched.

And it isolates. Each language gets its own run directory and each stage runs as its own
process, so a language that dies at stage 04 does not stop the language after it, and a
stage that dies does not stop the stages that do not depend on it. What survives is
recorded per language and per stage, and printed as a table at the end.

    python run_languages.py --model gemma --languages mr hi ta yo ru ar ja sr vi
    python run_languages.py --model gemma --languages sa --contexts en
    python run_languages.py --model gemma --languages mr --stages 00 01 02 03 --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# stage -> (script, needs a model, extra arguments)
STAGES = {
    "00": ("00_prepare_data.py", False, []),
    "01": ("01_run_behavior.py", True, []),
    "02": ("02_extract_states.py", True, []),
    "03": ("03_analyze_states.py", True, []),
    "04": ("04_run_causal_patching.py", True, []),
    "05": ("05_run_component_patching.py", True, ["--cumulative"]),
    "06": ("06_run_head_patching.py", True, []),
    "07": ("07_run_recovery.py", True,
           ["--normalise-vectors", "--tune-variant", "additive",
            "--alphas", "0.5", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0", "8.0",
            "--tuning-facts", "60"]),
    "08": ("08_summarize_results.py", False, []),
    "09": ("09_jacobian_lens.py", True, []),
    "10": ("10_name_surprisal.py", True, []),
    "11": ("11_name_order_swap.py", True, ["--patch"]),
    "12": ("12_romanization_schemes.py", True, []),
}
DEFAULT_STAGES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Stages that take --contexts, verified against the scripts' own argument parsers rather
# than assumed. 02 and 03 read the prompt languages from the tables 01 wrote and reject the
# flag outright; 06 accepts it and was wrongly being denied it.
TAKES_CONTEXTS = {"01", "04", "05", "06", "07", "09", "10", "11", "12"}

# Gemma accepts nearly every positive pair, so a gate defined on binary correctness selects
# nothing and stages 04 to 06 produce empty files. It needs the margin gate read on the
# polarity it actually fails. Without this the run silently repeats the 299-fact outcome.
MODEL_STAGE_ARGS = {
    "gemma": {"04": ["--gate", "margin", "--gate-polarity", "negative"]},
    "llama": {**{s: ["--scale-layers"] for s in ("04", "05", "06", "07", "09", "11")},
              "04": ["--scale-layers", "--gate", "either"]},
}
# Same for any deeper model added later; the band was fixed on a 28-layer network.
for key in ("qwen14", "aya", "gemma9"):
    MODEL_STAGE_ARGS.setdefault(key, {s: ["--scale-layers"] for s in
                                      ("04", "05", "06", "07", "09", "11")})


def preflight(language: str, corpus_root: Path, contexts: list) -> tuple:
    """Returns (ok, contexts_to_use, notes). Never launches anything that cannot work."""
    sys.path.insert(0, str(HERE))
    from pipeline_common import QUESTION_TEMPLATES, REVIEWED_LANGUAGES

    notes = []
    corpus = corpus_root / language / "corpus_selected.csv"
    negatives = corpus_root / language / "hard_negative_map.csv"
    if not corpus.exists() or not negatives.exists():
        return False, [], [f"no corpus at {corpus.parent}"]

    n_items = len(pd.read_csv(corpus))
    notes.append(f"{n_items} items")
    if n_items < 100:
        notes.append("under 100 items, so per-condition estimates will be noisy")

    wanted = contexts or [language, "en"]
    usable = []
    for context in wanted:
        if context not in QUESTION_TEMPLATES:
            notes.append(f"no question template for '{context}', dropped")
            continue
        if context not in REVIEWED_LANGUAGES:
            notes.append(f"'{context}' templates unreviewed by a speaker")
        usable.append(context)
    if not usable:
        return False, [], notes + ["no usable prompt language"]
    if language not in usable:
        notes.append(f"running English prompts only; the {language} script contrast is "
                     f"still measured, the prompt-language crossing is not")
    return True, usable, notes


def run_stage(stage: str, language: str, run_dir: Path, model: str, contexts: list,
              force: bool, dry: bool) -> tuple:
    script, needs_model, extra = STAGES[stage]
    command = [sys.executable, str(HERE / script), "--run-dir", str(run_dir)]
    if needs_model:
        command += ["--model", model]
    if stage in TAKES_CONTEXTS:
        command += ["--contexts", *contexts]
    if stage == "00":
        command += ["--corpus", str(run_dir.parent / "_corpus" / language / "corpus_selected.csv"),
                    "--negatives", str(run_dir.parent / "_corpus" / language / "hard_negative_map.csv")]
    command += extra
    command += MODEL_STAGE_ARGS.get(model, {}).get(stage, [])
    if force:
        command += ["--force"]

    if dry:
        print(f"      would run: {' '.join(command[1:])}")
        return "dry", 0.0, ""

    # Streamed rather than captured. State extraction and head screening run for hours, and
    # capturing their output means an unattended run shows nothing at all until a stage
    # ends, which makes a hung stage indistinguishable from a slow one. Each line goes to
    # the terminal and to a per-stage log at the same time, and the last few are kept so a
    # failure can still be reported in the summary.
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stage_{stage}.log"
    recent: list = []
    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       bufsize=1, cwd=str(HERE))
            for line in process.stdout:
                print(f"      | {line.rstrip()}", flush=True)
                log.write(line)
                recent.append(line.rstrip())
                if len(recent) > 40:
                    recent.pop(0)
            code = process.wait()
    except Exception as exc:                                             # noqa: BLE001
        return "error", time.time() - started, str(exc)[:200]

    elapsed = time.time() - started
    if code == 0:
        return "ok", elapsed, ""
    detail = next((l for l in reversed(recent) if l.strip()), f"exit {code}")
    return "failed", elapsed, detail[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--contexts", nargs="+", default=None,
                    help="prompt languages; defaults to the corpus language plus English")
    ap.add_argument("--corpus-root", type=Path,
                    default=ROOT / "multilingual" / "out" / "corpora")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES,
                    choices=list(STAGES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="abandon a language after its first failed stage. Off by default, "
                         "because a failure at 04 says nothing about whether 09 to 12 will "
                         "work, and those do not depend on it")
    args = ap.parse_args()

    print(f"model {args.model} | {len(args.languages)} languages | stages "
          f"{' '.join(args.stages)}")
    print(f"corpora from {args.corpus_root}")

    print("\npreflight")
    plan = {}
    for language in args.languages:
        ok, contexts, notes = preflight(language, args.corpus_root, args.contexts)
        mark = "ok  " if ok else "SKIP"
        print(f"  {mark} {language:<4} {'; '.join(notes) if notes else ''}")
        if ok:
            plan[language] = contexts
    if not plan:
        raise SystemExit("\nnothing to run")
    skipped = [l for l in args.languages if l not in plan]
    if skipped:
        print(f"\n  skipping {', '.join(skipped)}; the rest continue")

    rows = []
    for language, contexts in plan.items():
        run_dir = args.out / f"{args.model}_{language}"
        print(f"\n=== {language} -> {run_dir.name}  (prompts in {', '.join(contexts)}) ===")
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            # Stage 00 reads the corpus from a stable place next to the run dirs, so the
            # command it records in its own log stays valid if the source tree moves.
            staged = args.out / "_corpus" / language
            staged.mkdir(parents=True, exist_ok=True)
            for name in ("corpus_selected.csv", "hard_negative_map.csv"):
                source = args.corpus_root / language / name
                if source.exists():
                    (staged / name).write_bytes(source.read_bytes())

        failed_here = False
        for stage in args.stages:
            if failed_here and args.stop_on_failure:
                rows.append({"language": language, "stage": stage, "status": "not attempted",
                             "seconds": 0.0, "detail": "earlier stage failed"})
                continue
            status, elapsed, detail = run_stage(stage, language, run_dir, args.model,
                                                contexts, args.force, args.dry_run)
            rows.append({"language": language, "stage": stage, "status": status,
                         "seconds": round(elapsed, 1), "detail": detail})
            flag = {"ok": "ok", "dry": "--", "failed": "FAILED", "error": "ERROR"}[status]
            print(f"    {stage} {STAGES[stage][0]:<32} {flag:<7} {elapsed:6.1f}s"
                  + (f"  {detail}" if detail else ""))
            if status in ("failed", "error"):
                failed_here = True

    status = pd.DataFrame(rows)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        status.to_csv(args.out / f"run_status_{args.model}.csv", index=False)

    print("\n" + "=" * 78)
    table = status.pivot_table(index="language", columns="stage", values="status",
                               aggfunc="first")
    print(table.to_string())
    done = int((status.status == "ok").sum())
    bad = int(status.status.isin(["failed", "error"]).sum())
    print(f"\n{done} stages completed, {bad} failed, "
          f"{status.seconds.sum() / 3600:.2f} hours total")
    if bad:
        print("\nfailures, with the last line of each:")
        for row in status[status.status.isin(["failed", "error"])].itertuples():
            print(f"  {row.language} stage {row.stage}: {row.detail}")
        print("\nEvery other language and stage still ran. Rerun just the failures by "
              "naming them with --languages and --stages.")


if __name__ == "__main__":
    main()

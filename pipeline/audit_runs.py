"""What did each run actually produce, and what failed?

Run status alone is not enough: a stage can be marked failed after writing every one of
its outputs, which is what stage 01 did for Hindi, and a stage can exit clean having
written a file with the wrong contents. This reports both sides, so the decision to
rerun is made from artefacts rather than from a scrollback.

    python audit_runs.py                       # every run under results/
    python audit_runs.py --runs gemma_hi qwen_hi llama_hi
    python audit_runs.py --runs qwen_hi --log 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from verify_archive import PIPELINE_REQUIRED, PIPELINE_CAUSAL

# Which stage writes each artefact, so a missing file names the stage to rerun.
WRITTEN_BY = {
    "data/corpus.csv": "00", "data/manifest.json": "00",
    "tables/behavior.csv": "01", "tables/behavior_summary.csv": "01",
    "tables/knowledge_gate.csv": "01",
    "tables/context_script_interaction_per_fact.csv": "01",
    "tables/tokenization.csv": "03",
    "tables/isolated_retrieval_summary.csv": "03",
    "tables/pair_retrieval_summary.csv": "03",
    "tables/relation_probe_summary.csv": "03",
    "tables/relation_probe_heldout_summary.csv": "03",
    "tables/exact_failure_probe_summary.csv": "03",
    "tables/fragmentation_correlation.csv": "03",
    "tables/failure_prediction_oof_auc.csv": "03",
    "patching/causal_candidates.csv": "04",
    "patching/causal_patching_summary.csv": "04",
    "patching/component_patching_summary.csv": "05",
    "patching/head_confirm_summary.csv": "06",
    "recovery/prompt_anchor_summary.csv": "07",
    "recovery/vector_recovery_test_summary.csv": "07",
    "tables/name_surprisal_summary.csv": "10",
    "tables/name_order_swap.csv": "11",
}

EXTRA = ["tables/name_surprisal_summary.csv", "tables/name_order_swap.csv"]


def audit(run_dir: Path, show_log: str | None) -> bool:
    print(f"\n=== {run_dir.name} " + "=" * (58 - len(run_dir.name)))
    if not run_dir.is_dir():
        print("  run directory does not exist")
        return False

    wanted = PIPELINE_REQUIRED + PIPELINE_CAUSAL + EXTRA
    missing, empty = [], []
    for rel in wanted:
        p = run_dir / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size == 0:
            empty.append(rel)
        elif p.suffix == ".csv":
            try:
                if len(pd.read_csv(p, nrows=2)) == 0:
                    empty.append(rel)
            except Exception as exc:
                empty.append(f"{rel} (unreadable: {type(exc).__name__})")

    if not missing and not empty:
        print(f"  all {len(wanted)} expected artefacts present and non-empty")
    for rel in missing:
        print(f"  MISSING  {rel:<52}stage {WRITTEN_BY.get(rel, '?')}")
    for rel in empty:
        print(f"  EMPTY    {rel}")

    stages = sorted({WRITTEN_BY.get(r, "?") for r in missing})
    if stages:
        print(f"  -> rerun stages: {' '.join(stages)}")

    logs = run_dir / "logs"
    if logs.is_dir():
        names = sorted(p.name for p in logs.glob("stage_*.log"))
        print(f"  logs present: {', '.join(names) if names else 'none'}")
        if show_log:
            p = logs / f"stage_{show_log}.log"
            if p.exists():
                tail = p.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                print(f"\n  --- tail of stage_{show_log}.log ---")
                for line in tail:
                    print("  | " + line)
            else:
                print(f"  no stage_{show_log}.log in this run")
    return not (missing or empty)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--runs", nargs="*", help="run directory names; default is all")
    ap.add_argument("--log", help="also print the tail of this stage's log, e.g. 10")
    args = ap.parse_args()

    for f in sorted(args.results.glob("run_status_*.csv")):
        d = pd.read_csv(f)
        print(f"\n### {f.name}")
        print(d.to_string(index=False))

    names = args.runs or sorted(
        p.name for p in args.results.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    ok = [audit(args.results / n, args.log) for n in names]
    print(f"\n{sum(ok)} of {len(ok)} runs complete")


if __name__ == "__main__":
    main()

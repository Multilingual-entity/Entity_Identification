"""Confirm a run was prepared from the same corpus and splits as a reference run.

Stage 00 takes a default corpus path when none is given. On a machine that still holds
an earlier corpus at that path, a run prepares silently against the wrong data: the
stages all succeed, the gate counts look plausible, and nothing compares them to the
other runs until a number reaches the paper. This compares them.

    python check_prepared_corpus.py --run results/mistral24_mr --reference results/qwen_mr
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


def fingerprint(path: Path) -> tuple[int, str]:
    d = pd.read_csv(path, dtype={"fact_id": str})
    if "split" not in d.columns:
        raise SystemExit(f"{path} has no split column; was stage 00 run?")
    frame = d[["fact_id", "split"]].sort_values("fact_id")
    return len(d), hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_csv = args.run / "data" / "corpus.csv"
    ref_csv = args.reference / "data" / "corpus.csv"
    for p in (run_csv, ref_csv):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    n_run, h_run = fingerprint(run_csv)
    n_ref, h_ref = fingerprint(ref_csv)
    print(f"  {args.run.name:16} {n_run:>5} facts   splits {h_run}")
    print(f"  {args.reference.name:16} {n_ref:>5} facts   splits {h_ref}")

    if (n_run, h_run) != (n_ref, h_ref):
        print()
        print("These runs were prepared from different corpora or different splits.")
        print("Nothing downstream of stage 00 is comparable between them. Re-run stage 00")
        print("with an explicit --corpus and --negatives, then re-run state extraction and")
        print("patching with --force.")
        raise SystemExit(1)

    behaviour = args.run / "tables" / "behavior.csv"
    if behaviour.exists():
        b = pd.read_csv(behaviour, usecols=["fact_id"], dtype={"fact_id": str})
        if b.fact_id.nunique() != n_run:
            print()
            print(f"behavior.csv covers {b.fact_id.nunique()} facts but the prepared corpus "
                  f"holds {n_run}. The run directory is internally inconsistent: behaviour "
                  f"was measured on one corpus and stage 00 has since written another.")
            raise SystemExit(1)
        print(f"  behaviour covers all {n_run} prepared facts")

    print("\nsame corpus, same splits, consistent behaviour table")


if __name__ == "__main__":
    main()

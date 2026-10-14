"""Is a run actually complete, or does it only look complete?

A stage that dies partway leaves files behind. causal_patching.csv exists but holds half
the facts; pair_states.npy is allocated at full size whatever has been written into it;
behaviour tables survive a crash in the stage after them. Reading a log tail does not
catch any of that, and twice in this project a run was treated as finished when it was not.

This checks the things that distinguish the two: progress markers, partial files, and
whether the row counts are internally consistent.

    python check_run_complete.py results/mistral24_mr
    python check_run_complete.py results/qwen_mr --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS = ["single_factorial", "window_factorial", "cross_context_factorial",
               "joint_12_17", "wrong_entity_control", "wrong_position_control",
               "reverse_DD_into_LL"]


def say(state: str, stage: str, detail: str) -> None:
    print(f"  {state:9} {stage:22} {detail}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="e.g. results/mistral24_mr")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run = args.run
    if not run.is_dir():
        raise SystemExit(f"no such run directory: {run}")
    print(f"{run}")
    incomplete = []

    # ---- 00 prepare ----------------------------------------------------------
    corpus = run / "data" / "corpus.csv"
    if corpus.exists():
        c = pd.read_csv(corpus, dtype={"fact_id": str})
        say("done", "00 prepare", f"{len(c)} facts, splits "
            f"{dict(c.split.value_counts())}")
        n_facts = len(c)
    else:
        # Exit non-zero rather than returning: a run with no prepared corpus is not a
        # run that passed, and printing MISSING while exiting 0 is how a check reports
        # success on nothing.
        say("MISSING", "00 prepare", "no data/corpus.csv, so nothing else can be checked")
        raise SystemExit(1)

    # ---- 01 behaviour --------------------------------------------------------
    beh = run / "tables" / "behavior.csv"
    partial = run / "tables" / "behavior.partial.csv"
    if beh.exists():
        b = pd.read_csv(beh, dtype={"fact_id": str})
        covered = b.fact_id.nunique()
        ok = covered == n_facts
        say("done" if ok else "PARTIAL", "01 behaviour",
            f"{len(b)} rows over {covered} of {n_facts} facts"
            + ("" if ok else "  <-- fewer facts than prepared"))
        if not ok:
            incomplete.append("01 behaviour covers fewer facts than the corpus")
    else:
        say("not run", "01 behaviour", "")
    if partial.exists():
        say("WARNING", "01 behaviour", f"{partial.name} still present; a run was "
            f"interrupted and resumed or abandoned")

    # ---- 02 states -----------------------------------------------------------
    prog = run / "states" / "pair_states.progress.json"
    arr = run / "states" / "pair_states.npy"
    meta = run / "states" / "pair_state_metadata.csv"
    if arr.exists():
        gib = arr.stat().st_size / 2**30
        rows = len(pd.read_csv(meta)) if meta.exists() else None
        if prog.exists():
            p = json.loads(prog.read_text())
            nxt, shape = p.get("next_row"), p.get("shape")
            total = shape[0] if shape else rows
            ok = total is not None and nxt is not None and nxt >= total
            say("done" if ok else "PARTIAL", "02 states",
                f"{gib:.1f} GiB, wrote {nxt} of {total} rows"
                + ("" if ok else "  <-- allocated full size but not filled"))
            if not ok:
                incomplete.append(f"02 states stopped at row {nxt} of {total}")
        else:
            say("done", "02 states", f"{gib:.1f} GiB, {rows} rows, no progress file left")
    else:
        say("not run", "02 states", "no pair_states.npy")

    # ---- 04 patching ---------------------------------------------------------
    cp = run / "patching" / "causal_patching.csv"
    cpart = run / "patching" / "causal_patching.partial.csv"
    done = run / "patching" / "causal_patching.done.json"
    if cp.exists():
        d = pd.read_csv(cp, dtype={"fact_id": str})
        missing = [e for e in EXPERIMENTS if e not in set(d.experiment)]
        per = d.groupby("fact_id").size()
        uniform = per.nunique() <= 2  # 519 for one context, 1038 for facts in both
        ok = not missing and uniform and not cpart.exists()
        say("done" if ok else "PARTIAL", "04 patching",
            f"{len(d)} rows, {d.fact_id.nunique()} facts, "
            f"{len(set(d.experiment))}/7 experiments, rows per fact "
            f"{sorted(per.unique())[:4]}")
        if missing:
            say("", "", f"missing experiments: {missing}")
            incomplete.append(f"04 patching missing {missing}")
        if not uniform:
            say("", "", "rows per fact are not uniform; some facts were cut short")
            incomplete.append("04 patching has ragged per-fact row counts")
        if done.exists():
            say("", "", "causal_patching.done.json present")
    else:
        say("not run", "04 patching", "")
    if cpart.exists():
        say("WARNING", "04 patching", "causal_patching.partial.csv still present")
        incomplete.append("04 patching left a partial file")

    # ---- is a scaled run actually scaled? ------------------------------------
    wj = run / "patching" / "windows_used.json"
    if wj.exists():
        w = json.loads(wj.read_text())
        sweep = w.get("single_layer_sweep") or []
        n = w.get("n_layers")
        wins = w.get("windows") or {}
        if w.get("scaled") and sweep and n and wins:
            # The windows in this file are already scaled. A single-layer sweep built
            # from the same flags must therefore reach at least as deep as they do. When
            # --scale-layers reached the windows but not the sweep, the sweep stopped
            # short of them, which is the signature to look for. At the reference depth
            # the two coincide and this passes trivially, as it should.
            deepest_window = max(L for v in wins.values() for L in v)
            if sweep[-1] < deepest_window:
                say("MISMATCH", "scaled sweep",
                    f"single-layer sweep ends at {sweep[-1]} but the scaled windows "
                    f"reach {deepest_window} of {n}; --scale-layers reached the windows "
                    f"and not the sweep, so the per-layer profile is at the wrong depth")
                incomplete.append(
                    "single-layer sweep stops short of the scaled windows")
            else:
                say("done", "scaled sweep",
                    f"sweep {sweep[0]}--{sweep[-1]} covers the scaled windows, which "
                    f"reach {deepest_window} of {n}")

    # ---- 07 recovery ---------------------------------------------------------
    sel = run / "recovery" / "vector_hyperparameters_selected.csv"
    tsum = run / "recovery" / "vector_recovery_test_summary.csv"
    apart = run / "recovery" / "prompt_anchor_recovery.partial.csv"
    if sel.exists():
        s = pd.read_csv(sel)
        say("done", "07 recovery select", ", ".join(
            f"{r.context} L{int(r.layer)} a{r.alpha:g}" for r in s.itertuples()))
        say("done" if tsum.exists() else "skipped", "07 recovery test",
            "test summary present" if tsum.exists()
            else "no test evaluation (--skip-test, or it stopped before it)")
    else:
        say("not run", "07 recovery", "")
    if apart.exists():
        say("WARNING", "07 recovery", "prompt_anchor_recovery.partial.csv still present")

    print()
    if incomplete:
        for x in incomplete:
            print(f"INCOMPLETE  {x}")
        raise SystemExit(1)
    print("every stage present in this run looks complete")


if __name__ == "__main__":
    main()

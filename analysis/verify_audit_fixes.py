"""Regression guard for the defects recorded in audit.md.

The first version of this script asserted the broken state, so it started failing the
moment the defects were fixed. This one asserts the fixed state instead, which is what a
regression guard should do: it fails if a fix is ever undone.

Defects 4 and 7 are not checked, because they are open decisions for the author rather
than code properties: whether the three archive-era verifiers are retargeted or removed,
and what data fixed the layer 10 to 18 band.

    python analysis/verify_audit_fixes.py --results ../results_mr --results ../results_hi \
                                          --paper ../paper/final_draft.tex
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OK: list[bool] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    OK.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", type=Path, required=True)
    ap.add_argument("--paper", type=Path, required=True)
    args = ap.parse_args()

    tables = read("analysis/paper_tables.py")
    stage05 = read("pipeline/05_run_component_patching.py")
    stage08 = read("pipeline/08_summarize_results.py")
    readme = read("README.md")

    print("defect 1: the table checker covers the Hindi runs")
    check("run folders are resolved per family and prompt language",
          "FAMILIES = {" in tables and 'LANG = {"Marathi": ("mr", "mr")' in tables)
    check("the Hindi run supplies the Hindi columns",
          '("qwen_hi", "hi")' in tables and '("gemma_hi", "hi")' in tables)
    check("the gate polarity is keyed by family, not by run",
          'GATE_TRUTH = {"qwen": 1, "gemma": 0, "llama": 1}' in tables)
    out = subprocess.run(
        [sys.executable, str(HERE / "paper_tables.py"), "--check",
         *sum([["--results", str(r)] for r in args.results], []),
         "--paper", str(args.paper)],
        capture_output=True, text=True).stdout
    agree = re.search(r"(\d+) of (\d+) generated tables agree", out)
    check("the checker runs and reports on every generated table", agree is not None,
          agree.group(0) if agree else out[-200:])
    if agree:
        n, total = int(agree.group(1)), int(agree.group(2))
        # tab:dprime is expected to differ until the paper carries the Hindi rows,
        # which is TODO 1.10. Anything worse than that is a regression.
        check("at most one table differs, and it is tab:dprime", total - n <= 1,
              f"{total - n} differ")
        if total - n == 1:
            check("the differing table is tab:dprime", "MISMATCH tab:dprime" in out)

    print("defect 2: the held-out probe table is read at the question mark")
    check("the generator filters the position", 'd.position == "Q"' in tables)
    for results in args.results:
        for run, src, tgt, want_layer, want_value in (("gemma_mr", "mr", "en", 20, 0.838),
                                                      ("qwen_mr", "mr", "mr", 16, 0.925)):
            path = results / run / "tables/relation_probe_heldout_summary.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path)
            d = d[(d.source_context == src) & (d.target_context == tgt) & (d.position == "Q")]
            val = d[d.evaluation_split == "validation"]
            layer = int(val.loc[val.balanced_accuracy.idxmax(), "layer"])
            test = d[(d.evaluation_split == "test") & (d.layer == layer)]
            value = round(float(test.balanced_accuracy.iloc[0]), 3)
            check(f"{run} {src}->{tgt}: validation picks L{want_layer}, test {want_value}",
                  layer == want_layer and value == want_value, f"L{layer} {value}")

    print("defect 3: the checker runs from a clean clone")
    check("results roots are arguments, repeatable", '"--results", action="append"' in tables)
    check("the paper is an argument", '"--paper", type=Path' in tables)
    check("no default points at a file that does not exist",
          'HERE / "paperB_neurips.tex"' not in tables and 'HERE.parent / "results"' not in tables)
    check("the README gives the working command",
          "--results ../results_mr --results ../results_hi" in readme)

    print("defect 5: the dashboard selects on validation")
    check("the probe layer is chosen on validation",
          'val = base.loc[base["evaluation_split"] == "validation"]' in stage08
          and "relation_probe_test_balanced_at_selected" in stage08)
    check("the best window is chosen on validation only",
          'heldout = causal.loc[causal["split"] == "validation"]' in stage08)
    check("no dashboard key selects a maximum on test",
          "relation_probe_best_test_balanced" not in stage08)

    print("defect 6: component items are selected by an explicit split filter")
    check("split priority is an argument", '"--split-priority"' in stage05)
    check("selection is by rank, not by the alphabet",
          'sort_values(["context", "split_rank", "fact_id"])' in stage05
          and 'selected["split"].isin(args.split_priority)' in stage05)
    for results in args.results:
        for run in ("qwen_mr", "qwen_hi", "gemma_mr", "gemma_hi"):
            cpath = results / run / "patching/causal_candidates.csv"
            ppath = results / run / "patching/component_patching.csv"
            if not (cpath.exists() and ppath.exists()):
                continue
            cand = pd.read_csv(cpath)
            cand = cand[cand.stable == 1]
            priority = ["test", "train"]
            sel = cand[cand.split.isin(priority)].copy()
            sel["r"] = sel.split.map({s: i for i, s in enumerate(priority)})
            sel = (sel.sort_values(["context", "r", "fact_id"])
                      .groupby("context", group_keys=False).head(30))
            actual = pd.read_csv(ppath).groupby("context").fact_id.apply(
                lambda x: set(x.astype(str)))
            same = all(set(sel[sel.context == c].fact_id.astype(str)) == actual[c]
                       for c in actual.index)
            check(f"{run}: the explicit filter reproduces the items already patched", same)

    print(f"\n{sum(OK)} of {len(OK)} pass")
    if not all(OK):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

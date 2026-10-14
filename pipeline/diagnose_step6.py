"""Why did step 6 report missing frozen rows?

The step 6 check compared len(index.intersection) against len(rows). If the key is not
unique those are different things and the check fires on every run, whether or not
anything changed. This settles which it is before any table is regenerated.

    python diagnose_step6.py results/qwen_mr patching_cap60_20260930_0537 mr
    python diagnose_step6.py results/qwen_hi patching_cap60_20260930_0537 hi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEY = ["fact_id", "context", "split", "truth", "experiment", "source_condition",
       "site", "layer_or_window"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run directory, e.g. results/qwen_mr")
    ap.add_argument("archive", help="archive directory name beside patching/")
    ap.add_argument("context", help="the context whose rows to compare")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any archived fact is missing from the "
                         "rerun, so an unattended script can gate on it")
    a = ap.parse_args()
    run, archive_name, ctx, strict = a.run, a.archive, a.context, a.strict
    old = pd.read_csv(Path(run) / archive_name / "causal_patching.csv",
                      dtype={"fact_id": str})
    new = pd.read_csv(Path(run) / "patching" / "causal_patching.csv",
                      dtype={"fact_id": str})

    o = old[old.context == ctx]
    n = new[new.context == ctx]
    print(f"{ctx}: archived {len(o)} rows, new {len(n)} rows")

    # 1. Is the key unique? If not, the step 6 arithmetic was meaningless.
    for label, d in [("archived", o), ("new", n)]:
        dup = len(d) - len(d.drop_duplicates(KEY))
        print(f"  {label}: {dup} rows share a key with another row "
              f"({len(d.drop_duplicates(KEY))} unique keys)")

    # 2. Did the set of facts change? This is the question step 6 meant to ask.
    of, nf = set(o.fact_id), set(n.fact_id)
    print(f"  facts: archived {len(of)}, new {len(nf)}, "
          f"dropped {len(of - nf)}, added {len(nf - of)}")
    if of - nf:
        print(f"    dropped: {sorted(of - nf)[:10]}")
    if nf - of:
        print(f"    added:   {sorted(nf - of)[:10]}")

    # 3. Did the experiment grid change? A new experiment or site would move row counts
    #    without any fact changing.
    for col in ("experiment", "site", "source_condition", "layer_or_window"):
        ov, nv = set(o[col].astype(str)), set(n[col].astype(str))
        if ov != nv:
            print(f"  {col}: only archived {sorted(ov - nv)[:8]}, "
                  f"only new {sorted(nv - ov)[:8]}")
        else:
            print(f"  {col}: identical ({len(ov)} values)")

    # 4. On the rows that match by key, did any number move?
    om = o.drop_duplicates(KEY).set_index(KEY).sort_index()
    nm = n.drop_duplicates(KEY).set_index(KEY).sort_index()
    common = om.index.intersection(nm.index)
    if len(common):
        worst = (om.loc[common, "delta_margin"] - nm.loc[common, "delta_margin"]).abs().max()
        print(f"  {len(common)} keys in both; largest delta_margin change {worst:.6f}")
    print(f"  keys only in archived: {len(om.index.difference(nm.index))}")
    print(f"  keys only in new:      {len(nm.index.difference(om.index))}")

    # --strict makes this usable as a gate in an unattended script. A frozen context
    # must lose nothing: a cap can only keep fewer items, so a dropped fact means the
    # gate moved and the published numbers no longer describe the run.
    if strict:
        if of - nf:
            print(f"STRICT FAIL: {len(of - nf)} frozen {ctx} facts dropped")
            raise SystemExit(1)
        print(f"STRICT OK: no {ctx} fact dropped")


if __name__ == "__main__":
    main()

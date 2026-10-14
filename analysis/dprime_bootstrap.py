"""Bootstrap the Lat--Lat minus Dev--Dev d' gap, resampling items rather than prompts.

The paper reports these intervals in its sensitivity appendix. Prompts derived from one
item are not independent, so the resample is over items. Every pooled rate uses the
log-linear estimator (x + 0.5) / (n + 1), which is what reproduces the point estimates;
applying it only to rates of zero or one does not.

    python analysis/dprime_bootstrap.py --results results_mr --contexts mr en
    python analysis/dprime_bootstrap.py --results results_hi --contexts hi en
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pipeline_common import dprime  # noqa: E402

SEED = 20260813


def _gap(counted: pd.DataFrame, facts) -> float:
    """d' in Lat--Lat minus d' in Dev--Dev, over the given items."""
    out = {}
    for condition in ("DEVDEV", "LATLAT"):
        sub = counted.xs(condition, level=1).loc[facts]
        n_pos, n_neg = int(sub.npos.sum()), int(sub.nneg.sum())
        value, _ = dprime(sub.hit.sum() / max(n_pos, 1),
                          sub.fa.sum() / max(n_neg, 1), n_pos, n_neg)
        out[condition] = value
    return out["LATLAT"] - out["DEVDEV"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="a results directory holding <run>/tables/behavior.csv")
    parser.add_argument("--contexts", nargs="+", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for run in sorted(Path(args.results).iterdir()):
        path = run / "tables" / "behavior.csv"
        if not path.exists():
            continue
        behaviour = pd.read_csv(path, usecols=["fact_id", "context", "condition",
                                               "truth", "correct"])
        for context in args.contexts:
            frame = behaviour[(behaviour.context == context)
                              & (behaviour.condition.isin(["DEVDEV", "LATLAT"]))]
            if frame.empty:
                continue
            counted = (frame.assign(hit=(frame.truth == 1) & (frame.correct == 1),
                                    fa=(frame.truth == 0) & (frame.correct == 0),
                                    npos=(frame.truth == 1), nneg=(frame.truth == 0))
                       .groupby(["fact_id", "condition"])[["hit", "fa", "npos", "nneg"]]
                       .sum())
            facts = counted.index.get_level_values(0).unique().to_numpy()
            observed = _gap(counted, facts)
            draws = np.array([_gap(counted, rng.choice(facts, len(facts), replace=True))
                              for _ in range(args.n_boot)])
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append({"run": run.name, "context": context, "gap": observed,
                         "ci_low": low, "ci_high": high, "n_facts": len(facts),
                         "n_boot": args.n_boot, "seed": args.seed})
            print(f"{run.name:14s} {context:3s} {observed:+.3f} "
                  f"[{low:+.3f}, {high:+.3f}]   n={len(facts)}")

    if not rows:
        raise SystemExit("no behaviour tables found under --results")
    out = Path(args.results) / "dprime_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

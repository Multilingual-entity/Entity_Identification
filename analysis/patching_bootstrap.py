"""Bootstrap the patching effects, resampling held-out items.

Two quantities, and they need different conventions:

  site effect        must match the paper's site columns exactly, which average layers
                     10 to 18 and all three Latin-bearing donor conditions
  entity specificity is a paired contrast against the wrong-entity control, which exists
                     only with a Lat--Lat donor, so both arms are restricted to that
                     donor, to the same layers, and to the items present in both

Mixing those two conventions silently inflates the site effect and makes the contrast
compare different populations, so each is computed on its own terms.

    python analysis/patching_bootstrap.py --results results_mr --contexts mr en
    python analysis/patching_bootstrap.py --results results_hi --contexts hi en
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
BAND = (10, 18)
# Qwen and Llama fail positive pairs; Gemma fails negatives and is gated there.
GATE_TRUTH = {"gemma": 0}


def _item_means(frame: pd.DataFrame, low: int, high: int) -> pd.Series:
    """One number per item: its mean margin change across the band."""
    band = frame[frame.layer.between(low, high)]
    return band.groupby("fact_id").delta_margin.mean()


def _rng(seed: int, *parts) -> np.random.Generator:
    """A generator keyed to the quantity being resampled.

    One generator consumed in loop order makes every interval depend on how many runs
    and contexts happened to precede it, so the same cell changes value when a second
    context is passed on the command line. Keying the seed to the cell removes that.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8]
    return np.random.default_rng(seed + int.from_bytes(digest, "big") % (2 ** 32))


def _boot(values: np.ndarray, rng: np.random.Generator, n_boot: int):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.array([rng.choice(values, len(values), replace=True).mean()
                      for _ in range(n_boot)])
    return float(values.mean()), *np.quantile(draws, [0.025, 0.975])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--contexts", nargs="+", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split", default="test",
                        help="one split, or 'all' for every gated item that was patched. "
                             "'all' is better powered but is not held out: the layer band "
                             "was fixed on a pilot, so those items are not independent of "
                             "the analysis choices. Report it beside the test estimate, "
                             "never instead of it.")
    args = parser.parse_args()

    rows = []
    for run in sorted(Path(args.results).iterdir()):
        path = run / "patching" / "causal_patching.csv"
        if not path.exists():
            continue
        model = run.name.split("_")[0]
        truth = GATE_TRUTH.get(model, 1)
        sign = 1.0 if truth == 1 else -1.0          # report toward the correct answer
        d = pd.read_csv(path)
        d = d.assign(layer=pd.to_numeric(d.layer_or_window, errors="coerce"))
        d = d[d.truth == truth]
        if args.split != "all":
            d = d[d.split == args.split]

        for context in args.contexts:
            sub = d[d.context == context]
            if sub.empty:
                continue
            single = sub[sub.experiment == "single_factorial"]

            # site effects, averaged over all three Latin-bearing donors as the paper does
            for site in ("E1", "E2", "Q", "READOUT"):
                values = sign * _item_means(single[single.site == site], *BAND)
                mean, low, high = _boot(
                    values.to_numpy(),
                    _rng(args.seed, run.name, context, site, args.split),
                    args.n_boot)
                rows.append({"run": run.name, "context": context, "quantity": site,
                             "donors": "all three", "mean": mean, "ci_low": low,
                             "ci_high": high, "n_items": len(values)})

            # entity specificity: same donor, same layers, same items on both sides
            correct = sign * _item_means(
                single[(single.site == "E2") & (single.source_condition == "LATLAT")], *BAND)
            wrong = sign * _item_means(
                sub[sub.experiment == "wrong_entity_control"], *BAND)
            shared = correct.index.intersection(wrong.index)
            if len(shared):
                for label, values in (("E2, Lat--Lat donor", correct.loc[shared]),
                                      ("wrong-entity control", wrong.loc[shared]),
                                      ("E2 minus wrong-entity",
                                       correct.loc[shared] - wrong.loc[shared])):
                    mean, low, high = _boot(
                        values.to_numpy(),
                        _rng(args.seed, run.name, context, label, args.split),
                        args.n_boot)
                    rows.append({"run": run.name, "context": context, "quantity": label,
                                 "donors": "Lat--Lat", "mean": mean, "ci_low": low,
                                 "ci_high": high, "n_items": len(shared)})

    if not rows:
        raise SystemExit("no patching tables found under --results")

    out = pd.DataFrame(rows)
    for run, block in out.groupby("run"):
        print(f"\n{run}")
        for _, r in block.iterrows():
            star = "" if (r.ci_low <= 0 <= r.ci_high) else "  *"
            print(f"  {r.context:3s} {r.quantity:22s} {r['mean']:+6.2f} "
                  f"[{r.ci_low:+6.2f}, {r.ci_high:+6.2f}]  n={int(r.n_items):3d}{star}")
    print("\n  * interval excludes zero")
    path = Path(args.results) / "patching_bootstrap.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""How far each item would have to move along the correction direction, against the grid.

Section 7 adds a fixed direction at a tuned scale and recovers little. That null is only
interpretable if the scales searched were large enough. The patching experiments give a
reference: replacing the state outright works, and the coordinate of that replacement
along the correction direction is the distance the additive vector would have to cover to
match it.

For each held-out item this computes

    c_i = u . (h_LatLat - h_DevDev)

at the second entity and a given layer, where u is the unit correction direction learned
on the training split, the same object section 7 adds. If typical c_i sit far above the
largest alpha in the grid, "unresolved" understates the problem and the grid was simply
too short. If they sit inside it, scale is not the explanation and the site or the
direction is.

Runs where the states are, so on the GPU box rather than locally.

    python analysis/donor_coordinates.py --run-dir results/qwen_mr --contexts mr en \\
        --layers 12 13 14 15 16 17 --alphas 0.5 1.0 1.5 2.0 3.0 4.0 6.0 8.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS = ["E1", "E2", "Q", "READOUT"]


def learn_vectors(states: np.ndarray, meta: pd.DataFrame) -> dict:
    """The section 7 vectors, reproduced from 07_run_recovery.py."""
    vectors = {}
    train = meta.loc[meta["split"] == "train"]
    for context in sorted(train.context.unique()):
        subset = train.loc[train["context"] == context]
        idx = {
            condition: subset.loc[subset["condition"] == condition]
            .sort_values(["fact_id", "truth"])["state_row"].to_numpy(dtype=int)
            for condition in ("DEVDEV", "LATDEV", "DEVLAT", "LATLAT")
        }
        f32 = lambda rows: states[rows].astype(np.float32)
        full = f32(idx["LATLAT"]) - f32(idx["DEVDEV"])
        first = f32(idx["LATDEV"]) - f32(idx["DEVDEV"])
        second = f32(idx["DEVLAT"]) - f32(idx["DEVDEV"])
        vectors[(context, "full")] = full.mean(axis=0)
        vectors[(context, "first_entity")] = first.mean(axis=0)
        vectors[(context, "second_entity")] = second.mean(axis=0)
        vectors[(context, "additive")] = (vectors[(context, "first_entity")]
                                          + vectors[(context, "second_entity")])
    return vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--contexts", nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--variant", default="full",
                    choices=["full", "additive", "first_entity", "second_entity"])
    ap.add_argument("--split", default="test",
                    choices=["train", "validation", "test", "all"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    meta = pd.read_csv(args.run_dir / "states" / "pair_state_metadata.csv")
    states = np.load(args.run_dir / "states" / "pair_states.npy", mmap_mode="r")
    vectors = learn_vectors(states, meta)
    e2 = POSITIONS.index("E2")
    grid_max = max(args.alphas)

    rows = []
    print(f"correction direction: {args.variant}, learned on train, "
          f"evaluated on {args.split}")
    print(f"scale grid runs to {grid_max}\n")
    print(f"{'ctx':5}{'layer':>6}{'n':>6}{'median':>9}{'mean':>9}{'p90':>9}"
          f"{'|u|':>9}{'above grid':>12}")

    for context in args.contexts:
        u = vectors[(context, args.variant)][:, e2, :]        # [layers, d]
        sel = meta[meta.context == context]
        if args.split != "all":
            sel = sel[sel.split == args.split]
        pos = sel[(sel.condition == "LATLAT")].set_index(["fact_id", "truth"]).state_row
        neg = sel[(sel.condition == "DEVDEV")].set_index(["fact_id", "truth"]).state_row
        shared = pos.index.intersection(neg.index)
        if len(shared) == 0:
            print(f"{context:5}  no paired states in this split")
            continue

        for layer in args.layers:
            raw = u[layer].astype(np.float32)
            norm = float(np.linalg.norm(raw))
            if norm < 1e-8:
                continue
            unit = raw / norm
            diff = (states[pos.loc[shared].to_numpy(), layer, e2].astype(np.float32)
                    - states[neg.loc[shared].to_numpy(), layer, e2].astype(np.float32))
            c = diff @ unit
            frac = float((np.abs(c) > grid_max).mean())
            print(f"{context:5}{layer:>6}{len(c):>6}{np.median(c):>9.2f}{c.mean():>9.2f}"
                  f"{np.quantile(c, 0.9):>9.2f}{norm:>9.2f}{frac * 100:>11.0f}%")
            rows.append(dict(context=context, layer=layer, variant=args.variant,
                             split=args.split, n_items=len(c),
                             median_coord=float(np.median(c)), mean_coord=float(c.mean()),
                             p10_coord=float(np.quantile(c, 0.1)),
                             p90_coord=float(np.quantile(c, 0.9)),
                             vector_norm=norm, grid_max=grid_max,
                             fraction_above_grid=frac))

    if not rows:
        raise SystemExit("no coordinates computed; check --contexts and --layers")
    out = args.out or (args.run_dir / "recovery" / "donor_coordinates.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("\nRead it this way: the coordinate is in the same units as alpha, because the")
    print("direction is unit length in both. A median far above the grid maximum means")
    print("the search never reached the distance the whole patch covers, and the")
    print("section 7 null is about the grid. A median inside the grid means scale is not")
    print("the explanation.")


if __name__ == "__main__":
    main()

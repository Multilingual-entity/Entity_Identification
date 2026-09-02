"""Collect finished runs into the cross-language comparison.

Twenty-odd separate run directories are not a result. This turns them into one table:
the size of the script gap in every language, measured the same way, ranked.

The gap is reported as d-prime rather than raw accuracy. Models differ in how readily
they answer yes, and languages differ in how hard their entities are, so accuracy mixes
knowledge with response threshold and cannot be compared across cells. d-prime separates
the two. Entities also differ between languages under the hub-and-spoke design, which is
sound because the gap is a within-entity contrast: the same person, two scripts.

Expects run directories named <model>_<lang>, each holding tables/behavior.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent


def dprime(hit: float, fa: float, n: int) -> float:
    e = 0.5 / max(n, 1)
    return float(norm.ppf(np.clip(hit, e, 1 - e)) - norm.ppf(np.clip(fa, e, 1 - e)))


def bootstrap_gap(frame: pd.DataFrame, draws: int, rng) -> tuple:
    """Cluster bootstrap over entities, since prompts from one entity are not independent."""
    facts = frame.fact_id.unique()
    wide = (frame.groupby(["fact_id", "condition", "truth"]).correct.mean()
            .unstack(["condition", "truth"]).reindex(facts))
    need = [("LATLAT", 1), ("LATLAT", 0), ("DEVDEV", 1), ("DEVDEV", 0)]
    if any(c not in wide.columns for c in need):
        return float("nan"), float("nan"), float("nan")
    arr = {c: wide[c].to_numpy() for c in need}
    n = len(facts)
    point = (dprime(np.nanmean(arr[("LATLAT", 1)]), 1 - np.nanmean(arr[("LATLAT", 0)]), n)
             - dprime(np.nanmean(arr[("DEVDEV", 1)]), 1 - np.nanmean(arr[("DEVDEV", 0)]), n))
    out = np.empty(draws)
    for b in range(draws):
        k = rng.integers(0, n, n)
        out[b] = (dprime(np.nanmean(arr[("LATLAT", 1)][k]), 1 - np.nanmean(arr[("LATLAT", 0)][k]), n)
                  - dprime(np.nanmean(arr[("DEVDEV", 1)][k]), 1 - np.nanmean(arr[("DEVDEV", 0)][k]), n))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi


def summarise(run_dir: Path, draws: int, rng) -> list:
    beh = run_dir / "tables" / "behavior.csv"
    if not beh.exists():
        return []
    cols = ["fact_id", "context", "condition", "truth", "correct"]
    df = pd.read_csv(beh, usecols=lambda c: c in cols + ["origin"])
    meta_path = run_dir / "model_run_metadata.json"
    model = json.loads(meta_path.read_text())["model_key"] if meta_path.exists() else "?"

    name = run_dir.name
    lang = name.split("_", 1)[1] if "_" in name else name

    rows = []
    for prompt_lang, g in df.groupby("context"):
        acc = g.groupby(["condition", "truth"]).correct.mean()
        point, lo, hi = bootstrap_gap(g, draws, rng)
        row = {"language": lang, "model": model, "prompt_language": prompt_lang,
               "n_entities": g.fact_id.nunique(),
               "native_pos": round(acc.get(("DEVDEV", 1), np.nan) * 100, 1),
               "native_neg": round(acc.get(("DEVDEV", 0), np.nan) * 100, 1),
               "latin_pos": round(acc.get(("LATLAT", 1), np.nan) * 100, 1),
               "latin_neg": round(acc.get(("LATLAT", 0), np.nan) * 100, 1),
               "dprime_gap": round(point, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
               "excludes_zero": int(lo > 0)}
        if "origin" in g.columns:
            for origin, og in g.groupby("origin"):
                p, _, _ = bootstrap_gap(og, 200, rng)
                row[f"gap_{origin}"] = round(p, 3)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="directory holding the <model>_<lang> run directories")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "cross_language.csv")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dirs = sorted(d for d in args.results.iterdir() if d.is_dir())
    rows = []
    for d in dirs:
        got = summarise(d, args.draws, rng)
        print(f"  {d.name}: {len(got)} cell(s)" if got else f"  {d.name}: no behaviour table")
        rows.extend(got)
    if not rows:
        raise SystemExit("no runs found")

    out = pd.DataFrame(rows).sort_values(["prompt_language", "dprime_gap"], ascending=[True, False])
    out.to_csv(args.out, index=False)

    print()
    english = out[out.prompt_language == "en"]
    if len(english):
        print("=" * 74)
        print("SCRIPT GAP UNDER ENGLISH PROMPTS, ranked. The one comparison that holds the")
        print("prompt constant across every language.")
        print("=" * 74)
        cols = ["language", "model", "n_entities", "native_pos", "latin_pos",
                "dprime_gap", "ci_low", "ci_high", "excludes_zero"]
        print(english[[c for c in cols if c in english.columns]].to_string(index=False))

    native = out[out.prompt_language != "en"]
    if len(native):
        print()
        print("Under each language's own prompt:")
        cols = ["language", "model", "prompt_language", "dprime_gap", "ci_low", "ci_high"]
        print(native[[c for c in cols if c in native.columns]].to_string(index=False))

    if "gap_native" in out.columns and "gap_foreign" in out.columns:
        print()
        print("=" * 74)
        print("ORIGIN CONTRAST. The account predicts a positive gap for foreign entities,")
        print("whose canonical form is Latin, and a smaller or reversed gap for native")
        print("entities, whose canonical form is the native script.")
        print("=" * 74)
        oc = out.dropna(subset=["gap_native", "gap_foreign"]).copy()
        oc["reversed"] = (oc.gap_native < 0).astype(int)
        print(oc[["language", "model", "prompt_language", "gap_foreign",
                  "gap_native", "reversed"]].to_string(index=False))
        rev = sorted(oc[oc["reversed"] == 1].language.unique())
        if rev:
            print(f"\nreversed for native entities in: {', '.join(rev)}")
            print("  This is the prediction that distinguishes the canonical-form account")
            print("  from 'non-Latin script is simply harder'.")
        else:
            print("\nno language shows a reversal on native entities.")

    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

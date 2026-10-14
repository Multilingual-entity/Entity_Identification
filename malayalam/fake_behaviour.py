"""Write a synthetic behavior.csv so the analysis is exercised before the GPU run.

The numbers are invented. The point is that the shapes, columns, joins and the per-polarity
eligibility mask behave, and that the two known-in-advance cells come out right: the AUROC
of a script condition with a built-in effect must exceed chance, and the shared English
Lat--Lat arm must come out at exactly zero in the language contrast.

yes_minus_no_margin is the raw yes-logit minus no-logit, as score_prompt_frame computes it:
positive means the model answered yes, whatever the truth of the item. Getting this sign
wrong makes the AUROC come out near zero while d' looks healthy, which is how the first
version of this file read.

    python fake_behaviour.py --root <run root> --results <run root>/_fake
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# A script effect built in deliberately, so a contrast that reports nothing is a bug in
# the analysis rather than a property of the fake data. It shrinks the yes-margin of a
# same-person item written in the native script, which is the paper's finding in shape.
EFFECT = {"DEVDEV": -1.6, "DEVLAT": -0.7, "LATDEV": -0.5, "LATLAT": 0.0}
SHARED = ("LATLAT", "en")   # the arm the two runs ask with identical prompts


def synthesise(prompts: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = prompts.copy()
    entity = pd.Series(rng.normal(0, 0.8, prompts.fact_id.nunique()),
                       index=sorted(prompts.fact_id.unique()))
    # Positive items attract yes, negative items attract no, and the script penalty eats
    # into the positive evidence only.
    margin = (np.where(out.truth == 1, 1.8, -1.8)
              + out.fact_id.map(entity)
              + np.where(out.truth == 1, out.condition.map(EFFECT), 0.0)
              + rng.normal(0, 0.9, len(out)))
    out["yes_minus_no_margin"] = margin
    out["pred_semantic"] = np.where(margin > 0, "yes", "no")
    out["correct"] = (out.pred_semantic == out.correct_semantic).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()

    frames = {}
    for name, run, seed in (("ml", "qwen_ml", 7), ("mr380", "qwen_mr380", 11)):
        prompts = pd.read_csv(args.root / f"prompts_{name}.csv", dtype={"fact_id": str})
        frames[name] = synthesise(prompts, seed)

    # Reality: the English Lat--Lat prompts are byte-identical between the two runs, so
    # the model returns the same margins. Copying them across makes the analysis's
    # shared-arm check discriminating rather than an artefact of one shared seed.
    keys = ["fact_id", "truth", "paraphrase_id", "yes_letter"]
    ml, mr = frames["ml"], frames["mr380"]
    mask_ml = (ml.condition == SHARED[0]) & (ml.context == SHARED[1])
    mask_mr = (mr.condition == SHARED[0]) & (mr.context == SHARED[1])
    source = ml.loc[mask_ml].set_index(keys)
    target = mr.loc[mask_mr].set_index(keys)
    for column in ("yes_minus_no_margin", "pred_semantic", "correct"):
        target[column] = source[column]
    mr = pd.concat([mr.loc[~mask_mr], target.reset_index()], ignore_index=True)
    frames["mr380"] = mr

    for name, run in (("ml", "qwen_ml"), ("mr380", "qwen_mr380")):
        tables = args.results / run / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(tables / "behavior.csv", index=False, encoding="utf-8")
        print(f"  {run}: {len(frames[name])} synthetic rows -> "
              f"{tables / 'behavior.csv'}")
    print(f"  the {SHARED[0]}/{SHARED[1]} arm is shared between the two, as it is in a "
          f"real pair of runs")


if __name__ == "__main__":
    main()

"""Recompute the patching rows of tab:models under the convention tab:profile uses.

tab:models is one of the tables paper_tables.py does not generate, so nothing has ever
checked it. Its Qwen English peak (layer 12, 3.93) and Hindi peak (5.00) reproduce under
no convention tried here, while tab:profile and the Gemma appendix both give layer 16 at
4.63 and layer 17 at 6.41. This prints the values under the documented convention so the
table can be corrected from data rather than guessed at.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GATE_TRUTH = {"qwen": 1, "gemma": 0, "llama": 1}
RUNS = [("Qwen2.5-7B", "qwen"), ("Gemma-2-2b", "gemma"), ("Llama-3.1-8B", "llama")]


def peak(model: str, ctx: str, tree: str, run: str):
    path = ROOT / f"{tree}/{run}/patching/causal_patching.csv"
    if not path.exists():
        return None, None
    d = pd.read_csv(path)
    d = d.assign(layer=pd.to_numeric(d.layer_or_window, errors="coerce"))
    truth = GATE_TRUTH[model]
    sign = 1.0 if truth == 1 else -1.0
    s = d[(d.experiment == "single_factorial") & (d.truth == truth)
          & (d.context == ctx) & (d.site == "E2")
          & (d.source_condition == "LATLAT")]
    if s.empty:
        return None, None
    g = sign * s.groupby("layer").delta_margin.mean()
    return int(g.idxmax()), round(float(g.max()), 2)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Second-entity peak, Lat--Lat source, pooled over splits "
          "(the tab:profile convention)\n")
    print(f"{'row':44}{'Qwen':>16}{'Gemma':>16}{'Llama':>16}")
    for label, ctx, tree in (("Marathi", "mr", "results_mr"),
                             ("Hindi", "hi", "results_hi"),
                             ("English (Marathi run)", "en", "results_mr")):
        layers, effects = [], []
        for _, model in RUNS:
            suffix = "mr" if tree.endswith("mr") else "hi"
            L, v = peak(model, ctx, tree, f"{model}_{suffix}")
            layers.append("---" if L is None else str(L))
            effects.append("---" if v is None else f"{v:+.2f}")
        print(f"{'peak layer, ' + label:44}" + "".join(f"{x:>16}" for x in layers))
        print(f"{'peak effect, ' + label:44}" + "".join(f"{x:>16}" for x in effects))
    print()
    print("tab:models currently states: English peak layer 12 effect 3.93, "
          "Hindi peak 5.00.")
    print("tab:profile and the Gemma appendix state 4.63 (English, layer 16) and "
          "6.41 (Hindi, layer 17).")


if __name__ == "__main__":
    main()

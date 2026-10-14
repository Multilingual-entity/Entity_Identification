"""Emit tab:control and tab:site over all four arms, from the control CSVs.

The Marathi arms came first and the Hindi arms were run afterwards; both are built here
from their own files so the tables and the prose cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2] / "results_mr"

# unmodified held-out baselines, recomputed in verify_window_claims.py
BASE = {
    "mr": {"DD": (0.065, 0.994), "LL": (0.340, 0.997)},
    "en_mr": {"DD": (0.283, 1.000), "LL": (0.529, 1.000)},
    "hi": {"DD": (0.122, 1.000), "LL": (0.401, 1.000)},
    "en_hi": {"DD": (0.260, 1.000), "LL": (0.521, 1.000)},
}
# the published Dev--Dev window arm, from tab:winreplicate
PUB = {"mr": (0.335, 0.985), "en_mr": (0.590, 0.976),
       "hi": (0.546, 0.869), "en_hi": (0.585, 0.974)}

LABEL = {"mr": "Marathi", "en_mr": "English, mr corpus",
         "hi": "Hindi", "en_hi": "English, hi corpus"}


def cell(path: Path, context: str) -> tuple[float, float]:
    d = pd.read_csv(path)
    a = d[d.vector_kind == "additive"]
    pos = float(a[(a.context == context) & (a.truth == 1)].accuracy.iloc[0])
    neg = float(a[(a.context == context) & (a.truth == 0)].accuracy.iloc[0])
    return pos, neg


LL = {
    "mr": cell(ROOT / "qwen_mr" / "ctrl_latlat_summary.csv", "mr"),
    "en_mr": cell(ROOT / "qwen_mr" / "ctrl_latlat_summary.csv", "en"),
    "hi": cell(ROOT / "qwen_hi" / "hi_ctrl_latlat_hi.csv", "hi"),
    "en_hi": cell(ROOT / "qwen_hi" / "hi_ctrl_latlat_en.csv", "en"),
}
SITE = {}
for arm, (run, fname, ctx) in {
        "mr": ("qwen_mr", "ctrl_site_{s}.csv", "mr"),
        "en_mr": ("qwen_mr", "ctrl_site_{s}.csv", "en"),
        "hi": ("qwen_hi", "hi_site_{s}_hi.csv", "hi"),
        "en_hi": ("qwen_hi", "hi_site_{s}_en.csv", "en")}.items():
    SITE[arm] = {s: cell(ROOT / run / fname.format(s=s), ctx)
                 for s in ("E1", "Q", "READOUT")}

# --- tab:control -----------------------------------------------------------
gains = {a: (LL[a][0] - BASE[a]["LL"][0], PUB[a][0] - BASE[a]["DD"][0])
         for a in LABEL}
ll_only = [a for a in LABEL if gains[a][0] > gains[a][1]]

print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{The same intervention applied to the failing condition and to the healthy "
      r"one, on the 120 held-out entities of each arm, at the scale that arm's validation "
      r"selected. Positives are SAME pairs. The vector raises Lat--Lat acceptance in every "
      r"arm, by " + f"{min(g[0] for g in gains.values()):.3f} to "
      f"{max(g[0] for g in gains.values()):.3f}" + r", which is why we do not describe it "
      r"as closing the script gap: a recovery stated as a fraction of the gap to Lat--Lat "
      r"presumes that reference is fixed, and it is not. The two gains are not equal, "
      r"though. " + (LABEL[ll_only[0]] if len(ll_only) == 1 else "Some arms") +
      r" is the only arm where the healthy condition gains more than the failing one; in "
      r"the other three the Dev--Dev gain is larger, so we claim that the intervention is "
      r"not specific to the failing condition and not that the two conditions gain "
      r"equally.}")
print(r"\label{tab:control}")
print(r"\begin{tabular}{llcc}")
print(r"\toprule")
print(r"Arm & Condition & Positives & Negatives\\")
for arm in ("mr", "en_mr", "hi", "en_hi"):
    print(r"\midrule")
    print(r"\multirow{4}{*}{" + LABEL[arm] + r"}")
    print(f" & Dev--Dev, unmodified & {BASE[arm]['DD'][0]:.3f} & {BASE[arm]['DD'][1]:.3f}\\\\")
    print(f" & Dev--Dev, $+$ vector & \\textbf{{{PUB[arm][0]:.3f}}} & {PUB[arm][1]:.3f}\\\\")
    print(f" & Lat--Lat, unmodified & {BASE[arm]['LL'][0]:.3f} & {BASE[arm]['LL'][1]:.3f}\\\\")
    print(f" & Lat--Lat, $+$ vector & \\textbf{{{LL[arm][0]:.3f}}} & {LL[arm][1]:.3f}\\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# --- tab:site --------------------------------------------------------------
e1 = [SITE[a]["E1"][0] - BASE[a]["DD"][0] for a in LABEL]
ro_better = [a for a in LABEL
             if SITE[a]["READOUT"][0] >= PUB[a][0] - 0.01]

print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{The same vector, window and scale added at four token positions, "
      r"Dev--Dev inputs, 120 held-out entities per arm. The second entity is the position "
      r"the vector was estimated at and the one \S\ref{sec:patching} localises; the other "
      r"three are controls. One result holds in every arm: the first entity moves accuracy "
      r"by at most " + f"{max(e1):.3f}" + r", so the gain is not available anywhere in the "
      r"prompt. The readout is weaker and less consistent: it lowers accuracy in three "
      r"arms but matches the second entity under "
      + (LABEL[ro_better[0]] if len(ro_better) == 1 else "some") +
      r" prompts, " + f"{SITE['hi']['READOUT'][0]:.3f} against {PUB['hi'][0]:.3f}" +
      r" at comparable negatives, which we report without an account for it. We therefore "
      r"read the intervention as depending on position rather than as acting only at the "
      r"second entity.}")
print(r"\label{tab:site}")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{Native prompts} & \multicolumn{2}{c}{English prompts}\\")
print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
print(r"Site & positives & negatives & positives & negatives\\")
for corpus, native, eng in (("Marathi corpus", "mr", "en_mr"),
                            ("Hindi corpus", "hi", "en_hi")):
    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\em " + corpus + r"}\\")
    print(f"unmodified & {BASE[native]['DD'][0]:.3f} & {BASE[native]['DD'][1]:.3f}"
          f" & {BASE[eng]['DD'][0]:.3f} & {BASE[eng]['DD'][1]:.3f}\\\\")
    print(f"\\textbf{{second entity}} & \\textbf{{{PUB[native][0]:.3f}}} & {PUB[native][1]:.3f}"
          f" & \\textbf{{{PUB[eng][0]:.3f}}} & {PUB[eng][1]:.3f}\\\\")
    for s, name in (("E1", "first entity"), ("Q", "question mark"), ("READOUT", "readout")):
        print(f"{name} & {SITE[native][s][0]:.3f} & {SITE[native][s][1]:.3f}"
              f" & {SITE[eng][s][0]:.3f} & {SITE[eng][s][1]:.3f}\\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

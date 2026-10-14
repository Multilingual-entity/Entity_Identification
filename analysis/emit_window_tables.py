"""Emit the LaTeX for every table from the 2026-09-30 window run.

Written to a file and run with python, never pasted through a shell: the Bash tool
strips backslashes, and every line here is backslashes.

    python analysis/emit_window_tables.py > window_tables.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_mr" / "overnight_20260930_0537"
REC = ROOT / "results_mr" / "qwen_mr" / "recovery_window"
BS = chr(92)
NL = BS + BS


def row(*cells: str) -> str:
    return " & ".join(cells) + NL


def emit(text: str) -> None:
    print(text)


def sweep_table() -> None:
    t = pd.read_csv(OUT / "1d_tuning_grid.csv")
    g = t.groupby(["context", "alpha", "truth"]).correct.mean().unstack("truth")
    emit(BS + "begin{table}[h]")
    emit(BS + "centering" + BS + "small")
    emit(BS + "caption{Validation sweep for the vector applied across layers 12--17, "
         "Qwen, 60 tuning items. The published sweep applied the vector at a single "
         "layer and stopped at $" + BS + "alpha{=}8$; both changes are needed. At the "
         "published scale $" + BS + "alpha{=}6$ the window alone lifts Marathi from "
         "0.083 to 0.233, and the extended grid lifts it again to 0.392. Positive-pair "
         "accuracy peaks and turns over, which the published grid never did.}")
    emit(BS + "label{tab:winsweep}")
    emit(BS + "begin{tabular}{lcc|cc}")
    emit(BS + "toprule")
    emit(row("", BS + "multicolumn{2}{c|}{Marathi prompts}",
             BS + "multicolumn{2}{c}{English prompts}").rstrip(NL) + NL)
    emit(BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-5}")
    emit(row("Scale", "positives", "negatives", "positives", "negatives"))
    emit(BS + "midrule")
    # g is indexed (context, alpha) with truth as columns, so the peak positive-pair
    # scale per context is the argmax of column 1.
    best = {c: g.loc[c][1].idxmax() for c in ("mr", "en")}
    for a in sorted(t.alpha.unique()):
        m, e = g.loc[("mr", a)], g.loc[("en", a)]
        mp = f"{BS}textbf{{{m[1]:.3f}}}" if a == best["mr"] else f"{m[1]:.3f}"
        ep = f"{BS}textbf{{{e[1]:.3f}}}" if a == best["en"] else f"{e[1]:.3f}"
        emit(row(f"$_{BS}alpha{{=}}{a:g}$".replace("$_", "$"),
                 mp, f"{m[0]:.3f}", ep, f"{e[0]:.3f}"))
    emit(BS + "bottomrule")
    emit(BS + "end{tabular}")
    emit(BS + "end{table}")
    emit("")


def test_table() -> None:
    v = pd.read_csv(REC / "vector_recovery_test_summary.csv")
    p = pd.read_csv(REC / "prompt_anchor_summary.csv")
    p = p[p.split == "test"]

    def anchor(ctx: str, name: str, truth: int) -> float:
        s = p[(p.context == ctx) & (p.anchor == name) & (p.truth == truth)]
        return float(s.accuracy.iloc[0])

    def vec(ctx: str, kind: str, truth: int) -> float:
        s = v[(v.context == ctx) & (v.vector_kind == kind) & (v.truth == truth)]
        return float(s.accuracy.iloc[0])

    emit(BS + "begin{table}[h]")
    emit(BS + "centering" + BS + "small")
    emit(BS + "caption{Held-out test recovery with the vector applied across layers "
         "12--17, Qwen, 120 items, at the scale selected on validation ($"
         + BS + "alpha{=}12$ Marathi, $" + BS + "alpha{=}24$ English). The "
         "{" + BS + "em additive} row is the pre-registered variant, the one selection "
         "optimised; the others are reported for completeness and must not be read as "
         "the result. Marathi reaches the Lat--Lat reference and English exceeds it.}")
    emit(BS + "label{tab:wintest}")
    emit(BS + "begin{tabular}{lcc|cc}")
    emit(BS + "toprule")
    emit(row("", BS + "multicolumn{2}{c|}{Marathi}",
             BS + "multicolumn{2}{c}{English}").rstrip(NL) + NL)
    emit(BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-5}")
    emit(row("Condition", "positives", "negatives", "positives", "negatives"))
    emit(BS + "midrule")
    emit(row("No intervention",
             f"{anchor('mr', 'none', 1):.3f}", f"{anchor('mr', 'none', 0):.3f}",
             f"{anchor('en', 'none', 1):.3f}", f"{anchor('en', 'none', 0):.3f}"))
    emit(BS + "midrule")
    for kind, label in [("full", "Vector, full"),
                        ("first_entity", "Vector, first entity"),
                        ("second_entity", "Vector, second entity")]:
        emit(row(label,
                 f"{vec('mr', kind, 1):.3f}", f"{vec('mr', kind, 0):.3f}",
                 f"{vec('en', kind, 1):.3f}", f"{vec('en', kind, 0):.3f}"))
    emit(row(BS + "textbf{Vector, additive}",
             f"{BS}textbf{{{vec('mr', 'additive', 1):.3f}}}",
             f"{vec('mr', 'additive', 0):.3f}",
             f"{BS}textbf{{{vec('en', 'additive', 1):.3f}}}",
             f"{vec('en', 'additive', 0):.3f}"))
    emit(BS + "midrule")
    emit(row("Lat--Lat reference",
             f"{anchor('mr', 'latin_only_ceiling', 1):.3f}",
             f"{anchor('mr', 'latin_only_ceiling', 0):.3f}",
             f"{anchor('en', 'latin_only_ceiling', 1):.3f}",
             f"{anchor('en', 'latin_only_ceiling', 0):.3f}"))
    emit(BS + "bottomrule")
    emit(BS + "end{tabular}")
    emit(BS + "end{table}")
    emit("")
    # The percentages quoted in the prose, computed here so they cannot drift.
    for ctx in ("mr", "en"):
        base, ceil = anchor(ctx, "none", 1), anchor(ctx, "latin_only_ceiling", 1)
        got = vec(ctx, "additive", 1)
        pct = 100 * (got - base) / (ceil - base)
        emit(f"% {ctx}: {base:.3f} -> {got:.3f}, reference {ceil:.3f}, "
             f"{pct:.0f} percent of the gap")
    emit("")


def donor_table() -> None:
    d = pd.read_csv(OUT / "1e_donor_coords.csv")
    emit(BS + "begin{table}[h]")
    emit(BS + "centering" + BS + "small")
    emit(BS + "caption{How far each held-out item must move along the rank-one "
         "direction to reach its Lat--Lat counterpart, in the same units as the scale "
         "$" + BS + "alpha$. The published grid stopped at $" + BS + "alpha{=}8$, "
         "roughly half the required distance at every layer, which accounts for the "
         "null without any appeal to the direction being wrong. The scales selected on "
         "validation, 12 for Marathi and 24 for English, bracket these medians.}")
    emit(BS + "label{tab:donorcoord}")
    emit(BS + "begin{tabular}{lccc}")
    emit(BS + "toprule")
    emit(row("Prompt language", "Layer", "Median coordinate", "Beyond grid"))
    emit(BS + "midrule")
    for ctx, label in (("mr", "Marathi"), ("en", "English")):
        s = d[d.context == ctx].sort_values("layer")
        for i, r in enumerate(s.itertuples()):
            emit(row(label if i == 0 else "", f"{int(r.layer)}",
                     f"{r.median_coord:.1f}", f"{r.fraction_above_grid:.2f}"))
        if ctx == "mr":
            emit(BS + "midrule")
    emit(BS + "bottomrule")
    emit(BS + "end{tabular}")
    emit(BS + "end{table}")
    emit("")


def refarm_table() -> None:
    e = pd.read_csv(OUT / "1e_reference_summary.csv")
    b = pd.read_csv(OUT / "1b_summary.csv")
    c = pd.read_csv(OUT / "1c_summary.csv")
    emit(BS + "begin{table}[h]")
    emit(BS + "centering" + BS + "small")
    emit(BS + "caption{The reference arm: the rank-one direction applied across layers "
         "12--17 with each item's own donor coordinate, which is the most informed "
         "additive intervention short of replacing the state. Rows one and two validate "
         "the window implementation against the single-layer result of "
         + BS + "S" + BS + "ref{sec:mediator}. The upper bound moves margins but flips "
         "few gated items, which is the tension discussed in "
         + BS + "S" + BS + "ref{sec:repair}.}")
    emit(BS + "label{tab:refarm}")
    emit(BS + "begin{tabular}{llccccc}")
    emit(BS + "toprule")
    emit(row("Configuration", "Prompt", "$n$", "whole $" + BS + "Delta$",
             "rank-one $" + BS + "Delta$", "share", "rank-one flips"))
    emit(BS + "midrule")
    for label, df in (("Layer 16 only", b), ("Layers 12--17", c)):
        s = df[df.group == "gated"]
        for r in s.itertuples():
            emit(row(label, "Marathi", f"{int(r.n_items)}", f"{r.delta_whole:.2f}",
                     f"{r.delta_subspace:.2f}", f"{r.share:.2f}",
                     f"{r.subspace_flip:.2f}"))
    emit(BS + "midrule")
    s = e[e.group == "gated"]
    for r in s.itertuples():
        emit(row("Layers 12--17, held out",
                 "Marathi" if r.context == "mr" else "English",
                 f"{int(r.n_items)}", f"{r.delta_whole:.2f}",
                 f"{r.delta_subspace:.2f}", f"{r.share:.2f}",
                 f"{r.subspace_flip:.2f}"))
    emit(BS + "bottomrule")
    emit(BS + "end{tabular}")
    emit(BS + "end{table}")
    emit("")


def nonce_table() -> None:
    n = pd.read_csv(OUT / "4b_behavior_summary.csv")
    r = pd.read_csv(ROOT / "results_mr" / "qwen_mr" / "tables" / "behavior_summary.csv")

    def cell(df: pd.DataFrame, ctx: str, cond: str, truth: int) -> str:
        s = df[(df.context == ctx) & (df.condition == cond) & (df.truth == truth)]
        return f"{s.accuracy.mean():.3f}" if len(s) else "--"

    emit(BS + "begin{table}[h]")
    emit(BS + "centering" + BS + "small")
    emit(BS + "caption{Nonce-name control, Qwen. 600 invented people built from "
         "recombined attested name parts, sharing no fact identifier with the real "
         "corpus. Read the negatives column first: on nonce names the model answers "
         "{" + BS + "em different} to essentially everything, so positive-pair accuracy "
         "falls in every condition, not only Dev--Dev. The script gap shrinks as a "
         "consequence of that shift rather than independently of it, and Marathi is "
         "uninformative at 0.036 even in the Lat--Lat condition.}")
    emit(BS + "label{tab:nonce}")
    emit(BS + "begin{tabular}{llcc|cc}")
    emit(BS + "toprule")
    emit(row("", "", BS + "multicolumn{2}{c|}{Real corpus}",
             BS + "multicolumn{2}{c}{Nonce corpus}").rstrip(NL) + NL)
    emit(BS + "cmidrule(lr){3-4}" + BS + "cmidrule(lr){5-6}")
    emit(row("Prompt", "Condition", "positives", "negatives",
             "positives", "negatives"))
    emit(BS + "midrule")
    for ctx, label in (("mr", "Marathi"), ("en", "English")):
        for i, cond in enumerate(("DEVDEV", "DEVLAT", "LATDEV", "LATLAT")):
            pretty = cond[:3].title() + "--" + cond[3:].title()
            emit(row(label if i == 0 else "", pretty,
                     cell(r, ctx, cond, 1), cell(r, ctx, cond, 0),
                     cell(n, ctx, cond, 1), cell(n, ctx, cond, 0)))
        if ctx == "mr":
            emit(BS + "midrule")
    emit(BS + "bottomrule")
    emit(BS + "end{tabular}")
    emit(BS + "end{table}")
    emit("")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sweep_table()
    test_table()
    donor_table()
    refarm_table()
    nonce_table()


if __name__ == "__main__":
    main()

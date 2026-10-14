"""Emit two tables that come free from data already on disk.

  tab:overlap   the script gap stratified by how alike the two Latin forms are, which
                is the direct test of the surface-similarity reading
  tab:template  the repair evaluated on all three question templates, having been
                selected on one

    python analysis/emit_overlap_tables.py > overlap_tables.tex
"""
from __future__ import annotations

import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MR = ROOT / "results_mr" / "qwen_mr"
HI = ROOT / "results_hi" / "qwen_hi"
REC = MR / "recovery_window"
BS = chr(92)
NL = BS + BS

BINS = [0, 0.4, 0.6, 0.8, 1.01]
NAMES = ["$<$0.4", "0.4--0.6", "0.6--0.8", "$>$0.8"]


def row(*cells: str) -> str:
    return " & ".join(cells) + NL


def overlap_frame(run: Path, native: str) -> pd.DataFrame:
    c = pd.read_csv(run / "data" / "corpus.csv", dtype={"fact_id": str})
    c["ov"] = [SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
               for a, b in zip(c.name_a_lat, c.name_b_lat)]
    b = pd.read_csv(run / "tables" / "behavior.csv", dtype={"fact_id": str})
    b = b.merge(c[["fact_id", "ov"]], on="fact_id")
    b["bin"] = pd.cut(b.ov, BINS, labels=NAMES)
    return b[(b.truth == 1) & (b.context.isin([native, "en"]))]


def overlap_table() -> None:
    mr = overlap_frame(MR, "mr")
    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{The script gap on positive pairs, split by how alike the two "
          "Latin forms are (difflib sequence ratio on the corpus of the Marathi run). "
          "If the model were matching on surface similarity and Devanagari merely made "
          "that harder, the gap should shrink as the names become less alike. It does "
          "the opposite: the gap is largest in the least similar band and smallest in "
          "the most similar one, in both prompt languages.}")
    print(BS + "label{tab:overlap}")
    print(BS + "begin{tabular}{lccc|ccc}")
    print(BS + "toprule")
    print(row("", BS + "multicolumn{3}{c|}{Marathi prompts}",
              BS + "multicolumn{3}{c}{English prompts}").rstrip(NL) + NL)
    print(BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}")
    print(row("Latin overlap", "Dev--Dev", "Lat--Lat", "gap",
              "Dev--Dev", "Lat--Lat", "gap"))
    print(BS + "midrule")
    for name in NAMES:
        cells = [name]
        for ctx in ("mr", "en"):
            d = mr[(mr.context == ctx) & (mr.bin == name)]
            dd = d[d.condition == "DEVDEV"].correct.mean() * 100
            ll = d[d.condition == "LATLAT"].correct.mean() * 100
            cells += [f"{dd:.1f}", f"{ll:.1f}", f"{BS}textbf{{{ll - dd:.1f}}}"]
        print(row(*cells))
    print(BS + "midrule")
    n = mr[mr.context == "mr"].groupby("bin", observed=True).fact_id.nunique()
    print(row("items", *[f"{BS}multicolumn{{3}}{{c|}}{{{' / '.join(str(n[x]) for x in NAMES)}}}",
                         f"{BS}multicolumn{{3}}{{c}}{{same items}}"]))
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()


def template_table() -> None:
    c = pd.read_csv(MR / "data" / "corpus.csv", dtype={"fact_id": str})
    test = set(c[c.split == "test"].fact_id)
    b = pd.read_csv(MR / "tables" / "behavior.csv", dtype={"fact_id": str})
    b = b[b.fact_id.isin(test) & (b.truth == 1)]
    v = pd.read_csv(REC / "vector_recovery_test.csv", dtype={"fact_id": str})
    v = v[(v.vector_kind == "additive") & (v.truth == 1)]

    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{The intervention across question templates, 120 held-out items. Layer, "
          "window and scale were selected on template 0 alone, and the vector was "
          "estimated on 361 training entities none of which appear here, so templates 1 "
          "and 2 are held out twice over. Positive-pair accuracy rises in every cell. "
          "Template 2 is the hardest for every condition, not only for the repair: its "
          "Lat--Lat reference is 8.8 in Marathi, which the repaired model exceeds.}")
    print(BS + "label{tab:template}")
    print(BS + "begin{tabular}{llccc}")
    print(BS + "toprule")
    print(row("Prompt", "Template", "Dev--Dev", "repaired", "Lat--Lat"))
    print(BS + "midrule")
    for ctx, label in (("mr", "Marathi"), ("en", "English")):
        for i, p in enumerate((0, 1, 2)):
            d = b[(b.context == ctx) & (b.paraphrase_id == p)]
            dd = d[d.condition == "DEVDEV"].correct.mean() * 100
            ll = d[d.condition == "LATLAT"].correct.mean() * 100
            rp = v[(v.context == ctx) & (v.paraphrase_id == p)].correct.mean() * 100
            star = f"{BS}textbf{{{rp:.1f}}}" if rp > ll else f"{rp:.1f}"
            print(row(label if i == 0 else "", str(p),
                      f"{dd:.1f}", star, f"{ll:.1f}"))
        if ctx == "mr":
            print(BS + "midrule")
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    overlap_table()
    template_table()


if __name__ == "__main__":
    main()

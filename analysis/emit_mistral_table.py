"""Emit the depth-matched Mistral tables.

    python analysis/emit_mistral_table.py > mistral_table.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cross-script-entity-matching" / "pipeline"))
from pipeline_common import scale_layer  # noqa: E402

CSV = ROOT / "results_mr" / "mistral24_mr" / "causal_patching_scaled.csv"
BS = chr(92)
NL = BS + BS


def frame() -> pd.DataFrame:
    d = pd.read_csv(CSV, dtype={"fact_id": str})
    return d.assign(layer=pd.to_numeric(d.layer_or_window, errors="coerce"))


def profile_table(d: pd.DataFrame) -> None:
    s = d[(d.experiment == "single_factorial") & (d.truth == 1)
          & (d.source_condition == "LATLAT") & (d.site == "E2")]
    g = s.groupby(["context", "layer"]).delta_margin.mean()
    layers = sorted({int(x) for x in g.index.get_level_values("layer")})
    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{Mistral-Small second-entity margin change by layer, depth-matched "
          "sweep. The band is layers 11--29 of 40, which is Qwen's 8--20 of 28 rescaled, "
          "so it covers layer 23, the depth-matched equivalent of Qwen's layer 16 that "
          "the earlier sweep did not reach. Nothing is there: every layer from 21 to 29 "
          "is within 0.02 of zero in both prompt languages. The effect peaks at layer 11 "
          "under Marathi prompts and 18 under English, both shallower in relative depth "
          "than Qwen's 12--17 of 28. 60 gated items per prompt language, Lat--Lat donor.}")
    print(BS + "label{tab:mistraldepth}")
    print(BS + "begin{tabular}{ccc}")
    print(BS + "toprule")
    print("Layer & Marathi & English" + NL)
    print(BS + "midrule")
    peak = {c: max(g[(c, L)] for L in layers) for c in ("mr", "en")}
    for L in layers:
        cells = [f"{L}"]
        for ctx in ("mr", "en"):
            v = g[(ctx, L)]
            cell = f"$-${abs(v):.2f}" if v < 0 else f"{v:.2f}"
            cells.append(BS + "textbf{" + cell + "}" if v == peak[ctx] else cell)
        print(" & ".join(cells) + NL)
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()


def summary_table(d: pd.DataFrame) -> None:
    lo, hi = scale_layer(10, 40), scale_layer(18, 40)
    band = d[(d.layer >= lo) & (d.layer <= hi)]
    sf = band[(band.experiment == "single_factorial") & (band.truth == 1)
              & (band.source_condition == "LATLAT")]
    print(BS + "begin{table}[h]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{Mistral-Small site effects over two bands. The upper block is the "
          "whole swept band, layers 11--29 of 40. The lower block is the sub-band "
          f"depth-matched to the layers Qwen's Table~{BS}ref{{tab:patchci}} averages, "
          f"10--18 of 28 rescaled to {lo}--{hi} of 40. The first entity is at zero in "
          "both. The second entity clears the question mark over the swept band, but not "
          "in Marathi on the depth-matched sub-band, where it has already decayed below "
          "it: Mistral locates this effect shallower than Qwen does, so matching Qwen's "
          "depth window misses the Marathi peak rather than centring on it. Magnitudes "
          "are roughly tenfold smaller than Qwen's 2.42 on either band, which the "
          "depth-matched sweep was run to establish and does. The final column is the "
          "paired entity-specificity contrast at the second entity, correct donor minus "
          "wrong-entity donor over the items present in both arms.}")
    print(BS + "label{tab:mistralsites}")
    print(BS + "begin{tabular}{lccccc}")
    print(BS + "toprule")
    print("Prompt & E1 & E2 & Q & R & E2 minus wrong entity" + NL)
    print(BS + "midrule")
    full = d[(d.layer >= 11) & (d.layer <= 29)]
    for blocklabel, frm in (("Swept band, 11--29", full),
                            (f"Depth-matched, {lo}--{hi}", band)):
        print(BS + "multicolumn{6}{l}{" + BS + "em " + blocklabel + "}" + NL)
        rows = frm[(frm.experiment == "single_factorial") & (frm.truth == 1)
                   & (frm.source_condition == "LATLAT")]
        for ctx, label in (("mr", "Marathi"), ("en", "English")):
            t = rows[rows.context == ctx]
            cells = []
            for site in ("E1", "E2", "Q", "READOUT"):
                v = t[t.site == site].delta_margin.mean()
                cells.append(f"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}")
            cor = t[t.site == "E2"].groupby("fact_id").delta_margin.mean()
            wrong = frm[(frm.experiment == "wrong_entity_control") & (frm.truth == 1)
                        & (frm.site == "E2") & (frm.context == ctx)]
            wrong = wrong.groupby("fact_id").delta_margin.mean()
            common = cor.index.intersection(wrong.index)
            diff = (cor[common] - wrong[common]).mean()
            cells.append(f"{diff:+.3f}")
            print(BS + "quad " + label + " & " + " & ".join(cells) + NL)
        if blocklabel.startswith("Swept"):
            print(BS + "midrule")
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table}")
    print()
    for ctx in ("mr", "en"):
        t = sf[(sf.context == ctx) & (sf.site == "E2")]
        print(f"% {ctx}: depth-matched E2 {t.delta_margin.mean():.3f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = frame()
    profile_table(d)
    summary_table(d)


if __name__ == "__main__":
    main()

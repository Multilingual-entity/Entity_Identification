"""STALE: the paper's tab:calibration caption has been corrected by hand since
this emitter last matched it. The paper says the offset is held to a floor the
vector was not, labels the threshold column an oracle diagnostic, names the AUROC
"entity-averaged", and carries the script-gap ranking difference. Splicing this
script's output over the paper would revert all four. Update this file to match
the paper before using it again."""

"""Emit the calibration table.

    python analysis/emit_calibration_table.py > calibration_table.tex
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BS = chr(92)
NL = BS + BS

_spec = importlib.util.spec_from_file_location("cb", HERE / "calibration_baseline.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    arms = [cb.arm(*x) for x in cb.ARMS]

    print(BS + "begin{table*}[t]")
    print(BS + "centering" + BS + "small")
    print(BS + "caption{Does the correction change what the model discriminates, or only "
          "where its decision boundary sits? AUROC is computed as in "
          + BS + "S" + BS + "ref{sec:behaviour}: the six prompts of an entity are averaged "
          "first, so the unit is the entity and the bootstrap resamples entities, 2000 "
          "draws. It is threshold-free and does not move when a constant is added to every "
          "margin. True- and false-positive rates are prompt-level, as in the accuracy "
          "tables, so the false-positive rate is one minus the negative-pair accuracy "
          "reported there. The offset is one constant per arm added to the output margin, "
          "chosen on the same validation entities by the same objective as the scale "
          "sweep, balanced accuracy subject to the same negative-pair floor. The last "
          "column is what the unmodified model reaches by moving its threshold alone to "
          "the intervention's own false-positive rate. Each cell gives its own "
          "false-positive rate in parentheses: the offset column is at the rate its "
          "selected constant produces, and the threshold-only column is at the "
          "intervention's rate by construction. The {\em change} column is the "
          "difference of unrounded AUROC values, so it need not equal the difference of "
          "the rounded figures elsewhere. Held-out test split.}")
    print(BS + "label{tab:calibration}")
    print(BS + "begin{tabular}{lcc|ccc}")
    print(BS + "toprule")
    print(" & " + BS + "multicolumn{2}{c|}{AUROC, entity level} & "
          + BS + "multicolumn{3}{c}{TPR (FPR)}" + NL)
    print(BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-6}")
    print("Arm & Dev--Dev & change & vector & offset & threshold only" + NL)
    print(BS + "midrule")
    for a in arms:
        best = max(a["tpr_vec"], a["tpr_off"], a["tpr_thr"])
        def mark(x):
            s = f"{x:.3f}"
            return BS + "textbf{" + s + "}" if abs(x - best) < 1e-9 else s
        print(" & ".join([
            a["label"],
            f"{a['auc_DEVDEV']:.3f}",
            f"${a['d_vector']:+.3f}$ $[{a['d_vector_lo']:+.3f}, {a['d_vector_hi']:+.3f}]$",
            f"{mark(a['tpr_vec'])} ({a['fpr_vec']:.3f})",
            f"{mark(a['tpr_off'])} ({a['fpr_off']:.3f})",
            f"{mark(a['tpr_thr'])} ({a['fpr_vec']:.3f})",
        ]) + NL)
    print(BS + "bottomrule")
    print(BS + "end{tabular}")
    print(BS + "end{table*}")
    print()
    for a in arms:
        print(f"% {a['label']}: script gap LL-DD {a['d_script']:+.3f} "
              f"[{a['d_script_lo']:+.3f}, {a['d_script_hi']:+.3f}]; "
              f"LatDev AUROC {a['auc_LATDEV']:.3f} vs LatLat {a['auc_LATLAT']:.3f}; "
              f"offset c={a['offset']:+.2f}; vector FPR {a['fpr_vec']:.3f}")


if __name__ == "__main__":
    main()

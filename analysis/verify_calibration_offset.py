"""Recompute tab:calibration's offset column, and assert its selection never reads test.

This column was wrong once. The floor was set to 1 - fpr_vec, where fpr_vec is the
vector's false-positive rate on the test split, so the evaluation split informed the
selection and the Marathi cell read 0.371 (0.018) instead of 0.312 (0.012). The last check
here fails if that form comes back.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
PAPER = Path.home() / "Downloads" / "when_entities_align" / "final_draft.tex"
SRC = HERE / "calibration_baseline.py"

WANT = {  # arm: (vector tpr, vector fpr, offset tpr, offset fpr, threshold tpr)
    "Marathi": (0.335, 0.015, 0.312, 0.012, 0.336),
    "English (mr corpus)": (0.590, 0.024, 0.714, 0.037, 0.644),
    "Hindi": (0.546, 0.131, 0.656, 0.178, 0.594),
}
FLOORS = {"Marathi": 0.975, "English (mr corpus)": 0.975, "Hindi": 0.925}

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


sys.path.insert(0, str(HERE))
import importlib.util

spec = importlib.util.spec_from_file_location("cb", SRC)
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)

for args in cb.ARMS:
    label = args[0]
    a = cb.arm(*args)
    vt, vf, ot, of, tt = WANT[label]
    for name, got, want in (("vector tpr", a["tpr_vec"], vt),
                            ("vector fpr", a["fpr_vec"], vf),
                            ("offset tpr", a["tpr_off"], ot),
                            ("offset fpr", a["fpr_off"], of),
                            ("threshold tpr", a["tpr_thr"], tt)):
        check(f"tab:calibration {label} {name} = {want}",
              abs(got - want) < 0.0006, f"{got:.3f}")
    check(f"tab:calibration {label} offset floor = {FLOORS[label]}",
          abs(a["offset_floor"] - FLOORS[label]) < 1e-9, f"{a['offset_floor']}")

# the structural guard: the offset's floor must not be derived from fpr_vec
src = SRC.read_text(encoding="utf-8")
leak = re.search(r"best_offset\([^)]*1\.0\s*-\s*out\[.fpr_vec.\]", src)
check("the offset floor is not taken from the vector's test false-positive rate",
      leak is None, "clean" if leak is None else "LEAKAGE REINTRODUCED")

paper = PAPER.read_text(encoding="utf-8")
check("the leaky Marathi cell 0.371 (0.018) is not in the paper",
      "0.371} (0.018)" not in paper and "0.371 (0.018)" not in paper, "")
check("the caption states that selection does not read test",
      "No part of its selection reads the test split." in paper, "")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every tab:calibration number matches, and the offset selection is test-clean")

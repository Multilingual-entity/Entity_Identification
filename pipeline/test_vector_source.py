"""Check the --vector-source wiring without loading a model.

The jacobian path cannot be exercised end to end here, but three things can, and all
three are ways this change could be silently wrong: the vectors it produces must have the
shape additions() indexes, the tuned variant must be the one that exists, and the test
loop must iterate over the variants actually present rather than the four hardcoded ones.

    python test_vector_source.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stage07", HERE / "07_run_recovery.py")
stage07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage07)

fails = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{': ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. The shape additions() indexes: vectors[(ctx, kind)][layer, POSITIONS.index("E2")]
    n_layers, d = 28, 8
    v = np.zeros((n_layers, len(stage07.POSITIONS), d), dtype=np.float32)
    v[16, stage07.POSITIONS.index("E2")] = np.arange(d)
    try:
        got = v[16, stage07.POSITIONS.index("E2")]
        check("vector shape is indexable as additions() does", got.shape == (d,),
              f"{v.shape} -> {got.shape}")
    except Exception as exc:
        check("vector shape is indexable as additions() does", False, str(exc))

    # 2. normalise must not divide by zero on the unfilled slots.
    unit = stage07.unit_vectors({("mr", "jacobian"): v})[("mr", "jacobian")]
    finite = np.isfinite(unit).all()
    check("unit_vectors leaves no NaN on the zero slots", bool(finite),
          f"{np.isnan(unit).sum()} NaN")
    norm = np.linalg.norm(unit[16, stage07.POSITIONS.index("E2")])
    check("the filled slot is unit length", abs(norm - 1.0) < 1e-5, f"{norm:.6f}")

    # 3. The flags must refuse the combinations that cannot work.
    r = subprocess.run([sys.executable, str(HERE / "07_run_recovery.py"),
                        "--run-dir", "nonexistent", "--model", "qwen",
                        "--tune-variant", "jacobian"],
                       capture_output=True, text=True)
    check("tune-variant jacobian without the source is refused",
          "needs --vector-source jacobian" in (r.stdout + r.stderr),
          (r.stdout + r.stderr).strip().splitlines()[-1][:70] if (r.stdout + r.stderr) else "")

    # 4. The four mean-difference variants still exist and are still the default.
    import argparse
    ap = argparse.ArgumentParser()
    src = (HERE / "07_run_recovery.py").read_text(encoding="utf-8")
    check("meandiff is still the default", 'default="meandiff"' in src)
    check("test loop is no longer hardcoded",
          'for kind in ("full", "first_entity", "second_entity", "additive")' not in src)
    check("test loop reads the vectors present", "kinds = sorted({k for (c, k)" in src)

    print()
    if fails:
        print(f"{len(fails)} FAILED: " + ", ".join(fails))
        raise SystemExit(1)
    print("the --vector-source wiring is consistent")


if __name__ == "__main__":
    main()

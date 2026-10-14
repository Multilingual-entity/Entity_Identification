"""Recompute every number in tab:probematched from the two runs, paired on one split."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parents[2]
PAPER = Path.home() / "Downloads" / "when_entities_align" / "final_draft.tex"

WANT = {
    # context: map: (probe, baseline, probe - baseline, probe where baseline fails)
    "mr": {"shipped": (0.925, 0.933, -0.008, 0.250),
           "matched": (0.908, 0.796, +0.112, 0.653)},
    "en": {"shipped": (0.950, 0.933, +0.017, 0.375),
           "matched": (0.879, 0.796, +0.083, 0.755)},
}

failed: list[str] = []


def check(name: str, ok: bool, got: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}")
    if not ok:
        failed.append(name)


def parse(block: str) -> dict[str, float]:
    """Pull the four point estimates out of one printed block."""
    out = {}
    for line in block.splitlines():
        m = re.search(r"([-+]?\d\.\d{3})", line)
        if not m:
            continue
        v = float(m.group(1))
        if "probe balanced" in line:
            out["probe"] = v
        elif "surface balanced" in line:
            out["baseline"] = v
        elif "probe where surface fails" in line:
            out["fails"] = v
        elif "probe minus surface" in line:
            out["difference"] = v
    return out


paper = PAPER.read_text(encoding="utf-8")
for ctx in ("mr", "en"):
    proc = subprocess.run([sys.executable, str(HERE / "matched_probe_ci.py"), ctx],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode:
        raise SystemExit(f"matched_probe_ci.py {ctx} failed:\n{proc.stderr[-800:]}")
    parts = proc.stdout.split("negatives,")
    if len(parts) < 3:
        raise SystemExit(f"could not find both blocks in the {ctx} output")
    for which, idx in (("shipped", 1), ("matched", 2)):
        got = parse(parts[idx])
        w_probe, w_base, w_diff, w_fails = WANT[ctx][which]
        for label, want in (("probe", w_probe), ("baseline", w_base),
                            ("difference", w_diff), ("fails", w_fails)):
            if label not in got:
                check(f"tab:probematched {ctx} {which} {label}", False, "not parsed")
                continue
            check(f"tab:probematched {ctx} {which} {label} = {want}",
                  abs(got[label] - want) < 0.0015, f"{got[label]:.3f}")

    # the two directional claims the section rests on
    sh, ma = parse(parts[1]), parse(parts[2])
    check(f"{ctx}: the probe beats the baseline only under the matched map",
          ma["difference"] > 0 > sh["difference"] - 0.02 and ma["difference"] > 0.02,
          f"matched {ma['difference']:+.3f}, shipped {sh['difference']:+.3f}")
    check(f"{ctx}: the probe does better where the strings fail, matched vs shipped",
          ma["fails"] > sh["fails"], f"{ma['fails']:.3f} vs {sh['fails']:.3f}")

check("the unqualified 'decodes them about as well' claim is gone",
      "about as well, so this establishes" not in paper, "")
check("sec:probe carries the quantitative result, not just a forward pointer",
      "a difference of $+0.112$" in paper, "")
check("the scope is stated: 25 of 120 test entities meet the 0.05 tolerance",
      "25 meet the 0.05 matching tolerance" in paper, "")
check("the paper says both sides were refitted under the modified map",
      "Both sides were refitted under the modified map" in paper, "")
check("the claim is scoped to the five string features",
      "the five string features do not capture" in paper, "")

print()
if failed:
    raise SystemExit(f"{len(failed)} FAILED: " + ", ".join(failed))
print("every tab:probematched number matches the data")

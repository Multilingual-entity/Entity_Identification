"""Does check_run_script.py catch a continuation written as a literal escape?

This is the bug that reached the server: a line ending in backslash-n instead of a real
newline. bash reads it as the character "n", so the stage gets a stray positional
argument, and both `bash -n` and the flag audit pass. Every check that exists missed it,
so it needs its own test.

    python test_check_run_script.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_overnight.sh"
CHECKER = HERE / "check_run_script.py"

BACKSLASH = chr(92)


def run_checker(path: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(CHECKER), str(path)],
                       capture_output=True, text=True, cwd=HERE)
    return p.returncode, p.stdout + p.stderr


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    good = SCRIPT.read_text(encoding="utf-8")
    failures = []

    # 1. The real script must pass, and must contain no literal escape already.
    rc, out = run_checker(SCRIPT)
    print(f"1. real script: exit {rc}")
    if rc != 0:
        failures.append("the real script does not pass its own checker")
        print(out[-600:])

    # 2. Break one continuation the way the Bash heredoc broke it, and the checker must
    #    fail. Replace "<backslash><newline>" with "<backslash>n " on the 1a line.
    target = "--alphas 6.0 --skip-test " + BACKSLASH + "\n"
    if target not in good:
        failures.append("the 1a continuation is not where the test expects it")
    else:
        broken = good.replace(target, "--alphas 6.0 --skip-test " + BACKSLASH + "n ", 1)
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "broken.sh"
            bad.write_text(broken, encoding="utf-8", newline="\n")
            rc, out = run_checker(bad)
        print(f"2. broken continuation: exit {rc}")
        if rc == 0:
            failures.append("a literal backslash-n continuation was NOT caught")
        elif "literal" not in out:
            failures.append(f"caught, but not for the right reason: {out[-200:]}")

    # 3. No CR bytes: the shell refuses the whole file over one of them.
    raw = SCRIPT.read_bytes()
    print(f"3. CR bytes in {SCRIPT.name}: {raw.count(bytes([13]))}")
    if bytes([13]) in raw:
        failures.append(f"{SCRIPT.name} has CRLF line endings; bash will not run it")

    # 4. printf lines legitimately contain backslash-n, and must not be flagged.
    if "literal" in run_checker(SCRIPT)[1]:
        failures.append("the real script's printf lines are being flagged")
    print("4. printf lines not flagged: ok")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        raise SystemExit(1)
    print("all checks pass")


if __name__ == "__main__":
    main()

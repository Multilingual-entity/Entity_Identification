"""Check that every flag a runner script passes is one the stage actually accepts.

Two runs were lost to this: --layers given a range where a list was wanted, and --model
given to the two stages that deliberately do not take one. Both are visible without a
GPU, by reading the argparse setup, so they should never reach the server.

    python check_run_script.py run_review3.sh
"""
from __future__ import annotations

import ast
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = {"--run-dir", "--seed", "--force"}
COMMON_MODEL = {"--model", "--batch-size"}


def resolve(rel: str) -> Path | None:
    """Find a script whether its directory is a sibling or a child.

    The runner executes from the pipeline root on the GPU box, where analysis/ and
    corpus_construction/ are children. Locally they are siblings of pipeline/. Looking
    in both means the checker does not report a missing stage that is simply elsewhere.
    """
    for cand in (HERE / rel, HERE.parent / rel, HERE / Path(rel).name):
        if cand.exists():
            return cand
    return None


def flags_of(stage: Path) -> set[str] | None:
    """Every --flag the stage defines, including the ones add_common_args contributes."""
    if not stage.exists():
        return None
    src = stage.read_text(encoding="utf-8")
    found = set(re.findall(r"add_argument\(\s*[\"'](--[a-z0-9-]+)", src))
    m = re.search(r"add_common_args\([^)]*\)", src)
    if m:
        found |= COMMON
        if "include_model=False" not in m.group(0):
            found |= COMMON_MODEL
    return found


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    script = Path(sys.argv[1] if len(sys.argv) > 1 else "run_review3.sh")
    text = script.read_text(encoding="utf-8")
    # Join shell line continuations, then keep the lines that invoke a stage.
    text = re.sub(r"\\\s*\n\s*", " ", text)

    problems, checked, unparsed = [], 0, []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("python ") or line.startswith("#"):
            continue
        # Redirects are not arguments, and their quoted targets break the shell
        # variable substitution below, so remove them before anything else.
        # A literal backslash-n inside a command is almost always a line continuation
        # that was written as an escape instead of a real newline. bash accepts it as
        # the character "n", so the stage receives a stray positional argument and both
        # `bash -n` and the flag check below pass. This is the only place it shows up.
        if re.search(r"\\n(?=\s|$)", line):
            problems.append(f"literal \\n in an invocation (broken continuation): "
                            f"{line[:70]}")
        cleaned = re.sub(r"\s+\d?>[>&]?\s*\S+", " ", line)
        cleaned = re.sub(r"\s+2>&1", " ", cleaned)
        # Shell expansions are opaque to a static check; keep the flag, drop the value.
        cleaned = re.sub(r"\$\([^)]*\)", "X", cleaned)
        cleaned = re.sub(r'"\$\{?\w+\}?[^"]*"', "X", cleaned)
        cleaned = re.sub(r"\$\{?\w+\}?", "X", cleaned)
        try:
            parts = shlex.split(cleaned)
        except ValueError as exc:
            unparsed.append((line[:70], str(exc)))
            continue
        if len(parts) < 2 or not parts[1].endswith((".py", ".sh")):
            continue
        stage = resolve(parts[1])
        accepted = flags_of(stage) if stage else None
        if accepted is None:
            problems.append(f"{parts[1]}: no such stage file")
            continue
        checked += 1
        used = {p for p in parts if p.startswith("--")}
        unknown = sorted(used - accepted)
        status = "ok" if not unknown else "UNKNOWN " + ", ".join(unknown)
        print(f"  {parts[1]:32}{len(used):>3} flags   {status}")
        problems += [f"{parts[1]} does not accept {f}" for f in unknown]

    print(f"\n{checked} invocation(s) checked")
    if unparsed:
        print(f"{len(unparsed)} line(s) could NOT be parsed and were not checked:")
        for line, why in unparsed:
            print(f"  {line}  ({why})")
        problems += [f"unparsed: {line}" for line, _ in unparsed]
    if problems:
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("every flag is accepted by the stage it is passed to")


if __name__ == "__main__":
    main()

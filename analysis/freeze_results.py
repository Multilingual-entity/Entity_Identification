r"""Freeze the result directory the paper is built from, and detect later drift.

The paper went out with a main text from one corpus and an appendix from another. The
reason was not carelessness about any single number: it was that "the results" had no
fixed referent, so a table edited in July and a table edited in September could both look
current. This writes a manifest of every result file with its size, row count and a hash,
so that "the frozen run" is a thing that can be checked rather than assumed.

    python freeze_results.py --write      once, to create the manifest
    python freeze_results.py              thereafter, to verify nothing moved

Any mismatch means a result file changed after the paper's numbers were taken from it,
and every table sourced from that file has to be regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import os

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE.parent / "results"))
MANIFEST = HERE / "results_manifest.json"


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest()[:16],
        "bytes": len(data),
        # Row count rather than byte count alone, because a re-export with different
        # float formatting is a different file but the same result, and the distinction
        # matters when deciding whether a table needs regenerating.
        "rows": max(0, data.count(b"\n") - 1) if path.suffix == ".csv" else None,
    }


def collect() -> dict:
    out = {}
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".npz"}:
            # The raw per-prompt dumps are large and are not read by any table.
            if path.stat().st_size > 50_000_000:
                continue
            out[str(path.relative_to(RESULTS)).replace("\\", "/")] = digest(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="create or replace the manifest instead of checking it")
    args = ap.parse_args()

    current = collect()
    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=1, sort_keys=True), encoding="utf-8")
        print(f"froze {len(current)} result files -> {MANIFEST.name}")
        return

    if not MANIFEST.exists():
        raise SystemExit("no manifest yet; run with --write first")
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))

    added = sorted(set(current) - set(frozen))
    removed = sorted(set(frozen) - set(current))
    changed = sorted(k for k in set(current) & set(frozen)
                     if current[k]["sha256"] != frozen[k]["sha256"])

    print(f"{len(frozen)} files frozen, {len(current)} present")
    for label, items in (("changed since freeze", changed),
                         ("missing", removed), ("new", added)):
        if items:
            print(f"\n{label}: {len(items)}")
            for k in items[:12]:
                if label == "changed since freeze":
                    print(f"  {k}  rows {frozen[k]['rows']} -> {current[k]['rows']}")
                else:
                    print(f"  {k}")
    if not (added or removed or changed):
        print("\nunchanged: every table in the paper is sourced from this exact state")
    else:
        raise SystemExit("\nresults have drifted; regenerate any table sourced from the above")


if __name__ == "__main__":
    main()

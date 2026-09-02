"""Record what a run directory actually contains, so an old run stays interpretable.

The corpus is about to be rescaled from 299 facts to roughly 800. That invalidates the
splits, the causal gate, and every tuned hyperparameter, so the new run cannot be compared
with the old one row by row. If the new run is not finished, the old numbers are what the
paper stands on, and they have to remain readable months later without anyone reconstructing
which corpus and which settings produced them.

Nothing here recovers information that was never written down. It reads what a finished run
already carries -- the data manifest with its source hashes and seed, the model metadata,
and the shape and checksum of every table -- and freezes it into one file that can be read
without the run directory being intact.

    python provenance.py --snapshot ../../shareable_results --label "qwen 299-fact run"
    python provenance.py --compare ../../shareable_results ../../results/qwen_800

Run --snapshot on the existing runs BEFORE rebuilding the corpus. Afterwards the source CSV
may have been replaced, and the hash recorded in the old manifest is then the only evidence
of which file was used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path, limit_mb: int = 512) -> str:
    """Checksum, skipping files too large to be worth hashing on every call."""
    if path.stat().st_size > limit_mb * 1024 * 1024:
        return f"skipped-{path.stat().st_size}-bytes"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_tables(run: Path) -> list:
    """Every CSV in the run, with its shape and checksum.

    Shape matters as much as the checksum. A table with 598 candidate rows came from a
    299-fact corpus; one with 1600 did not. That single number identifies which corpus a
    run used even if the manifest is missing.
    """
    rows = []
    for folder in ("data", "tables", "patching", "recovery", "states"):
        directory = run / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:                                            # noqa: BLE001
                continue
            entry = {
                "path": path.relative_to(run).as_posix(),
                "n_rows": int(len(frame)),
                "n_cols": int(frame.shape[1]),
                "sha256": sha256_file(path),
            }
            if "fact_id" in frame.columns:
                entry["n_facts"] = int(frame.fact_id.nunique())
            rows.append(entry)
    return rows


def snapshot(run: Path, label: str) -> dict:
    record = {
        "label": label,
        "run_dir": str(run.resolve()),
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": None,
        "model": None,
        "invocations": None,
        "tables": describe_tables(run),
    }
    manifest = run / "data" / "manifest.json"
    if manifest.exists():
        record["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))

    # Model metadata is written by whichever stage loaded the weights; its name has varied.
    for candidate in ("model_run_metadata.json", "model_metadata.json",
                      "run_metadata.json", "metadata.json"):
        for folder in ("logs", "tables", "."):
            path = run / folder / candidate
            if path.exists():
                record["model"] = json.loads(path.read_text(encoding="utf-8"))
                break
        if record["model"]:
            break

    state = run / "states" / "state_manifest.json"
    if state.exists():
        record["states"] = json.loads(state.read_text(encoding="utf-8"))

    log = run / "logs" / "invocations.jsonl"
    if log.exists():
        record["invocations"] = [json.loads(line) for line in
                                 log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return record


def print_snapshot(record: dict) -> None:
    print(f"\n{record['label']}  ({record['run_dir']})")
    print(f"captured {record['captured_utc']}")
    manifest = record.get("manifest")
    if manifest:
        print("\ncorpus")
        for key in ("n_source", "n_prepared", "n_train", "n_validation", "n_test", "seed"):
            if key in manifest:
                print(f"    {key:<16} {manifest[key]}")
        for key in ("source_corpus", "source_corpus_sha256"):
            if key in manifest:
                print(f"    {key:<16} {manifest[key]}")
    else:
        print("\ncorpus: no manifest. The fact count in data/corpus.csv below is the only "
              "record of which corpus this run used.")
    model = record.get("model")
    if model:
        print("\nmodel")
        for key in ("model_id", "model_revision", "n_layers", "dtype", "gpu",
                    "transformers_version"):
            if key in model:
                print(f"    {key:<20} {model[key]}")
    if record.get("invocations"):
        print(f"\n{len(record['invocations'])} recorded invocations "
              f"(the exact command line of each stage)")
    else:
        print("\nno invocation log: this run predates command-line recording, so the flags "
              "each stage used are not recoverable from the run directory.")
    print(f"\n{len(record['tables'])} tables")
    for entry in record["tables"][:40]:
        facts = f"  {entry['n_facts']} facts" if "n_facts" in entry else ""
        print(f"    {entry['path']:<52} {entry['n_rows']:>8} rows{facts}")
    if len(record["tables"]) > 40:
        print(f"    ... and {len(record['tables']) - 40} more")


def compare(a: dict, b: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{a['label']}   against   {b['label']}")
    print("=" * 78)
    ma, mb = a.get("manifest") or {}, b.get("manifest") or {}
    for key in ("n_prepared", "n_train", "n_validation", "n_test", "seed",
                "source_corpus_sha256"):
        va, vb = ma.get(key), mb.get(key)
        flag = "  <-- differs" if va != vb else ""
        print(f"  {key:<22} {str(va):<24} {str(vb):<24}{flag}")
    by_path_a = {t["path"]: t for t in a["tables"]}
    by_path_b = {t["path"]: t for t in b["tables"]}
    only_a = sorted(set(by_path_a) - set(by_path_b))
    only_b = sorted(set(by_path_b) - set(by_path_a))
    if only_a:
        print(f"\n  only in {a['label']}: {', '.join(only_a[:10])}")
    if only_b:
        print(f"\n  only in {b['label']}: {', '.join(only_b[:10])}")
    print("\n  tables present in both, where the row count changed:")
    changed = 0
    for path in sorted(set(by_path_a) & set(by_path_b)):
        ra, rb = by_path_a[path]["n_rows"], by_path_b[path]["n_rows"]
        if ra != rb:
            changed += 1
            print(f"    {path:<50} {ra:>8} -> {rb:>8}")
    if not changed:
        print("    none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, nargs="+",
                    help="run directories to record")
    ap.add_argument("--label", default=None,
                    help="human-readable name; defaults to the directory name")
    ap.add_argument("--compare", type=Path, nargs=2,
                    help="two run directories to diff")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write provenance.json; defaults to inside each run")
    args = ap.parse_args()

    if args.compare:
        records = [snapshot(p, p.name) for p in args.compare]
        compare(*records)
        return
    if not args.snapshot:
        raise SystemExit("pass --snapshot or --compare")

    for run in args.snapshot:
        if not run.exists():
            print(f"skipping {run}: does not exist")
            continue
        record = snapshot(run, args.label or run.name)
        print_snapshot(record)
        target = args.out or (run / "provenance.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten: {target}")


if __name__ == "__main__":
    main()

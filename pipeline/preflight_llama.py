"""Pre-flight checks for the Llama run. No GPU required."""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

OK, BAD = [], []


def check(label: str, passed: bool, detail: str = "") -> None:
    (OK if passed else BAD).append(f"{'PASS' if passed else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--run-dir", default="results/llama_a100")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))

    # 1. model registered
    try:
        import pipeline_common as pc
        importlib.reload(pc)
        spec = pc.MODEL_SPECS.get(args.model)
        check("model registered in MODEL_SPECS", spec is not None,
              f"available: {sorted(pc.MODEL_SPECS)}")
    except Exception as exc:  # noqa: BLE001
        check("import pipeline_common", False, repr(exc))
        report(); return

    if spec is None:
        report(); return
    print(f"model_id = {spec.model_id}  | batch_size = {spec.batch_size}")

    # 2. scale_layer present and sane
    has_scale = hasattr(pc, "scale_layer")
    check("scale_layer helper present", has_scale)
    if has_scale:
        ident = all(pc.scale_layer(i, 28) == i for i in range(28))
        check("scale_layer is identity at 28 layers", ident)
        w = {k: (pc.scale_layer(a, 32), pc.scale_layer(b, 32))
             for k, (a, b) in {"12_17": (12, 17), "18_20": (18, 20)}.items()}
        check("scale_layer maps 12_17 -> 14_19 at 32 layers", w["12_17"] == (14, 19), str(w))

    # 3. CLI flags exist
    for script, flags in [("04_run_causal_patching.py", ["--scale-layers"]),
                          ("07_run_recovery.py", ["--scale-layers", "--alphas"])]:
        src = (here / script).read_text(encoding="utf-8")
        for flag in flags:
            check(f"{script} accepts {flag}", flag in src)

    # 4. all stages compile
    for f in sorted(here.glob("*.py")):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)], capture_output=True)
        check(f"compiles: {f.name}", r.returncode == 0, r.stderr.decode()[:120])

    # 5. tokenizer contract
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
        for label in ("A", "B"):
            ids = tok.encode(label, add_special_tokens=False)
            check(f"label {label!r} is a single token", len(ids) == 1, str(ids))
        has_tpl = getattr(tok, "chat_template", None) is not None
        check("tokenizer has a chat template", has_tpl)
        if has_tpl:
            rendered = tok.apply_chat_template([{"role": "user", "content": "x"}],
                                               tokenize=False, add_generation_prompt=True)
            check("chat template renders", isinstance(rendered, str) and len(rendered) > 0)
        dev = tok.encode("लेडी गागा", add_special_tokens=False)
        lat = tok.encode("Lady Gaga", add_special_tokens=False)
        print(f"fertility probe: 'Lady Gaga' -> {len(lat)} tokens | Devanagari -> {len(dev)} tokens")
        check("Devanagari tokenizes without error", len(dev) > 0)
    except Exception as exc:  # noqa: BLE001
        check("tokenizer loads", False, repr(exc))

    # 6. inputs the run needs
    csv_dir = here / "csv_qwen7b_instruct"
    check("source corpus directory present", csv_dir.is_dir(), str(csv_dir))

    # 7. disk
    total, used, free = shutil.disk_usage(str(here))
    gb = free / 1e9
    check("at least 30 GB free (weights + states)", gb >= 30, f"{gb:.1f} GB free")

    report()


def report() -> None:
    print("\n" + "=" * 64)
    for line in OK:
        print(" ", line)
    for line in BAD:
        print(" ", line)
    print("=" * 64)
    print(f"{len(OK)} passed, {len(BAD)} failed")
    print("READY TO LAUNCH" if not BAD else "NOT READY - fix the FAIL lines above")
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()

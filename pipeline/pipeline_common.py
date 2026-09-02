"""Shared utilities for the cross-script entity-identity A100 pipeline.

All experiment scripts live in the same directory and import this module.  The
functions here deliberately avoid notebook state and write no files unless the
caller asks them to.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SEED = 20260813
MAX_LENGTH = 384

CONDITIONS = {
    "DEVDEV": ("dev", "dev"),
    "DEVLAT": ("dev", "lat"),
    "LATDEV": ("lat", "dev"),
    "LATLAT": ("lat", "lat"),
}

TEMPLATE_FILE = Path(__file__).resolve().parent / "question_templates.json"


def _load_templates() -> tuple[dict, str, set]:
    """Question templates keyed by prompt language, from the shared JSON file.

    Falls back to the original hard-coded Marathi and English set if the file is absent,
    so an older checkout still runs.
    """
    fallback = {
        "mr": [
            "[{a}] आणि [{b}] हे एकच व्यक्ती आहेत का?",
            "[{a}] आणि [{b}] ही नावे एकाच व्यक्तीला दर्शवतात का?",
            "[{a}] हे [{b}] यांचे दुसरे नाव आहे का?",
        ],
        "en": [
            "Do [{a}] and [{b}] refer to the same person?",
            "Are [{a}] and [{b}] names for the same person?",
            "Is [{a}] another name for [{b}]?",
        ],
    }
    default_instruction = "\n\nAnswer only A or B. {yes} = yes. {no} = no."
    if not TEMPLATE_FILE.exists():
        return fallback, default_instruction, {"mr", "en"}

    blob = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
    templates, reviewed = {}, set()
    for lang, entry in blob["languages"].items():
        rows = entry["templates"]
        if len(rows) != 3:
            raise ValueError(f"{lang}: expected 3 paraphrases, found {len(rows)}")
        templates[lang] = rows
        if entry.get("reviewed"):
            reviewed.add(lang)
    return templates, blob.get("instruction", default_instruction), reviewed


QUESTION_TEMPLATES, ANSWER_INSTRUCTION, REVIEWED_LANGUAGES = _load_templates()
SUPPORTED_CONTEXTS = sorted(QUESTION_TEMPLATES)


def contexts_in(frame, column: str = "context") -> list:
    """Prompt languages actually present in a frame, rather than a hard-coded pair."""
    if column not in getattr(frame, "columns", []):
        return list(SUPPORTED_CONTEXTS)
    return sorted(str(c) for c in frame[column].dropna().unique())


def contexts_in_gate(gate, suffix: str = "_known_any") -> list:
    """Prompt languages present in a knowledge-gate frame, read off its column names."""
    found = sorted({c[: -len(suffix)] for c in gate.columns if c.endswith(suffix)})
    return found or list(SUPPORTED_CONTEXTS)


def check_contexts(contexts) -> None:
    """Reject unknown prompt languages; warn about unreviewed ones."""
    unknown = [c for c in contexts if c not in QUESTION_TEMPLATES]
    if unknown:
        raise SystemExit(
            f"no question templates for {unknown}. "
            f"Available: {', '.join(SUPPORTED_CONTEXTS)}. "
            f"Add them to {TEMPLATE_FILE.name}."
        )
    unreviewed = [c for c in contexts if c not in REVIEWED_LANGUAGES]
    if unreviewed:
        print(
            f"WARNING: question templates for {unreviewed} have not been checked by a "
            f"native speaker. A poor template produces a null result indistinguishable "
            f"from a model that cannot do the task."
        )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str | None
    batch_size: int


MODEL_SPECS = {
    "qwen": ModelSpec(
        key="qwen",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        batch_size=24,
    ),
    "gemma": ModelSpec(
        key="gemma",
        model_id="google/gemma-2-2b-it",
        revision=None,
        batch_size=32,
    ),
    # The NousResearch mirror, not meta-llama. It reproduces the Meta release, it is what
    # the finished 299-fact run actually used, and it is what the weights on the run machine
    # are. Pointing at meta-llama would fail the licence gate and, if that were cleared,
    # would silently change the provenance of a result the paper already reports.
    "llama": ModelSpec(
        key="llama",
        model_id="NousResearch/Meta-Llama-3.1-8B-Instruct",
        revision=None,
        batch_size=24,
    ),
    # Candidates for a fourth model. Llama was added for a second causal replication and
    # did not provide one: its Marathi arm is degenerate, with a sensitivity of 0.02 and
    # paraphrases returning exactly 50.0 percent, so the causal gate has almost nothing to
    # select from. What is needed is not simply a larger model but one that can do the
    # task in Marathi at all, which points at either the two families that already worked
    # or a model with explicit Indic coverage.
    #
    # None of these has been run. Verify each with preflight_llama.py --model <key>, which
    # checks that the identifier resolves, that A and B are single tokens, and that a chat
    # template exists, then spend one cheap behaviour-only run before committing to states
    # and patching. A model that cannot clear the knowledge gate in Marathi cannot
    # replicate the causal result no matter how much compute follows.
    "gemma9": ModelSpec(
        key="gemma9",
        model_id="google/gemma-2-9b-it",
        revision=None,
        batch_size=24,
    ),
    "qwen14": ModelSpec(
        key="qwen14",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        revision=None,
        batch_size=16,
    ),
    "aya": ModelSpec(
        key="aya",
        model_id="CohereForAI/aya-expanse-8b",
        revision=None,
        batch_size=24,
    ),
}

# The layer bands used throughout were fixed on Qwen2.5-7B-Instruct, which has 28
# layers.  For a model of a different depth the same absolute layer numbers point at a
# different part of the network, so they are rescaled proportionally when --scale-layers
# is passed.  Default behaviour is unchanged for qwen and gemma.
REFERENCE_N_LAYERS = 28


def scale_layer(layer: int, n_layers: int, reference: int = REFERENCE_N_LAYERS) -> int:
    """Map a layer index defined on the reference model onto a model of another depth."""
    scaled = int(round(layer * n_layers / reference))
    return max(0, min(n_layers - 1, scaled))


def add_common_args(parser: argparse.ArgumentParser, include_model: bool = True) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    if include_model:
        parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="qwen")
        parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _log_invocation(paths: dict) -> None:
    """Append this command line, and the code that ran it, to the run's log.

    The existing runs cannot say which flags produced them, because nothing recorded the
    flags. That was tolerable while there was one corpus; with a 299-fact run and an
    800-fact run side by side it is not, since the difference between them lives entirely
    in settings that leave no trace in the output tables.

    Hashing the stage scripts matters as much as the arguments. Several of them gained
    flags that change what is computed -- a margin gate instead of an accuracy one, a
    cumulative component patch instead of a single-layer one -- so two runs of the same
    command against different versions of the code produce different numbers under
    identical filenames.

    Never allowed to interrupt a run: a provenance record is worth less than the run it
    would abort.
    """
    import sys
    from datetime import datetime, timezone

    try:
        here = Path(__file__).resolve().parent
        code = {}
        for script in sorted(here.glob("*.py")):
            digest = hashlib.sha256()
            digest.update(script.read_bytes())
            code[script.name] = digest.hexdigest()[:16]
        record = {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
            "code_sha256_16": code,
        }
        with (paths["logs"] / "invocations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:                                                    # noqa: BLE001
        pass


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "root": run_dir,
        "data": run_dir / "data",
        "states": run_dir / "states",
        "tables": run_dir / "tables",
        "patching": run_dir / "patching",
        "recovery": run_dir / "recovery",
        "logs": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    _log_invocation(paths)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prepared_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus_path = run_dir / "data" / "corpus.csv"
    negatives_path = run_dir / "data" / "hard_negative_map.csv"
    if not corpus_path.exists() or not negatives_path.exists():
        raise FileNotFoundError("Run 00_prepare_data.py before this script.")
    corpus = pd.read_csv(corpus_path)
    negatives = pd.read_csv(negatives_path)
    for col in ("name_a_dev", "name_a_lat", "name_b_dev", "name_b_lat"):
        corpus[col] = corpus[col].fillna("").astype(str)
    return corpus, negatives


class PromptBuilder:
    def __init__(self, corpus: pd.DataFrame, negatives: pd.DataFrame):
        self.by_id = corpus.set_index("fact_id", drop=False)
        self.negative_of = dict(
            zip(negatives["fact_id"].astype(str), negatives["negative_fact_id"].astype(str))
        )

    @staticmethod
    def answer_instruction(yes_letter: str) -> tuple[str, str]:
        if yes_letter not in {"A", "B"}:
            raise ValueError("yes_letter must be A or B")
        no_letter = "B" if yes_letter == "A" else "A"
        return f"Answer only A or B. {yes_letter} = yes. {no_letter} = no.", no_letter

    def names(
        self,
        fact_id: str,
        condition: str,
        truth: int,
        anchor: str = "none",
    ) -> tuple[str, str]:
        a_script, b_script = CONDITIONS[condition]
        arow = self.by_id.loc[fact_id]
        brow = arow if int(truth) == 1 else self.by_id.loc[self.negative_of[str(fact_id)]]

        a = str(arow[f"name_a_{a_script}"])
        b = str(brow[f"name_b_{b_script}"])
        if anchor != "none":
            a = self._anchor_name(arow, "a", anchor, a_script)
            b = self._anchor_name(brow, "b", anchor, b_script)
        return a, b

    @staticmethod
    def _auto_romanize(text: str) -> str:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate

            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            return text

    def _anchor_name(self, row: pd.Series, role: str, anchor: str, script: str) -> str:
        base = str(row[f"name_{role}_{script}"])
        if script != "dev":
            return base
        if anchor == "latin_a" and role != "a":
            return base
        if anchor == "latin_b" and role != "b":
            return base
        if anchor in {"latin_a", "latin_b", "latin_both"}:
            return f"{base} ({row[f'name_{role}_lat']})"
        if anchor == "auto_both":
            return f"{base} ({self._auto_romanize(base)})"
        raise ValueError(f"Unknown anchor: {anchor}")

    def make(
        self,
        fact_id: str,
        context: str,
        condition: str,
        truth: int,
        paraphrase_id: int = 0,
        yes_letter: str = "A",
        anchor: str = "none",
    ) -> dict[str, object]:
        if context not in QUESTION_TEMPLATES:
            raise ValueError(f"Unknown context: {context}")
        a, b = self.names(fact_id, condition, truth, anchor=anchor)
        question = QUESTION_TEMPLATES[context][paraphrase_id].format(a=a, b=b)
        instruction, no_letter = self.answer_instruction(yes_letter)
        return {
            "fact_id": str(fact_id),
            "context": context,
            "condition": condition,
            "truth": int(truth),
            "paraphrase_id": int(paraphrase_id),
            "yes_letter": yes_letter,
            "no_letter": no_letter,
            "correct_semantic": "yes" if int(truth) == 1 else "no",
            "a_text": a,
            "b_text": b,
            "anchor": anchor,
            "question": question,
            "prompt": question + "\n\n" + instruction,
        }


def build_prompt_frame(
    corpus: pd.DataFrame,
    negatives: pd.DataFrame,
    contexts: Sequence[str] = ("mr", "en"),
    conditions: Sequence[str] = tuple(CONDITIONS),
    truths: Sequence[int] = (1, 0),
    paraphrases: Sequence[int] = (0, 1, 2),
    yes_letters: Sequence[str] = ("A", "B"),
    anchors: Sequence[str] = ("none",),
) -> pd.DataFrame:
    builder = PromptBuilder(corpus, negatives)
    records: list[dict[str, object]] = []
    split_of = dict(zip(corpus["fact_id"].astype(str), corpus["split"].astype(str)))
    for fid in corpus["fact_id"].astype(str):
        for context in contexts:
            for condition in conditions:
                for truth in truths:
                    for paraphrase_id in paraphrases:
                        for yes_letter in yes_letters:
                            for anchor in anchors:
                                rec = builder.make(
                                    fid,
                                    context,
                                    condition,
                                    truth,
                                    paraphrase_id,
                                    yes_letter,
                                    anchor,
                                )
                                rec["split"] = split_of[fid]
                                records.append(rec)
    frame = pd.DataFrame(records)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def load_model_and_tokenizer(model_key: str, batch_size: int | None = None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the experiment scripts.")
    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based semantic positions.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = get_layers(model)
    device = next(model.parameters()).device
    effective_batch = batch_size or spec.batch_size
    metadata = {
        "model_key": model_key,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "n_layers": len(layers),
        "d_model": int(model.config.hidden_size),
        "batch_size": effective_batch,
    }
    return model, tokenizer, device, effective_batch, metadata


def load_tokenizer_only(model_key: str):
    """The tokenizer without the weights, for analyses that count tokens.

    Tokenizer-level questions -- fertility, how a romanization is chopped -- need no GPU
    and no 16 GB download, so they should not be gated behind one.
    """
    from transformers import AutoTokenizer

    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise TypeError(f"Unsupported model architecture: {type(model).__name__}")


def format_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def label_token_ids(tokenizer) -> tuple[int, int]:
    ids: dict[str, int] = {}
    for label in ("A", "B"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Label {label!r} is not one token: {encoded}")
        ids[label] = int(encoded[0])
    return ids["A"], ids["B"]


def token_covering_char(offsets: Sequence[Sequence[int]], char_index: int) -> int:
    matches = [
        i for i, (start, end) in enumerate(offsets) if end > start and start <= char_index < end
    ]
    if not matches:
        raise RuntimeError(f"No token offset covers character index {char_index}.")
    return matches[-1]


def semantic_char_positions(formatted: str, a_text: str, b_text: str) -> dict[str, int]:
    a_start = formatted.find(a_text)
    if a_start < 0:
        raise ValueError(f"First entity not found in formatted prompt: {a_text!r}")
    e1 = formatted.find("]", a_start + len(a_text))
    b_start = formatted.find(b_text, e1 + 1)
    if b_start < 0:
        raise ValueError(f"Second entity not found in formatted prompt: {b_text!r}")
    e2 = formatted.find("]", b_start + len(b_text))
    question = formatted.rfind("?")
    if min(e1, e2, question) < 0:
        raise ValueError("Could not locate E1, E2, or question mark.")
    return {"E1": e1, "E2": e2, "Q": question}


def semantic_token_positions(tokenizer, prompt: str, a_text: str, b_text: str) -> dict[str, int]:
    formatted = format_chat(tokenizer, prompt)
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    offsets = encoded["offset_mapping"][0].tolist()
    chars = semantic_char_positions(formatted, a_text, b_text)
    positions = {name: token_covering_char(offsets, index) for name, index in chars.items()}
    positions["READOUT"] = int(encoded["input_ids"].shape[1] - 1)
    return positions


def score_prompt_frame(
    model,
    tokenizer,
    device,
    frame: pd.DataFrame,
    batch_size: int,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Score A/B prompts, resuming from a row-id keyed checkpoint when present."""
    import torch
    from tqdm.auto import tqdm

    a_id, b_id = label_token_ids(tokenizer)
    finished = pd.DataFrame()
    done: set[int] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        finished = pd.read_csv(checkpoint_path)
        done = set(finished["row_id"].astype(int))
    pending = frame.loc[~frame["row_id"].isin(done)].copy()
    chunks: list[pd.DataFrame] = [finished] if len(finished) else []

    for start in tqdm(range(0, len(pending), batch_size), desc="A/B scoring"):
        chunk = pending.iloc[start : start + batch_size].copy()
        texts = [format_chat(tokenizer, prompt) for prompt in chunk["prompt"]]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
        logit_a = logits[:, a_id].detach().cpu().numpy()
        logit_b = logits[:, b_id].detach().cpu().numpy()
        yes_is_a = chunk["yes_letter"].to_numpy() == "A"
        margin = np.where(yes_is_a, logit_a - logit_b, logit_b - logit_a)
        chunk["logit_a"] = logit_a
        chunk["logit_b"] = logit_b
        chunk["yes_minus_no_margin"] = margin
        chunk["pred_semantic"] = np.where(margin > 0, "yes", "no")
        chunk["correct"] = (
            chunk["pred_semantic"].to_numpy() == chunk["correct_semantic"].to_numpy()
        ).astype(np.int8)
        chunks.append(chunk)
        if checkpoint_path is not None:
            combined = pd.concat(chunks, ignore_index=True).sort_values("row_id")
            combined.to_csv(checkpoint_path, index=False)
            chunks = [combined]
        del encoded, logits
        gc.collect()

    if not chunks:
        return frame.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True).drop_duplicates("row_id").sort_values("row_id")


def fixed_yes_margin(model, tokenizer, device, prompt: str) -> float:
    import torch

    a_id, b_id = label_token_ids(tokenizer)
    encoded = tokenizer(
        format_chat(tokenizer, prompt),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, -1].float()
    value = float((logits[a_id] - logits[b_id]).item())
    del encoded, logits
    return value


def capture_prompt_states(model, tokenizer, device, record: pd.Series | dict) -> tuple[np.ndarray, dict[str, int]]:
    """Return block-output states [layers, positions, d_model] for E1/E2/Q/readout."""
    import torch

    prompt = str(record["prompt"])
    a_text = str(record["a_text"])
    b_text = str(record["b_text"])
    positions = semantic_token_positions(tokenizer, prompt, a_text, b_text)
    order = ["E1", "E2", "Q", "READOUT"]
    encoded = tokenizer(
        format_chat(tokenizer, prompt),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    with torch.inference_mode():
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    array = torch.stack(
        [
            torch.stack([hidden[0, positions[name]] for name in order], dim=0)
            for hidden in output.hidden_states[1:]
        ],
        dim=0,
    ).detach().to(torch.float16).cpu().numpy()
    del encoded, output
    return array, positions


def patch_margin(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: Sequence[tuple[int, int, object]],
) -> float:
    """Patch block outputs. Each patch is (layer, token_position, vector)."""
    import torch

    layers = get_layers(model)
    by_layer: dict[int, list[tuple[int, object]]] = {}
    for layer, position, vector in patches:
        by_layer.setdefault(int(layer), []).append((int(position), vector))
    handles = []

    def make_hook(items):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = (output[0] if is_tuple else output).clone()
            for position, vector in items:
                hidden[:, position, :] = torch.as_tensor(
                    vector, device=hidden.device, dtype=hidden.dtype
                )
            return (hidden, *output[1:]) if is_tuple else hidden

        return hook

    for layer, items in by_layer.items():
        handles.append(layers[layer].register_forward_hook(make_hook(items)))
    try:
        return fixed_yes_margin(model, tokenizer, device, prompt)
    finally:
        for handle in handles:
            handle.remove()


def add_vector_margin(
    model,
    tokenizer,
    device,
    prompt: str,
    additions: Sequence[tuple[int, int, object, float]],
) -> float:
    """Add vectors to block outputs: (layer, position, vector, alpha)."""
    import torch

    layers = get_layers(model)
    by_layer: dict[int, list[tuple[int, object, float]]] = {}
    for layer, position, vector, alpha in additions:
        by_layer.setdefault(int(layer), []).append((int(position), vector, float(alpha)))
    handles = []

    def make_hook(items):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = (output[0] if is_tuple else output).clone()
            for position, vector, alpha in items:
                hidden[:, position, :] += alpha * torch.as_tensor(
                    vector, device=hidden.device, dtype=hidden.dtype
                )
            return (hidden, *output[1:]) if is_tuple else hidden

        return hook

    for layer, items in by_layer.items():
        handles.append(layers[layer].register_forward_hook(make_hook(items)))
    try:
        return fixed_yes_margin(model, tokenizer, device, prompt)
    finally:
        for handle in handles:
            handle.remove()


def cleanup_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation. Avoids a scipy dependency on the cluster image."""
    from math import log, sqrt

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    low, high = 0.02425, 1 - 0.02425
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    if p < low:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > high:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def dprime(hit_rate: float, false_alarm_rate: float, n_signal: int, n_noise: int):
    """Sensitivity with a log-linear correction, plus the criterion.

    Accuracy alone confounds sensitivity with the model's willingness to say yes, and the
    paper's Gemma result turned on exactly that distinction, so the order comparison is
    reported the same way.
    """
    def z(p, n):
        corrected = (p * n + 0.5) / (n + 1.0)
        return inverse_normal_cdf(corrected)

    zh, zf = z(hit_rate, n_signal), z(false_alarm_rate, n_noise)
    return float(zh - zf), float(-0.5 * (zh + zf))



def bootstrap_mean_by_fact(
    frame: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    n_boot: int = 2000,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    grouper = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    for key, sub in frame.groupby(grouper, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        fact_means = sub.groupby("fact_id")[value_col].mean().to_numpy(dtype=float)
        observed = float(np.mean(fact_means))
        boot = np.array(
            [np.mean(rng.choice(fact_means, size=len(fact_means), replace=True)) for _ in range(n_boot)]
        )
        record = dict(zip(group_cols, key))
        record.update(
            mean=observed,
            ci_low=float(np.quantile(boot, 0.025)),
            ci_high=float(np.quantile(boot, 0.975)),
            n_facts=int(len(fact_means)),
        )
        rows.append(record)
    return pd.DataFrame(rows)


def append_records(path: Path, records: Iterable[dict[str, object]]) -> None:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)

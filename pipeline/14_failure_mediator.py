"""Failure-conditioned causal mediator: a low-rank subspace at the second entity.

The patching results say that replacing the whole second-entity state with its Latin
counterpart repairs the Devanagari failure. This stage asks whether a small subspace
of that state is enough, and whether it is *specific* to the failure:

    h_{D<-L}^U  =  h_D  +  U U^T (h_L - h_D)

U is a d_model-by-r orthonormal matrix learned by gradient descent through the frozen
model, with the objective conditioned on failure: push gated items (fail Dev--Dev,
succeed Lat--Lat) toward the correct answer, and penalise any change on items that do
not need repair (matched negatives and robust positives). Standard DAS is run with the
same parametrisation and the closure loss alone, so the difference between the two is
the conditioning and the specificity term, nothing else.

Untrained baselines at the same rank and layer: a random subspace, PCA of the donor
minus native differences, the mean difference, the probe direction, the Jacobian
direction, and the full residual patch as an approximate upper bound.

Splits. Fitting uses the train split only. Evaluation uses validation by default. The
test split is refused unless --allow-test is passed, which is meant for the single
frozen evaluation at the end of the project and nothing before it.

Groups. The `robust` control is nearly empty in native-script contexts, because a model
that fails almost every Devanagari positive has almost no item that is correct in both
scripts: 0 items for Qwen Marathi validation, 3 for Qwen Hindi, 0 for Gemma Marathi.
Specificity there rests on `negatives`, the same items at the other polarity, and on
`never`, items wrong in both scripts. English has a usable `robust` group. Read the
group sizes this script prints before believing any specificity number.

Templates. Cached states cover paraphrase 0 with answer key A. Fitting uses the
paraphrases in --train-paraphrases and evaluation those in --eval-paraphrases, so the
default (0 1 against 2) is leave-one-template-out; donor states for paraphrases other
than 0 are captured on the fly and cached in memory.

Outputs, under <run_dir>/mediator/:
    mediator_items.csv       one row per config, group, item, paraphrase
    mediator_summary.csv     per config and group: closure, flips, specificity, reverse
    subspace_<context>_L<layer>_r<rank>_<method>.npy   the learned or derived U
    mediator.done.json       resume state

Precision. The residual stream is bfloat16, and a low-rank correction is a small
perturbation to a large vector, so the subspace move is quantised at roughly six percent
of its own magnitude. The whole-state patch does not pay this, being a direct
assignment, so closure carries a noise floor that grows as rank falls. Computing the
projection in float32 does not help, because the cost is in storing the result back.
--precision-check measures what it is worth on the margin itself: a rank-1 subspace
aligned with the donor-minus-native difference is algebraically the whole patch, so any
margin difference between the two is quantisation and nothing else.

Cost. Each trained configuration is epochs x examples forward-backward passes, and the
script prints the total before it starts. Run it in two phases:

    # phase 1, layer sweep at one rank
    python 14_failure_mediator.py --run-dir results/qwen_mr --model qwen --contexts mr en

    # phase 2, rank sweep at the layer phase 1 selected on validation
    python 14_failure_mediator.py --run-dir results/qwen_mr --model qwen --contexts mr en \
        --layers 16 --ranks 1 2 4 8

Nothing here writes to the test split unless --allow-test is passed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    PromptBuilder,
    add_common_args,
    append_records,
    capture_prompt_states,
    check_contexts,
    ensure_run_dirs,
    format_chat,
    get_layers,
    label_token_ids,
    load_model_and_tokenizer,
    load_prepared_data,
    read_json,
    scale_layer,
    seed_everything,
    semantic_token_positions,
    write_json,
    MAX_LENGTH,
)

POSITIONS = ["E1", "E2", "Q", "READOUT"]
E2 = POSITIONS.index("E2")
# Settings that make two rows comparable. Layers, ranks and methods are deliberately
# absent: phase 1 and phase 2 differ in those by design and their rows belong in one
# file. Anything else differing means the rows measure different quantities.
# Deliberately excludes the paraphrase settings: every row records which fold it came
# from, so folds are distinguishable inside one file and are meant to accumulate there.
SETTINGS_KEYS = ("model", "fit_split", "eval_split", "epochs", "lr", "lambda_spec",
                 "keep_groups", "train_batch", "max_train_facts", "scale_layers", "seed",
                 "coefficients")
TRAINED = ("mediator", "das")
DERIVED = ("random", "pca", "meandiff", "probe", "jacobian")
METHODS = TRAINED + DERIVED + ("fullpatch",)


# --------------------------------------------------------------------------------
# item groups
# --------------------------------------------------------------------------------

def behavioural_groups(candidates: pd.DataFrame, context: str, split: str,
                       robust: float = 0.8, never: float = 0.2) -> dict[str, list[tuple[str, int]]]:
    """The four groups the stop conditions name, as (fact_id, truth) pairs.

    gated      fails native script, succeeds Latin, at the gate polarity: what U repairs
    negatives  the other polarity of the gated items: must stay correct
    robust     correct in both scripts at the gate polarity: must not move
    never      wrong in both scripts: U should not manufacture an answer
    """
    c = candidates.loc[(candidates.context == context) & (candidates.split == split)]
    if c.empty:
        return {k: [] for k in ("gated", "negatives", "robust", "never")}
    polarity = str(c.polarity.iloc[0])
    truth = 0 if polarity == "negative" else 1
    native = c["pos_DEVDEV"] if truth == 1 else c["neg_DEVDEV"]
    latin = c["pos_LATLAT"] if truth == 1 else c["neg_LATLAT"]
    gated = c.loc[c.stable == 1, "fact_id"].astype(str).tolist()
    gated_set = set(gated)
    robust_ids = c.loc[(native >= robust) & (latin >= robust), "fact_id"].astype(str).tolist()
    never_ids = c.loc[(native <= never) & (latin <= never), "fact_id"].astype(str).tolist()
    # Both controls exclude the gated items explicitly. Under the margin gate an item can
    # fail in both scripts by accuracy while the donor still moves its margin by the gate
    # threshold, so it would otherwise qualify as gated and as never-correct at once and
    # receive contradictory training signals. That does not happen in the current runs,
    # which is precisely why relying on it would be unsafe.
    return {
        "gated": [(f, truth) for f in gated],
        "negatives": [(f, 1 - truth) for f in gated],
        "robust": [(f, truth) for f in robust_ids if f not in gated_set],
        "never": [(f, truth) for f in never_ids if f not in gated_set],
    }


def toward_correct(margin, truth: int):
    """Signed margin: positive means the model leans toward the right answer."""
    return margin if int(truth) == 1 else -margin


def fold_tag(train_paraphrases, eval_paraphrases) -> str:
    """Names one leave-one-template-out fold, e.g. t12e0 for fit on 1 and 2, test on 0.

    It is part of a configuration's identity: the same layer, rank and method fitted on
    different templates is a different result, and both belong in the same file.
    """
    return ("t" + "".join(str(p) for p in train_paraphrases)
            + "e" + "".join(str(p) for p in eval_paraphrases))


def stable_seed(*parts) -> int:
    """A seed that depends on the arguments and not on execution order.

    Python's hash() is salted per process, so a resumed run would draw different random
    baselines from an uninterrupted one. This does not.
    """
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def derangement(n: int, rng) -> np.ndarray:
    """A permutation with no fixed point, so that no item keeps its own coefficient.

    A plain shuffle leaves about one item in every run holding its own amount, which
    would weaken the control by exactly the effect it is meant to remove.
    """
    if n < 2:
        return np.zeros(n, dtype=int)
    for _ in range(1000):
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            return order
    order = np.roll(np.arange(n), 1)   # always a derangement for n >= 2
    return order


def settings_of(args) -> dict:
    out = {}
    for key in SETTINGS_KEYS:
        value = getattr(args, key)
        out[key] = list(value) if isinstance(value, list) else value
    return out


# --------------------------------------------------------------------------------
# states and hooks
# --------------------------------------------------------------------------------

class StateSource:
    """Second-entity states for any prompt. Returns float32 numpy vectors [d_model].

    Stage 02's cache is used when it is present, and everything is captured live when it
    is not. The cache holds paraphrase 0 with answer key A only, so a held-out-template
    run captures most of its states either way; making it optional means this stage does
    not depend on a multi-gigabyte artefact that is gitignored and absent from both
    results trees.
    """

    def __init__(self, model, tokenizer, device, builder: PromptBuilder, paths: dict):
        self.model, self.tokenizer, self.device, self.builder = model, tokenizer, device, builder
        meta_path = paths["states"] / "pair_state_metadata.csv"
        array_path = paths["states"] / "pair_states.npy"
        self.cached = None
        self.index = {}
        if meta_path.exists() and array_path.exists():
            meta = pd.read_csv(meta_path)
            self.cached = np.load(array_path, mmap_mode="r")
            self.index = {
                (str(r.fact_id), r.context, r.condition, int(r.truth)): int(r.state_row)
                for r in meta.itertuples()
            }
            print(f"state cache: {array_path.name}, {len(self.index)} cached prompts")
        else:
            print("state cache: absent, capturing every state live "
                  "(run stage 02 to speed this up)")
        self.live: dict[tuple, np.ndarray] = {}
        self.positions: dict[tuple, int] = {}

    def prompt(self, fid: str, context: str, condition: str, truth: int, paraphrase: int) -> dict:
        return self.builder.make(fid, context, condition, truth, paraphrase, "A")

    def e2_position(self, rec: dict) -> int:
        """Token index of the second entity, cached: the same prompt is re-tokenised
        once per example per epoch otherwise."""
        key = (rec["fact_id"], rec["context"], rec["condition"], rec["truth"], rec["paraphrase_id"])
        if key not in self.positions:
            self.positions[key] = semantic_token_positions(
                self.tokenizer, rec["prompt"], rec["a_text"], rec["b_text"]
            )["E2"]
        return self.positions[key]

    def e2_state(self, fid: str, context: str, condition: str, truth: int,
                 paraphrase: int, layer: int) -> np.ndarray:
        cache_key = (fid, context, condition, truth)
        if paraphrase == 0 and self.cached is not None and cache_key in self.index:
            row = self.index[cache_key]
            return np.asarray(self.cached[row, layer, E2], dtype=np.float32)
        key = (fid, context, condition, truth, paraphrase)
        if key not in self.live:
            arr, _ = capture_prompt_states(self.model, self.tokenizer, self.device,
                                           self.prompt(fid, context, condition, truth, paraphrase))
            self.live[key] = arr[:, E2, :].astype(np.float32)  # [layers, d]
        return self.live[key][layer]


class SubspacePatch:
    """Forward hook: at one layer and one token position, replace the U-coordinates of
    the running state with those of a donor vector. Differentiable in U."""

    def __init__(self, model, layer: int):
        self.block = get_layers(model)[layer]
        self.handle = self.block.register_forward_hook(self)
        self.off()

    def off(self):
        """Disable the hook. Called between every measurement so that a mode set for
        one configuration cannot leak into the next."""
        self.position = None
        self.donor = None
        self.U = None          # [d, r] tensor, orthonormal columns
        self.full = False      # replace the whole vector instead
        self.coeff = None      # supplied coordinates, used instead of the donor's

    def subspace(self, position: int, donor, U):
        self.position, self.donor, self.U, self.full = position, donor, U, False
        self.coeff = None

    def directional(self, position: int, U, coeff):
        """Move along the columns of U by coordinates supplied from outside, rather than
        by the coordinates this item's own donor implies. This is what separates the
        direction from the amount: the direction is U for every item, while `coeff` may
        be this item's own, another item's, or one value shared by all of them."""
        self.position, self.donor, self.U, self.full = position, None, U, False
        self.coeff = coeff

    def whole(self, position: int, donor):
        self.position, self.donor, self.U, self.full = position, donor, None, True
        self.coeff = None

    def __call__(self, _module, _inputs, output):
        if self.position is None:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        h = hidden[:, self.position, :]
        if self.full:
            new = self.donor.to(hidden.dtype).expand_as(h)
        elif self.coeff is not None:
            U = self.U.to(hidden.dtype)
            new = h + (U @ self.coeff.to(hidden.dtype)).expand_as(h)
        else:
            U = self.U.to(hidden.dtype)
            donor = self.donor.to(hidden.dtype)
            new = h + (donor - h) @ U @ U.T
        hidden = torch_cat_replace(hidden, self.position, new)
        return (hidden, *output[1:]) if is_tuple else hidden

    def remove(self):
        self.handle.remove()


def torch_cat_replace(hidden, position: int, new):
    """hidden with hidden[:, position, :] replaced by new, without in-place ops so that
    autograd can trace through new."""
    import torch
    before, after = hidden[:, :position, :], hidden[:, position + 1:, :]
    return torch.cat([before, new.unsqueeze(1), after], dim=1)


def margin_with_grad(model, tokenizer, device, prompt: str, grad: bool):
    """Yes-minus-no margin (A minus B logit) for a single prompt, with or without a graph.

    Every prompt here is built with yes_letter="A", so the margin is the A-minus-B
    logit difference and needs no per-prompt sign flip.
    """
    import torch
    if not hasattr(tokenizer, "_ab_ids"):
        tokenizer._ab_ids = label_token_ids(tokenizer)
    a_id, b_id = tokenizer._ab_ids
    encoded = tokenizer(format_chat(tokenizer, prompt), return_tensors="pt",
                        truncation=True, max_length=MAX_LENGTH).to(device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(**encoded, use_cache=False).logits[0, -1].float()
        return logits[a_id] - logits[b_id]


# --------------------------------------------------------------------------------
# derived subspaces
# --------------------------------------------------------------------------------

def orthonormal(M: np.ndarray, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(M[:, :rank].astype(np.float64))
    q = q[:, :rank].astype(np.float32)
    if q.shape[1] != rank:
        # Silently returning fewer columns would compare a short subspace against a
        # random baseline that always has its full rank, which flatters the baseline.
        raise ValueError(
            f"asked for rank {rank} but only {q.shape[1]} independent directions are "
            f"available; see usable_rank()")
    return q


def usable_rank(n_diffs: int) -> int:
    """How many directions the difference matrix can actually support.

    PCA centres before decomposing, which costs one degree of freedom, so n difference
    vectors give at most n-1 meaningful components. Beyond that the columns are noise
    at the 1e-7 level and the QR that orthonormalises them is arbitrary.
    """
    return max(1, n_diffs - 1)


def pad_with_pca(first: np.ndarray, diffs: np.ndarray, rank: int) -> np.ndarray:
    """A rank-1 direction padded to `rank` with the leading principal components of the
    differences after removing that direction."""
    d = first / (np.linalg.norm(first) + 1e-8)
    if rank == 1:
        return d[:, None].astype(np.float32)
    resid = diffs - np.outer(diffs @ d, d)
    _, _, vt = np.linalg.svd(resid - resid.mean(0), full_matrices=False)
    return orthonormal(np.column_stack([d, vt[:rank - 1].T]), rank)


def derived_subspace(method: str, rank: int, diffs: np.ndarray, rng: np.random.Generator,
                     probe_dir: np.ndarray | None, jac_dir: np.ndarray | None) -> np.ndarray:
    d_model = diffs.shape[1]
    if method == "random":
        return orthonormal(rng.standard_normal((d_model, rank)), rank)
    if method == "pca":
        _, _, vt = np.linalg.svd(diffs - diffs.mean(0), full_matrices=False)
        return orthonormal(vt[:rank].T, rank)
    if method == "meandiff":
        return pad_with_pca(diffs.mean(0), diffs, rank)
    if method == "probe":
        return pad_with_pca(probe_dir, diffs, rank)
    if method == "jacobian":
        return pad_with_pca(jac_dir, diffs, rank)
    raise ValueError(method)


def probe_direction(source: StateSource, corpus: pd.DataFrame, context: str, layer: int,
                    alpha: float = 10.0, max_facts: int = 120) -> np.ndarray:
    """Ridge direction separating SAME from DIFFERENT on Lat--Lat train states at E2,
    the same estimator stage 03 uses, read at the intervention site.

    Capped at max_facts because without a state cache each fact costs two forward
    passes, and the direction is stable long before the whole train split is used.
    """
    from sklearn.linear_model import RidgeClassifier
    train = corpus.loc[corpus.split == "train", "fact_id"].astype(str).tolist()[:max_facts]
    X, y = [], []
    for fid in train:
        for truth in (1, 0):
            X.append(source.e2_state(fid, context, "LATLAT", truth, 0, layer))
            y.append(truth)
    X = np.stack(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    clf = RidgeClassifier(alpha=alpha).fit(X, np.array(y))
    return clf.coef_.ravel().astype(np.float32)


def jacobian_direction(model, tokenizer, device, source: StateSource, items, context: str,
                       layer: int, paraphrase: int) -> np.ndarray:
    """Mean gradient of the toward-correct margin with respect to the E2 state at the
    layer, over the gated train items in Dev--Dev."""
    import torch
    grads = []
    block = get_layers(model)[layer]
    for fid, truth in items:
        rec = source.prompt(fid, context, "DEVDEV", truth, paraphrase)
        pos = source.e2_position(rec)
        holder = {}

        def hook(_m, _i, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            # The model's parameters are frozen, so nothing upstream requires grad and
            # this tensor is not in a graph. Making it a leaf that requires grad is what
            # lets the backward pass deposit d(margin)/d(state) on it.
            hidden = hidden.detach().requires_grad_(True)
            holder["h"] = hidden
            return (hidden, *output[1:]) if is_tuple else hidden

        handle = block.register_forward_hook(hook)
        try:
            m = margin_with_grad(model, tokenizer, device, rec["prompt"], grad=True)
            toward_correct(m, truth).backward()
            grads.append(holder["h"].grad[0, pos].float().cpu().numpy())
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    return np.mean(grads, axis=0).astype(np.float32)


# --------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------

def train_subspace(model, tokenizer, device, source: StateSource, patch: SubspacePatch,
                   context: str, layer: int, rank: int, groups: dict, paraphrases: list[int],
                   method: str, epochs: int, lr: float, lam: float, batch: int,
                   init: np.ndarray, keep_groups: tuple, seed: int, log) -> np.ndarray:
    """Learn U.

    `mediator`: logistic loss toward the correct answer on gated items, plus lam times
    the absolute margin change on the keep groups, which is what makes the subspace
    specific to the failure rather than a general push toward yes.

    `das`: the same parametrisation and the closure loss alone, on every group, with no
    penalty. The only differences from the mediator are the conditioning and the
    penalty, so a gap between the two is attributable to those and not to the fit.
    """
    import torch
    d_model = init.shape[0]
    W = torch.nn.Parameter(torch.as_tensor(init, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([W], lr=lr)

    def U_of(W):
        q, _ = torch.linalg.qr(W)
        return q

    if method == "mediator":
        repair = [(f, t, "gated") for f, t in groups["gated"]]
        keep = [(f, t, g) for g in keep_groups for f, t in groups[g]]
        if not keep:
            log(f"    WARNING: no items in keep groups {keep_groups}; the specificity "
                f"term is inactive and this run is DAS with extra steps")
    else:
        repair = [(f, t, g) for g in ("gated",) + tuple(keep_groups) for f, t in groups[g]]
        keep = []
    examples = [(f, t, g, p) for p in paraphrases for f, t, g in repair + keep]
    baseline = {}
    for f, t, g, p in examples:
        if method == "mediator" and g != "gated":
            rec = source.prompt(f, context, "DEVDEV", t, p)
            patch.off()
            baseline[(f, t, p)] = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
    # Seeded from the run seed and this configuration, so --seed changes the fit and a
    # resumed run reproduces an uninterrupted one.
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(examples))
        total, n = 0.0, 0
        for start in range(0, len(order), batch):
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for idx in order[start:start + batch]:
                f, t, g, p = examples[idx]
                rec = source.prompt(f, context, "DEVDEV", t, p)
                pos = source.e2_position(rec)
                patch.subspace(
                    pos,
                    torch.as_tensor(source.e2_state(f, context, "LATLAT", t, p, layer), device=device),
                    U_of(W),
                )
                m = margin_with_grad(model, tokenizer, device, rec["prompt"], grad=True)
                s = toward_correct(m, t)
                if method == "mediator" and g != "gated":
                    loss = lam * (m - baseline[(f, t, p)]).abs()
                else:
                    loss = torch.nn.functional.softplus(-s)
                loss = loss / min(batch, len(order) - start)
                loss.backward()
                loss_sum += float(loss.detach())
            opt.step()
            total += loss_sum
            n += 1
        log(f"    {method} L{layer} r{rank} epoch {epoch + 1}/{epochs} loss {total / max(n, 1):.4f}")
    patch.off()
    with torch.no_grad():
        return U_of(W).float().cpu().numpy()


# --------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------

def coefficients_for(source: StateSource, device, context: str, layer: int, U_t,
                     items, paraphrases: list[int]) -> dict:
    """The U-coordinates of each item's donor-minus-native difference.

    These are the amounts the matched arm moves by. Holding U fixed and swapping these
    between items is what tests whether the amount carries item-specific information or
    is merely a scale.
    """
    import torch
    out = {}
    U32 = U_t.to(torch.float32)
    for fid, truth in items:
        for p in paraphrases:
            donor = torch.as_tensor(
                source.e2_state(fid, context, "LATLAT", truth, p, layer),
                device=device, dtype=torch.float32)
            native = torch.as_tensor(
                source.e2_state(fid, context, "DEVDEV", truth, p, layer),
                device=device, dtype=torch.float32)
            out[(fid, truth, p)] = U32.T @ (donor - native)
    return out


def evaluate(model, tokenizer, device, source: StateSource, patch: SubspacePatch,
             context: str, layer: int, layer_label: int, rank: int, method: str,
             U: np.ndarray | None, groups: dict, paraphrases: list[int], split: str,
             full_cache: dict | None = None, reverse_cache: dict | None = None,
             fold: str = "", coefficient: str = "matched", fixed_coeff=None,
             seed: int = 0) -> list[dict]:
    """`layer` is the layer actually patched; `layer_label` is the reference-model layer
    the row is filed under. They differ under --scale-layers, and reading the donor from
    one while patching at the other was a real bug on Llama.

    `full_cache` holds whole-state-patch margins, which depend on the item and the layer
    but not on the method or the rank, so they are computed once per layer rather than
    once per method.
    """
    import torch
    rows = []
    full_cache = {} if full_cache is None else full_cache
    reverse_cache = {} if reverse_cache is None else reverse_cache
    U_t = None if U is None else torch.as_tensor(U, device=device)
    if coefficient not in ("matched", "shuffled", "fixed"):
        raise ValueError(f"unknown coefficient mode {coefficient!r}")
    if coefficient == "fixed" and fixed_coeff is None and method != "fullpatch":
        raise ValueError("the fixed arm needs a coefficient fitted on the training items")
    for group, items in groups.items():
        # The shuffled arm reassigns amounts within a group, so that the comparison is
        # against items facing the same intervention, not across groups.
        swap = {}
        if coefficient == "shuffled" and method != "fullpatch" and U_t is not None:
            pool = coefficients_for(source, device, context, layer, U_t, items, paraphrases)
            keys = sorted(pool)
            if len(keys) > 1:
                rng = np.random.default_rng(stable_seed("shuffle", context, layer, rank,
                                                       method, group, fold, seed))
                order = derangement(len(keys), rng)
                swap = {k: pool[keys[j]] for k, j in zip(keys, order)}
            else:
                # One item cannot be given anyone else's amount, and giving it its own
                # would quietly turn the control back into the matched arm.
                swap = {}
        for fid, truth in items:
            for p in paraphrases:
                rec = source.prompt(fid, context, "DEVDEV", truth, p)
                pos = source.e2_position(rec)
                donor = torch.as_tensor(source.e2_state(fid, context, "LATLAT", truth, p, layer), device=device)
                cache_key = (context, layer, fid, truth, p)
                if cache_key in full_cache:
                    base, full = full_cache[cache_key]
                else:
                    patch.off()
                    base = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
                    patch.whole(pos, donor)
                    full = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
                    full_cache[cache_key] = (base, full)
                if method == "fullpatch":
                    med = full
                elif coefficient == "matched":
                    patch.subspace(pos, donor, U_t)
                    med = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
                else:
                    c = fixed_coeff if coefficient == "fixed" else swap.get((fid, truth, p))
                    if c is None:
                        patch.off()
                        continue
                    patch.directional(pos, U_t, c)
                    med = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
                patch.off()
                row = {
                    "context": context, "split": split, "fold": fold,
                    "layer": layer_label,
                    "rank": 0 if method == "fullpatch" else rank,
                    "method": method,
                    "coefficient": "matched" if method == "fullpatch" else coefficient,
                    "group": group, "fact_id": fid, "truth": truth,
                    "paraphrase_id": p, "baseline_margin": base, "fullpatch_margin": full,
                    "mediated_margin": med,
                    "delta_full": toward_correct(full - base, truth),
                    "delta_mediated": toward_correct(med - base, truth),
                    "abs_change": abs(med - base),
                    "baseline_correct": int(toward_correct(base, truth) > 0),
                    "mediated_correct": int(toward_correct(med, truth) > 0),
                    "reverse_drop": np.nan,
                }
                # Reverse corruption on gated items: Devanagari coordinates into the
                # Latin run. Sufficiency of U to break the run, not necessity.
                # It uses the donor's own coordinates, so it does not depend on the
                # coefficient arm. Measuring it once avoids paying for it three times
                # and avoids filing one measurement under three arms as though each had
                # been measured separately.
                if group == "gated" and method != "fullpatch" and coefficient == "matched":
                    lrec = source.prompt(fid, context, "LATLAT", truth, p)
                    lpos = source.e2_position(lrec)
                    ddonor = torch.as_tensor(source.e2_state(fid, context, "DEVDEV", truth, p, layer), device=device)
                    if cache_key in reverse_cache:
                        lbase = reverse_cache[cache_key]
                    else:
                        patch.off()
                        lbase = float(margin_with_grad(model, tokenizer, device, lrec["prompt"], False))
                        reverse_cache[cache_key] = lbase
                    patch.subspace(lpos, ddonor, U_t)
                    lrev = float(margin_with_grad(model, tokenizer, device, lrec["prompt"], False))
                    patch.off()
                    row["reverse_drop"] = toward_correct(lbase - lrev, truth)
                rows.append(row)
    return rows


# Closure is a fraction of the whole-state effect, so it means nothing where that
# effect is absent. The never and robust groups are exactly the ones the whole patch
# does not move, and an unguarded ratio reported values around 1e8 there.
MIN_FULL_DELTA = 0.5


def closure_of(mediated, full, min_full: float = MIN_FULL_DELTA):
    """mediated / full, undefined unless the whole patch repairs by at least min_full
    logits in the direction of the correct answer."""
    full = np.asarray(full, dtype=float)
    mediated = np.asarray(mediated, dtype=float)
    return np.where(full >= min_full, mediated / np.where(full >= min_full, full, 1.0), np.nan)


def row_key(row) -> str:
    """The configuration a row belongs to, in the form used by the resume set.

    The coefficient arm is part of the key. If it were left out, every row would fail to
    match a resume set written with it and the next run would delete the lot.
    """
    get = row.get if hasattr(row, "get") else (lambda _k, _d=None: _d)
    fold = get("fold", "") or ""
    arm = get("coefficient", "matched") or "matched"
    return (f"{row['context']}:L{row['layer']}:r{row['rank']}:{row['method']}"
            f":{fold}:{arm}")


def drop_orphan_rows(items_path: Path, completed: set) -> None:
    """Remove rows whose configuration is not marked complete.

    Rows are appended before the key is recorded, so a crash between the two leaves rows
    on disk that the resume does not know about. The configuration would then run again
    and append a second copy, and every average over that configuration would count it
    twice.
    """
    if not items_path.exists():
        return
    items = pd.read_csv(items_path)
    if items.empty:
        return
    # fullpatch is stored with rank 0, and its resume key carries the rank it was
    # launched under, so it is matched by prefix rather than by the full key.
    def is_complete(row) -> bool:
        get = row.get if hasattr(row, "get") else (lambda _k, _d=None: _d)
        fold = get("fold", "") or ""
        arm = get("coefficient", "matched") or "matched"
        # A resume set written before the arms existed has no arm suffix. Those runs
        # were all matched, so accept the older spelling rather than delete their rows.
        suffixes = [f":{arm}"] + ([""] if arm == "matched" else [])
        if row["method"] == "fullpatch":
            prefix = f"{row['context']}:L{row['layer']}:"
            return any(k.startswith(prefix) and k.endswith(f":fullpatch:{fold}{s}")
                       for s in suffixes for k in completed)
        full = row_key(row)
        return any((full if s else full.rsplit(":", 1)[0]) in completed
                   for s in suffixes)

    keep = items.apply(is_complete, axis=1)
    if bool(keep.all()):
        return
    dropped = int((~keep).sum())
    orphans = sorted({row_key(r) for _, r in items[~keep].iterrows()})
    items[keep].to_csv(items_path, index=False)
    print(f"dropped {dropped} rows from an interrupted run, in {len(orphans)} "
          f"configuration(s): {', '.join(orphans[:4])}"
          + (" ..." if len(orphans) > 4 else ""))


def summarise(items: pd.DataFrame, min_full: float = MIN_FULL_DELTA) -> pd.DataFrame:
    by = ["context", "split", "layer", "rank", "method", "group"]
    # Files written before the coefficient arms existed have one implicit arm.
    if "coefficient" in items.columns:
        by.insert(5, "coefficient")
    if "fold" in items.columns:
        by.insert(2, "fold")
    g = items.groupby(by, as_index=False)
    s = g.agg(n_items=("fact_id", "nunique"), n_rows=("fact_id", "size"),
              mean_delta_full=("delta_full", "mean"),
              mean_delta_mediated=("delta_mediated", "mean"),
              mean_abs_change=("abs_change", "mean"),
              baseline_acc=("baseline_correct", "mean"),
              mediated_acc=("mediated_correct", "mean"),
              reverse_drop=("reverse_drop", "mean"))
    s["closure"] = closure_of(s.mean_delta_mediated, s.mean_delta_full, min_full)
    s["closure_defined"] = (s.mean_delta_full >= min_full).astype(int)
    return s


def precision_check(model, tokenizer, device, source: StateSource, patch: SubspacePatch,
                    context: str, layer: int, items, paraphrase: int, n: int) -> None:
    """How much of the closure signal is bfloat16 rounding?

    A rank-1 subspace spanning the donor-minus-native difference satisfies
    U U^T (h_L - h_D) = h_L - h_D exactly, so the subspace patch is algebraically the
    whole-state patch. Any difference in the resulting margin is quantisation of the
    residual stream, measured on the quantity that is actually reported.
    """
    import torch
    gaps, fulls = [], []
    for fid, truth in list(items)[:n]:
        rec = source.prompt(fid, context, "DEVDEV", truth, paraphrase)
        pos = source.e2_position(rec)
        h = source.e2_state(fid, context, "DEVDEV", truth, paraphrase, layer)
        d = source.e2_state(fid, context, "LATLAT", truth, paraphrase, layer)
        donor = torch.as_tensor(d, device=device)
        patch.off()
        base = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
        patch.whole(pos, donor)
        full = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
        direction = d - h
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        U = torch.as_tensor((direction / norm)[:, None], device=device)
        patch.subspace(pos, donor, U)
        rank1 = float(margin_with_grad(model, tokenizer, device, rec["prompt"], False))
        patch.off()
        gaps.append(abs(rank1 - full))
        fulls.append(abs(full - base))
    if not gaps:
        return
    gap, effect = float(np.mean(gaps)), float(np.mean(fulls))
    print(f"  precision check L{layer}: a rank-1 subspace spanning the difference is "
          f"algebraically the whole patch; measured margin gap {gap:.3f} logits against "
          f"a whole-patch effect of {effect:.3f} ({100 * gap / max(effect, 1e-9):.1f}% "
          f"of the effect). That is the bfloat16 floor on closure.")


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--contexts", nargs="+", default=["mr", "en"])
    # Phase 1 by default: sweep the layer at one rank. Phase 2 is the rank sweep at
    # whichever layer phase 1 selects on validation, which is a separate invocation.
    # Running both grids at once costs about four times as much and answers the rank
    # question at layers that were never going to be used.
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 13, 14, 15, 16, 17])
    ap.add_argument("--ranks", nargs="+", type=int, default=[4])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--out-name", default="mediator",
                    help="subdirectory of the run directory to write into. Use a new "
                         "name when settings change, rather than --force.")
    ap.add_argument("--coefficients", nargs="+", default=["matched"],
                    choices=["matched", "shuffled", "fixed"],
                    help="how far to move along the fitted direction. matched uses this "
                         "item's own donor difference; shuffled gives it another item's, "
                         "within the same group and never its own; fixed gives every "
                         "item the mean over the training items. Comparing them holds "
                         "the direction constant and varies only the amount.")
    ap.add_argument("--fit-split", default="train", choices=["train"])
    ap.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    ap.add_argument("--allow-test", action="store_true",
                    help="Required to evaluate on test. Meant for the one frozen evaluation.")
    ap.add_argument("--train-paraphrases", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--eval-paraphrases", nargs="+", type=int, default=[2],
                    help="Default holds template 2 out entirely: leave-one-template-out.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda-spec", type=float, default=1.0)
    ap.add_argument("--keep-groups", nargs="+", default=["negatives", "robust"],
                    choices=["negatives", "robust", "never"],
                    help="Groups the specificity penalty protects. `robust` is empty in "
                         "native-script contexts on these models, so `negatives` carries "
                         "it there; add `never` to also penalise manufacturing an answer "
                         "on items the model gets wrong in both scripts.")
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--max-train-facts", type=int, default=60)
    ap.add_argument("--scale-layers", action="store_true")
    ap.add_argument("--min-full-delta", type=float, default=MIN_FULL_DELTA,
                    help="Closure is left undefined where the whole-state patch repairs "
                         "by less than this many logits, because the ratio is meaningless "
                         "when its denominator is noise.")
    ap.add_argument("--precision-check", type=int, default=8, metavar="N",
                    help="Items per layer used to measure the bfloat16 floor on closure. "
                         "0 disables it.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the item groups and the config grid, load no model.")
    args = ap.parse_args()
    check_contexts(args.contexts)
    if args.eval_split == "test" and not args.allow_test:
        raise SystemExit("evaluation on test is refused without --allow-test; "
                         "freeze every choice first, then run once")
    if set(args.train_paraphrases) & set(args.eval_paraphrases):
        print("warning: train and eval paraphrases overlap; this is not a held-out-template run")
    seed_everything(args.seed)
    fold = fold_tag(args.train_paraphrases, args.eval_paraphrases)
    print(f"fold {fold}: fit on paraphrases {args.train_paraphrases}, "
          f"evaluate on {args.eval_paraphrases}")
    paths = ensure_run_dirs(args.run_dir)
    # A separate name lets an experiment with different settings live beside an earlier
    # one instead of forcing --force, which would delete results that are still wanted.
    out_dir = paths["root"] / args.out_name
    out_dir.mkdir(exist_ok=True)
    items_path = out_dir / "mediator_items.csv"
    done_path = out_dir / "mediator.done.json"
    if args.force:
        for p in (items_path, done_path, out_dir / "mediator_summary.csv"):
            if p.exists():
                p.unlink()

    # Rows already in mediator_items.csv were produced under the settings recorded in
    # done.json. Appending rows from different settings to the same file, and then
    # averaging them together, would be silent and wrong.
    settings = settings_of(args)
    previous = read_json(done_path) if done_path.exists() else {}
    stored = previous.get("settings")
    if previous and stored is None:
        raise SystemExit(
            f"{done_path.name} records completed work but no settings, so the rows in "
            f"{items_path.name} cannot be shown to be comparable with this run. "
            f"Use --force to start again.")
    if stored is not None and stored != settings:
        differing = sorted(k for k in settings if stored.get(k) != settings.get(k))
        raise SystemExit(
            "these results were produced under different settings: "
            + "; ".join(f"{k}: {stored.get(k)!r} now {settings[k]!r}" for k in differing)
            + f".\nAppending to {items_path.name} would mix them. Use --force to start "
              "again, or point --run-dir at a different directory."
        )

    corpus, negatives = load_prepared_data(args.run_dir)
    candidates = pd.read_csv(paths["patching"] / "causal_candidates.csv")
    fit_groups = {c: behavioural_groups(candidates, c, args.fit_split) for c in args.contexts}
    eval_groups = {c: behavioural_groups(candidates, c, args.eval_split) for c in args.contexts}
    for c in args.contexts:
        capped = []
        for g in fit_groups[c]:
            full = len(fit_groups[c][g])
            fit_groups[c][g] = fit_groups[c][g][:args.max_train_facts]
            if len(fit_groups[c][g]) < full:
                capped.append(f"{g} {full}->{len(fit_groups[c][g])}")
        print(f"{c}: fit on {args.fit_split} "
              + ", ".join(f"{g}={len(v)}" for g, v in fit_groups[c].items())
              + f" | eval on {args.eval_split} "
              + ", ".join(f"{g}={len(v)}" for g, v in eval_groups[c].items()))
        if capped:
            print(f"  capped by --max-train-facts {args.max_train_facts}: "
                  + ", ".join(capped) + " (evaluation groups are never capped)")
        empty = [g for g in args.keep_groups if not fit_groups[c][g]]
        if empty:
            print(f"  WARNING {c}: keep groups {empty} are empty on the fit split. The "
                  f"specificity penalty will rest on what remains; if nothing remains, "
                  f"the mediator reduces to DAS. See the docstring.")
    # The subspace methods are fitted to the gated differences, so a rank larger than
    # those differences can support is not a smaller result, it is a meaningless one.
    # Checked here so it fails before the weights load rather than hours in.
    subspace_methods = set(args.methods) & set(DERIVED + TRAINED)
    for c in args.contexts if subspace_methods else ():
        n_diffs = len(fit_groups[c]["gated"]) * len(args.train_paraphrases)
        if not n_diffs:
            continue
        too_big = [r for r in args.ranks if r > usable_rank(n_diffs)]
        if too_big:
            raise SystemExit(
                f"{c}: {len(fit_groups[c]['gated'])} gated fit items x "
                f"{len(args.train_paraphrases)} paraphrases give {n_diffs} difference "
                f"vectors, which support rank {usable_rank(n_diffs)} at most. "
                f"Ranks {too_big} are not estimable. Drop them, add a training "
                f"paraphrase, or raise --max-train-facts.")

    grid = [(c, L, r, m) for c in args.contexts for L in args.layers for r in args.ranks
            for m in args.methods if not (m == "fullpatch" and r != args.ranks[0])]

    # What this will cost, before it is spent, counted the way the code actually runs.
    #
    # Training: one forward and one backward per example per epoch, for the trained
    # methods only, plus one unpatched baseline forward per keep example for the
    # mediator.
    #
    # Evaluation: the unpatched and whole-state margins are cached per context and
    # layer, so they cost two forwards per item-paraphrase once, not once per method.
    # Each further method costs one forward per item-paraphrase, and two more on gated
    # items for the reverse-corruption arm. The precision check adds three per item.
    n_trained = sum(1 for c, L, r, m in grid if m in TRAINED)
    n_par_fit, n_par_eval = len(args.train_paraphrases), len(args.eval_paraphrases)
    train_ex = {c: n_par_fit * sum(len(fit_groups[c][g])
                                   for g in ("gated",) + tuple(args.keep_groups))
                for c in args.contexts}
    train_passes = sum(args.epochs * train_ex[c] for c, L, r, m in grid if m in TRAINED)

    layers_per_context = {c: len({L for cc, L, r, m in grid if cc == c}) for c in args.contexts}
    n_all = {c: n_par_eval * sum(len(v) for v in eval_groups[c].values()) for c in args.contexts}
    n_gated = {c: n_par_eval * len(eval_groups[c]["gated"]) for c in args.contexts}
    # Per layer: two passes per item-paraphrase to fill the unpatched and whole-state
    # cache, and one per gated item to fill the reverse-corruption baseline. Per
    # configuration after that: one pass per item-paraphrase, and one more per gated
    # item for the corrupted Latin run.
    eval_passes = sum((2 * n_all[c] + n_gated[c]) * layers_per_context[c]
                      for c in args.contexts)
    # Each coefficient arm re-evaluates every item along the same direction. The
    # reverse-corruption pass does not depend on the arm and is measured once.
    n_arms = max(1, len(args.coefficients))
    eval_passes += sum(n_arms * n_all[c] + n_gated[c]
                       for c, L, r, m in grid if m != "fullpatch")
    eval_passes += sum(3 * min(args.precision_check, len(eval_groups[c]["gated"]))
                       * layers_per_context[c] for c in args.contexts)
    print(f"{len(grid)} configs ({n_trained} trained); paraphrases fit "
          f"{args.train_paraphrases} eval {args.eval_paraphrases}; "
          f"coefficient arms {list(args.coefficients)}")
    print(f"cost: about {train_passes:,} training passes (forward+backward) and "
          f"{eval_passes:,} evaluation forward passes")
    if args.dry_run:
        return

    import torch
    model, tokenizer, device, _, meta = load_model_and_tokenizer(args.model, args.batch_size)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = meta["n_layers"]
    layer_map = {L: (scale_layer(L, n_layers) if args.scale_layers else L) for L in args.layers}
    builder = PromptBuilder(corpus, negatives)
    source = StateSource(model, tokenizer, device, builder, paths)
    completed = set(previous.get("completed", []))
    drop_orphan_rows(items_path, completed)
    log = print

    for context in args.contexts:
        if not fit_groups[context]["gated"] or not eval_groups[context]["gated"]:
            print(f"{context}: no gated items on one side; skipped")
            continue
        for L_ref in args.layers:
            L = layer_map[L_ref]
            if L >= n_layers:
                print(f"  {context} L{L_ref}: beyond the model's {n_layers} layers; "
                      f"skipped. Use --scale-layers for a model of a different depth.")
                continue
            # What is left to do at this layer, decided before any of the work that
            # feeds it. A resumed run must not pay for a probe direction it will throw
            # away.
            pending = [(r, m) for r in args.ranks for m in args.methods
                       if not (m == "fullpatch" and r != args.ranks[0])
                       and f"{context}:L{L_ref}:r{r}:{m}:{fold}" not in completed]
            if not pending:
                print(f"  {context} L{L_ref}: already complete")
                continue
            pending_methods = {m for _, m in pending}

            # Differences and directions shared by the derived baselines at this layer.
            # Only the subspace methods need them; a run of fullpatch alone needs none.
            needs_diffs = bool(pending_methods & set(DERIVED + TRAINED))
            diffs = np.stack([
                source.e2_state(f, context, "LATLAT", t, p, L) - source.e2_state(f, context, "DEVDEV", t, p, L)
                for p in args.train_paraphrases for f, t in fit_groups[context]["gated"]
            ]) if needs_diffs else np.zeros((0, 0), dtype=np.float32)
            probe_dir = (probe_direction(source, corpus, context, L)
                         if "probe" in pending_methods else None)
            jac_dir = (jacobian_direction(model, tokenizer, device, source, fit_groups[context]["gated"],
                                          context, L, args.train_paraphrases[0])
                       if "jacobian" in pending_methods else None)
            patch = SubspacePatch(model, L)
            full_cache: dict = {}
            reverse_cache: dict = {}
            try:
                if args.precision_check:
                    precision_check(model, tokenizer, device, source, patch, context, L,
                                    eval_groups[context]["gated"], args.eval_paraphrases[0],
                                    args.precision_check)
                for rank in args.ranks:
                    for method in args.methods:
                        if method == "fullpatch" and rank != args.ranks[0]:
                            continue
                        arms = (["matched"] if method == "fullpatch"
                                else list(args.coefficients))
                        key = f"{context}:L{L_ref}:r{rank}:{method}:{fold}"
                        todo = [a for a in arms
                                if f"{key}:{a}" not in completed]
                        if not todo:
                            continue
                        # Seeded from the configuration rather than from a generator
                        # advanced in grid order, so resuming reproduces a full run.
                        config_seed = stable_seed(args.seed, key)
                        U = None
                        if method in DERIVED:
                            U = derived_subspace(method, rank, diffs,
                                                 np.random.default_rng(config_seed),
                                                 probe_dir, jac_dir)
                        elif method in TRAINED:
                            init = derived_subspace("pca", rank, diffs,
                                                    np.random.default_rng(config_seed),
                                                    None, None)
                            U = train_subspace(model, tokenizer, device, source, patch, context, L, rank,
                                               fit_groups[context], args.train_paraphrases, method,
                                               args.epochs, args.lr, args.lambda_spec, args.train_batch,
                                               init, tuple(args.keep_groups), config_seed, log)
                        if U is not None:
                            np.save(out_dir / f"subspace_{context}_L{L_ref}_r{rank}_{method}_{fold}.npy", U)
                        # One direction, several amounts. The fixed amount is the mean
                        # over the items U was fitted on, so no evaluation item informs
                        # it.
                        fixed_coeff = None
                        if "fixed" in todo and U is not None:
                            import torch
                            pool = coefficients_for(
                                source, device, context, L,
                                torch.as_tensor(U, device=device),
                                fit_groups[context].get("gated", []),
                                args.train_paraphrases)
                            if pool:
                                fixed_coeff = torch.stack(list(pool.values())).mean(0)
                        rows = []
                        for coefficient in todo:
                            if coefficient == "fixed" and fixed_coeff is None \
                                    and method != "fullpatch":
                                log(f"  {key}: no training items, skipping the fixed arm")
                                continue
                            produced = evaluate(model, tokenizer, device, source, patch, context, L,
                                                L_ref, rank, method, U, eval_groups[context],
                                                args.eval_paraphrases, args.eval_split, full_cache,
                                                reverse_cache, fold, coefficient, fixed_coeff,
                                                args.seed)
                            if not produced:
                                # The shuffled arm cannot run on a group of one, and an
                                # arm recorded as complete with no rows would later look
                                # like a missing result with no explanation.
                                log(f"  {key} [{coefficient}]: no rows produced, "
                                    f"most likely too few items to reassign amounts within")
                            rows += produced
                            completed.add(f"{key}:{coefficient}")
                        if not rows:
                            continue
                        append_records(items_path, rows)
                        write_json(done_path, {"completed": sorted(completed),
                                               "settings": settings,
                                               "layer_map": {str(k): v for k, v in layer_map.items()}})
                        g = pd.DataFrame(rows)
                        # Reported per arm. Averaging the arms together would blend an
                        # intervention with the controls that exist to bound it.
                        for arm, part in g.groupby("coefficient"):
                            gated = part[part.group == "gated"]
                            if gated.empty:
                                continue
                            c = float(closure_of(gated.delta_mediated.mean(),
                                                 gated.delta_full.mean(),
                                                 args.min_full_delta))
                            shown = "n/a" if np.isnan(c) else f"{c:.2f}"
                            print(f"  {key} [{arm}]: gated closure {shown} "
                                  f"(whole patch {gated.delta_full.mean():+.2f}), "
                                  f"acc {gated.baseline_correct.mean():.2f}"
                                  f"->{gated.mediated_correct.mean():.2f}")
            finally:
                patch.remove()

    if not items_path.exists():
        # Reachable: a model whose causal gate never fires in the native script, which
        # is Llama in Marathi, skips every context and produces no rows at all.
        print(f"no results were produced: every prompt language was skipped for want of "
              f"gated items. Nothing written to {out_dir}.")
        return
    items = pd.read_csv(items_path)
    summarise(items, args.min_full_delta).to_csv(out_dir / "mediator_summary.csv", index=False)
    print("Saved:", out_dir / "mediator_summary.csv")


if __name__ == "__main__":
    main()

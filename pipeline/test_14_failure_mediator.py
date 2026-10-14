"""CPU tests for stage 14. No GPU, no model weights, no network.

The parts worth testing without a GPU are the ones that are wrong silently: the
subspace algebra in the hook, whether a gradient actually reaches U, whether the
derived subspaces are orthonormal and of the requested rank, and whether the item
groups are disjoint in the ways the specificity argument needs.

    python test_14_failure_mediator.py                      # algebra and groups
    python test_14_failure_mediator.py --run-dir ../../results_mr/qwen_mr
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("stage14", HERE / "14_failure_mediator.py")
stage14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage14)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


class TinyBlock:
    """Stands in for one transformer block: takes and returns a hidden state tuple."""

    def __init__(self):
        import torch.nn as nn
        self.mod = nn.Identity()


class TinyModel:
    """get_layers() finds `model.model.layers`, so that is what this provides."""

    def __init__(self, n_layers=3):
        import torch.nn as nn
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])


def test_hook_algebra():
    import torch
    torch.manual_seed(0)
    d, r, T, pos = 16, 3, 5, 2
    model = TinyModel()
    patch = stage14.SubspacePatch(model, 1)
    block = model.model.layers[1]

    h = torch.randn(1, T, d)
    donor = torch.randn(d)
    U = torch.linalg.qr(torch.randn(d, r))[0][:, :r]

    patch.off()
    out = block(h)
    check("hook off is the identity", torch.allclose(out, h))

    patch.subspace(pos, donor, U)
    out = block(h)
    other = [i for i in range(T) if i != pos]
    check("subspace patch touches only the target position",
          torch.allclose(out[:, other, :], h[:, other, :]))
    expected = h[:, pos, :] + (donor - h[:, pos, :]) @ U @ U.T
    check("subspace patch equals h + UU^T(donor - h)",
          torch.allclose(out[:, pos, :], expected, atol=1e-5))
    moved = out[:, pos, :] - h[:, pos, :]
    resid = moved - moved @ U @ U.T
    check("the change lies inside the subspace", float(resid.norm()) < 1e-5,
          f"out-of-subspace norm {float(resid.norm()):.2e}")

    patch.whole(pos, donor)
    out = block(h)
    check("whole patch replaces the state", torch.allclose(out[:, pos, :], donor.expand(1, d)))
    check("whole patch touches only the target position",
          torch.allclose(out[:, other, :], h[:, other, :]))

    # A full-rank subspace must reproduce the whole patch, which is the sanity check
    # that ties the two modes together.
    patch.subspace(pos, donor, torch.eye(d))
    out = block(h)
    check("rank-d subspace equals the whole patch",
          torch.allclose(out[:, pos, :], donor.expand(1, d), atol=1e-5))

    # Mode leakage: the bug the off()/subspace()/whole() helpers exist to prevent.
    patch.whole(pos, donor)
    patch.subspace(pos, donor, U)
    out = block(h)
    check("subspace after whole is not still a whole patch",
          torch.allclose(out[:, pos, :], expected, atol=1e-5))
    patch.remove()


def test_gradient_reaches_u():
    import torch
    torch.manual_seed(0)
    d, r, T, pos = 12, 2, 4, 1
    model = TinyModel()
    patch = stage14.SubspacePatch(model, 0)
    block = model.model.layers[0]
    W = torch.nn.Parameter(torch.randn(d, r))
    readout = torch.randn(d)

    h = torch.randn(1, T, d)
    donor = torch.randn(d)
    U = torch.linalg.qr(W)[0]
    patch.subspace(pos, donor, U)
    margin = (block(h)[:, pos, :] @ readout).squeeze()
    margin.backward()
    check("a gradient reaches the subspace parameter",
          W.grad is not None and float(W.grad.abs().sum()) > 0,
          f"|grad| {float(W.grad.abs().sum()):.3e}")

    # The projection is what carries it: with the donor equal to the state there is
    # nothing to move, so the gradient must vanish.
    W2 = torch.nn.Parameter(torch.randn(d, r))
    patch.subspace(pos, h[0, pos, :].clone(), torch.linalg.qr(W2)[0])
    (block(h)[:, pos, :] @ readout).squeeze().backward()
    check("no gradient when the donor equals the state",
          W2.grad is None or float(W2.grad.abs().sum()) < 1e-6)
    patch.remove()


def test_derived_subspaces():
    rng = np.random.default_rng(0)
    d, n = 32, 40
    diffs = rng.standard_normal((n, d)).astype(np.float32) + 2.0
    probe = rng.standard_normal(d).astype(np.float32)
    jac = rng.standard_normal(d).astype(np.float32)
    for method in stage14.DERIVED:
        for rank in (1, 2, 4, 8):
            U = stage14.derived_subspace(method, rank, diffs, rng, probe, jac)
            gram = U.T @ U
            ok = U.shape == (d, rank) and np.allclose(gram, np.eye(rank), atol=1e-4)
            check(f"{method} r={rank} is d-by-r and orthonormal", ok, str(U.shape))
    # The rank-1 methods must actually point along the thing they are named for.
    u = stage14.derived_subspace("meandiff", 1, diffs, rng, probe, jac)[:, 0]
    m = diffs.mean(0) / np.linalg.norm(diffs.mean(0))
    check("meandiff rank 1 is the mean difference", abs(abs(float(u @ m)) - 1) < 1e-4)
    u = stage14.derived_subspace("probe", 1, diffs, rng, probe, jac)[:, 0]
    check("probe rank 1 is the probe direction",
          abs(abs(float(u @ (probe / np.linalg.norm(probe)))) - 1) < 1e-4)
    u4 = stage14.derived_subspace("probe", 4, diffs, rng, probe, jac)
    check("probe rank 4 keeps the probe direction as its first column",
          abs(abs(float(u4[:, 0] @ (probe / np.linalg.norm(probe)))) - 1) < 1e-4)


def test_toward_correct():
    check("toward_correct passes positives through", stage14.toward_correct(2.0, 1) == 2.0)
    check("toward_correct flips negatives", stage14.toward_correct(2.0, 0) == -2.0)


def test_summarise():
    rows = []
    for group, dm, df in (("gated", 1.0, 2.0), ("robust", 0.0, 2.0)):
        for i in range(3):
            rows.append({"context": "mr", "split": "validation", "layer": 14, "rank": 2,
                         "method": "mediator", "group": group, "fact_id": f"q{i}", "truth": 1,
                         "paraphrase_id": 2, "baseline_margin": -1.0, "fullpatch_margin": 1.0,
                         "mediated_margin": 0.0, "delta_full": df, "delta_mediated": dm,
                         "abs_change": abs(dm), "baseline_correct": 0,
                         "mediated_correct": int(dm > 0), "reverse_drop": np.nan})
    s = stage14.summarise(pd.DataFrame(rows))
    gated = s[s.group == "gated"].iloc[0]
    check("closure is mediated over full", abs(float(gated.closure) - 0.5) < 1e-9,
          f"{float(gated.closure):.3f}")
    check("item counts survive the paraphrase rows", int(gated.n_items) == 3)


def test_groups(run_dir: Path):
    path = run_dir / "patching" / "causal_candidates.csv"
    if not path.exists():
        print(f"  SKIP  groups: {path} not present")
        return
    cand = pd.read_csv(path)
    for context in sorted(cand.context.unique()):
        for split in ("train", "validation", "test"):
            g = stage14.behavioural_groups(cand, context, split)
            gated = {f for f, _ in g["gated"]}
            robust = {f for f, _ in g["robust"]}
            neg_truth = {t for _, t in g["negatives"]}
            gate_truth = {t for _, t in g["gated"]}
            check(f"{context}/{split}: gated and robust are disjoint", not (gated & robust),
                  f"gated={len(gated)} robust={len(robust)} never={len(g['never'])}")
            if gated:
                check(f"{context}/{split}: negatives are the other polarity",
                      neg_truth == {1 - t for t in gate_truth})
                stable = set(cand.loc[(cand.context == context) & (cand.split == split)
                                      & (cand.stable == 1), "fact_id"].astype(str))
                check(f"{context}/{split}: gated equals the stable set", gated == stable,
                      f"{len(gated)} vs {len(stable)}")


def test_stable_seed():
    a = stage14.stable_seed(20260813, "mr:L14:r4:random")
    b = stage14.stable_seed(20260813, "mr:L14:r4:random")
    c = stage14.stable_seed(20260813, "mr:L15:r4:random")
    check("stable_seed is deterministic", a == b)
    check("stable_seed separates configurations", a != c)
    # The bug this replaced: a generator advanced in grid order, so a resumed run drew
    # a different random subspace from an uninterrupted one.
    diffs = np.random.default_rng(1).standard_normal((30, 64)).astype(np.float32)

    def draw(key):
        return stage14.derived_subspace(
            "random", 4, diffs, np.random.default_rng(stage14.stable_seed(7, key)), None, None)

    check("the random baseline survives a resume", np.allclose(draw("c"), draw("c")))
    check("the random baseline differs between configurations",
          not np.allclose(draw("b"), draw("c")))


def test_closure_guard():
    import math
    check("closure is the plain ratio when the whole patch repairs",
          abs(float(stage14.closure_of(1.0, 2.0)) - 0.5) < 1e-9)
    check("closure is undefined when the whole patch does nothing",
          math.isnan(float(stage14.closure_of(0.1, 1e-9))))
    check("closure is undefined when the whole patch hurts",
          math.isnan(float(stage14.closure_of(-0.5, -2.0))))
    rows = [{"context": "mr", "split": "validation", "layer": 14, "rank": 4,
             "method": "pca", "group": "never", "fact_id": f"q{i}", "truth": 1,
             "paraphrase_id": 2, "baseline_margin": -8.0, "fullpatch_margin": -8.0,
             "mediated_margin": -7.9, "delta_full": 1e-9, "delta_mediated": 0.1,
             "abs_change": 0.1, "baseline_correct": 0, "mediated_correct": 0,
             "reverse_drop": np.nan} for i in range(5)]
    s = stage14.summarise(pd.DataFrame(rows))
    check("a group the whole patch never moves reports no closure",
          bool(s.closure.isna().all()) and int(s.closure_defined.iloc[0]) == 0)


def test_source_level_guards():
    """Two defects that need a GPU to exercise but can be pinned in the source.

    The scaled-layer bug: evaluate() was given the reference layer where training was
    given the scaled one, so on a --scale-layers model the donor came from layer 12 and
    was injected at layer 14. The call must pass the patched layer first, the label
    second.
    """
    src = (HERE / "14_failure_mediator.py").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", src)
    check("evaluate is called with the patched layer, then the label",
          "evaluate(model, tokenizer, device, source, patch, context, L, L_ref, rank, "
          "method, U" in flat)
    check("evaluate takes both layers", "layer: int, layer_label: int," in flat)
    check("the row is filed under the label", '"layer": layer_label,' in flat)
    check("fullpatch records no rank",
          '"rank": 0 if method == "fullpatch" else rank,' in flat)
    check("the settings guard refuses a mismatched resume",
          "these results were produced under different settings" in flat)
    check("DAS does not compute baselines it never uses",
          'if method == "mediator" and g != "gated":' in flat)
    check("the whole-state patch is cached across methods", "full_cache[cache_key]" in flat)
    check("training takes an explicit seed", "keep_groups: tuple, seed: int, log" in flat)


def test_settings_fingerprint():
    class A:
        pass
    a = A()
    for k in stage14.SETTINGS_KEYS:
        setattr(a, k, [1, 2] if "paraphrases" in k or k == "keep_groups" else 3)
    d = stage14.settings_of(a)
    check("every comparable setting is fingerprinted",
          set(d) == set(stage14.SETTINGS_KEYS))
    check("layers, ranks and methods are not fingerprinted",
          not {"layers", "ranks", "methods"} & set(stage14.SETTINGS_KEYS))
    import json
    check("the fingerprint survives a JSON round trip", json.loads(json.dumps(d)) == d)


def test_rank_is_estimable():
    """A short subspace compared against a full-rank random baseline flatters the
    baseline, so a rank the differences cannot support must be refused, not truncated."""
    rng = np.random.default_rng(0)
    check("usable rank is one less than the number of differences",
          stage14.usable_rank(9) == 8 and stage14.usable_rank(5) == 4)
    check("usable rank never drops below one", stage14.usable_rank(1) == 1)
    for n, rank, estimable in ((40, 8, True), (9, 8, True), (5, 8, False), (3, 4, False)):
        diffs = rng.standard_normal((n, 128)).astype(np.float32)
        probe = rng.standard_normal(128).astype(np.float32)
        results = []
        for method in stage14.DERIVED:
            try:
                U = stage14.derived_subspace(method, rank, diffs, rng, probe, probe)
                results.append(U.shape[1] == rank)
            except ValueError:
                results.append(None)
        short = [m for m, r in zip(stage14.DERIVED, results) if r is False]
        if estimable:
            check(f"n={n} rank={rank}: every method delivers the full rank",
                  not short and all(r for r in results), str(short))
        else:
            check(f"n={n} rank={rank}: a short subspace raises instead of being returned",
                  not short, f"silently short: {short}")
    # The random baseline must never be the only one at full rank.
    diffs = rng.standard_normal((5, 128)).astype(np.float32)
    r_ok = stage14.derived_subspace("random", 8, diffs, rng, None, None).shape[1] == 8
    try:
        stage14.derived_subspace("pca", 8, diffs, rng, None, None)
        pca_ok = True
    except ValueError:
        pca_ok = False
    check("random alone at full rank is now impossible to reach silently",
          r_ok and not pca_ok)


def test_no_rows_is_not_a_crash():
    src = (HERE / "14_failure_mediator.py").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", src)
    check("a run that produces no rows reports instead of raising",
          "if not items_path.exists():" in flat
          and "every prompt language was skipped for want of" in flat)
    check("a completed layer is skipped before its directions are computed",
          flat.index("pending = [(r, m)") < flat.index("probe_dir = (probe_direction")
          and "already complete" in flat)
    check("a direction is computed only when a method that needs it is pending",
          '"probe" in pending_methods' in flat and '"jacobian" in pending_methods' in flat)
    check("the reverse-corruption baseline is cached", "reverse_cache[cache_key] = lbase" in flat)
    check("a done.json without settings is refused",
          "records completed work but no settings" in flat)


def test_orphan_rows_are_dropped(tmp: Path):
    """Rows are appended before the key is recorded, so an interrupted run leaves rows
    the resume cannot see. Running that configuration again would append a second copy
    and every average over it would count it twice."""
    path = tmp / "mediator_items.csv"

    def rows(context, layer, rank, method, fold="t12e0", n=2):
        return [{"context": context, "split": "validation", "fold": fold, "layer": layer,
                 "rank": rank, "method": method, "group": "gated", "fact_id": f"q{i}",
                 "truth": 1, "paraphrase_id": 0, "baseline_margin": -1.0,
                 "fullpatch_margin": 1.0, "mediated_margin": 0.0, "delta_full": 2.0,
                 "delta_mediated": 1.0, "abs_change": 1.0, "baseline_correct": 0,
                 "mediated_correct": 1, "reverse_drop": np.nan} for i in range(n)]

    frame = pd.DataFrame(rows("mr", 14, 4, "pca") + rows("mr", 14, 4, "mediator")
                         + rows("mr", 14, 0, "fullpatch"))
    frame.to_csv(path, index=False)
    # The mediator configuration was interrupted after its rows landed.
    completed = {"mr:L14:r4:pca:t12e0", "mr:L14:r4:fullpatch:t12e0"}
    stage14.drop_orphan_rows(path, completed)
    left = pd.read_csv(path)
    check("the interrupted configuration's rows are gone",
          set(left.method) == {"pca", "fullpatch"}, str(sorted(set(left.method))))
    check("complete configurations are untouched", len(left) == 4, str(len(left)))
    check("fullpatch survives despite being stored at rank 0",
          "fullpatch" in set(left.method))
    stage14.drop_orphan_rows(path, completed)
    check("dropping twice is a no-op", len(pd.read_csv(path)) == 4)
    stage14.drop_orphan_rows(tmp / "absent.csv", completed)
    check("a missing items file is not an error", True)

    # The resume key gained an arm suffix. If row_key and the recorded keys disagree,
    # nothing matches and a resume silently deletes every row it has.
    armed = tmp / "armed.csv"
    with_arm = []
    for arm in ("matched", "shuffled"):
        for r in rows("mr", 16, 1, "jacobian") + rows("mr", 16, 0, "fullpatch"):
            with_arm.append({**r, "coefficient": arm})
    pd.DataFrame(with_arm).to_csv(armed, index=False)
    done = {"mr:L16:r1:jacobian:t12e0:matched", "mr:L16:r1:fullpatch:t12e0:matched"}
    stage14.drop_orphan_rows(armed, done)
    left = pd.read_csv(armed)
    check("arms are matched against the recorded keys",
          len(left) == 4 and set(left.coefficient) == {"matched"},
          f"{len(left)} rows, arms {sorted(set(left.coefficient))}")
    check("an unrecorded arm is dropped, the recorded one is kept",
          set(left.method) == {"jacobian", "fullpatch"})

    # A directory written before the arms existed must not be wiped by the new key.
    legacy = tmp / "legacy.csv"
    pd.DataFrame(rows("mr", 16, 1, "jacobian")).to_csv(legacy, index=False)
    stage14.drop_orphan_rows(legacy, {"mr:L16:r1:jacobian:t12e0"})
    check("a pre-arms resume set still matches its own rows",
          len(pd.read_csv(legacy)) == 2, f"{len(pd.read_csv(legacy))} rows")

    # Three folds are meant to accumulate in one file. The same layer, rank and method
    # fitted on different templates is a different result, so one must not evict another.
    path2 = tmp / "folds.csv"
    pd.DataFrame(rows("mr", 17, 1, "jacobian", "t12e0")
                 + rows("mr", 17, 1, "jacobian", "t02e1")
                 + rows("mr", 17, 1, "jacobian", "t01e2")).to_csv(path2, index=False)
    stage14.drop_orphan_rows(path2, {"mr:L17:r1:jacobian:t12e0",
                                     "mr:L17:r1:jacobian:t02e1",
                                     "mr:L17:r1:jacobian:t01e2"})
    kept = pd.read_csv(path2)
    check("three folds of one configuration coexist",
          len(kept) == 6 and set(kept.fold) == {"t12e0", "t02e1", "t01e2"},
          str(sorted(set(kept.fold))))
    stage14.drop_orphan_rows(path2, {"mr:L17:r1:jacobian:t12e0"})
    check("an unrecorded fold is dropped without touching the others",
          set(pd.read_csv(path2).fold) == {"t12e0"})


def test_fold_tag():
    check("fold tag names the split", stage14.fold_tag([1, 2], [0]) == "t12e0")
    check("fold tags are distinct",
          len({stage14.fold_tag(t, e) for t, e in
               (([1, 2], [0]), ([0, 2], [1]), ([0, 1], [2]))}) == 3)
    check("paraphrase settings left the fingerprint",
          "train_paraphrases" not in stage14.SETTINGS_KEYS
          and "eval_paraphrases" not in stage14.SETTINGS_KEYS)
    check("the settings that do change comparability are still fingerprinted",
          {"epochs", "lr", "max_train_facts", "eval_split"} <= set(stage14.SETTINGS_KEYS))


def test_groups_are_disjoint_by_construction():
    """The margin gate can admit an item that also fails in both scripts, which would
    put it in gated and never at once. It does not happen in the current runs, which is
    why relying on the data rather than on the construction would be unsafe."""
    cand = pd.DataFrame({
        "fact_id": ["a", "b", "c"],
        "context": ["mr"] * 3, "split": ["train"] * 3,
        "polarity": ["negative"] * 3, "gate": ["margin"] * 3,
        # every item fails in both scripts; the first is nonetheless gated
        "stable": [1, 0, 0],
        "pos_DEVDEV": [0.9, 0.9, 0.9], "pos_LATLAT": [0.9, 0.9, 0.95],
        "neg_DEVDEV": [0.1, 0.1, 0.9], "neg_LATLAT": [0.1, 0.15, 0.9],
    })
    g = stage14.behavioural_groups(cand, "mr", "train")
    S = {k: {f for f, _ in v} for k, v in g.items()}
    check("a gated item cannot also be never-correct", not (S["gated"] & S["never"]),
          str(S["gated"] & S["never"]))
    check("a gated item cannot also be robust", not (S["gated"] & S["robust"]))
    check("the remaining never-correct item is still kept", S["never"] == {"b"}, str(S["never"]))
    check("the robust item is still kept", S["robust"] == {"c"}, str(S["robust"]))


def test_derangement():
    """No item may keep its own coefficient, or the control is diluted by exactly the
    effect it exists to remove."""
    rng = np.random.default_rng(0)
    worst = 0
    for n in (2, 3, 5, 12, 40):
        for _ in range(200):
            order = stage14.derangement(n, rng)
            worst = max(worst, int((order == np.arange(n)).sum()))
            if sorted(order) != list(range(n)):
                check("a derangement is a permutation", False, f"n={n}")
                return
    check("a derangement is a permutation", True, "n in 2..40, 1000 draws")
    check("no item keeps its own coefficient", worst == 0, f"max fixed points {worst}")
    check("n<2 degrades without raising", len(stage14.derangement(1, rng)) == 1)


def test_directional_matches_subspace():
    """The matched arm must be the same intervention as the original subspace patch.

    h + UU^T(donor - h) is h + U c with c = U^T(donor - h). If these ever disagree, the
    shuffled and fixed arms are not comparable to the result they are meant to explain.
    """
    rng = np.random.default_rng(7)
    d, r = 64, 3
    for _ in range(50):
        U, _ = np.linalg.qr(rng.normal(size=(d, r)))
        h = rng.normal(size=d)
        donor = rng.normal(size=d)
        by_projection = h + (donor - h) @ U @ U.T
        by_coefficient = h + U @ (U.T @ (donor - h))
        if not np.allclose(by_projection, by_coefficient, atol=1e-10):
            check("matched coefficients reproduce the subspace patch", False)
            return
    check("matched coefficients reproduce the subspace patch", True, "50 random draws")

    # And a shuffled coefficient must actually change the intervention, otherwise the
    # control would trivially agree with the thing it controls.
    U, _ = np.linalg.qr(rng.normal(size=(d, r)))
    h, a, b = (rng.normal(size=d) for _ in range(3))
    mine = h + U @ (U.T @ (a - h))
    theirs = h + U @ (U.T @ (b - h))
    check("a swapped coefficient changes the result",
          not np.allclose(mine, theirs, atol=1e-6))


def test_arm_bookkeeping():
    """Things that would silently weaken the controls rather than fail loudly."""
    import inspect
    ev = inspect.getsource(stage14.evaluate)
    main = inspect.getsource(stage14.main)

    # A group of one cannot be deranged. Handing the item its own coefficient would
    # turn the shuffled control back into the matched arm without any error.
    check("a group of one yields no shuffled rows",
          "swap = {}" in ev and "swap = {k: pool[k] for k in keys}" not in ev)

    # The reverse-corruption measurement uses the donor's own coordinates, so it does
    # not vary by arm; running it per arm would triple its cost and imply three
    # measurements where there is one.
    check("reverse corruption is measured on the matched arm only",
          'coefficient == "matched"' in ev)

    # Progress output must not average an intervention with its own controls.
    check("progress is reported per arm", 'groupby("coefficient")' in main)

    # The direction must not depend on the arm, or the arms are not comparable.
    seed_line = [l for l in main.splitlines() if "config_seed = stable_seed" in l]
    check("the direction's seed does not depend on the arm",
          bool(seed_line) and "coefficient" not in seed_line[0], seed_line[0].strip()
          if seed_line else "not found")

    # An arm marked complete with no rows would look like a missing result later.
    check("an empty arm is reported", "no rows produced" in main)


def test_coefficient_modes_are_rejected_when_unknown():
    """A typo in --coefficients must not silently fall through to the matched arm."""
    import inspect
    src = inspect.getsource(stage14.evaluate)
    check("evaluate rejects an unknown coefficient mode",
          "unknown coefficient mode" in src)
    check("the fixed arm refuses to run without a fitted coefficient",
          "needs a coefficient fitted on the training items" in src)
    sig = inspect.signature(stage14.evaluate)
    check("evaluate defaults to the matched arm",
          sig.parameters["coefficient"].default == "matched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    print("hook algebra")
    test_hook_algebra()
    print("gradients")
    test_gradient_reaches_u()
    print("derived subspaces")
    test_derived_subspaces()
    print("scalars and summary")
    test_toward_correct()
    test_summarise()
    print("review fixes")
    test_stable_seed()
    test_closure_guard()
    test_settings_fingerprint()
    test_source_level_guards()
    test_rank_is_estimable()
    test_no_rows_is_not_a_crash()
    test_groups_are_disjoint_by_construction()
    test_fold_tag()
    print("coefficient arms")
    test_derangement()
    test_directional_matches_subspace()
    test_arm_bookkeeping()
    test_coefficient_modes_are_rejected_when_unknown()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_orphan_rows_are_dropped(Path(d))
    if args.run_dir:
        print(f"item groups from {args.run_dir}")
        test_groups(args.run_dir)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  failed: {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

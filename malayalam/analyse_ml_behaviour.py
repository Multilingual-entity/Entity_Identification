"""The Malayalam behavioural table, with the paired Marathi comparator.

Reads tables/behavior.csv from a Malayalam run and, optionally, from the matched 380-item
Marathi run, and reports what the main-body addition needs: accuracy by truth polarity,
balanced accuracy, entity-paired script gaps with item-clustered intervals, entity-averaged
AUROC of the margin, and d' where the false-alarm rate is off the floor.

Three populations are reported for every arm, in this order, and the paper should quote the
third:

  all          all 380 scored entities, descriptive only
  eligible     the entities the Malayalam audit keeps; DIFFERENT prompts additionally
               require the distractor to be kept, so the two polarities have different
               denominators and both are printed
  test         the eligible entities in the frozen 78-entity test split

Scoring happens before any subsetting, so no model output decides which rows survive.
Nothing here fits a threshold: the gaps are differences of rates and of AUROC, and a
threshold-selected number belongs to the validation split, not here.

    python analyse_ml_behaviour.py --results <results root> --ml qwen_ml --mr qwen_mr380 \
        --eligibility <run root>/ml_eligibility.csv

pipeline_common.py and analysis/margin_auc.py are found in either the nested local
checkout or the flat layout on the run machine; set PAPERB_PIPELINE if neither matches.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _locate(filename: str, required: bool = True) -> Path | None:
    """Find a pipeline or analysis file in either layout.

    The local checkout nests these under cross-script-entity-matching/; the run machine
    has pipeline_common.py and analysis/ beside each other at the top level. Importing
    from a guessed path works in one and fails in the other, which is a crash after the
    GPU time has already been spent.
    """
    nested = HERE.parent / "cross-script-entity-matching"
    for candidate in (nested / "pipeline", nested / "analysis",
                      HERE, HERE.parent, HERE.parent / "analysis",
                      HERE.parent / "pipeline"):
        if (candidate / filename).exists():
            return candidate.resolve()
    env = os.environ.get("PAPERB_PIPELINE")
    if env and (Path(env) / filename).exists():
        return Path(env).resolve()
    if required:
        raise SystemExit(f"cannot find {filename}. Set PAPERB_PIPELINE to the directory "
                         f"holding it.")
    return None


sys.path.insert(0, str(_locate("pipeline_common.py")))

# margin_auc.py is wanted only to assert that the fast AUC below reproduces the paper's
# implementation. That check is worth running where the file is present, and is not a
# reason to refuse to analyse a run where it is not.
_reference = _locate("margin_auc.py", required=False)
if _reference is not None:
    sys.path.insert(0, str(_reference))

from pipeline_common import dprime  # noqa: E402

if _reference is not None:
    from margin_auc import auc as reference_auc  # noqa: E402  the paper's own
else:
    reference_auc = None

SEED = 20260813
DRAWS = 2000
CONDITIONS = ("DEVDEV", "DEVLAT", "LATDEV", "LATLAT")
# What the paper prints. The pipeline's column names are frozen legacy labels.
LABEL = {"DEVDEV": "Mlym--Mlym", "DEVLAT": "Mlym--Lat",
         "LATDEV": "Lat--Mlym", "LATLAT": "Lat--Lat"}
MR_LABEL = {"DEVDEV": "Dev--Dev", "DEVLAT": "Dev--Lat",
            "LATDEV": "Lat--Dev", "LATLAT": "Lat--Lat"}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-averaged Mann-Whitney AUC, same definition as the paper's margin_auc.auc.

    Rewritten in numpy because the original builds a DataFrame per call, and the paired
    bootstraps here call it tens of thousands of times. _check_auc below asserts the two
    agree, including on ties, so this is a speed change and not a method change.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    uniq, inverse, counts = np.unique(values[order], return_inverse=True,
                                      return_counts=True)
    positions = np.arange(1, len(values) + 1, dtype=float)
    average = np.bincount(inverse, weights=positions) / counts
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = average[inverse]
    rank_sum = ranks[:len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _check_auc() -> None:
    if reference_auc is None:
        print("note: analysis/margin_auc.py not found, so the AUC was not cross-checked "
              "against the paper's implementation. The definition is the same "
              "tie-averaged Mann-Whitney statistic.")
        return
    rng = np.random.default_rng(0)
    for _ in range(25):
        # integers, so ties are common and the tie handling is actually exercised
        p = rng.integers(0, 6, rng.integers(2, 40)).astype(float)
        n = rng.integers(0, 6, rng.integers(2, 40)).astype(float)
        if abs(auc(p, n) - reference_auc(p, n)) > 1e-12:
            raise SystemExit("the fast AUC disagrees with the paper's implementation")


_check_auc()


def per_entity(frame: pd.DataFrame, column: str = "correct") -> pd.Series:
    """One number per entity, averaged over that entity's prompts first.

    Six prompts of one entity are not six observations, so every mean and every resample
    below is over entities.
    """
    return frame.groupby("fact_id")[column].mean()


def boot_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.array([values[rng.integers(0, n, n)].mean() for _ in range(DRAWS)])
    return float(values.mean()), *np.quantile(draws, [0.025, 0.975])


def boot_paired_gap(a: pd.Series, b: pd.Series, seed: int):
    """Mean of a minus b over the entities both share, resampling entities."""
    shared = a.index.intersection(b.index)
    diff = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=float)
    point, lo, hi = boot_mean(diff, seed)
    return point, lo, hi, len(shared)


def arm_auc(frame: pd.DataFrame, seed: int):
    """Entity-averaged AUROC of the yes-minus-no margin, with an item bootstrap.

    The margin is the raw yes-logit minus no-logit, so a same-person item should sit
    above a different-person one and the AUROC should exceed 0.5. Returns the two
    margin series still indexed by fact_id, because the script gap below has to pair
    conditions on entities rather than on position.
    """
    pos = per_entity(frame[frame.truth == 1], "yes_minus_no_margin")
    neg = per_entity(frame[frame.truth == 0], "yes_minus_no_margin")
    shared = pos.index.intersection(neg.index)
    pos, neg = pos.loc[shared], neg.loc[shared]
    point = auc(pos.to_numpy(), neg.to_numpy())
    rng = np.random.default_rng(seed)
    k = len(shared)
    p, n = pos.to_numpy(), neg.to_numpy()
    draws = [auc(p[i], n[i]) for i in (rng.integers(0, k, k) for _ in range(DRAWS))]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi), k, pos, neg


def auc_gap(latlat: dict, devdev: dict, seed: int):
    """AUROC in Lat--Lat minus AUROC in the native condition, paired on entities.

    Indexing two conditions' arrays with one set of positions assumes they hold the same
    entities in the same order. They normally do, and silently do not when a condition
    loses an entity, so the entities are intersected first.
    """
    shared = latlat["pos"].index.intersection(devdev["pos"].index)
    p1 = latlat["pos"].loc[shared].to_numpy()
    n1 = latlat["neg"].loc[shared].to_numpy()
    p0 = devdev["pos"].loc[shared].to_numpy()
    n0 = devdev["neg"].loc[shared].to_numpy()
    k = len(shared)
    rng = np.random.default_rng(seed)
    draws = [auc(p1[i], n1[i]) - auc(p0[i], n0[i])
             for i in (rng.integers(0, k, k) for _ in range(DRAWS))]
    gap = auc(p1, n1) - auc(p0, n0)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return gap, float(lo), float(hi), k


def arm_dprime(frame: pd.DataFrame):
    hits = frame[frame.truth == 1]["correct"]
    fas = frame[frame.truth == 0]["correct"]
    n_pos, n_neg = len(hits), len(fas)
    if not n_pos or not n_neg:
        return float("nan"), float("nan"), float("nan")
    hit_rate = hits.mean()
    fa_rate = 1.0 - fas.mean()          # a false alarm is saying yes to a different person
    value, criterion = dprime(hit_rate, fa_rate, n_pos, n_neg)
    return value, criterion, fa_rate


def populations(behaviour: pd.DataFrame, eligibility: pd.DataFrame | None):
    """(name, frame) pairs. Subsetting is applied after scoring, never before."""
    yield "all", behaviour
    if eligibility is None or eligibility.empty:
        return
    same_ok = set(eligibility.loc[eligibility.eligible_same == 1, "fact_id"])
    both_ok = set(eligibility.loc[eligibility.eligible_different == 1, "fact_id"])
    # A kept entity whose distractor was removed keeps its SAME prompts and loses its
    # DIFFERENT ones, so the mask is applied per polarity rather than per entity.
    keep = ((behaviour.truth == 1) & behaviour.fact_id.isin(same_ok)) | \
           ((behaviour.truth == 0) & behaviour.fact_id.isin(both_ok))
    eligible = behaviour.loc[keep]
    yield "eligible", eligible
    yield "test", eligible.loc[eligible.split == "test"]


def report(behaviour: pd.DataFrame, label: dict, contexts, eligibility, tag: str):
    rows = []
    for population, frame in populations(behaviour, eligibility):
        for context in contexts:
            arm = frame[frame.context == context]
            if arm.empty:
                continue
            print(f"\n  {tag} | {population} | prompts in {context}")
            print(f"    {'condition':12}{'SAME':>8}{'DIFF':>8}{'bal':>8}"
                  f"{'AUROC':>8}{'95% CI':>18}{'d-prime':>9}{'items':>7}")
            store = {}
            for condition in CONDITIONS:
                sub = arm[arm.condition == condition]
                if sub.empty:
                    continue
                same = per_entity(sub[sub.truth == 1])
                diff = per_entity(sub[sub.truth == 0])
                balanced = 0.5 * (same.mean() + diff.mean())
                point, lo, hi, k, p, n = arm_auc(sub, SEED)
                d, criterion, fa = arm_dprime(sub)
                store[condition] = dict(same=same, diff=diff, pos=p, neg=n)
                print(f"    {label[condition]:12}{same.mean():>8.3f}{diff.mean():>8.3f}"
                      f"{balanced:>8.3f}{point:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
                      f"{d:>9.2f}{k:>7}")
                # Every row carries the same estimate/ci_low/ci_high triple, whatever the
                # metric is, so a gap can never be read out of a column named for an
                # accuracy. The descriptive columns are filled only where they apply.
                rows.append(dict(run=tag, population=population, context=context,
                                 condition=label[condition], metric="auroc",
                                 estimate=point, ci_low=lo, ci_high=hi,
                                 same_accuracy=same.mean(), diff_accuracy=diff.mean(),
                                 balanced_accuracy=balanced, dprime=d,
                                 criterion=criterion, false_alarm_rate=fa,
                                 n_entities=k))
                if fa <= 0.0:
                    print(f"      note: no false alarms in {label[condition]}, so its "
                          f"d-prime rests on the log-linear correction alone")
            if "LATLAT" not in store or "DEVDEV" not in store:
                continue

            # The script gap, paired by entity. Accuracy and AUROC answer different
            # questions: a gap in accuracy can be a criterion shift, a gap in AUROC
            # cannot be.
            native = label["DEVDEV"]
            for what, key in (("SAME accuracy", "same"), ("DIFF accuracy", "diff")):
                point, lo, hi, k = boot_paired_gap(store["LATLAT"][key],
                                                   store["DEVDEV"][key], SEED + 2)
                verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
                print(f"    gap in {what:<16}{point:>+8.3f}  "
                      f"[{lo:+.3f}, {hi:+.3f}]  {k} entities, {verdict}")
                rows.append(dict(run=tag, population=population, context=context,
                                 condition=f"Lat--Lat minus {native}",
                                 metric=f"gap_{key}_accuracy", estimate=point,
                                 ci_low=lo, ci_high=hi, n_entities=k))

            gap, lo, hi, k = auc_gap(store["LATLAT"], store["DEVDEV"], SEED + 3)
            verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
            print(f"    gap in AUROC          {gap:>+8.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]  {k} entities, {verdict}")
            rows.append(dict(run=tag, population=population, context=context,
                             condition=f"Lat--Lat minus {native}", metric="gap_auroc",
                             estimate=gap, ci_low=lo, ci_high=hi, n_entities=k))
    return rows


def narrowest(behaviour: pd.DataFrame, eligibility) -> tuple[str, pd.DataFrame]:
    """The last population: eligible test entities, or all of them if no audit was given.

    Naming the population in the heading rather than assuming it means a run without an
    eligibility file does not print "eligible, test split" over numbers covering
    everything.
    """
    return list(populations(behaviour, eligibility))[-1]


def per_template(behaviour: pd.DataFrame, label: dict, contexts, eligibility, tag: str):
    """Template-specific results. A single weak paraphrase can carry a whole arm."""
    population, frame = narrowest(behaviour, eligibility)
    scope = ("eligible entities, test split" if population == "test"
             else "all scored entities")
    print(f"\n  {tag} | per paraphrase, {scope}")
    rows = []
    for context in contexts:
        for paraphrase in sorted(frame.paraphrase_id.unique()):
            sub = frame[(frame.context == context) & (frame.paraphrase_id == paraphrase)]
            if sub.empty:
                continue
            line = f"    {context} P{paraphrase + 1}  "
            for condition in CONDITIONS:
                cell = sub[sub.condition == condition]
                if cell.empty:
                    continue
                same = per_entity(cell[cell.truth == 1]).mean()
                line += f"{label[condition]} {same:.3f}   "
                rows.append(dict(run=tag, population=population, context=context,
                                 paraphrase=paraphrase + 1,
                                 condition=label[condition],
                                 metric="same_accuracy", estimate=same,
                                 n_entities=cell.fact_id.nunique()))
            print(line)
    return rows


def language_contrast(ml: pd.DataFrame, mr: pd.DataFrame, eligibility):
    """Malayalam against Marathi on the same 380 entities, paired entity by entity.

    The two runs share their English Lat--Lat arm by construction, so that arm is the
    place to check that the comparison is sound: it should be flat. A difference there is
    not a language effect, it is a run difference.
    """
    population, subset = narrowest(ml, eligibility)
    scope = ("eligible entities, test split" if population == "test"
             else "all scored entities")
    print(f"\n  Malayalam against the matched Marathi subset, paired by entity "
          f"({scope})")
    # The comparator is restricted to the same entities and the same split, so the two
    # sides of every difference are the same people.
    entities = set(subset.fact_id)
    comparator = mr.loc[mr.fact_id.isin(entities)]
    if population == "test":
        comparator = comparator.loc[comparator.split == "test"]
    rows = []
    for condition in CONDITIONS:
        a = per_entity(subset[(subset.condition == condition)
                              & (subset.truth == 1)
                              & (subset.context == "en")])
        b = per_entity(comparator[(comparator.condition == condition)
                                  & (comparator.truth == 1)
                                  & (comparator.context == "en")])
        if a.empty or b.empty:
            continue
        point, lo, hi, k = boot_paired_gap(a, b, SEED + 4)
        note = ""
        if condition == "LATLAT":
            note = ("   <- shared prompts; anything but 0.000 means the two runs differ, "
                    "not the languages")
        print(f"    {condition:8} English prompts, Malayalam minus Marathi "
              f"{point:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {k} entities{note}")
        rows.append(dict(run="ml_minus_mr", population=population, context="en",
                         condition=condition, metric="gap_same_accuracy",
                         estimate=point, ci_low=lo, ci_high=hi, n_entities=k))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--ml", default="qwen_ml")
    ap.add_argument("--mr", default=None, help="the matched Marathi run, if it has been scored")
    ap.add_argument("--eligibility", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    eligibility = None
    if args.eligibility is not None and args.eligibility.exists():
        eligibility = pd.read_csv(args.eligibility, dtype={"fact_id": str})
        print(f"eligibility from {args.eligibility.name}: "
              f"{int(eligibility.eligible_same.sum())} entities for SAME, "
              f"{int(eligibility.eligible_different.sum())} for both polarities")
        print("  this is a structural audit, not a native-speaker review")
    else:
        print("no eligibility file given; reporting all scored entities only")

    rows = []
    ml_path = args.results / args.ml / "tables" / "behavior.csv"
    if not ml_path.exists():
        raise SystemExit(f"no behaviour table at {ml_path}; run stage 01 first")
    ml = pd.read_csv(ml_path, dtype={"fact_id": str})
    contexts = [c for c in ("ml", "en") if c in set(ml.context)]
    rows += report(ml, LABEL, contexts, eligibility, args.ml)
    rows += per_template(ml, LABEL, contexts, eligibility, args.ml)

    if args.mr:
        mr_path = args.results / args.mr / "tables" / "behavior.csv"
        if mr_path.exists():
            mr = pd.read_csv(mr_path, dtype={"fact_id": str})
            mr_contexts = [c for c in ("mr", "en") if c in set(mr.context)]
            if eligibility is not None and not eligibility.empty:
                print(f"\n  the comparator below is masked by the same Malayalam audit, "
                      f"so both sides of every paired difference cover the same "
                      f"entities. It is not a judgment on the Marathi fields, which "
                      f"carry their own frozen review.")
            rows += report(mr, MR_LABEL, mr_contexts, eligibility, args.mr)
            rows += language_contrast(ml, mr, eligibility)
        else:
            print(f"\n  no comparator table at {mr_path}; "
                  f"the language contrast needs the Marathi run scored too")

    out = args.out or (args.results / args.ml / "tables" / "ml_behaviour_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

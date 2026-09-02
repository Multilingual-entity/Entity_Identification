"""Analyze entity trajectories, relation probes, exact failures, and predictors."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoTokenizer

from pipeline_common import (
    contexts_in_gate,
    contexts_in,
    MODEL_SPECS,
    add_common_args,
    ensure_run_dirs,
    load_prepared_data,
    seed_everything,
)


POSITIONS = ["E1", "E2", "Q", "READOUT"]
ROLE_PAIRS = {
    "a": [("dev", "DEVDEV", "LATDEV"), ("lat", "DEVLAT", "LATLAT")],
    "b": [("dev", "DEVDEV", "DEVLAT"), ("lat", "LATDEV", "LATLAT")],
}


def centered_ranks(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = query.astype(np.float32)
    candidates = candidates.astype(np.float32)
    query -= query.mean(axis=0, keepdims=True)
    candidates -= candidates.mean(axis=0, keepdims=True)
    query = normalize(query)
    candidates = normalize(candidates)
    similarity = query @ candidates.T
    order = np.argsort(-similarity, axis=1)
    ranks = np.empty(len(query), dtype=np.int32)
    for i in range(len(query)):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    return ranks, np.diag(similarity)


def isolated_retrieval(states: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, rank_rows = [], []
    for role in ("a", "b"):
        dev = meta.loc[(meta["role"] == role) & (meta["script"] == "dev")].sort_values("fact_id")
        lat = meta.loc[(meta["role"] == role) & (meta["script"] == "lat")].sort_values("fact_id")
        if not np.array_equal(dev["fact_id"].to_numpy(), lat["fact_id"].to_numpy()):
            raise RuntimeError("Isolated Dev/Latin metadata are not aligned.")
        for layer in range(states.shape[1]):
            ranks, cosine = centered_ranks(
                states[dev["state_row"].to_numpy(), layer],
                states[lat["state_row"].to_numpy(), layer],
            )
            summary_rows.append(
                {
                    "role": role,
                    "layer": layer,
                    "top1": float(np.mean(ranks == 1)),
                    "top5": float(np.mean(ranks <= 5)),
                    "mrr": float(np.mean(1.0 / ranks)),
                    "mean_rank": float(np.mean(ranks)),
                    "chance_top5": min(5 / len(ranks), 1.0),
                }
            )
            rank_rows.extend(
                {
                    "fact_id": fid,
                    "role": role,
                    "layer": layer,
                    "rank": int(rank),
                    "correct_cosine": float(cos),
                }
                for fid, rank, cos in zip(dev["fact_id"], ranks, cosine)
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(rank_rows)


def pair_retrieval(states: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, rank_rows = [], []
    positive = meta.loc[meta["truth"] == 1]
    for context in contexts_in(positive):
        for role in ("a", "b"):
            for other_script, dev_condition, lat_condition in ROLE_PAIRS[role]:
                dev = positive.loc[
                    (positive["context"] == context) & (positive["condition"] == dev_condition)
                ].sort_values("fact_id")
                lat = positive.loc[
                    (positive["context"] == context) & (positive["condition"] == lat_condition)
                ].sort_values("fact_id")
                if not np.array_equal(dev["fact_id"].to_numpy(), lat["fact_id"].to_numpy()):
                    raise RuntimeError("Pair-prompt Dev/Latin metadata are not aligned.")
                for position_index, position in enumerate(POSITIONS):
                    for layer in range(states.shape[1]):
                        ranks, cosine = centered_ranks(
                            states[dev["state_row"].to_numpy(), layer, position_index],
                            states[lat["state_row"].to_numpy(), layer, position_index],
                        )
                        summary_rows.append(
                            {
                                "context": context,
                                "role": role,
                                "other_entity_script": other_script,
                                "dev_condition": dev_condition,
                                "lat_condition": lat_condition,
                                "position": position,
                                "layer": layer,
                                "top1": float(np.mean(ranks == 1)),
                                "top5": float(np.mean(ranks <= 5)),
                                "mrr": float(np.mean(1.0 / ranks)),
                                "mean_rank": float(np.mean(ranks)),
                            }
                        )
                        rank_rows.extend(
                            {
                                "fact_id": fid,
                                "context": context,
                                "role": role,
                                "other_entity_script": other_script,
                                "position": position,
                                "layer": layer,
                                "rank": int(rank),
                                "correct_cosine": float(cos),
                            }
                            for fid, rank, cos in zip(dev["fact_id"], ranks, cosine)
                        )
    return pd.DataFrame(summary_rows), pd.DataFrame(rank_rows)


def relation_probe_oof(
    states: np.ndarray,
    meta: pd.DataFrame,
    source_context: str,
    target_context: str,
    position: str,
    alpha: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_index = POSITIONS.index(position)
    source = meta.loc[
        (meta["context"] == source_context) & (meta["condition"] == "LATLAT")
    ]
    target = meta.loc[
        (meta["context"] == target_context) & (meta["condition"] == "DEVDEV")
    ]
    facts = np.array(sorted(meta["fact_id"].unique()))
    splitter = GroupKFold(n_splits=5)
    summary_rows, prediction_rows = [], []
    for layer in range(states.shape[1]):
        layer_predictions = []
        for fold, (train_index, test_index) in enumerate(
            splitter.split(np.zeros(len(facts)), groups=facts)
        ):
            train_facts = set(facts[train_index])
            test_facts = set(facts[test_index])
            source_train = source.loc[source["fact_id"].isin(train_facts)]
            target_test = target.loc[target["fact_id"].isin(test_facts)]
            x_train = normalize(
                states[source_train["state_row"].to_numpy(), layer, position_index].astype(np.float32)
            )
            y_train = source_train["truth"].to_numpy(dtype=int)
            x_test = normalize(
                states[target_test["state_row"].to_numpy(), layer, position_index].astype(np.float32)
            )
            y_test = target_test["truth"].to_numpy(dtype=int)
            classifier = RidgeClassifier(alpha=alpha).fit(x_train, y_train)
            pred = classifier.predict(x_test)
            decision = classifier.decision_function(x_test)
            for local, (_, row) in enumerate(target_test.iterrows()):
                layer_predictions.append(
                    {
                        "source_context": source_context,
                        "target_context": target_context,
                        "position": position,
                        "layer": layer,
                        "fold": fold,
                        "fact_id": row["fact_id"],
                        "split": row["split"],
                        "truth": int(y_test[local]),
                        "probe_pred": int(pred[local]),
                        "probe_decision": float(decision[local]),
                        "probe_correct": int(pred[local] == y_test[local]),
                    }
                )
        layer_frame = pd.DataFrame(layer_predictions)
        summary_rows.append(
            {
                "source_context": source_context,
                "target_context": target_context,
                "position": position,
                "layer": layer,
                "balanced_accuracy": balanced_accuracy_score(
                    layer_frame["truth"], layer_frame["probe_pred"]
                ),
                "same_recall": recall_score(
                    layer_frame["truth"], layer_frame["probe_pred"], pos_label=1
                ),
                "different_recall": recall_score(
                    layer_frame["truth"], layer_frame["probe_pred"], pos_label=0
                ),
            }
        )
        prediction_rows.extend(layer_predictions)
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def relation_probe_heldout(
    states: np.ndarray,
    meta: pd.DataFrame,
    source_context: str,
    target_context: str,
    position: str,
    alpha: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train only on LATLAT train facts; evaluate DEVDEV validation and test facts."""
    position_index = POSITIONS.index(position)
    source_train = meta.loc[
        (meta["context"] == source_context)
        & (meta["condition"] == "LATLAT")
        & (meta["split"] == "train")
    ]
    target = meta.loc[
        (meta["context"] == target_context) & (meta["condition"] == "DEVDEV")
    ]
    summary_rows, prediction_rows = [], []
    for layer in range(states.shape[1]):
        x_train = normalize(
            states[source_train["state_row"].to_numpy(), layer, position_index].astype(np.float32)
        )
        y_train = source_train["truth"].to_numpy(dtype=int)
        classifier = RidgeClassifier(alpha=alpha).fit(x_train, y_train)
        for evaluation_split in ("validation", "test"):
            target_eval = target.loc[target["split"] == evaluation_split]
            x_eval = normalize(
                states[target_eval["state_row"].to_numpy(), layer, position_index].astype(np.float32)
            )
            y_eval = target_eval["truth"].to_numpy(dtype=int)
            pred = classifier.predict(x_eval)
            decision = classifier.decision_function(x_eval)
            summary_rows.append(
                {
                    "source_context": source_context,
                    "target_context": target_context,
                    "position": position,
                    "evaluation_split": evaluation_split,
                    "layer": layer,
                    "balanced_accuracy": balanced_accuracy_score(y_eval, pred),
                    "same_recall": recall_score(y_eval, pred, pos_label=1),
                    "different_recall": recall_score(y_eval, pred, pos_label=0),
                    "n_facts": target_eval["fact_id"].nunique(),
                }
            )
            prediction_rows.extend(
                {
                    "source_context": source_context,
                    "target_context": target_context,
                    "position": position,
                    "evaluation_split": evaluation_split,
                    "layer": layer,
                    "fact_id": row.fact_id,
                    "truth": int(truth),
                    "probe_pred": int(prediction),
                    "probe_decision": float(score),
                    "probe_correct": int(prediction == truth),
                }
                for row, truth, prediction, score in zip(
                    target_eval.itertuples(), y_eval, pred, decision
                )
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def subgroup_retrieval_tables(
    pair_ranks: pd.DataFrame, gate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = []
    for context in contexts_in_gate(gate):
        groups = gate[["fact_id", f"{context}_LL_success_DD_fail", f"{context}_robust_all"]].copy()
        groups["group"] = np.where(
            groups[f"{context}_LL_success_DD_fail"] == 1,
            "script_sensitive",
            np.where(groups[f"{context}_robust_all"] == 1, "robust", "other"),
        )
        context_ranks = pair_ranks.loc[pair_ranks["context"] == context].merge(
            groups[["fact_id", "group"]], on="fact_id", how="left"
        )
        labeled.append(context_ranks)
    labeled_ranks = pd.concat(labeled, ignore_index=True)
    subgroup = (
        labeled_ranks.loc[labeled_ranks["group"].isin(["script_sensitive", "robust"])]
        .assign(top1=lambda x: (x["rank"] <= 1).astype(int), top5=lambda x: (x["rank"] <= 5).astype(int))
        .groupby(
            [
                "context",
                "group",
                "role",
                "other_entity_script",
                "position",
                "layer",
            ],
            as_index=False,
        )
        .agg(top1=("top1", "mean"), top5=("top5", "mean"), mean_rank=("rank", "mean"), n=("fact_id", "nunique"))
    )
    joint_source = labeled_ranks.loc[
        (labeled_ranks["other_entity_script"] == "dev")
        & labeled_ranks["group"].isin(["script_sensitive", "robust"])
    ].copy()
    joint_source["top5"] = (joint_source["rank"] <= 5).astype(int)
    joint_per_fact = (
        joint_source.groupby(["fact_id", "context", "group", "position", "layer"])["top5"]
        .agg(both_top5=lambda values: int(len(values) == 2 and np.all(np.asarray(values) == 1)))
        .reset_index()
    )
    joint = (
        joint_per_fact.groupby(["context", "group", "position", "layer"], as_index=False)
        .agg(both_top5=("both_top5", "mean"), n=("fact_id", "nunique"))
    )
    return subgroup, joint


def tokenization_table(corpus: pd.DataFrame, model_key: str) -> pd.DataFrame:
    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
    rows = []
    for _, row in corpus.iterrows():
        for role in ("a", "b"):
            for script in ("dev", "lat"):
                text = str(row[f"name_{role}_{script}"])
                rows.append(
                    {
                        "fact_id": row["fact_id"],
                        "role": role,
                        "script": script,
                        "text": text,
                        "n_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                        "n_chars": len(text),
                    }
                )
    return pd.DataFrame(rows)


def predictive_breakdown(
    corpus: pd.DataFrame,
    gate: pd.DataFrame,
    tokenization: pd.DataFrame,
    pair_ranks: pd.DataFrame,
    probe_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, scores = [], []
    token_wide = tokenization.pivot_table(
        index=["fact_id", "role"], columns="script", values="n_tokens", aggfunc="first"
    ).reset_index()
    token_wide["ratio"] = token_wide["dev"] / token_wide["lat"].clip(lower=1)
    token_features = token_wide.pivot(index="fact_id", columns="role", values="ratio")
    token_features.columns = [f"token_ratio_{role}" for role in token_features.columns]

    for context in contexts_in_gate(gate):
        labels = gate.set_index("fact_id")[[
            f"{context}_LL_success_DD_fail", f"{context}_robust_all"
        ]]
        labels = labels.loc[labels.sum(axis=1) == 1].copy()
        labels["sensitive"] = labels[f"{context}_LL_success_DD_fail"].astype(int)
        feature = labels[["sensitive"]].join(token_features, how="left")
        generated = corpus.set_index("fact_id")[["generated_lat"]].copy()
        generated["generated_lat"] = (
            generated["generated_lat"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "1": 1, "false": 0, "0": 0})
            .fillna(0)
            .astype(int)
        )
        feature = feature.join(generated, how="left")
        chosen = pair_ranks.loc[
            (pair_ranks["context"] == context)
            & (pair_ranks["other_entity_script"] == "dev")
            & (pair_ranks["position"].isin(["E2", "Q", "READOUT"]))
            & (pair_ranks["layer"].isin([3, 8, 12, 18]))
        ].copy()
        chosen["feature"] = (
            "rank_" + chosen["role"] + "_" + chosen["position"] + "_l" + chosen["layer"].astype(str)
        )
        rank_features = chosen.pivot_table(
            index="fact_id", columns="feature", values="rank", aggfunc="mean"
        )
        feature = feature.join(rank_features, how="left")
        probe = probe_predictions.loc[
            (probe_predictions["source_context"] == context)
            & (probe_predictions["target_context"] == context)
            & (probe_predictions["position"] == "Q")
            & (probe_predictions["truth"] == 1)
            & (probe_predictions["layer"].isin([8, 12, 18]))
        ].copy()
        probe["feature"] = "probe_q_l" + probe["layer"].astype(str)
        probe_features = probe.pivot_table(
            index="fact_id", columns="feature", values="probe_decision", aggfunc="first"
        )
        feature = feature.join(probe_features, how="left")
        feature["context"] = context
        rows.append(feature.reset_index())

        feature_sets = {
            "surface": [c for c in feature if c.startswith("token_")] + ["generated_lat"],
            "surface_plus_early": [c for c in feature if c.startswith("token_")]
            + ["generated_lat"]
            + [c for c in feature if c.endswith("_l3")],
            "surface_plus_persistence": [c for c in feature if c.startswith("token_")]
            + ["generated_lat"]
            + [c for c in feature if c.startswith("rank_")],
            "all_including_relation": [c for c in feature if c != "sensitive" and c != "context"],
        }
        y = feature["sensitive"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260813)
        for name, columns in feature_sets.items():
            columns = [col for col in columns if col in feature.columns]
            x = feature[columns].replace([np.inf, -np.inf], np.nan)
            x = x.fillna(x.median(numeric_only=True)).to_numpy(dtype=float)
            estimator = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            )
            probability = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]
            scores.append(
                {
                    "context": context,
                    "feature_set": name,
                    "oof_auc": roc_auc_score(y, probability),
                    "n": len(y),
                    "n_sensitive": int(y.sum()),
                    "n_features": len(columns),
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    paths = ensure_run_dirs(args.run_dir)
    corpus, _ = load_prepared_data(args.run_dir)
    gate = pd.read_csv(paths["tables"] / "knowledge_gate.csv")
    isolated_meta = pd.read_csv(paths["states"] / "isolated_state_metadata.csv")
    pair_meta = pd.read_csv(paths["states"] / "pair_state_metadata.csv")
    isolated_states = np.load(paths["states"] / "isolated_states.npy", mmap_mode="r")
    pair_states = np.load(paths["states"] / "pair_states.npy", mmap_mode="r")

    isolated_summary, isolated_ranks = isolated_retrieval(isolated_states, isolated_meta)
    isolated_summary.to_csv(paths["tables"] / "isolated_retrieval_summary.csv", index=False)
    isolated_ranks.to_csv(paths["tables"] / "isolated_retrieval_ranks.csv", index=False)
    pair_summary, pair_ranks = pair_retrieval(pair_states, pair_meta)
    pair_summary.to_csv(paths["tables"] / "pair_retrieval_summary.csv", index=False)
    pair_ranks.to_csv(paths["tables"] / "pair_retrieval_ranks.csv", index=False)
    subgroup, joint_subgroup = subgroup_retrieval_tables(pair_ranks, gate)
    subgroup.to_csv(paths["tables"] / "pair_retrieval_subgroups.csv", index=False)
    joint_subgroup.to_csv(paths["tables"] / "pair_joint_retrieval_subgroups.csv", index=False)

    probe_summaries, probe_predictions = [], []
    heldout_summaries, heldout_predictions = [], []
    _ctxs = contexts_in(pair_meta)
    for source_context, target_context, position in product(_ctxs, _ctxs, ("Q", "READOUT")):
        summary, predictions = relation_probe_oof(
            pair_states,
            pair_meta,
            source_context,
            target_context,
            position,
        )
        probe_summaries.append(summary)
        probe_predictions.append(predictions)
        heldout_summary, heldout_prediction = relation_probe_heldout(
            pair_states,
            pair_meta,
            source_context,
            target_context,
            position,
        )
        heldout_summaries.append(heldout_summary)
        heldout_predictions.append(heldout_prediction)
    probe_summary = pd.concat(probe_summaries, ignore_index=True)
    probe_predictions_df = pd.concat(probe_predictions, ignore_index=True)
    probe_summary.to_csv(paths["tables"] / "relation_probe_summary.csv", index=False)
    probe_predictions_df.to_csv(paths["tables"] / "relation_probe_oof_predictions.csv", index=False)
    pd.concat(heldout_summaries, ignore_index=True).to_csv(
        paths["tables"] / "relation_probe_heldout_summary.csv", index=False
    )
    pd.concat(heldout_predictions, ignore_index=True).to_csv(
        paths["tables"] / "relation_probe_heldout_predictions.csv", index=False
    )

    exact_rows = []
    for context in contexts_in_gate(gate):
        sensitive = set(
            gate.loc[gate[f"{context}_LL_success_DD_fail"] == 1, "fact_id"].astype(str)
        )
        subset = probe_predictions_df.loc[
            (probe_predictions_df["source_context"] == context)
            & (probe_predictions_df["target_context"] == context)
            & (probe_predictions_df["truth"] == 1)
            & (probe_predictions_df["fact_id"].astype(str).isin(sensitive))
        ]
        for (position, layer), group in subset.groupby(["position", "layer"]):
            exact_rows.append(
                {
                    "context": context,
                    "position": position,
                    "layer": layer,
                    "n_exact_failures": group["fact_id"].nunique(),
                    "probe_same_recall_exact_failures": float((group["probe_pred"] == 1).mean()),
                    "mean_probe_decision_exact_failures": float(group["probe_decision"].mean()),
                }
            )
    pd.DataFrame(exact_rows).to_csv(paths["tables"] / "exact_failure_probe_summary.csv", index=False)

    tokenization = tokenization_table(corpus, args.model)
    tokenization.to_csv(paths["tables"] / "tokenization.csv", index=False)
    ratios = tokenization.pivot_table(
        index=["fact_id", "role"], columns="script", values="n_tokens", aggfunc="first"
    ).reset_index()
    ratios["dev_lat_token_ratio"] = ratios["dev"] / ratios["lat"].clip(lower=1)
    ignition_rows = []
    for (fid, role), group in isolated_ranks.groupby(["fact_id", "role"]):
        ordered = group.sort_values("layer")
        values = ordered["rank"].to_numpy()
        layers = ordered["layer"].to_numpy()
        ignition = np.nan
        for index in range(max(0, len(values) - 2)):
            if np.all(values[index : index + 3] == 1):
                ignition = int(layers[index])
                break
        ignition_rows.append({"fact_id": fid, "role": role, "stable_top1_ignition": ignition})
    fragmentation = ratios.merge(pd.DataFrame(ignition_rows), on=["fact_id", "role"])
    fragmentation.to_csv(paths["tables"] / "fragmentation_vs_ignition.csv", index=False)
    valid = fragmentation.dropna(subset=["stable_top1_ignition"])
    rho, p_value = spearmanr(valid["dev_lat_token_ratio"], valid["stable_top1_ignition"])
    pd.DataFrame([{"rho": rho, "p_value": p_value, "n": len(valid)}]).to_csv(
        paths["tables"] / "fragmentation_correlation.csv", index=False
    )

    feature_rows, feature_scores = predictive_breakdown(
        corpus, gate, tokenization, pair_ranks, probe_predictions_df
    )
    feature_rows.to_csv(paths["tables"] / "failure_prediction_features.csv", index=False)
    feature_scores.to_csv(paths["tables"] / "failure_prediction_oof_auc.csv", index=False)
    print("Best within-context relation probes:")
    print(
        probe_summary.loc[probe_summary["source_context"] == probe_summary["target_context"]]
        .groupby(["source_context", "position"])["balanced_accuracy"]
        .max()
        .to_string()
    )
    print("\nPredictive breakdown:")
    print(feature_scores.to_string(index=False))


if __name__ == "__main__":
    main()

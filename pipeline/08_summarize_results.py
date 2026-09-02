"""Create a compact dashboard and shareable ZIP (large hidden states excluded)."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import add_common_args, contexts_in, ensure_run_dirs, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, include_model=False)
    args = parser.parse_args()
    paths = ensure_run_dirs(args.run_dir)
    dashboard: dict[str, object] = {}

    behavior = pd.read_csv(paths["tables"] / "behavior.csv")
    canonical = behavior.loc[behavior["paraphrase_id"] == 0]
    behavior_summary = (
        canonical.groupby(["context", "condition", "truth"])["correct"].mean().reset_index()
    )
    for row in behavior_summary.itertuples():
        dashboard[f"behavior_{row.context}_{row.condition}_truth{row.truth}"] = float(row.correct)
    dashboard["n_facts"] = int(behavior["fact_id"].nunique())
    dashboard["n_behavior_prompts"] = int(len(behavior))

    interaction_path = paths["tables"] / "context_script_interaction_per_fact.csv"
    if interaction_path.exists():
        interaction = pd.read_csv(interaction_path)
        dashboard["mean_context_by_script_margin_interaction"] = float(
            interaction["context_by_script_interaction"].mean()
        )

    pair_path = paths["tables"] / "pair_retrieval_summary.csv"
    if pair_path.exists():
        pair = pd.read_csv(pair_path)
        for context in contexts_in(pair):
            subset = pair.loc[
                (pair["context"] == context)
                & (pair["other_entity_script"] == "dev")
                & (pair["position"] == "Q")
            ]
            dashboard[f"pair_Q_peak_top5_{context}"] = float(subset["top5"].max())

    probe_path = paths["tables"] / "relation_probe_summary.csv"
    if probe_path.exists():
        probe = pd.read_csv(probe_path)
        for context in contexts_in(probe, "source_context"):
            subset = probe.loc[
                (probe["source_context"] == context)
                & (probe["target_context"] == context)
                & (probe["position"] == "Q")
            ]
            dashboard[f"relation_probe_best_balanced_{context}"] = float(
                subset["balanced_accuracy"].max()
            )

    heldout_probe_path = paths["tables"] / "relation_probe_heldout_summary.csv"
    if heldout_probe_path.exists():
        heldout_probe = pd.read_csv(heldout_probe_path)
        for context in contexts_in(heldout_probe, "source_context"):
            subset = heldout_probe.loc[
                (heldout_probe["source_context"] == context)
                & (heldout_probe["target_context"] == context)
                & (heldout_probe["position"] == "Q")
                & (heldout_probe["evaluation_split"] == "test")
            ]
            dashboard[f"relation_probe_best_test_balanced_{context}"] = float(
                subset["balanced_accuracy"].max()
            )

    exact_path = paths["tables"] / "exact_failure_probe_summary.csv"
    if exact_path.exists():
        exact = pd.read_csv(exact_path)
        band = exact.loc[(exact["layer"] >= 10) & (exact["layer"] <= 18)]
        for (context, position), group in band.groupby(["context", "position"]):
            dashboard[f"exact_failure_same_recall_10_18_{context}_{position}"] = float(
                group["probe_same_recall_exact_failures"].mean()
            )

    causal_path = paths["patching"] / "causal_patching.csv"
    if causal_path.exists():
        causal = pd.read_csv(causal_path)
        heldout = causal.loc[causal["split"].isin(["validation", "test"])]
        for context in heldout["context"].unique():
            same = heldout.loc[
                (heldout["context"] == context)
                & (heldout["truth"] == 1)
                & (heldout["experiment"] == "window_factorial")
                & (heldout["source_condition"] == "LATLAT")
            ]
            if len(same):
                grouped = same.groupby("layer_or_window").agg(
                    mean_delta=("delta_margin", "mean"), flip_rate=("patched_correct", "mean")
                )
                best_name = grouped["mean_delta"].idxmax()
                dashboard[f"causal_best_window_{context}"] = str(best_name)
                dashboard[f"causal_best_window_delta_{context}"] = float(
                    grouped.loc[best_name, "mean_delta"]
                )
                dashboard[f"causal_best_window_flip_rate_{context}"] = float(
                    grouped.loc[best_name, "flip_rate"]
                )

    component_path = paths["patching"] / "component_patching_summary.csv"
    if component_path.exists():
        component = pd.read_csv(component_path)
        heldout = component.loc[component["split"].isin(["validation", "test"])]
        for (context, kind), group in heldout.loc[heldout["truth"] == 1].groupby(
            ["context", "component"]
        ):
            dashboard[f"component_mean_delta_{context}_{kind}"] = float(group["mean_delta"].mean())

    prompt_path = paths["recovery"] / "prompt_anchor_summary.csv"
    if prompt_path.exists():
        prompt = pd.read_csv(prompt_path)
        test = prompt.loc[prompt["split"] == "test"]
        for (context, truth, anchor), group in test.groupby(["context", "truth", "anchor"]):
            dashboard[f"prompt_recovery_{context}_truth{truth}_{anchor}"] = float(
                group["accuracy"].mean()
            )

    vector_path = paths["recovery"] / "vector_recovery_test_summary.csv"
    if vector_path.exists():
        vector = pd.read_csv(vector_path)
        for row in vector.itertuples():
            dashboard[
                f"vector_recovery_{row.context}_truth{row.truth}_{row.vector_kind}"
            ] = float(row.accuracy)

    write_json(args.run_dir / "results_dashboard.json", dashboard)
    lines = ["# Cross-script entity pipeline results", ""]
    for key, value in dashboard.items():
        rendered = f"{value:.4f}" if isinstance(value, float) and np.isfinite(value) else str(value)
        lines.append(f"- `{key}`: {rendered}")
    (args.run_dir / "results_dashboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive_path = args.run_dir / "shareable_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in args.run_dir.rglob("*"):
            if not path.is_file() or path == archive_path:
                continue
            if path.suffix in {".npy", ".npz"} or ".partial" in path.name:
                continue
            archive.write(path, path.relative_to(args.run_dir))
    print(json.dumps(dashboard, indent=2, ensure_ascii=False))
    print("\nShare this archive for inference:", archive_path)


if __name__ == "__main__":
    main()

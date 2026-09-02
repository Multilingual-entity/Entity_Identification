#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:-qwen}"
RUN_DIR="${2:-results/${MODEL_KEY}_a100}"

python 00_prepare_data.py --run-dir "${RUN_DIR}"
python 01_run_behavior.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 02_extract_states.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 03_analyze_states.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 04_run_causal_patching.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 05_run_component_patching.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 06_run_head_patching.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 07_run_recovery.py --model "${MODEL_KEY}" --run-dir "${RUN_DIR}"
python 08_summarize_results.py --run-dir "${RUN_DIR}"

echo "Pipeline complete: ${RUN_DIR}"
echo "Share: ${RUN_DIR}/shareable_results.zip"


#!/usr/bin/env bash
# Run the stages that Gemma is missing, on the 299-fact corpus.
#
# Gemma's accuracy gate selected zero items, so stages 04 to 06 never produced anything and
# the paper reports all causal results as single-model. The margin gate read on the negative
# polarity is what changes that: Gemma accepts nearly every positive pair, so its failures
# are on the negatives and a gate defined on positive correctness cannot fire.
#
# Deliberately does NOT stop at the first failure. A failure at 04 says nothing about
# whether 09 to 12 will work, and those do not depend on it. Every stage is resumable, so a
# stage that dies can be rerun with the same command and picks up from its checkpoint.
#
#   bash run_gemma.sh                      # gemma, default run dir
#   bash run_gemma.sh qwen results/qwen_a100
#
# Run it inside tmux: stage 06 alone is hours.

set -u

MODEL="${1:-gemma}"
RUN_DIR="${2:-results/${MODEL}_a100}"
LOG_DIR="${LOG_DIR:-logs_${MODEL}}"

mkdir -p "$LOG_DIR"

echo "model    : $MODEL"
echo "run dir  : $RUN_DIR"
echo "logs     : $LOG_DIR/"
echo

if [ ! -f "$RUN_DIR/data/corpus.csv" ]; then
    echo "ERROR: no corpus at $RUN_DIR/data/corpus.csv"
    echo "       stage 00 has not been run for this run directory."
    exit 1
fi
echo "corpus   : $(( $(wc -l < "$RUN_DIR/data/corpus.csv") - 1 )) items"

# Fail early rather than after the first stage has loaded five gigabytes of weights.
if ! python -c "import pipeline_common, transformers, torch" 2>/dev/null; then
    echo "ERROR: the environment is not ready. Activate the conda env first:"
    echo "         source ~/miniconda3/etc/profile.d/conda.sh && conda activate multi-lingual"
    exit 1
fi

# stage:extra arguments. Gemma needs the margin gate; the others take no special flags.
STAGES=(
  "04:04_run_causal_patching.py:--gate margin --gate-polarity negative"
  "05:05_run_component_patching.py:--cumulative"
  "06:06_run_head_patching.py:"
  "09:09_jacobian_lens.py:"
  "10:10_name_surprisal.py:"
  "11:11_name_order_swap.py:--patch"
  "12:12_romanization_schemes.py:"
)
# Only Gemma needs the gate override, because only Gemma's acceptance bias stops the
# default gate from firing. Everything else runs 04 on its own default.
if [ "$MODEL" != "gemma" ]; then
    STAGES[0]="04:04_run_causal_patching.py:"
fi

# Models deeper or shallower than the 28-layer reference need the preregistered bands
# rescaled, or the same absolute layer numbers point at a different part of the network.
# Llama has 32 layers; without this its patching is measured in the wrong place.
case "$MODEL" in
  llama|qwen14|aya|gemma9)
    for i in "${!STAGES[@]}"; do
      stage="${STAGES[$i]%%:*}"
      case "$stage" in
        04|05|06|07|09|11) STAGES[$i]="${STAGES[$i]} --scale-layers" ;;
      esac
    done
    echo "depth-rescaling enabled: --scale-layers added to stages 04 05 06 07 09 11"
    ;;
esac

declare -a RESULTS=()
STARTED_ALL=$(date +%s)

for entry in "${STAGES[@]}"; do
    stage="${entry%%:*}"
    rest="${entry#*:}"
    script="${rest%%:*}"
    extra="${rest#*:}"

    echo
    echo "=============================================================="
    echo "stage $stage  $script  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)

    # shellcheck disable=SC2086
    python "$script" --model "$MODEL" --run-dir "$RUN_DIR" $extra 2>&1 \
        | tee "$LOG_DIR/stage_${stage}.log"
    code=${PIPESTATUS[0]}

    elapsed=$(( $(date +%s) - started ))
    if [ "$code" -eq 0 ]; then
        RESULTS+=("$stage ok      ${elapsed}s")
        echo "stage $stage finished in ${elapsed}s"
    else
        RESULTS+=("$stage FAILED  ${elapsed}s  (exit $code)")
        echo "stage $stage FAILED after ${elapsed}s; continuing with the next stage"
    fi
done

echo
echo "=============================================================="
echo "summary for $MODEL"
echo "=============================================================="
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "  total $(( ( $(date +%s) - STARTED_ALL ) / 60 )) minutes"
echo
echo "The number that matters is in $LOG_DIR/stage_04.log: how many items passed the"
echo "gate. Zero means the flags did not take effect and stages 05 and 06 produced"
echo "nothing. Around a hundred means this model has a causal arm."
grep -iE "stable|gate=" "$LOG_DIR/stage_04.log" 2>/dev/null | head -3

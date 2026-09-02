#!/usr/bin/env bash
# Finish Qwen and run Llama end to end, unattended.
#
# Sequential on purpose. Two models patching at once share a card another user is
# already saturating, so nothing finishes sooner, and both write hidden states to a
# disk that has already filled once tonight.
#
# Nothing here stops at the first failure. A stage that dies says nothing about the
# stages that do not depend on it, and every stage is resumable, so a failed one can be
# rerun with the same command afterwards.
#
#   bash overnight_mr.sh                 # qwen 05-12, then llama 00-12
#   bash overnight_mr.sh 2>&1 | tee overnight_mr.log
#
# Run it inside tmux, or it dies with the ssh session:
#   tmux new -s overnight
#   bash overnight_mr.sh

set -u

CORPUS_ROOT="${CORPUS_ROOT:-corpora}"
# Not LANG: that is the shell locale variable, and overwriting it changes how
# Python decodes text. A Devanagari corpus is the worst place to find that out.
CORPUS_LANG="${CORPUS_LANG:-mr}"
LOG_DIR="${LOG_DIR:-logs_overnight}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

mkdir -p "$LOG_DIR"

# The run that failed tonight died with ENOSPC in the middle of writing hidden states,
# which left truncated checkpoints that the next run would have resumed onto. Checking
# before each stage costs nothing and turns a corrupt run into a clean stop.
check_disk () {
    local free
    free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo
        echo "STOPPING: only ${free}G free, below the ${MIN_FREE_GB}G floor."
        echo "Nothing is corrupt. Free space and rerun the same command."
        exit 1
    fi
    echo "  disk ${free}G free"
}

run_stages () {
    local model="$1"; shift
    local stages="$*"
    local log="$LOG_DIR/${model}_$(echo "$stages" | tr ' ' '-').log"

    echo
    echo "=============================================================="
    echo "$model  stages $stages   $(date '+%F %H:%M:%S')"
    echo "=============================================================="
    check_disk

    local started
    started=$(date +%s)
    # shellcheck disable=SC2086
    python run_languages.py --languages "$CORPUS_LANG" --model "$model" \
        --corpus-root "$CORPUS_ROOT" --stages $stages 2>&1 | tee "$log"
    local code=${PIPESTATUS[0]}
    local mins=$(( ( $(date +%s) - started ) / 60 ))

    if [ "$code" -eq 0 ]; then
        RESULTS+=("$model  $stages  ok      ${mins}m")
    else
        RESULTS+=("$model  $stages  FAILED  ${mins}m  (exit $code)")
    fi
    echo "$model stages $stages finished in ${mins}m (exit $code)"
}

# Same as run_stages but overwrites a completed stage instead of skipping it.
run_stages_forced () {
    local model="$1"; shift
    local stages="$*"
    echo
    echo "=============================================================="
    echo "$model  stages $stages  (forced rerun)   $(date '+%F %H:%M:%S')"
    echo "=============================================================="
    check_disk
    local started; started=$(date +%s)
    # shellcheck disable=SC2086
    python run_languages.py --languages "$CORPUS_LANG" --model "$model"         --corpus-root "$CORPUS_ROOT" --stages $stages --force 2>&1         | tee "$LOG_DIR/${model}_$(echo "$stages" | tr ' ' '-')_forced.log"
    local code=${PIPESTATUS[0]}
    local mins=$(( ( $(date +%s) - started ) / 60 ))
    if [ "$code" -eq 0 ]; then
        RESULTS+=("$model  $stages  ok (forced)  ${mins}m")
    else
        RESULTS+=("$model  $stages  FAILED       ${mins}m  (exit $code)")
    fi
}

declare -a RESULTS=()
ALL_STARTED=$(date +%s)

echo "corpus root : $CORPUS_ROOT"
echo "language    : $CORPUS_LANG"
echo "logs        : $LOG_DIR/"
echo "disk floor  : ${MIN_FREE_GB}G"
check_disk

# Gemma's stage 07 ran while the scale grid still stopped at 3.0. The grid was extended
# afterwards, and a recovery result is only comparable across models if all three
# searched the same space, so Gemma's is redone here. --force is required because the
# stage is resumable and would otherwise see its own finished output and skip.
# 08 as well as 07: the dashboard reads vector_recovery_test_summary.csv, so leaving
# it alone would report the old grid's numbers beside the new grid's results.
# Stages 09 to 12 are independent of both and do not need redoing.
run_stages_forced gemma 07 08

# Qwen's 00-04 are already done; only the rest is left.
run_stages qwen 05 06 07 08 09 10 11 12

# Llama from scratch. run_languages.py adds --scale-layers for it automatically: the
# preregistered layer bands were fixed on a 28-layer network and Llama has 32, so
# without rescaling the patching is measured in the wrong part of the model.
#
# Split at 04 so a gate that selects nothing is visible in its own log rather than
# buried, and so the stages that do not need the gate still run.
run_stages llama 00 01 02 03
run_stages llama 04
run_stages llama 05 06 07 08 09 10 11 12

echo
echo "=============================================================="
echo "summary   $(date '+%F %H:%M:%S')"
echo "=============================================================="
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "  total $(( ( $(date +%s) - ALL_STARTED ) / 3600 )) hours"
df -h "$HOME" | tail -1

echo
echo "Gate counts, the number that decides whether 05 and 06 produced anything:"
grep -h "stable {" "$LOG_DIR"/*.log 2>/dev/null | sed 's/^ *| */  /' | sort -u

echo
echo "Llama's Marathi is expected to be weak: the paper reports d' = 0.02 in Dev-Dev"
echo "and excludes Marathi for that model. A thin gate there replicates a known result."

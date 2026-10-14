#!/usr/bin/env bash
# Runs for the third review. Stop after each stage and look before going on.
#
#   bash run_review3.sh arms_validation     # item 4, safe, validation only
#   bash run_review3.sh arms_test           # item 4, burns one test evaluation
#   bash run_review3.sh mistral             # item 5, no new data needed
#
# Nothing here touches the corpus, and nothing overwrites an existing mediator
# directory: the coefficient arms write to mediator_arms/ instead.
set -euo pipefail
cd "$(dirname "$0")"
STAGE="${1:-}"

# ---------------------------------------------------------------- item 4 ----
# One direction, three amounts. The question is whether an item's own donor
# coordinate does anything a stranger's coordinate would not.
arms_validation() {
  for RUN in results/qwen_mr:mr results/qwen_hi:hi; do
    DIR="${RUN%%:*}"; CTX="${RUN##*:}"
    for TRAIN_EVAL in "1 2:0" "0 2:1" "0 1:2"; do
      TR="${TRAIN_EVAL%%:*}"; EV="${TRAIN_EVAL##*:}"
      echo "=== $DIR  $CTX  train[$TR] eval[$EV]"
      python 14_failure_mediator.py \
        --run-dir "$DIR" --model qwen --contexts "$CTX" en \
        --out-name mediator_arms \
        --methods jacobian meandiff random fullpatch \
        --coefficients matched shuffled fixed \
        --layers 16 --ranks 1 \
        --train-paraphrases $TR --eval-paraphrases "$EV"
    done
  done
  echo
  echo "now look, before running arms_test:"
  echo "  python ../analysis/coefficient_arms.py \\"
  echo "      --items results/qwen_mr/mediator_arms/mediator_items.csv \\"
  echo "              results/qwen_hi/mediator_arms/mediator_items.csv"
}

# Only after the validation arms have been read. One evaluation, then stop.
arms_test() {
  python 14_failure_mediator.py \
    --run-dir results/qwen_mr --model qwen --contexts mr en \
    --out-name mediator_arms_test \
    --methods jacobian meandiff random fullpatch \
    --coefficients matched shuffled fixed \
    --layers 16 --ranks 1 \
    --train-paraphrases 0 1 --eval-paraphrases 2 \
    --eval-split test --allow-test
}

# ---------------------------------------------------------------- item 5 ----
# Mistral-Small-24B already shows Qwen-like positive-pair rejection behaviourally.
# This asks whether the causal picture recurs. Sites and the gate come from the
# validation split; the test split is not touched here.
mistral() {
  RUN=results/mistral24_mr
  # Stage 00 falls back to csv_qwen7b_instruct/, which on this machine still holds the
  # earlier 299-fact corpus. Passing the corpus explicitly is what the other runs do,
  # via run_languages.py, and the guard below refuses to continue if they disagree.
  CORPUS=results/_corpus/mr/corpus_selected.csv
  NEGATIVES=results/_corpus/mr/hard_negative_map.csv
  test -f "$CORPUS" || { echo "no corpus at $CORPUS; point CORPUS at the 601-fact file"; exit 1; }
  # 00 and 08 take no --model: 00 is model-independent, which is why every run gets
  # the same deterministic splits, and 08 only reads what earlier stages wrote.
  python 00_prepare_data.py  --run-dir "$RUN" --corpus "$CORPUS" --negatives "$NEGATIVES" --force
  python check_prepared_corpus.py --run "$RUN" --reference results/qwen_mr || exit 1
  python 01_run_behavior.py  --run-dir "$RUN" --model mistral24
  python 02_extract_states.py --run-dir "$RUN" --model mistral24
  python 03_analyze_states.py --run-dir "$RUN" --model mistral24
  # Layers are given in Qwen reference space, so every stage that takes them needs
  # --scale-layers on a model of a different depth, including stage 04. Mistral has 40
  # layers against Qwen's 28, so a literal 8-20 covers 20 to 50 percent of this model
  # against 29 to 71 percent of Qwen, and the two profiles would not be comparable.
  python 04_run_causal_patching.py --run-dir "$RUN" --model mistral24 \
      --contexts mr en --layer-low 8 --layer-high 20 --scale-layers \
      --gate either --gate-polarity positive
  python 09_jacobian_lens.py --run-dir "$RUN" --model mistral24 \
      --contexts mr en --site E2 --scale-layers
  python 11_name_order_swap.py --run-dir "$RUN" --model mistral24 \
      --contexts mr en --patch --patch-sites E1 E2 \
      --layer-low 8 --layer-high 20 --scale-layers
  python 08_summarize_results.py --run-dir "$RUN"
  echo
  echo "read 04 and 11 before running the mediator on this model:"
  echo "  the E2 band there decides --layers below, do not reuse the Qwen value of 16"
}

# The mediator on Mistral, once its own E2 band is known. Pass the layer.
mistral_mediator() {
  L="${2:?give the layer chosen from the Mistral patching profile}"
  python 14_failure_mediator.py \
    --run-dir results/mistral24_mr --model mistral24 --contexts mr en \
    --out-name mediator_arms \
    --methods jacobian meandiff random fullpatch \
    --coefficients matched shuffled fixed \
    --layers "$L" --ranks 1 \
    --train-paraphrases 0 1 --eval-paraphrases 2
}

case "$STAGE" in
  arms_validation)  arms_validation ;;
  arms_test)        arms_test ;;
  mistral)          mistral ;;
  mistral_mediator) mistral_mediator "$@" ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac

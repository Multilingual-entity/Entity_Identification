#!/usr/bin/env bash
# Everything still owed after the 2026-09-30 overnight run.
#
#   cd ~/paperB_pipeline && bash run_night2.sh 2>&1 | tee night2.log
#
# What is left, and why:
#
#   6. The English causal cap for Qwen. Last night's attempt passed "--gate either",
#      which is the mistral24 setting from run_review3.sh. It widened the gate for every
#      context, dropped 29 of the 53 frozen Marathi facts and added 92 new ones. This
#      passes NO --gate flag, so the default "accuracy" applies, which is what produced
#      the published {en: 116, mr: 53}. Marathi must come back untouched.
#   5. Mistral. Blocked twice over: the 24B weights are not in the HF cache (44 GB) and
#      results/mistral24_mr/states was deleted to free disk (16 GB to re-extract). It is
#      attempted only if both the disk and the card can actually take it, and skipped
#      with a note otherwise rather than failing at 3am.
#
# Every destructive step restores from the pre-tonight archive first and runs once. The
# double-launch that corrupted the last attempt is prevented by an explicit check.

set -euo pipefail
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d_%H%M)
OUT="night2_${STAMP}"
mkdir -p "$OUT"
NOTES="$OUT/DECISIONS_NEEDED.txt"
: > "$NOTES"

# The archive step 6 took before it ran with the wrong gate. This is the published
# state, and every restore below comes from it rather than from a later copy.
FROZEN_TAG="cap60_20260930_0537"

step() { printf '\n\033[1m=== %s ===\033[0m  %s\n' "$1" "$(date +%H:%M:%S)"; }
note() { printf '%s\n' "$1" >> "$NOTES"; printf '  note: %s\n' "$1"; }
die()  { printf '\n\033[1;31mABORT: %s\033[0m\n' "$1"; exit 1; }

free_gib() { df -P -BG "$1" | awk 'NR==2 {gsub("G","",$4); print $4+0}'; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# Restore a run's patching directory from the published archive, so the rerun starts
# from the same place every time no matter what the previous attempt left behind.
restore() {
  local run="$1" src="results/$1/patching_${FROZEN_TAG}"
  [ -d "$src" ] || die "missing $src; nothing to restore from"
  rm -rf "results/$run/patching" "results/$run/patching_frozen"
  cp -r "$src" "results/$run/patching"
  cp -r "$src" "results/$run/patching_frozen"
  printf '  restored %s from %s\n' "$run" "$src"
}

# ----------------------------------------------------------------- preflight ----
step "preflight"
if pgrep -f 04_run_causal_patching > /dev/null; then
  die "a causal patching process is already running. Two writers on the same partial
  file is what corrupted the last attempt; kill it before starting this."
fi
python check_run_script.py run_night2.sh
printf '  disk: %s GiB free\n' "$(free_gib .)"
printf '  gpu:  %s MiB free\n' "$(gpu_free_mib)"
[ "$(free_gib .)" -ge 2 ] || die "under 2 GiB free"
if [ "$(gpu_free_mib)" -lt 22000 ]; then
  die "under 22 GiB free on the card; Qwen needs about 20. Wait for the other job."
fi

# ------------------------------------------------- 6a. Qwen Marathi corpus ------
step "6a  qwen_mr, English cap 60 -> 116, default accuracy gate"
restore qwen_mr
python 04_run_causal_patching.py --run-dir results/qwen_mr --model qwen \
    --contexts mr en --layer-low 8 --layer-high 20 \
    --max-facts-per-context 116 --force \
    > "$OUT/6a_qwen_mr.log" 2>&1
grep -E "^gate=|^selected by split" "$OUT/6a_qwen_mr.log" | tee -a "$NOTES"
# The gate line must read {'en': 116, 'mr': 53}. Anything else means a --gate flag crept
# back in, and the Marathi rows would no longer be the published ones.
grep -q "stable {'en': 116, 'mr': 53}" "$OUT/6a_qwen_mr.log" \
    || die "the gate did not return {'en': 116, 'mr': 53}; see 6a_qwen_mr.log"
python diagnose_step6.py results/qwen_mr patching_frozen mr --strict \
    | tee "$OUT/6a_diagnose_mr.txt" || die "frozen Marathi facts were dropped"
python diagnose_step6.py results/qwen_mr patching_frozen en \
    | tee "$OUT/6a_diagnose_en.txt"

# ---------------------------------------------------- 6b. Qwen Hindi corpus -----
step "6b  qwen_hi, English cap 60 -> 107, default accuracy gate"
restore qwen_hi
python 04_run_causal_patching.py --run-dir results/qwen_hi --model qwen \
    --contexts hi en --layer-low 8 --layer-high 20 \
    --max-facts-per-context 107 --force \
    > "$OUT/6b_qwen_hi.log" 2>&1
grep -E "^gate=|^selected by split" "$OUT/6b_qwen_hi.log" | tee -a "$NOTES"
grep -q "stable {'en': 107, 'hi': 50}" "$OUT/6b_qwen_hi.log" \
    || die "the gate did not return {'en': 107, 'hi': 50}; see 6b_qwen_hi.log"
python diagnose_step6.py results/qwen_hi patching_frozen hi --strict \
    | tee "$OUT/6b_diagnose_hi.txt" || die "frozen Hindi facts were dropped"
python diagnose_step6.py results/qwen_hi patching_frozen en \
    | tee "$OUT/6b_diagnose_en.txt"

cp results/qwen_mr/patching/causal_patching.csv "$OUT/6a_causal_patching_mr.csv"
cp results/qwen_hi/patching/causal_patching.csv "$OUT/6b_causal_patching_hi.csv"
note "6: regenerate tab:patchci with emit_patchci_table.py --write and drop the caption note about the 60-item cap. English n rises from 60 to 116 (Marathi corpus) and 107 (Hindi corpus); Marathi 53 and Hindi 50 are unchanged."

# ------------------------------------------------------------- 5. Mistral -------
# Needs 44 GB of weights plus 16 GB of re-extracted states, and about 50 GiB on the
# card. Checked rather than attempted, because a half-finished 44 GB download on a full
# volume is worse than not starting.
step "5  Mistral, attempted only if it can finish"
NEED_DISK=62
HAVE_DISK=$(free_gib .)
HAVE_GPU=$(gpu_free_mib)
WEIGHTS=$(find "${HF_HOME:-$HOME/.cache/huggingface}" -maxdepth 3 \
          -name "*Mistral-Small-24B*" 2>/dev/null | head -1)
if [ -n "$WEIGHTS" ]; then
  printf '  weights present: %s\n' "$WEIGHTS"
  NEED_DISK=18
fi
if [ "$HAVE_DISK" -lt "$NEED_DISK" ]; then
  note "5: SKIPPED. ${HAVE_DISK} GiB free, needs ${NEED_DISK} GiB (44 for the weights if absent, 16 for re-extracted states). Free space and run 02_extract_states then the selection pass by hand."
elif [ "$HAVE_GPU" -lt 52000 ]; then
  note "5: SKIPPED. ${HAVE_GPU} MiB free on the card, the 24B model needs about 50 GiB. device_map=auto would offload to CPU rather than fail, so this is refused deliberately."
else
  step "5a  re-extract Mistral states"
  python 02_extract_states.py --run-dir results/mistral24_mr --model mistral24 \
      > "$OUT/5a_states.log" 2>&1
  step "5b  Mistral selection pass on already-spent items"
  SEEN=results/mistral24_mr/patching_unscaled_8_20/seen_fact_ids.csv
  [ -f "$SEEN" ] || die "missing $SEEN"
  python 04_run_causal_patching.py --run-dir results/mistral24_mr --model mistral24 \
      --contexts mr en --layer-low 8 --layer-high 20 --scale-layers \
      --gate either --gate-polarity positive \
      --fact-ids "$SEEN" --max-facts-per-context 141 --force \
      > "$OUT/5b_mistral.log" 2>&1
  cp results/mistral24_mr/patching/causal_patching.csv "$OUT/5b_causal_patching.csv" || true
  note "5: selection pass done. Pick the E2 peak layer from 5b_mistral.log, write it down, then run the evaluation pass on the 29 Marathi and 120 English never-patched items."
fi

# ---------------------------------------------------------------- wrap up -------
step "done"
cp "$NOTES" "DECISIONS_NEEDED_${STAMP}.txt" 2>/dev/null || true
printf '\nlogs: %s\n' "$OUT"
printf 'disk now: %s GiB free\n\n' "$(free_gib .)"
cat "$NOTES"
printf '\nMarathi and Hindi rows were restored from the published archive before each\n'
printf 'rerun and verified unchanged afterwards. No --gate flag was passed to either\n'
printf 'Qwen call, so both used the default accuracy rule the paper reports.\n'

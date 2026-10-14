#!/usr/bin/env bash
# Unattended overnight run. Everything that does not need a human decision.
#
#   cd ~/paperB_pipeline && bash run_overnight.sh 2>&1 | tee overnight.log
#
# Three principles, because nobody is watching:
#
#   1. Every sanity check compares against a known value and aborts the whole run on a
#      mismatch. A check whose result is only printed is not a check at 3am.
#   2. No --allow-test anywhere, and no step spends a holdout that is still unspent.
#      Stage 07 runs with --skip-test, so no scale is ever scored on test. Stage 04's
#      published causal tables already pool all three splits, so step 6 widening the
#      English gate stays inside that convention; the wrap-up says exactly which items
#      each step touched rather than claiming test was avoided everywhere.
#   3. Anything needing a judgement -- which layer Mistral peaks at, whether a gate is
#      worth pursuing -- stops and says so rather than guessing.
#
# Safe to re-run: destructive steps archive first, and stages that support --force are
# given it explicitly so a half-finished run does not silently resume into a mixture.

set -euo pipefail
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d_%H%M)
OUT="overnight_${STAMP}"
mkdir -p "$OUT"
NOTES="$OUT/DECISIONS_NEEDED.txt"
: > "$NOTES"

step() { printf '\n\033[1m=== %s ===\033[0m  %s\n' "$1" "$(date +%H:%M:%S)"; }
note() { printf '%s\n' "$1" >> "$NOTES"; printf '  note: %s\n' "$1"; }
die()  { printf '\n\033[1;31mABORT: %s\033[0m\n' "$1"; exit 1; }

# free_gib <path>: whole gibibytes available on the filesystem holding <path>.
free_gib() { df -P -BG "$1" | awk 'NR==2 {gsub("G","",$4); print $4+0}'; }

# need_disk <gib> <label>: refuse to start work that cannot finish. The nonce states
# alone are about 8 GiB, and a partial write at 3am looks like a result in the morning.
need_disk() {
  local have; have=$(free_gib .)
  printf '  disk: %s GiB free, %s needs %s GiB\n' "$have" "$2" "$1"
  [ "$have" -ge "$1" ] || die "only ${have} GiB free; $2 needs ${1} GiB"
}

# Archive a results directory before a --force run deletes what is in it. The Mistral
# step already did this; the Qwen runs need it more, because their patching directory is
# what six of the paper's tables are computed from.
archive() {
  local src="$1" dst="$1_$2"
  if [ ! -d "$src" ]; then printf '  nothing to archive at %s\n' "$src"; return 0; fi
  if [ -d "$dst" ]; then printf '  archive exists, kept: %s\n' "$dst"; return 0; fi
  cp -r "$src" "$dst"
  printf '  archived %s -> %s\n' "$src" "$dst"
}

# A reporting block must never end the night. These blocks only print and interpret; if
# a column is named differently than expected the run should lose the commentary, not
# the remaining hours of GPU work. Used as:  report "label" <<'PY' ... PY
report() {
  local label="$1"
  if python - | tee -a "$NOTES"; then return 0; fi
  note "$label: the reporting block failed; the results are on disk, read them by hand."
}

# check_csv <file> <python expression over df> <expected> <tolerance> <label>
# Fatal. Use it only where continuing would produce results that cannot be read.
check_csv() { check_value "$@" || die "check failed: $5"; }

# The same comparison, reporting instead of aborting. Sections 4, 5 and 6 do not depend
# on stage 15, so a stage 15 mismatch should cost stage 15 and nothing else.
soft_check() {
  if check_value "$@"; then return 0; fi
  note "MISMATCH in $5. The dependent step is skipped; everything independent continues."
  return 1
}

check_value() {
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import sys, pandas as pd
path, expr, want, tol, label = sys.argv[1:6]
df = pd.read_csv(path)
got = float(eval(expr, {"df": df, "pd": pd}))
want, tol = float(want), float(tol)
ok = abs(got - want) <= tol
print(f"  {label}: got {got:.4f}, expected {want:.4f} +/- {tol}  {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
}

# ----------------------------------------------------------------- preflight ----
step "preflight"
# 2 GiB covers the archives and every CSV and log this run writes. Section 4 needs far
# more and is checked again at its own step, so a full volume stops the cheap work here
# rather than halfway through writing an 8 GiB array.
need_disk 2 "the archives and logs"
python check_run_script.py run_overnight.sh
python check_run_script.py run_review3.sh
# A self-comparison cannot prove the corpus is right: the fingerprints are taken from
# the same file. What it does prove is the second half, that behavior.csv covers every
# prepared fact, which is what catches a run whose stage 00 was rerun under it.
python check_prepared_corpus.py --run results/qwen_mr --reference results/qwen_mr
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true

# Stage 07 rewrites the whole recovery directory, and paper_tables.py and
# audit_paper_coverage.py read it. 1a and 1d both run stage 07, so the published
# section 7 files have to be copied out of the way before either does.
archive results/qwen_mr/recovery "paper_${STAMP}"

# In stage 07, --force controls one thing only: whether the prompt-anchor block is
# recomputed. The vector grid, which is the whole point of 1a and 1d, runs either way.
# That block scores all 57,696 prompts across every split, so forcing it costs about
# half an hour per call and is the one remaining place where this run reads test.
#
# Reusing the existing file is only safe if it was built from the corpus now prepared,
# which is the assumption that put the wrong corpus into the Mistral run. So check it
# rather than assume it, and fall back to recomputing when it does not match.
ANCHOR_FORCE=""
if python - <<'PY'
import sys
from pathlib import Path
import pandas as pd
anchors = Path("results/qwen_mr/recovery/prompt_anchor_recovery.csv")
if not anchors.exists():
    print("  no prompt_anchor_recovery.csv; it has to be computed")
    sys.exit(1)
a = pd.read_csv(anchors, usecols=["fact_id", "split"], dtype={"fact_id": str})
c = pd.read_csv("results/qwen_mr/data/corpus.csv", usecols=["fact_id"], dtype={"fact_id": str})
if set(a.fact_id) != set(c.fact_id):
    print(f"  anchors cover {a.fact_id.nunique()} facts, the prepared corpus holds "
          f"{len(c)}, and the sets differ; they must be recomputed")
    sys.exit(1)
print(f"  prompt anchors match the prepared corpus on all {len(c)} facts, reusing them")
PY
then
  printf '  stage 07 runs without --force, so its anchor block never reads test\n'
else
  ANCHOR_FORCE="--force"
  note "prompt anchors are missing or built from another corpus, so stage 07 recomputes them. That adds roughly half an hour to each of 1a and 1d, and the anchor block scores every split including test. Nothing is tuned on it, but the wrap-up's test summary is weaker for this run."
fi

# ------------------------------------------------- 1a. single-layer sanity ------
# Three settings here are NOT argparse defaults, and the published run used all three.
# Reconstructing them cost three failed attempts, so they are recorded here:
#
#   --tune-variant additive    default is "full"
#   --normalise-vectors        default is off; without it the vectors come out about
#                              150 times longer and nothing is comparable
#   --tuning-facts 60          default is 30; the published tuning file holds 240 rows
#                              per cell, which is 60 facts x 2 truths x 2 letters
#
# The evidence for each: the published log line reads "tuning on additive | normalised",
# and the archived vector_tuning_validation.csv has 240 rows where 30 facts give 120.
# Changing any one of them changes the accuracy, so a mismatch in 1a means the command
# drifted from the paper's, not that the model or the states did.
# The window change must not have disturbed the path that produced the published
# number. If it has, everything downstream is built on a different intervention.
step "1a  recovery, single layer, must reproduce 0.083"
python 07_run_recovery.py --run-dir results/qwen_mr --model qwen \
    --contexts mr --layers 16 --layer-window 1 --alphas 6.0 --skip-test \
    --tune-variant additive --normalise-vectors --tuning-facts 60 $ANCHOR_FORCE \
    > "$OUT/1a_recovery_w1.log" 2>&1
cp results/qwen_mr/recovery/vector_hyperparameters_selected.csv "$OUT/1a_selected.csv"
cp results/qwen_mr/recovery/vector_tuning_validation.csv "$OUT/1a_tuning.csv"
# Check the size of the evaluation before its result. An accuracy that differs tells you
# nothing about why; a row count that differs names the cause, because the count is
# tuning_facts x 2 truths x 2 letters and nothing else. Reconstructing the published
# settings took three attempts for want of exactly this line.
check_csv "$OUT/1a_tuning.csv" \
    "len(df[(df.context=='mr')&(df.layer==16)&(df.alpha==6.0)])" 240 0.5 \
    "evaluations per cell (240 = 60 tuning facts x 2 truths x 2 letters)"
check_csv results/qwen_mr/recovery/vector_hyperparameters_selected.csv \
    "df.loc[df.context=='mr','same_accuracy'].iloc[0]" 0.0833 0.002 \
    "Marathi same-pair accuracy at layer 16 alpha 6"
check_csv results/qwen_mr/recovery/vector_hyperparameters_selected.csv \
    "df.loc[df.context=='mr','layer'].iloc[0]" 16 0.5 "selected layer"

# ------------------------------------------ 1b. stage 15, one layer -------------
# With a single layer the window patch reduces to the tested single-layer hook, so a
# mismatch here is stage 15's own plumbing rather than hook composition. It also
# exercises live state capture, because the default fold evaluates paraphrase 2 and the
# cache holds paraphrase 0 only.
step "1b  stage 15 at one layer, must reproduce stage 14 on fold t01e2"
python 15_window_subspace.py --run-dir results/qwen_mr --model qwen \
    --contexts mr --layers 16 --rank 1 --split validation \
    > "$OUT/1b_stage15_single.log" 2>&1
cp results/qwen_mr/patching/window_subspace_summary.csv "$OUT/1b_summary.csv"
cp results/qwen_mr/patching/window_subspace_items.csv "$OUT/1b_items.csv"
STAGE15_OK=1
soft_check "$OUT/1b_summary.csv" \
    "df.loc[df.group=='gated','delta_whole'].iloc[0]" 2.44 0.05 \
    "whole patch, layer 16" || STAGE15_OK=0
soft_check "$OUT/1b_summary.csv" \
    "df.loc[df.group=='gated','delta_subspace'].iloc[0]" 1.38 0.05 \
    "rank-one subspace" || STAGE15_OK=0

# ------------------------------------------ 1c. stage 15, six hooks -------------
# Only interpretable once 1b passes. --eval-paraphrases 0 is required: the patching
# tables hold paraphrase 0 with answer key A only, so any other paraphrase is not
# comparable with them.
step "1c  stage 15 across 12-17, whole patch must match the windows table"
python 15_window_subspace.py --run-dir results/qwen_mr --model qwen \
    --contexts mr --layers 12 13 14 15 16 17 --rank 1 \
    --eval-paraphrases 0 --split validation \
    > "$OUT/1c_stage15_window.log" 2>&1
cp results/qwen_mr/patching/window_subspace_summary.csv "$OUT/1c_summary.csv"
cp results/qwen_mr/patching/window_subspace_items.csv "$OUT/1c_items.csv"
# Tolerance 0.3 rather than 0.1: this composes six hooks and captures states live,
# against a cached single-pass reference, and bfloat16 over 28 blocks does not reproduce
# to two decimals. A real composition failure is a margin point or more, not a tenth.
soft_check "$OUT/1c_summary.csv" \
    "df.loc[df.group=='gated','delta_whole'].iloc[0]" 13.82 0.30 \
    "six-hook whole patch on validation" || STAGE15_OK=0
soft_check "$OUT/1c_summary.csv" \
    "df.loc[df.group=='gated','whole_flip'].iloc[0]" 0.70 0.02 \
    "whole-patch flip rate" || STAGE15_OK=0

note "1c: only delta_whole in 1c_summary.csv is a result. Its delta_subspace fits on paraphrases 0 and 1 and evaluates on 0, so that column is leaky by construction and exists only because the whole-patch comparison needs paraphrase 0. The clean subspace numbers are 1b and 1e."

if [ "$STAGE15_OK" -eq 1 ]; then
  printf '\n  stage 15 agrees with stage 14 at one layer and across six\n'
else
  note "stage 15 did not reproduce stage 14, so 1e is skipped. 1d does not use stage 15 and still runs."
fi

# ---------------------------------------------------- 1d. the repair sweep ------
step "1d  repair sweep across 12-17, extended scale grid"
python 07_run_recovery.py --run-dir results/qwen_mr --model qwen \
    --contexts mr en --layers 17 --layer-window 6 \
    --alphas 0.5 1.0 1.5 2.0 3.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 --skip-test \
    --tune-variant additive --normalise-vectors --tuning-facts 60 $ANCHOR_FORCE \
    > "$OUT/1d_sweep.log" 2>&1
cp results/qwen_mr/recovery/vector_hyperparameters_selected.csv "$OUT/1d_selected.csv"
cp results/qwen_mr/recovery/vector_tuning_validation.csv "$OUT/1d_tuning_grid.csv"
report "1d" <<'PY'
import pandas as pd
s = pd.read_csv("results/qwen_mr/recovery/vector_hyperparameters_selected.csv")
print("1d: selected scale per context ->")
for r in s.itertuples():
    edge = " AT GRID EDGE, the curve still has not turned over" if r.alpha >= 32 else ""
    print(f"    {r.context}: layer {int(r.layer)} alpha {r.alpha} "
          f"same {r.same_accuracy:.3f} different {r.different_accuracy:.3f}{edge}")
PY

# ------------------------------------ 1e. reference arm and coordinates ---------
# A null from 1d is unreadable without these. The reference arm is the upper bound: if
# per-item amounts also fail across the window, a constant vector never had a chance.
if [ "$STAGE15_OK" -ne 1 ]; then
  note "1e: skipped because 1b or 1c did not reproduce stage 14."
else
step "1e  reference arm across 12-17"
python 15_window_subspace.py --run-dir results/qwen_mr --model qwen \
    --contexts mr en --layers 12 13 14 15 16 17 --rank 1 --split validation \
    > "$OUT/1e_reference_arm.log" 2>&1
cp results/qwen_mr/patching/window_subspace_summary.csv "$OUT/1e_reference_summary.csv"
cp results/qwen_mr/patching/window_subspace_items.csv "$OUT/1e_reference_items.csv"
fi

step "1e  donor coordinate distribution against the scale grid"
python analysis/donor_coordinates.py --run-dir results/qwen_mr \
    --contexts mr en --layers 12 13 14 15 16 17 \
    --alphas 0.5 1.0 1.5 2.0 3.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 \
    > "$OUT/1e_donor_coords.log" 2>&1
cp results/qwen_mr/recovery/donor_coordinates.csv "$OUT/1e_donor_coords.csv"
report "1e" <<'PY'
import pandas as pd
d = pd.read_csv("results/qwen_mr/recovery/donor_coordinates.csv")
worst = d.loc[d.fraction_above_grid.idxmax()]
print(f"1e: largest share of items beyond the grid: {worst.fraction_above_grid:.0%} "
      f"({worst.context} layer {int(worst.layer)}, median coordinate "
      f"{worst.median_coord:.2f} against grid max {worst.grid_max})")
if worst.fraction_above_grid > 0.25:
    print("    the grid is still too short to reach where the whole patch goes; a null")
    print("    from 1d is about the search range, not about the direction")
PY

# ------------------------------------------- 4. nonce corpus, behaviour ---------
# Answers a question the paper cannot: does the Dev--Dev collapse appear on name-shaped
# strings with no entity behind them? Also decides whether its causal arm can run.
step "4a  build the nonce corpus"
python corpus_construction/build_nonce_corpus.py \
    --source results/qwen_mr/data/corpus.csv \
    --out corpora/nonce_mr --n 600 --surplus 10 \
    > "$OUT/4a_build_nonce.log" 2>&1
grep -E "separation|overlap gap" "$OUT/4a_build_nonce.log" || true

# 4b writes behaviour tables, but 4c writes pair_states.npy for 600 facts, which is
# about 8 GiB. Check before 4b rather than before 4c: there is no point measuring
# behaviour on a corpus whose causal arm cannot then be extracted.
need_disk 10 "the nonce states in 4c"

step "4b  nonce behaviour"
python 00_prepare_data.py --run-dir results/nonce_mr \
    --corpus corpora/nonce_mr/corpus_selected.csv \
    --negatives corpora/nonce_mr/hard_negative_map.csv \
    > "$OUT/4b_prepare.log" 2>&1
# check_prepared_corpus compares two runs, and there is no second nonce run to compare
# against, so it would fingerprint this file against itself and pass on anything. The
# real question here is whether stage 00 took the nonce corpus or silently fell back to
# a default, and that is answered by the size and by the fact ids being new ones.
python - <<'PY' || die "the nonce run was not prepared from the nonce corpus"
import sys
import pandas as pd
n = pd.read_csv("results/nonce_mr/data/corpus.csv", dtype={"fact_id": str})
r = pd.read_csv("results/qwen_mr/data/corpus.csv", dtype={"fact_id": str})
shared = set(n.fact_id) & set(r.fact_id)
print(f"  nonce corpus: {len(n)} facts, {len(shared)} fact ids shared with the real corpus")
if len(n) < 500 or shared:
    print("  stage 00 did not prepare from corpora/nonce_mr; check 4b_prepare.log")
    sys.exit(1)
print(f"  splits: {n.split.value_counts().to_dict()}")
PY
python 01_run_behavior.py --run-dir results/nonce_mr --model qwen \
    > "$OUT/4b_behaviour.log" 2>&1
python 08_summarize_results.py --run-dir results/nonce_mr \
    > "$OUT/4b_summary.log" 2>&1
cp results/nonce_mr/tables/behavior_summary.csv "$OUT/4b_behavior_summary.csv"
report "4b" <<'PY'
import pandas as pd
b = pd.read_csv("results/nonce_mr/tables/behavior_summary.csv")
g = b.groupby(["context", "condition", "truth"]).accuracy.mean() * 100
for ctx in sorted(b.context.unique()):
    try:
        dd, ll = g[(ctx, "DEVDEV", 1)], g[(ctx, "LATLAT", 1)]
    except KeyError:
        continue
    print(f"4b: nonce {ctx} same-pair Dev-Dev {dd:.1f} vs Lat-Lat {ll:.1f}, "
          f"gap {ll - dd:.1f}  (real corpus: 8.5 vs 35.1, gap 26.7)")
    if ll - dd > 15:
        print("    a comparable collapse on names with nobody behind them means the")
        print("    script gap is string-level and the framing changes")
PY

step "4c  nonce causal gate"
python 02_extract_states.py --run-dir results/nonce_mr --model qwen \
    > "$OUT/4c_states.log" 2>&1
set +e
# --splits is not optional. The selection sorts by split and caps, and "test" sorts
# before "train", so without it the first 60 items taken are the held-out ones. That is
# the mistake the lessons section records, and it is silent.
python 04_run_causal_patching.py --run-dir results/nonce_mr --model qwen \
    --contexts mr --layer-low 8 --layer-high 20 \
    --gate either --gate-polarity positive --splits train validation \
    > "$OUT/4c_patching.log" 2>&1
PATCH_RC=$?
set -e
grep -E "^gate=" "$OUT/4c_patching.log" | tee -a "$NOTES" || true
if [ $PATCH_RC -ne 0 ]; then
  note "4c: patching exited $PATCH_RC, most likely an empty gate. See 4c_patching.log."
fi

# ---------------------------------------------- 6. English causal cap -----------
step "6  raise the English cap for Qwen"
# --force unlinks causal_patching.csv, and that file is what emit_patchci_table.py,
# paper_tables.py, patching_bootstrap.py, verify_key_numbers.py, verify_declined_tables.py
# and both recompute_*_table.py read. If this step dies partway, the server would
# otherwise be left with no copy of the numbers already in the paper.
archive results/qwen_mr/patching "cap60_${STAMP}"
archive results/qwen_hi/patching "cap60_${STAMP}"
# NO --gate here. The default is "accuracy", the original rule, and that is what the
# published Qwen runs used: their causal_candidates.csv records mr 53 and en 116, which
# is what "accuracy" gives. "--gate either" belongs to the mistral24 invocation in
# run_review3.sh, and passing it here widened the gate for every context: Marathi went
# from 53 gated facts to 116, dropping 29 of the frozen items and adding 92 new ones.
# A cap can only ever keep fewer items, so any change in the item set means the gate
# moved, not the cap. Raising the English sample is only a cap change.
python 04_run_causal_patching.py --run-dir results/qwen_mr --model qwen \
    --contexts mr en --layer-low 8 --layer-high 20 \
    --max-facts-per-context 116 --force \
    > "$OUT/6_qwen_mr_cap.log" 2>&1
python 04_run_causal_patching.py --run-dir results/qwen_hi --model qwen \
    --contexts hi en --layer-low 8 --layer-high 20 \
    --max-facts-per-context 107 --force \
    > "$OUT/6_qwen_hi_cap.log" 2>&1
# The Marathi rows are frozen. This rerun recomputes them as a side effect of raising
# the English cap, because stage 04 takes a context pair. They must come back identical.
report "6" <<PY
import pandas as pd
KEY = ["fact_id", "context", "split", "truth", "experiment", "source_condition",
       "site", "layer_or_window"]
for run, ctx in [("qwen_mr", "mr"), ("qwen_hi", "hi")]:
    old = pd.read_csv(f"results/{run}/patching_cap60_${STAMP}/causal_patching.csv")
    new = pd.read_csv(f"results/{run}/patching/causal_patching.csv")
    o = old[old.context == ctx].set_index(KEY).delta_margin.sort_index()
    n = new[new.context == ctx].set_index(KEY).delta_margin.sort_index()
    common = o.index.intersection(n.index)
    worst = (o.loc[common] - n.loc[common]).abs().max()
    print(f"6: {ctx} rows kept {len(common)} of {len(o)}, largest delta_margin change {worst:.4f}")
    if len(common) < len(o):
        print(f"    {len(o) - len(common)} frozen {ctx} rows are MISSING from the rerun.")
        print(f"    STOP. A cap can only keep fewer items, never add new ones, so a")
        print(f"    changed {ctx} item set means the GATE moved, not the cap. Check that")
        print(f"    this call passed no --gate flag: the published Qwen runs used the")
        print(f"    default 'accuracy', and '--gate either' widens every context.")
        print(f"    Do not regenerate any table; patching_cap60_${STAMP} is the record.")
    if worst > 0.01:
        print(f"    {ctx} VALUES MOVED. These are frozen; do not update the paper from "
              f"this run, and keep patching_cap60_${STAMP} as the record.")
PY
note "6: tab:patchci must be regenerated with emit_patchci_table.py --write, and its caption note about the 60-item cap removed."

# --------------------------------- 5. Mistral, selection pass only --------------
# The evaluation pass needs a layer chosen from this sweep, which is a judgement. This
# archives the old run and produces the profile, then stops.
step "5  Mistral, archive then depth-matched selection pass"
if [ ! -d results/mistral24_mr/patching_unscaled_8_20 ]; then
  cp -r results/mistral24_mr/patching results/mistral24_mr/patching_unscaled_8_20
  python - <<'PY'
import pandas as pd
d = pd.read_csv("results/mistral24_mr/patching/causal_patching.csv")
ids = d[["context", "fact_id", "split"]].drop_duplicates()
ids.to_csv("results/mistral24_mr/patching_unscaled_8_20/seen_fact_ids.csv", index=False)
print("archived; seen items by split:")
print(ids.groupby(["context", "split"]).size())
PY
fi
# The selection pass is now driven by this file, so it has to exist. If an earlier
# archive was taken by hand it may not, and the run must not delete the only copy of
# the patching directory before finding that out.
SEEN=results/mistral24_mr/patching_unscaled_8_20/seen_fact_ids.csv
[ -f "$SEEN" ] || die "missing $SEEN; rebuild it from the archived causal_patching.csv"
rm -rf results/mistral24_mr/patching
# The never-patched set is 29 Marathi and 120 English items, and all of them sit in
# train and validation. Selecting over those splits would therefore consume the entire
# holdout and leave nothing for the evaluation pass this run is supposed to set up.
# Selecting over the items the earlier sweep already patched spends nothing new.
python 04_run_causal_patching.py --run-dir results/mistral24_mr --model mistral24 \
    --contexts mr en --layer-low 8 --layer-high 20 --scale-layers \
    --gate either --gate-polarity positive \
    --fact-ids results/mistral24_mr/patching_unscaled_8_20/seen_fact_ids.csv \
    --max-facts-per-context 141 \
    > "$OUT/5_mistral_selection.log" 2>&1
# This directory is deleted by the rm above on any later run, so keep the results here.
cp results/mistral24_mr/patching/causal_patching.csv "$OUT/5_mistral_patching.csv" || true
cp results/mistral24_mr/patching/windows_used.json "$OUT/5_windows_used.json" || true
report "5" <<'PY'
import json, pandas as pd
from pathlib import Path
# An empty gate makes stage 04 return before it writes anything, so read the directory
# rather than assume the sweep produced a profile.
if not Path("results/mistral24_mr/patching/windows_used.json").exists():
    # Exit zero so this lands in the notes file as a finding rather than as a crash.
    print("5: no windows_used.json, so stage 04 returned before patching. The gate was "
          "probably empty; read 5_mistral_selection.log before rerunning.")
    raise SystemExit(0)
w = json.load(open("results/mistral24_mr/patching/windows_used.json"))
print(f"5: windows_used scaled={w['scaled']} over {w['n_layers']} layers")
if not w["scaled"]:
    print("    NOT SCALED. The band is wrong again; do not use this run.")
d = pd.read_csv("results/mistral24_mr/patching/causal_patching.csv")
d["layer"] = pd.to_numeric(d.layer_or_window, errors="coerce")
s = d[(d.experiment == "single_factorial") & (d.truth == 1) & (d.site == "E2")
      & (d.source_condition == "LATLAT")]
for ctx, grp in s.groupby("context"):
    prof = grp.groupby("layer").delta_margin.mean()
    print(f"    {ctx}: E2 peaks at layer {int(prof.idxmax())} with {prof.max():.2f}")
print("5: DECISION NEEDED. Pick the layer from the profile above, write it down, then")
print("   run the single evaluation pass on the untouched items in the morning.")
PY

# --------------------------------------------------------------- wrap up --------
step "done"
cp "$NOTES" "$OUT/../DECISIONS_NEEDED_${STAMP}.txt" 2>/dev/null || true
printf '\nresults and logs: %s\n' "$OUT"
printf 'decisions waiting: %s\n\n' "$NOTES"
cat "$NOTES"
printf '\nWhat this run spent, exactly:\n'
printf '  no --allow-test anywhere, so stage 14 and stage 15 never saw test\n'
if [ -z "$ANCHOR_FORCE" ]; then
  printf '  stage 07 ran with --skip-test and reused its prompt anchors, so it did not\n'
  printf '    read the test split at all\n'
else
  printf '  stage 07 ran with --skip-test, so no scale was scored on test. Its anchor\n'
  printf '    block was recomputed and does score every split, test included. Nothing\n'
  printf '    is tuned on it, so it spends nothing, but the claim to keep for this run\n'
  printf '    is "no scale was scored on test", not "stage 07 never touched test".\n'
fi
printf '  4c used --splits train validation, so the nonce holdout is untouched\n'
printf '  5 patched only items the earlier Mistral sweep had already spent, so its\n'
printf '    29 Marathi and 120 English never-patched items are still available\n'
printf '  6 widened the gate across all three splits, which is what the published\n'
printf '    causal tables already do; it spent no split those tables had not\n'

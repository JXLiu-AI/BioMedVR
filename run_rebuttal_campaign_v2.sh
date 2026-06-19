#!/bin/bash
# Tighter ECCV 2026 BioMedVR rebuttal campaign — 200 epoch budget
# Prioritized by reviewer impact. ~35 runs total, ~3.5 days on 1 H20.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0

EPOCHS=200
PY="python experiments/fs_BiomedVR_V8_rebuttal2.py --epoch $EPOCHS"

# Datasets: priority subsets per experiment
DS_ALL=(busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina)
DS_CORE4=(busi btmri kvasir knee_xray)         # most-cited datasets in rebuttal
DS_PAIR=(busi btmri)                            # for noise tests

START=$(date +%s)
run() {
    local desc="$1"; shift
    local t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] START: $desc ====="
    echo "  CMD: $*"
    "$@" >> rebuttal_campaign_out/per_run.log 2>&1
    local t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s = $(((t1-START)/60))min) ====="
}

mkdir -p rebuttal_campaign_out

# ---------- Tier A: extra seed (s=2) for 3-seed CI on 11 datasets ----------
# Each run also logs CS-activation rate (free), so this also covers Tier C.
for ds in "${DS_ALL[@]}"; do
    run "A.seed2 ds=$ds" $PY --dataset $ds --seed 2 \
        --log_dir logs_rebuttal_5seed --exp_tag 5seed_s2
done

# ---------- Tier B: Adversarial negatives (random + shuffled × 4 core datasets) ----------
for mode in random shuffled; do
  for ds in "${DS_CORE4[@]}"; do
    run "B.neg=$mode ds=$ds" $PY --dataset $ds --seed 1 --neg_mode $mode \
        --log_dir logs_rebuttal_negmode --exp_tag neg_${mode}
  done
done

# ---------- Tier C: Mask sensitivity (3 non-default sizes × 2 datasets) ----------
for mb in 8 24 32; do
  for ds in "${DS_PAIR[@]}"; do
    run "C.mask=$mb ds=$ds" $PY --dataset $ds --seed 1 --mask_border $mb \
        --log_dir logs_rebuttal_mask --exp_tag mask_b${mb}
  done
done

# ---------- Tier D: Per-class gating (4 core datasets) ----------
for ds in "${DS_CORE4[@]}"; do
  run "D.perclass ds=$ds" $PY --dataset $ds --seed 1 --gating_type per_class \
      --log_dir logs_rebuttal_perclass --exp_tag perclass
done

# ---------- Tier E: Positive-expert noise robustness (3 rates × 2 datasets × {full, no_cs}) ----------
for rate in 0.25 0.50 0.75; do
  for ds in "${DS_PAIR[@]}"; do
    run "E.pos_corrupt=$rate ds=$ds full" $PY --dataset $ds --seed 1 --pos_corrupt_rate $rate \
        --log_dir logs_rebuttal_poscorrupt --exp_tag poscorrupt_r${rate}_full
    run "E.pos_corrupt=$rate ds=$ds no_cs" $PY --dataset $ds --seed 1 --pos_corrupt_rate $rate --no_cs \
        --log_dir logs_rebuttal_poscorrupt --exp_tag poscorrupt_r${rate}_nocs
  done
done

# ---------- Tier F: CT-Kidney attribute deduplication ----------
# (handled separately by ct_kidney_dedup.py if it exists; keep placeholder)

echo "===== ALL DONE total $(( $(date +%s) - START ))s ====="

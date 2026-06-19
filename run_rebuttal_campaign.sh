#!/bin/bash
# Comprehensive ECCV 2026 BioMedVR rebuttal campaign
# Runs sequentially on GPU 0, logs to logs_rebuttal_<exp>/, ckpts to results/fs_BioMedVR_rebuttal2/<exp>/
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0

DATASETS_ALL=(busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina)
DATASETS_MASK=(busi btmri kvasir knee_xray)            # for mask-sensitivity
DATASETS_NOISE=(busi btmri)                            # for pos-expert noise
DATASETS_CSLOG=(busi knee_xray kvasir btmri ctkidney dermamnist)  # for CS-activation logging
DATASETS_CSLOG_NATURAL=(caltech101)                    # natural-image baseline for activation comparison

PY="python experiments/fs_BiomedVR_V8_rebuttal2.py"
START=$(date +%s)

run() {
    local desc="$1"; shift
    local t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] START: $desc ====="
    echo "  CMD: $*"
    "$@" 2>&1 | tail -15
    local t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
}

# ---------- Tier 1: 5-seed CIs (3 new seeds × 11 datasets) ----------
for seed in 2 3 4; do
  for ds in "${DATASETS_ALL[@]}"; do
    run "5seed s=$seed ds=$ds" $PY --dataset $ds --seed $seed \
        --log_dir logs_rebuttal_5seed --exp_tag 5seed_s${seed}
  done
done

# ---------- Tier 2: Mask sensitivity (3 non-default sizes × 4 datasets) ----------
for mb in 8 24 32; do
  for ds in "${DATASETS_MASK[@]}"; do
    run "mask=$mb ds=$ds" $PY --dataset $ds --seed 1 --mask_border $mb \
        --log_dir logs_rebuttal_mask --exp_tag mask_b${mb}
  done
done

# ---------- Tier 3: Adversarial negatives (random + shuffled × 11 datasets) ----------
for mode in random shuffled; do
  for ds in "${DATASETS_ALL[@]}"; do
    run "neg=$mode ds=$ds" $PY --dataset $ds --seed 1 --neg_mode $mode \
        --log_dir logs_rebuttal_negmode --exp_tag neg_${mode}
  done
done

# ---------- Tier 4: Per-class gating (1 seed × 11 datasets) ----------
for ds in "${DATASETS_ALL[@]}"; do
  run "perclass-gate ds=$ds" $PY --dataset $ds --seed 1 --gating_type per_class \
      --log_dir logs_rebuttal_perclass --exp_tag perclass
done

# ---------- Tier 5: Positive-expert noise robustness (3 rates × 2 datasets × {full, no_cs}) ----------
for rate in 0.25 0.50 0.75; do
  for ds in "${DATASETS_NOISE[@]}"; do
    run "pos_corrupt=$rate ds=$ds full" $PY --dataset $ds --seed 1 --pos_corrupt_rate $rate \
        --log_dir logs_rebuttal_poscorrupt --exp_tag poscorrupt_r${rate}_full
    run "pos_corrupt=$rate ds=$ds no_cs" $PY --dataset $ds --seed 1 --pos_corrupt_rate $rate --no_cs \
        --log_dir logs_rebuttal_poscorrupt --exp_tag poscorrupt_r${rate}_nocs
  done
done

# ---------- Tier 6: CS-activation rate on 6 medical + 1 natural ----------
# (these reuse default settings + per-epoch CS logging which is already in the script)
for ds in "${DATASETS_CSLOG[@]}"; do
    run "cslog ds=$ds (medical)" $PY --dataset $ds --seed 1 \
        --log_dir logs_rebuttal_cslog --exp_tag cslog_medical
done
# Natural image dataset for comparison (Caltech101 — already has _confuse not for natural,
# so we skip if missing; the code will fall back to none-mode which still logs).
for ds in "${DATASETS_CSLOG_NATURAL[@]}"; do
    run "cslog ds=$ds (natural)" $PY --dataset $ds --seed 1 \
        --log_dir logs_rebuttal_cslog --exp_tag cslog_natural || echo "skipped $ds"
done

echo "===== ALL DONE total $(( $(date +%s) - START ))s ====="

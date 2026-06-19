#!/bin/bash
# Multi-seed BG-LM-BLMP + AttriPrompt × 11 medical ds × seeds {2, 3}
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=${BGLM_GPU:-0}

DS_ALL=(busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina)
mkdir -p rebuttal_campaign_out
START=$(date +%s)

# BG-LM-BLMP × 11 ds × 2 more seeds = 22 runs
for s in 2 3; do
  for ds in "${DS_ALL[@]}"; do
    t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] BGLM-MS START: ds=$ds map=blmp seed=$s ====="
    python run_bglm_medical.py --epoch 200 --eval_every 5 \
        --dataset $ds --mapping blmp --seed $s --log_dir logs_baselines_bglm \
        >> rebuttal_campaign_out/per_run_bglm_ms.log 2>&1 || echo "  FAILED"
    t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
  done
done

# AttriPrompt × 11 ds × 2 more seeds = 22 runs
for s in 2 3; do
  for ds in "${DS_ALL[@]}"; do
    t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] AP-MS START: ds=$ds seed=$s ====="
    python run_attriprompt.py --epoch 200 --eval_every 5 \
        --dataset $ds --seed $s --log_dir logs_baselines_attriprompt \
        >> rebuttal_campaign_out/per_run_attriprompt_ms.log 2>&1 || echo "  FAILED"
    t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
  done
done

echo "===== BASELINE-MS ALL DONE total $(( $(date +%s) - START ))s ====="

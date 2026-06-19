#!/bin/bash
# Extend BG-LM (3 mapping × 7 ds = 21 run) + AttriPrompt (7 ds) to all 11 medical datasets
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=${BGLM_GPU:-0}

REM_DS=(kvasir lung_colon octmnist chmnist covid_19 ctkidney retina)
BGLM_PY="python run_bglm_medical.py --epoch 200 --eval_every 5"
ATTRIPROMPT_PY="python run_attriprompt.py --epoch 200 --eval_every 5"

mkdir -p rebuttal_campaign_out
START=$(date +%s)

# BG-LM remaining 7 × 3
for ds in "${REM_DS[@]}"; do
  for m in ilm blm blmp; do
    t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] BGLM-EXT START: ds=$ds map=$m ====="
    $BGLM_PY --dataset $ds --mapping $m --seed 1 --log_dir logs_baselines_bglm \
        >> rebuttal_campaign_out/per_run_bglm_ext.log 2>&1 || echo "  FAILED ds=$ds map=$m"
    t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
  done
done

# AttriPrompt remaining 7
for ds in "${REM_DS[@]}"; do
    t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] AP-EXT START: ds=$ds ====="
    $ATTRIPROMPT_PY --dataset $ds --seed 1 --log_dir logs_baselines_attriprompt \
        >> rebuttal_campaign_out/per_run_attriprompt_ext.log 2>&1 || echo "  FAILED ds=$ds"
    t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
done

echo "===== BASELINE-EXT ALL DONE total $(( $(date +%s) - START ))s ====="

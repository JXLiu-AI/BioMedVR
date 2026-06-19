#!/bin/bash
# BG-LM baseline on 4 representative medical datasets, 3 mappings each = 12 runs.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=${BGLM_GPU:-0}

DATASETS=(busi btmri knee_xray dermamnist)
MAPPINGS=(ilm blm blmp)

mkdir -p rebuttal_campaign_out
PY="python run_bglm_medical.py --epoch 200 --eval_every 5"
START=$(date +%s)
for ds in "${DATASETS[@]}"; do
  for m in "${MAPPINGS[@]}"; do
    t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] BGLM START: ds=$ds map=$m ====="
    $PY --dataset $ds --mapping $m --seed 1 --log_dir logs_baselines_bglm \
        >> rebuttal_campaign_out/per_run_bglm.log 2>&1 || echo "  FAILED ds=$ds map=$m"
    t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
  done
done
echo "===== BGLM ALL DONE total $(( $(date +%s) - START ))s ====="

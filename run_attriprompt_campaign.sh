#!/bin/bash
# AttriPrompt baseline on 4 representative medical datasets.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=${BGLM_GPU:-0}

DATASETS=(busi btmri knee_xray dermamnist)
mkdir -p rebuttal_campaign_out
PY="python run_attriprompt.py --epoch 200 --eval_every 5"
START=$(date +%s)
for ds in "${DATASETS[@]}"; do
    t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] AttriPrompt START: ds=$ds ====="
    $PY --dataset $ds --seed 1 --log_dir logs_baselines_attriprompt \
        >> rebuttal_campaign_out/per_run_attriprompt.log 2>&1 || echo "  FAILED ds=$ds"
    t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s) ====="
done
echo "===== AttriPrompt ALL DONE total $(( $(date +%s) - START ))s ====="

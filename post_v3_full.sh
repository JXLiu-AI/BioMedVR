#!/bin/bash
# Unified post-v3: wait for v3 -> Caltech101 train -> consolidated ECE on all ckpts.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=1
V3_PID=$1
echo "[post-v3-full] waiting for v3 PID=$V3_PID..."
while kill -0 $V3_PID 2>/dev/null; do sleep 60; done
echo "[post-v3-full] v3 done at $(date)."

# ---- 1. Caltech101 CS-activation training run ----
echo "[post-v3-full] starting Caltech101 train..."
python experiments/fs_BiomedVR_V8_rebuttal3.py --epoch 200 --dataset caltech101 --seed 1 \
    --log_dir logs_rebuttal_cslog --exp_tag cslog_natural >> rebuttal_campaign_out/per_run_caltech.log 2>&1
echo "[post-v3-full] Caltech101 train done at $(date)."

# ---- 2. Consolidated ECE across ALL ckpts (V8 + rebuttal2) ----
echo "[post-v3-full] running consolidated ECE..."
> rebuttal_campaign_out/calibration_all.jsonl
for ckpt in $(find results/fs_BioMedVR-V8 results/fs_BioMedVR_rebuttal2 -name 'best.pth' 2>/dev/null); do
    base=$(basename $(dirname $ckpt))
    ds=$(echo $base | sed -E 's/^[0-9]+([a-z_]+)k[0-9].*/\1/')
    case "$ds" in
        busi|knee_xray|kvasir|lung_colon|octmnist|btmri|chmnist|covid_19|ctkidney|dermamnist|retina|caltech101) ;;
        *) echo "  skip $ckpt"; continue ;;
    esac
    echo "  ckpt=$ckpt ds=$ds"
    python experiments/eval_calibration.py --ckpt "$ckpt" --dataset $ds >> rebuttal_campaign_out/calibration_all.jsonl 2>&1 || echo "    failed $ckpt"
done
echo "[post-v3-full] all done at $(date)."

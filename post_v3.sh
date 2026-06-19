#!/bin/bash
# Post-v3: ECE/MCE/NLL on all v3 checkpoints.
# Waits for v3 PID to end, then runs calibration eval.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=1
V3_PID=$1
echo "[post-v3] waiting for v3 PID=$V3_PID..."
while kill -0 $V3_PID 2>/dev/null; do sleep 60; done
echo "[post-v3] v3 done at $(date). Running ECE on v3 ckpts."

# ECE eval on every v3 best.pth
> rebuttal_campaign_out/calibration_v3.jsonl
for ckpt in $(find results/fs_BioMedVR_rebuttal2 -name 'best.pth' 2>/dev/null); do
    base=$(basename $(dirname $ckpt))
    ds=$(echo $base | sed -E 's/^[0-9]+([a-z_]+)k[0-9].*/\1/')
    case "$ds" in
        busi|knee_xray|kvasir|lung_colon|octmnist|btmri|chmnist|covid_19|ctkidney|dermamnist|retina) ;;
        *) echo "  skip $ckpt"; continue ;;
    esac
    echo "  ckpt=$ckpt ds=$ds"
    python experiments/eval_calibration.py --ckpt "$ckpt" --dataset $ds >> rebuttal_campaign_out/calibration_v3.jsonl 2>&1 || echo "    failed"
done
echo "[post-v3] all done at $(date)."

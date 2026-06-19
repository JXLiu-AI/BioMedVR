#!/bin/bash
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0
OUTLOG=rebuttal_campaign_out/disagreement.log
echo "=== disagreement campaign $(date) ===" > $OUTLOG
for ds in busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina; do
    CKPT="results/fs_BioMedVR_rebuttal2/5seed_s2/16${ds}k3a0.5s2/best.pth"
    if [ -f "$CKPT" ]; then
        echo "[$(date +%H:%M:%S)] $ds" >> $OUTLOG
        python experiments/measure_disagreement.py --ckpt "$CKPT" --dataset $ds >> $OUTLOG 2>&1
    fi
done
echo "=== done $(date) ===" >> $OUTLOG

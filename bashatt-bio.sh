#!/usr/bin/env bash
# Sweep all 11 biomedical datasets with the canonical BioMedVR config.
# (Reproduces Table 1, BioMedVR row, on CLIP ViT-B/16.)

export HF_ENDPOINT=https://hf-mirror.com
GPU="${GPU:-0}"

for ds in busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina; do
    echo "===== $ds ====="
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py --method biomedvr --dataset $ds --shot 16 --epoch 400
done

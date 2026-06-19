#!/bin/bash
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=1
echo "=== covid_19 gating tracking start: $(date) ==="
python experiments/fs_BiomedVR_V8_rebuttal3.py \
    --dataset covid_19 --seed 2 --epoch 200 \
    --log_dir logs_rebuttal_5seed --exp_tag 5seed_s2 \
    > rebuttal_campaign_out/covid_gating.log 2>&1
echo "=== covid_19 done: $(date) ==="

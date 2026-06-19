#!/bin/bash
# CT-Kidney attribute deduplication ablation
# Uses pre-built attributes/gpt3/gpt3_dedup/ctkidney_confuse.json
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs_rebuttal_ctk_dedup attributes_temp

# Symlink dedup attributes into a fake dataset name 'ctkidneyD' so the loader will find them
ln -sf $(pwd)/attributes/gpt3/gpt3_dedup/ctkidney_confuse.json $(pwd)/attributes/gpt3/gpt3/_ctkidneyD_confuse.json 2>/dev/null

# Just point to the deduped path by overriding via a custom dataset?
# Simpler: temporarily backup the original and swap
cp attributes/gpt3/gpt3/ctkidney_confuse.json attributes/gpt3/gpt3/ctkidney_confuse.json.orig_bak
cp attributes/gpt3/gpt3_dedup/ctkidney_confuse.json attributes/gpt3/gpt3/ctkidney_confuse.json

python experiments/fs_BiomedVR_V8_rebuttal2.py --epoch 200 --dataset ctkidney --seed 1 \
    --log_dir logs_rebuttal_ctk_dedup --exp_tag ctk_dedup 2>&1 | tail -10

# Restore
mv attributes/gpt3/gpt3/ctkidney_confuse.json.orig_bak attributes/gpt3/gpt3/ctkidney_confuse.json
echo "restored original ctkidney_confuse.json"

#!/bin/bash
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=1
OUTLOG=rebuttal_campaign_out/crosstask_v1.log
echo "=== crosstask start: $(date) ===" > $OUTLOG

# Pairs: (src, tgt) — chosen for variety: same-organ, same-modality, cross-modality, cross-organ
PAIRS=(
    "octmnist:retina"        # same organ (eye), different modality (OCT vs fundus)
    "chmnist:lung_colon"     # same modality (histopath), different tissue
    "btmri:ctkidney"         # cross-organ (brain MRI -> kidney CT)
    "busi:btmri"             # cross-modality + cross-organ
    "kvasir:lung_colon"      # endoscopy -> histopath
)

for pair in "${PAIRS[@]}"; do
    IFS=':' read -r src tgt <<< "$pair"
    CKPT="results/fs_BioMedVR_rebuttal2/5seed_s2/16${src}k3a0.5s2/best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "MISS ckpt: $CKPT" >> $OUTLOG; continue
    fi
    echo "[$(date +%H:%M:%S)] EVAL ${src} -> ${tgt}" >> $OUTLOG
    python experiments/eval_crosstask.py --src_ckpt "$CKPT" --tgt_dataset $tgt \
        >> $OUTLOG 2>&1
done

echo "=== crosstask done: $(date) ===" >> $OUTLOG

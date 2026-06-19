#!/bin/bash
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=1
OUTLOG=rebuttal_campaign_out/crosstask_v2.log
echo "=== crosstask v2 start: $(date) ===" > $OUTLOG

PAIRS=(
    "lung_colon:chmnist"     # 同病理
    "ctkidney:btmri"         # 反向：器官断层
    "retina:octmnist"        # 同眼科
)

for pair in "${PAIRS[@]}"; do
    IFS=':' read -r src tgt <<< "$pair"
    CKPT="results/fs_BioMedVR_rebuttal2/5seed_s2/16${src}k3a0.5s2/best.pth"
    if [ ! -f "$CKPT" ]; then echo "MISS ckpt: $CKPT" >> $OUTLOG; continue; fi
    echo "[$(date +%H:%M:%S)] EVAL ${src} -> ${tgt}" >> $OUTLOG
    python experiments/eval_crosstask.py --src_ckpt "$CKPT" --tgt_dataset $tgt >> $OUTLOG 2>&1
done
echo "=== done: $(date) ===" >> $OUTLOG

#!/bin/bash
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0

EPOCHS=200
PY="python experiments/fs_BiomedVR_V8_rebuttal3.py --epoch $EPOCHS"
LOGDIR=logs_rebuttal_v4
OUTDIR=rebuttal_campaign_out
mkdir -p $LOGDIR $OUTDIR
START=$(date +%s)

run() {
    local desc="$1"; shift
    local t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] V4 START: $desc ====="
    echo "  CMD: $*"
    "$@" >> $OUTDIR/per_run_v4.log 2>&1
    local t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s = $(((t1-START)/60))min) ====="
}

# ---------- #5: CT-K LLM-audit dedup ----------
echo "## STAGE 1: CT-K audit ##"
ATTR_DEFAULT=attributes/gpt3/gpt3/ctkidney_confuse.json
ATTR_AUDIT=attributes/gpt3/gpt3_audit/ctkidney_confuse.json
ATTR_BACKUP=$OUTDIR/ctkidney_confuse.orig.json
cp $ATTR_DEFAULT $ATTR_BACKUP
cp $ATTR_AUDIT  $ATTR_DEFAULT
run "5.audit ctkidney" $PY --dataset ctkidney --seed 1 --log_dir $LOGDIR --exp_tag audit_ctk
cp $ATTR_BACKUP $ATTR_DEFAULT
echo "## STAGE 1 DONE ##"

# ---------- #6: beta-sweep on remaining 7 ds ----------
DS_REM7=(octmnist btmri chmnist covid_19 ctkidney dermamnist retina)
BETAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
echo "## STAGE 2: beta-sweep ##"
for ds in "${DS_REM7[@]}"; do
    for b in "${BETAS[@]}"; do
        run "6.beta=$b $ds" $PY --dataset $ds --seed 1 --beta $b \
            --log_dir $LOGDIR --exp_tag beta${b}_${ds}
    done
done
echo "## ALL V4 DONE ($(((($(date +%s) - START))/60)) min) ##"

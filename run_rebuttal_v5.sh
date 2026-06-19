#!/bin/bash
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0
EPOCHS=200
OUTDIR=rebuttal_campaign_out
mkdir -p $OUTDIR
START=$(date +%s)

run() {
    local desc="$1"; shift
    local t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] V5: $desc ====="
    "$@" >> $OUTDIR/per_run_v5.log 2>&1
    echo "  END ($((($(date +%s)-t0)))s, total $((($(date +%s)-START)/60))min)"
}

# ---------- Stage A: train AttrVR on 4 missing ds (seed 0, 16-shot) ----------
echo '## STAGE A: AttrVR train (4 ds × seed 0) ##'
ATTRVR="python experiments/fs_attrvr_rebuttal.py --shot 16 --epoch $EPOCHS --seed 0"
for ds in octmnist covid_19 ctkidney dermamnist; do
    run "A.attrvr $ds" $ATTRVR --dataset $ds
done

# ---------- Stage A.eval: AttrVR ECE/MCE on all 11 ds ----------
echo '## STAGE A.eval: AttrVR ECE eval (11 ds) ##'
EVAL="python experiments/eval_calibration_attrvr.py --shot 16 --seed 0"
for ds in busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina; do
    CKPT="results/fs_attrvr/16${ds}k3a0.5s0/best.pth"
    if [ -f "$CKPT" ]; then
        run "A.eval $ds" $EVAL --dataset $ds --ckpt "$CKPT"
    else
        echo "  SKIP $ds: $CKPT missing"
    fi
done
echo "## V5 ALL DONE: $((($(date +%s)-START)/60)) min ##"

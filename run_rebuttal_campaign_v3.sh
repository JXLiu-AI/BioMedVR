#!/bin/bash
# v3: fill remaining gaps after v2.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=1

EPOCHS=200
PY="python experiments/fs_BiomedVR_V8_rebuttal3.py --epoch $EPOCHS"

DS_ALL=(busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina)
DS_REM7=(lung_colon octmnist chmnist covid_19 ctkidney dermamnist retina)  # not in v2 Tier B/D
DS_CORE4=(busi btmri kvasir knee_xray)
DS_PAIR=(busi btmri)

START=$(date +%s)
mkdir -p rebuttal_campaign_out
run() {
    local desc="$1"; shift
    local t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] V3 START: $desc ====="
    echo "  CMD: $*"
    "$@" >> rebuttal_campaign_out/per_run_v3.log 2>&1
    local t1=$(date +%s)
    echo "===== END ($((t1-t0))s, total $((t1-START))s = $(((t1-START)/60))min) ====="
}

# ---------- A2: seed=3 × 11 ds ----------
for ds in "${DS_ALL[@]}"; do
    run "A2.s3 $ds" $PY --dataset $ds --seed 3 --log_dir logs_rebuttal_5seed --exp_tag 5seed_s3
done
# ---------- A3: seed=4 × 11 ds ----------
for ds in "${DS_ALL[@]}"; do
    run "A3.s4 $ds" $PY --dataset $ds --seed 4 --log_dir logs_rebuttal_5seed --exp_tag 5seed_s4
done

# ---------- B2: random + shuffled × remaining 7 ds ----------
for mode in random shuffled; do
    for ds in "${DS_REM7[@]}"; do
        run "B2.neg=$mode $ds" $PY --dataset $ds --seed 1 --neg_mode $mode \
            --log_dir logs_rebuttal_negmode --exp_tag neg_${mode}
    done
done

# ---------- C2: mask kvasir + full-overlay × 2 + learned-mask × 2 ----------
for mb in 8 24 32; do
    run "C2.mask=$mb kvasir" $PY --dataset kvasir --seed 1 --mask_border $mb \
        --log_dir logs_rebuttal_mask --exp_tag mask_b${mb}
done
for ds in busi btmri; do
    run "C2.full $ds" $PY --dataset $ds --seed 1 --mask_mode full \
        --log_dir logs_rebuttal_mask --exp_tag mask_full
    run "C2.learned $ds" $PY --dataset $ds --seed 1 --mask_mode learned \
        --log_dir logs_rebuttal_mask --exp_tag mask_learned
done

# ---------- D2: per-class × remaining 7 ds + per-sample × 4 core ds ----------
for ds in "${DS_REM7[@]}"; do
    run "D2.perclass $ds" $PY --dataset $ds --seed 1 --gating_type per_class \
        --log_dir logs_rebuttal_perclass --exp_tag perclass
done
for ds in "${DS_CORE4[@]}"; do
    run "D2.persample $ds" $PY --dataset $ds --seed 1 --gating_type per_sample \
        --log_dir logs_rebuttal_persample --exp_tag persample
done

# ---------- G: --no_cs baseline × 9 remaining ds ----------
for ds in kvasir knee_xray lung_colon octmnist chmnist covid_19 ctkidney dermamnist retina; do
    run "G.no_cs $ds" $PY --dataset $ds --seed 1 --no_cs \
        --log_dir logs_rebuttal_nocs --exp_tag nocs
done

echo "===== V3 ALL DONE total $(( $(date +%s) - START ))s ====="

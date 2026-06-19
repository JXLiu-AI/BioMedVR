#!/bin/bash
# Post-campaign add-on runs: CT-Kidney dedup, cross-task transfer, ECE eval
# Runs after main campaign finishes.
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs_rebuttal_post

START=$(date +%s)
PY="python"

# ---------- CT-Kidney dedup (1 run) ----------
echo "===== POST.1: CT-Kidney dedup ====="
cp attributes/gpt3/gpt3/ctkidney_confuse.json attributes/gpt3/gpt3/ctkidney_confuse.json.bak
cp attributes/gpt3/gpt3_dedup/ctkidney_confuse.json attributes/gpt3/gpt3/ctkidney_confuse.json
$PY experiments/fs_BiomedVR_V8_rebuttal2.py --epoch 200 --dataset ctkidney --seed 1 \
    --log_dir logs_rebuttal_post --exp_tag ctk_dedup
mv attributes/gpt3/gpt3/ctkidney_confuse.json.bak attributes/gpt3/gpt3/ctkidney_confuse.json
echo "CT-Kidney attrs restored."

# ---------- Cross-task transfer (load existing src ckpts; fast inference-only) ----------
echo "===== POST.2: Cross-task transfer ====="
declare -a PAIRS=(
    "results/fs_BioMedVR-V8/16busik3a0.5s1/best.pth:btmri"
    "results/fs_BioMedVR-V8/16knee_xrayk3a0.5s1/best.pth:covid_19"
    "results/fs_BioMedVR-V8/16kvasirk3a0.5s0/best.pth:lung_colon"
    "results/fs_BioMedVR-V8/16kvasirk3a0.5s1/best.pth:chmnist"
    "results/fs_BioMedVR-V8/16lung_colonk3a0.5s0/best.pth:chmnist"
)
for pair in "${PAIRS[@]}"; do
    src=${pair%:*}
    tgt=${pair#*:}
    echo "  -- src=$src tgt=$tgt --"
    $PY experiments/eval_crosstask.py --src_ckpt "$src" --tgt_dataset "$tgt" >> logs_rebuttal_post/crosstask.jsonl 2>&1
done
echo "cross-task done."

# ---------- ECE post-hoc on all available best.pth ----------
echo "===== POST.3: Calibration ECE/MCE/NLL ====="
> logs_rebuttal_post/calibration.jsonl
for ckpt in $(find results/fs_BioMedVR-V8 results/fs_BioMedVR_rebuttal2 -name 'best.pth'); do
    # Parse dataset name from path (e.g., '16busik3a0.5s1' -> 'busi')
    base=$(basename $(dirname $ckpt))
    # Extract dataset: between leading shot-number digits and 'k'
    ds=$(echo $base | sed -E 's/^[0-9]+([a-z_]+)k[0-9].*/\1/')
    # Sanity check
    case "$ds" in
        busi|knee_xray|kvasir|lung_colon|octmnist|btmri|chmnist|covid_19|ctkidney|dermamnist|retina) ;;
        *) echo "  skip unknown dataset '$ds' from $ckpt"; continue ;;
    esac
    echo "  ckpt=$ckpt ds=$ds"
    $PY experiments/eval_calibration.py --ckpt "$ckpt" --dataset $ds >> logs_rebuttal_post/calibration.jsonl 2>&1 || echo "    failed"
done
echo "calibration done."

echo "===== POST-CAMPAIGN ALL DONE total $(( $(date +%s) - START ))s ====="

# ---------- POST.4: BG-LM baselines ----------
echo '===== POST.4: BG-LM baselines ====='
bash run_bglm_campaign.sh
echo '===== POST.4 done ====='

# ---------- POST.5: AttriPrompt baselines ----------
echo '===== POST.5: AttriPrompt baselines ====='
bash run_attriprompt_campaign.sh
echo '===== POST.5 done ====='

# ---------- POST.6: cross-vendor LLM (Qwen + DeepSeek) ----------
echo '===== POST.6: Cross-vendor LLM (Qwen + DeepSeek) ====='
bash run_llm_crossvendor.sh
echo '===== POST.6 done ====='

# ---------- POST.7: extend BG-LM+AttriPrompt to 11 ds ----------
echo '===== POST.7: Baseline extension (BG-LM + AttriPrompt × 7 more ds) ====='
bash run_baselines_extend.sh
echo '===== POST.7 done ====='

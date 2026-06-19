#!/bin/bash
# Cross-vendor LLM attribute robustness:
# Train BioMedVR with confuse attrs from Qwen3-235B and DeepSeek-Chat (11 ds × 2 LLMs = 22 runs)
set -u
cd ~/bio/AttrVR-main
source ~/miniconda3/bin/activate reprogram
export CUDA_VISIBLE_DEVICES=${LLMCV_GPU:-0}

DATASETS=(busi knee_xray kvasir lung_colon octmnist btmri chmnist covid_19 ctkidney dermamnist retina)
PY="python experiments/fs_BiomedVR_V8_rebuttal3.py --epoch 200 --eval_every 5"

mkdir -p rebuttal_campaign_out logs_rebuttal_llm_crossvendor
START=$(date +%s)

# Helper: temporarily swap confuse.json for one ds, run, restore
run_with_attr() {
    local llm=$1 ds=$2
    local orig=attributes/gpt3/gpt3/${ds}_confuse.json
    local alt=attributes/gpt3/gpt3_${llm}/${ds}_confuse.json
    if [ ! -f "$alt" ]; then echo "  alt attr missing: $alt; skip"; return; fi
    cp "$orig" "${orig}.crossvendor.bak"
    cp "$alt" "$orig"
    local t0=$(date +%s)
    echo "===== [$(date +%H:%M:%S)] LLMCV START: llm=$llm ds=$ds ====="
    $PY --dataset $ds --seed 1 \
        --log_dir logs_rebuttal_llm_crossvendor --exp_tag llmcv_${llm} \
        >> rebuttal_campaign_out/per_run_llmcv.log 2>&1 || echo "  FAILED ds=$ds llm=$llm"
    local t1=$(date +%s)
    echo "===== END ($((t1-t0))s) ====="
    mv "${orig}.crossvendor.bak" "$orig"
}

for llm in qwen deepseek; do
  for ds in "${DATASETS[@]}"; do
    run_with_attr $llm $ds
  done
done

echo "===== LLMCV ALL DONE total $(( $(date +%s) - START ))s ====="

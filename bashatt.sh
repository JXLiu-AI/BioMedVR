#!/usr/bin/env bash
# Sweep natural-image benchmarks for any chosen method.
# Usage: METHOD=biomedvr GPU=0 bash bashatt.sh
METHOD="${METHOD:-biomedvr}"
GPU="${GPU:-0}"
for ds in caltech101 food101 dtd eurosat oxford_pets oxford_flowers ucf101; do
    echo "===== $METHOD · $ds ====="
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py --method $METHOD --dataset $ds --shot 16 --epoch 400
done

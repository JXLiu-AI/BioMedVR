# CUDA_VISIBLE_DEVICES=0 python3 experiments/fs_ar_rebuttal_lora.py --dataset busi
# CUDA_VISIBLE_DEVICES=0 python3 experiments/fs_ar_rebuttal_lora.py --dataset knee_xray 
# CUDA_VISIBLE_DEVICES=0 python3 experiments/fs_ar_rebuttal_lora.py --dataset kvasir

CUDA_VISIBLE_DEVICES=1 python3 experiments/fs_BiomedVR-V8-rebuttal.py --dataset busi 
# CUDA_VISIBLE_DEVICES=1 python3 experiments/fs_BiomedVR-V8.py --dataset knee_xray 
# CUDA_VISIBLE_DEVICES=1 python3 experiments/fs_BiomedVR-V8.py --dataset kvasir
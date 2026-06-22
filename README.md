# BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for Biomedical Visual Reprogramming

Official code release.

> Jiaxiang Liu, Tianxiang Hu, Juwei Guan, Yujie Wu, Yusong Wang, Yao Mu, Zuozhu Liu, Mingkun Xu

[![Paper (arXiv)](https://img.shields.io/badge/arXiv-coming%20soon-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## TL;DR

BioMedVR is the **first visual-reprogramming (VR) framework tailored for biomedical imaging**. It tackles the confusion problem common in fine-grained medical recognition by:

1. **Mixture-of-Prompt Experts (MoPE)** — a *positive* expert for discriminative alignment + a *negative* expert that focuses on confusion suppression, balanced by an adaptive gating vector;
2. **Confusion-aware Attributes (ConfAttrs)** — LLM-generated text prompts that describe *visually similar but semantically incorrect* categories per class;
3. **Confusion-Suppression Loss (CS Loss)** — a margin-based loss that explicitly penalizes false-positive alignment with confusable classes.

The whole stack is **input-space**, **VLM-architecture-agnostic**, and **privacy-preserving** (frozen backbones), making it well suited for clinical adaptation.

## Repository layout

```
.
├── train.py                          # ← Unified launcher: python train.py --method ...
├── experiments/                      # method-specific training entrypoints
│   ├── biomedvr.py                   # canonical script that produced Table 1 (CLIP ViT-B/16)
│   ├── biomedvr_biomedclip.py        # BioMedVR on BiomedCLIP backbone
│   ├── biomedvr_v9_ablation.py       # MoPE-gating ablation
│   ├── biomedvr_wo_confuse.py        # ablation: BioMedVR without the confusion mechanism
│   ├── baseline_ar.py                # AR baseline
│   ├── baseline_attrvr.py            # AttrVR baseline (CLIP ViT-B/16)
│   ├── baseline_attrvr_biomedclip.py # AttrVR baseline (BiomedCLIP backbone)
│   ├── baseline_vp.py                # VP (Visual Prompt) baseline
│   ├── baseline_biomedclip.py        # BiomedCLIP linear-probe baseline
│   ├── eval_calibration.py           # eval: calibration / ECE
│   └── eval_crosstask.py             # eval: cross-task transfer
├── methods/                          # visual reprogramming + MoPE modules
├── datasets/                         # PyTorch wrappers for 11 medical + 7 natural benchmarks
├── attributes/                       # GPT-generated positive/descriptive attributes per dataset (json)
├── attributes_corrupted/             # corrupted-attribute ablations
├── attributes_neg_variants/          # confusion-aware attribute variants (ConfAttrs)
├── gen_confuse.py                    # LLM script: generate ConfAttrs per class via OpenAI API
├── generate_attributes.py            # LLM script: generate positive attributes
├── eval_confusion.py                 # diagnostics: per-class confusion matrices, calibration
├── tools.py                          # helpers (set_seed, schedulers, metric utils)
├── cfg.py                            # global DOWNSTREAM_PATH (set this before running)
├── bashatt.sh / bashatt-bio.sh       # convenience launchers
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Python env
conda create -n biomedvr python=3.10 -y
conda activate biomedvr

# 2. Dependencies
pip install -r requirements.txt
```

GPU: a single RTX 3090 / A6000 / H20 (8 GB+) is enough for 16-shot ViT-B/16.

## Data preparation

BioMedVR uses **11 biomedical** and **7 natural-image** benchmarks. See `datasets/build_loader.py` for the expected directory tree.

| Modality | Datasets |
|---|---|
| Medical | BUSI · Knee X-ray · Kvasir · LungColon · OCTMNIST · BTMRI · CHMNIST · COVID-19 · CT-Kidney · DermaMNIST · Retina |
| Natural | Caltech101 · Food101 · DTD · EuroSAT · Oxford-Pets · Oxford-Flowers · UCF101 |

Update the global path in `cfg.py`:

```python
DOWNSTREAM_PATH = "/path/to/your/datasets"
```

The medical datasets are not redistributed here. Download links and split files are documented per dataset in `datasets/*.py`.

## Quick start

All experiments are dispatched through the unified launcher **`train.py`**,
which forwards arguments to the matching script under `experiments/`.

```bash
# Canonical BioMedVR (paper Table 1), single dataset, 16-shot
python train.py --method biomedvr --dataset busi --shot 16 --epoch 400

# Variants
python train.py --method biomedvr-biomedclip  --dataset busi   # BiomedCLIP backbone
python train.py --method biomedvr-no-confuse  --dataset busi   # CS-Loss ablation
python train.py --method biomedvr-v9          --dataset busi   # MoPE-gating ablation

# Baselines
python train.py --method attrvr --dataset busi
python train.py --method vp     --dataset busi
python train.py --method ar     --dataset busi

# Sweep all 11 medical datasets via the bundled bash script
bash bashatt-bio.sh
```

You can also invoke a specific script directly (legacy):

```bash
python experiments/biomedvr.py --dataset busi --shot 16 --epoch 400
```

**Key arguments** (defaults match the paper):

| arg | default | meaning |
|---|---|---|
| `--dataset` | `dtd` | one of the 18 datasets listed above |
| `--shot` | `16` | few-shot training set size per class |
| `--epoch` | `400` | total epochs |
| `--lr` | `40` | initial LR (cosine annealing) |
| `--input_size` | `192` | reprogrammed-region size (224 input, padding frame = 16) |
| `--alpha` | `0.5` | λ balance weight between positive / negative expert |
| `--beta` | `0.3` | CS-loss weight (β in paper) |
| `--margin` | `0.5` | CS-loss margin (m in paper) |
| `--num_attr` | `20` | number of descriptive attributes per class |
| `--k` | `3` | top-k for kNN attribute selection |
| `--seed` | `1` | RNG seed |

### Generating ConfAttrs from scratch

```bash
export OPENAI_API_KEY=sk-...
python3 gen_confuse.py busi --num 5
```

This writes `attributes/gpt3/<dataset>_confuse.json` containing the top-5 confusion-aware
negative descriptions per class.

## Citation

```bibtex
@inproceedings{liu2026biomedvr,
  title     = {BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for
               Biomedical Visual Reprogramming},
  author    = {Liu, Jiaxiang and Hu, Tianxiang and Guan, Juwei and Wu, Yujie
               and Wang, Yusong and Mu, Yao and Liu, Zuozhu and Xu, Mingkun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

This code is released under the MIT License (see `LICENSE`). It builds on
[AttrVR](https://github.com/AttrVR/AttrVR-main) (the visual-reprogramming
backbone) and [CLIP](https://github.com/openai/CLIP) — both MIT-licensed.

## Acknowledgements

We thank the maintainers of AttrVR, CLIP, BiomedCLIP, and all the dataset
contributors that made this work possible. Compute generously provided by the
Guangdong Institute of Intelligence Science and Technology (GDIIST) and
Zhejiang University.

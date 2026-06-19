# BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for Biomedical Visual Reprogramming

**ECCV 2026** · Official code release

> Jiaxiang Liu, Tianxiang Hu, Juwei Guan, Yujie Wu, Yusong Wang, Yao Mu, Zuozhu Liu, Mingkun Xu

[![Paper (arXiv)](https://img.shields.io/badge/arXiv-coming%20soon-red)]()
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-blue)]()
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
├── experiments/                 # training / eval entry points (see Usage)
│   └── fs_BiomedVR-V8.py        # canonical script that produced Table 1 numbers
├── methods/                     # visual reprogramming + MoPE modules
├── datasets/                    # PyTorch dataset wrappers for 11 medical + 7 natural benchmarks
├── attributes/                  # GPT-generated positive/descriptive attributes per dataset (json)
├── attributes_corrupted/        # corrupted-attribute ablations
├── attributes_neg_variants/     # confusion-aware attribute variants (ConfAttrs)
├── gen_confuse.py               # LLM script: generate ConfAttrs per class via the OpenAI API
├── generate_attributes.py       # LLM script: generate positive attributes
├── eval_confusion.py            # diagnostics: per-class confusion matrices, calibration
├── bashatt-bio.sh               # convenience launcher for all 11 medical datasets
├── tools.py                     # helpers (set_seed, schedulers, metric utils)
├── cfg.py                       # global DOWNSTREAM_PATH (set this before running)
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

The script that produced Table 1 of the paper is `experiments/fs_BiomedVR-V8.py`.

```bash
# Single-dataset, 16-shot, default hyperparameters
python3 experiments/fs_BiomedVR-V8.py --dataset busi --shot 16 --epoch 400

# All 11 medical datasets
bash bashatt-bio.sh
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

## Reproduction (16-shot, ViT-B/16)

| Dataset | Paper (Table 1) | This repo |
|---|---:|---:|
| BUSI       | 82.60 | _to be filled by reproduction run_ |
| Knee X-ray | 45.74 | _to be filled_ |
| Kvasir     | 80.22 | _to be filled_ |

(Hyperparameters: `--shot 16 --epoch 400 --seed 1`, all other defaults as above.)

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

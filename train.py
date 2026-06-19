#!/usr/bin/env python3
"""
Unified entry point for BioMedVR training.

Dispatches to the matching script under ``experiments/`` based on ``--method``.
This is a thin wrapper that forwards all the unknown arguments to the chosen
script, so any ``--dataset``, ``--shot``, ``--seed`` etc. keep working.

Examples
--------
    # BioMedVR (canonical, paper Table 1)
    python train.py --method biomedvr --dataset busi --shot 16

    # Ablation: BioMedVR w/o confusion-aware attributes
    python train.py --method biomedvr-no-confuse --dataset busi

    # Baseline: AttrVR
    python train.py --method attrvr --dataset busi

    # Same hyperparameters but on the BiomedCLIP backbone
    python train.py --method biomedvr-biomedclip --dataset busi
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Map user-friendly method name → experiment script
METHOD_SCRIPTS: dict[str, str] = {
    # BioMedVR family
    "biomedvr": "experiments/biomedvr.py",  # canonical, CLIP ViT-B/16
    "biomedvr-biomedclip": "experiments/biomedvr_biomedclip.py",  # BiomedCLIP backbone
    "biomedvr-v9": "experiments/biomedvr_v9_ablation.py",  # MoPE ablation
    "biomedvr-no-confuse": "experiments/biomedvr_wo_confuse.py",  # CS-Loss ablation
    # Baselines
    "ar": "experiments/baseline_ar.py",
    "attrvr": "experiments/baseline_attrvr.py",
    "attrvr-biomedclip": "experiments/baseline_attrvr_biomedclip.py",
    "vp": "experiments/baseline_vp.py",
    "biomedclip": "experiments/baseline_biomedclip.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioMedVR unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available methods:\n  "
            + "\n  ".join(f"{k:<22s} → {v}" for k, v in METHOD_SCRIPTS.items())
        ),
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHOD_SCRIPTS),
        help="Training method to run (each maps to one script in experiments/).",
    )
    known, forwarded = parser.parse_known_args()

    script = REPO_ROOT / METHOD_SCRIPTS[known.method]
    if not script.exists():
        raise FileNotFoundError(f"Could not find {script}")

    # Forward all unknown args (--dataset, --shot, --seed, ...) to the chosen script.
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

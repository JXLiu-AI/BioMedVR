"""Update F3B_baselines.csv with 3-seed mean for BG-LM-BLMP and AttriPrompt."""

import csv
import os
import statistics
from collections import defaultdict

ROOT = os.path.expanduser("~/bio/AttrVR-main")
DATASETS_11 = [
    "busi",
    "knee_xray",
    "kvasir",
    "lung_colon",
    "octmnist",
    "btmri",
    "chmnist",
    "covid_19",
    "ctkidney",
    "dermamnist",
    "retina",
]


def best_from_log(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        last = None
        for line in f:
            if "Best Acc=" in line:
                last = line
    return float(last.split("Best Acc=")[1].split(",")[0]) * 100 if last else None


# baseline_path = use original v8 + new 5-seed for BioMedVR
def biomedvr_5seed_mean(ds):
    vals = []
    for s, d, fmt in [
        (0, "logs_BiomedVR_all_h20_v9_new", "plain"),
        (1, "logs_BiomedVR_all_h20_v8_new", "plain"),
    ] + [(s, "logs_rebuttal_5seed", "seed") for s in [2, 3, 4]]:
        p = f"{ROOT}/{d}/{ds}.log" if fmt == "plain" else f"{ROOT}/{d}/{ds}_seed{s}.log"
        v = best_from_log(p)
        if v is not None:
            vals.append(v)
    return statistics.mean(vals) if vals else None


out = f"{ROOT}/rebuttal_figdata/F3B_baselines_v2.csv"
with open(out, "w") as fout:
    fout.write("method,dataset,best_acc\n")
    for ds in DATASETS_11:
        # BioMedVR (5-seed mean)
        v = biomedvr_5seed_mean(ds)
        if v is not None:
            fout.write(f"BioMedVR,{ds},{v:.2f}\n")
        # AttriPrompt (3-seed mean)
        ap_vals = [
            best_from_log(f"{ROOT}/logs_baselines_attriprompt/{ds}_seed{s}.log") for s in [1, 2, 3]
        ]
        ap_vals = [v for v in ap_vals if v is not None]
        if ap_vals:
            fout.write(f"AttriPrompt,{ds},{statistics.mean(ap_vals):.2f}\n")
        # BG-LM (3 mappings; ILM/BLM 1-seed, BLMP 3-seed)
        for m in ["ilm", "blm"]:
            v = best_from_log(f"{ROOT}/logs_baselines_bglm/{ds}_{m}_seed1.log")
            if v is not None:
                fout.write(f"BG-LM-{m.upper()},{ds},{v:.2f}\n")
        # BLMP 3-seed mean
        blmp_vals = [
            best_from_log(f"{ROOT}/logs_baselines_bglm/{ds}_blmp_seed{s}.log") for s in [1, 2, 3]
        ]
        blmp_vals = [v for v in blmp_vals if v is not None]
        if blmp_vals:
            fout.write(f"BG-LM-BLMP,{ds},{statistics.mean(blmp_vals):.2f}\n")

print(f"wrote {out}")

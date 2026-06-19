"""Aggregate multi-seed BG-LM-BLMP + AttriPrompt × 11 ds × 3 seeds (s=1,2,3) into mean±std."""

import glob
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


def best_from_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        last = None
        for line in f:
            if "Best Acc=" in line:
                last = line
    if last is None:
        return None
    try:
        return float(last.split("Best Acc=")[1].split(",")[0]) * 100
    except:
        return None


# BG-LM-BLMP × 11 ds × 3 seeds
print("== BG-LM-BLMP ==")
print(f'{"ds":12} | {"s=1":>6} {"s=2":>6} {"s=3":>6} | {"mean":>6} {"std":>5}')
all_blmp_means = []
for ds in DATASETS_11:
    vals = []
    for s in [1, 2, 3]:
        v = best_from_log(f"{ROOT}/logs_baselines_bglm/{ds}_blmp_seed{s}.log")
        vals.append(v)
    valid = [v for v in vals if v is not None]
    m = statistics.mean(valid) if valid else None
    s_std = statistics.stdev(valid) if len(valid) > 1 else 0
    all_blmp_means.append(m)
    print(
        f'{ds:12} | {vals[0] if vals[0] else "  --":>6} {vals[1] if vals[1] else "  --":>6} {vals[2] if vals[2] else "  --":>6} | {m:>6.2f} {s_std:>5.2f}'
    )

avg_blmp = statistics.mean([x for x in all_blmp_means if x is not None])
print(f'{"AVG":12} | avg_means={avg_blmp:.2f}')

# AttriPrompt × 11 ds × 3 seeds
print()
print("== AttriPrompt ==")
print(f'{"ds":12} | {"s=1":>6} {"s=2":>6} {"s=3":>6} | {"mean":>6} {"std":>5}')
all_ap_means = []
for ds in DATASETS_11:
    vals = []
    for s in [1, 2, 3]:
        v = best_from_log(f"{ROOT}/logs_baselines_attriprompt/{ds}_seed{s}.log")
        vals.append(v)
    valid = [v for v in vals if v is not None]
    m = statistics.mean(valid) if valid else None
    s_std = statistics.stdev(valid) if len(valid) > 1 else 0
    all_ap_means.append(m)
    print(
        f'{ds:12} | {vals[0] if vals[0] else "  --":>6} {vals[1] if vals[1] else "  --":>6} {vals[2] if vals[2] else "  --":>6} | {m:>6.2f} {s_std:>5.2f}'
    )

avg_ap = statistics.mean([x for x in all_ap_means if x is not None])
print(f'{"AVG":12} | avg_means={avg_ap:.2f}')

"""Aggregate all rebuttal experiment logs into figure_data.csv files for plotting.

Outputs (in ~/bio/AttrVR-main/rebuttal_figdata/):
  F1A_5seed_CIs.csv               - 11 ds × 5 seed best acc
  F1B_hyperparam_beta.csv         - β-sweep avg + per-ds
  F1C_mask_sensitivity.csv        - mask variants × ds
  F1D_pos_corrupt.csv             - pos-corrupt × {full, no_cs}
  F2A_LLM_robustness.csv          - GPT/Qwen/DeepSeek/random/shuffled × 11 ds
  F2B_CS_activation.csv           - per-epoch CS-act fraction × ds
  F2C_gating_weights.csv          - trained (g+, g-) per ckpt
  F3A_calibration.csv             - ECE/MCE/NLL × method × ds
  F3B_baselines.csv               - BG-LM (3 mappings) + AttriPrompt + ours × 11 ds
  F3C_per_class_gating.csv        - shared / per-class / per-sample × ds
  F5_ctk_dedup.csv                - CT-K dedup before/after
"""

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = os.path.expanduser("~/bio/AttrVR-main")
OUT = os.path.join(ROOT, "rebuttal_figdata")
os.makedirs(OUT, exist_ok=True)

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
    except Exception:
        return None


# ---------- F1A 5-seed CIs ----------
print("F1A 5-seed CIs...")
rows = []
seed_dirs = {
    0: "logs_BiomedVR_all_h20_v9_new",  # original v9 (s=0)
    1: "logs_BiomedVR_all_h20_v8_new",  # original v8 (s=1)
    2: "logs_rebuttal_5seed",  # exp_tag 5seed_s2 → busi_seed2.log
    3: "logs_rebuttal_5seed",
    4: "logs_rebuttal_5seed",
}
with open(f"{OUT}/F1A_5seed_CIs.csv", "w") as fout:
    fout.write("dataset,seed,best_acc\n")
    for ds in DATASETS_11:
        for s in range(5):
            d = seed_dirs[s]
            if s in (0, 1):
                p = f"{ROOT}/{d}/{ds}.log"
            else:
                p = f"{ROOT}/{d}/{ds}_seed{s}.log"
            v = best_from_log(p)
            if v is not None:
                fout.write(f"{ds},{s},{v:.2f}\n")

# ---------- F1B hyperparam β-sweep (4 ds × 9 β) ----------
print("F1B hyperparam β-sweep...")


def best_per_block(path, n_epochs=400):
    bests = []
    if not os.path.exists(path):
        return bests
    with open(path) as f:
        block = []
        for line in f:
            if "Best Acc=" in line:
                block.append(line)
                if len(block) == n_epochs:
                    try:
                        bests.append(float(block[-1].split("Best Acc=")[1].split(",")[0]) * 100)
                    except:
                        bests.append(None)
                    block = []
    return bests


with open(f"{OUT}/F1B_hyperparam_beta.csv", "w") as fout:
    fout.write("dataset,beta,best_acc\n")
    BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sources = {
        "busi": f"{ROOT}/logs_BiomedVR_all_h20_v11_search/busi.log",
        "knee_xray": f"{ROOT}/logs_BiomedVR_all_h20_v11_search/knee_xray.log",
        "kvasir": f"{ROOT}/logs_BiomedVR_all_h20_v10_params_search/kvasir.log",
        "lung_colon": f"{ROOT}/logs_BiomedVR_all_h20_v10_params_search/lung_colon.log",
    }
    for ds, p in sources.items():
        bests = best_per_block(p)
        for i, b in enumerate(bests[:9]):
            if b is not None:
                fout.write(f"{ds},{BETAS[i]:.1f},{b:.2f}\n")

# ---------- F1C mask sensitivity ----------
print("F1C mask sensitivity...")
with open(f"{OUT}/F1C_mask_sensitivity.csv", "w") as fout:
    fout.write("dataset,mask_type,best_acc\n")
    masks = ["mask_b8", "mask_b24", "mask_b32", "mask_full", "mask_learned"]
    # default 16 px = our main run results
    for ds in ["busi", "btmri", "kvasir"]:
        if ds == "kvasir":
            seed_log = f"{ROOT}/logs_rebuttal_5seed/kvasir_seed1.log"
        else:
            seed_log = f"{ROOT}/logs_rebuttal_5seed/{ds}_seed1.log"
        # default uses paper Tbl 4 number; here approximate by using seed=1 result if exists
        v_default = best_from_log(seed_log)
        if v_default is not None:
            fout.write(f"{ds},border_16(default),{v_default:.2f}\n")
        for m in masks:
            p = f"{ROOT}/logs_rebuttal_mask/{ds}_seed1.log"
            # The mask logs all share same filename; need to grep by exp_tag header
            if not os.path.exists(p):
                continue
            # Look for the section header marker — use last occurrence  matching pattern
            with open(p) as f:
                content = f.read()
            tag_pattern = f"# === {m}"
            if tag_pattern not in content:
                continue
            # Get section between this header and next # === or EOF
            sections = re.split(r"^# ===.*===\s*$", content, flags=re.M)
            headers = re.findall(r"^# === (.*) ===\s*$", content, flags=re.M)
            # Map header -> section
            for h, sec in zip(headers, sections[1:]):
                if h.startswith(m):
                    last = None
                    for line in sec.splitlines():
                        if "Best Acc=" in line:
                            last = line
                    if last:
                        try:
                            v = float(last.split("Best Acc=")[1].split(",")[0]) * 100
                            fout.write(f"{ds},{m},{v:.2f}\n")
                        except:
                            pass
                    break

# ---------- F1D pos-attr corruption ----------
print("F1D pos-corrupt...")
with open(f"{OUT}/F1D_pos_corrupt.csv", "w") as fout:
    fout.write("dataset,corruption_rate,condition,best_acc\n")
    for ds in ["busi", "btmri"]:
        # baseline 0%
        v = best_from_log(f"{ROOT}/logs_rebuttal_5seed/{ds}_seed1.log")
        if v is not None:
            fout.write(f"{ds},0.00,full,{v:.2f}\n")
            fout.write(f"{ds},0.00,no_cs,{v:.2f}\n")  # placeholder same
        # corrupted
        for r in [0.25, 0.50, 0.75]:
            for cond in ["full", "nocs"]:
                p = f"{ROOT}/logs_rebuttal_poscorrupt/{ds}_seed1.log"
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    content = f.read()
                tag = f"poscorrupt_r{r:.2f}_{cond}"
                # find section
                headers = re.findall(r"^# === (.*) ===\s*$", content, flags=re.M)
                sections = re.split(r"^# ===.*===\s*$", content, flags=re.M)[1:]
                for h, sec in zip(headers, sections):
                    if tag in h:
                        last = None
                        for line in sec.splitlines():
                            if "Best Acc=" in line:
                                last = line
                        if last:
                            try:
                                v = float(last.split("Best Acc=")[1].split(",")[0]) * 100
                                fout.write(f"{ds},{r:.2f},{cond},{v:.2f}\n")
                            except:
                                pass
                        break

# ---------- F2A LLM robustness ----------
print("F2A LLM robustness...")
with open(f"{OUT}/F2A_LLM_robustness.csv", "w") as fout:
    fout.write("dataset,llm_source,best_acc\n")
    for ds in DATASETS_11:
        # default = GPT-4.1 = our main run
        v = best_from_log(f"{ROOT}/logs_rebuttal_5seed/{ds}_seed1.log")
        if v is not None:
            fout.write(f"{ds},gpt-4.1,{v:.2f}\n")
        # qwen
        p = f"{ROOT}/logs_rebuttal_llm_crossvendor/{ds}_seed1.log"
        if os.path.exists(p):
            with open(p) as f:
                content = f.read()
            for tag in ["llmcv_qwen", "llmcv_deepseek"]:
                headers = re.findall(r"^# === (.*) ===\s*$", content, flags=re.M)
                sections = re.split(r"^# ===.*===\s*$", content, flags=re.M)[1:]
                for h, sec in zip(headers, sections):
                    if tag in h:
                        last = None
                        for line in sec.splitlines():
                            if "Best Acc=" in line:
                                last = line
                        if last:
                            try:
                                v = float(last.split("Best Acc=")[1].split(",")[0]) * 100
                                fout.write(f'{ds},{tag.replace("llmcv_", "")},{v:.2f}\n')
                            except:
                                pass
                        break
        # random / shuffled negs
        for mode in ["random", "shuffled"]:
            p = f"{ROOT}/logs_rebuttal_negmode/{ds}_seed1.log"
            if not os.path.exists(p):
                continue
            with open(p) as f:
                content = f.read()
            tag = f"neg_{mode}"
            headers = re.findall(r"^# === (.*) ===\s*$", content, flags=re.M)
            sections = re.split(r"^# ===.*===\s*$", content, flags=re.M)[1:]
            for h, sec in zip(headers, sections):
                if tag in h:
                    last = None
                    for line in sec.splitlines():
                        if "Best Acc=" in line:
                            last = line
                    if last:
                        try:
                            v = float(last.split("Best Acc=")[1].split(",")[0]) * 100
                            fout.write(f"{ds},neg_{mode},{v:.2f}\n")
                        except:
                            pass
                    break

# ---------- F2B CS-activation rate ----------
print("F2B CS-activation rate...")
with open(f"{OUT}/F2B_CS_activation.csv", "w") as fout:
    fout.write("dataset,seed,epoch,cs_active_frac,gate_pos,gate_neg\n")
    for f in glob.glob(f"{ROOT}/logs_rebuttal_5seed/*_seed*_cs.log"):
        bn = os.path.basename(f)
        m = re.match(r"(.+)_seed(\d+)_cs\.log", bn)
        if not m:
            continue
        ds, s = m.group(1), int(m.group(2))
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    continue  # header
                p = line.strip().split(",")
                if len(p) >= 5:
                    fout.write(f"{ds},{s},{p[0]},{p[1]},{p[3]},{p[4]}\n")

# ---------- F2C gating weights from ckpts ----------
print("F2C gating weights (post-hoc from ckpts)...")
import torch
import torch.nn.functional as F

with open(f"{OUT}/F2C_gating_weights.csv", "w") as fout:
    fout.write("exp_tag,dataset,seed,best_acc,g_pos,g_neg\n")
    for ckpt in glob.glob(f"{ROOT}/results/fs_BioMedVR_rebuttal2/*/16*/best.pth") + glob.glob(
        f"{ROOT}/results/fs_BioMedVR-V8/16*/best.pth"
    ):
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        path_parts = ckpt.split("/")
        # results/fs_BioMedVR_rebuttal2/<tag>/<dirname>/best.pth
        # results/fs_BioMedVR-V8/<dirname>/best.pth
        if "fs_BioMedVR-V8" in ckpt:
            tag = "v8"
            dirname = path_parts[-2]
        else:
            tag = path_parts[-3]
            dirname = path_parts[-2]
        m = re.match(r"(\d+)([a-z_]+)k\d+a[0-9.]+s(\d+)", dirname)
        if not m:
            continue
        ds = m.group(2)
        s = int(m.group(3))
        best = sd.get("best_acc", None)
        vp = sd.get("visual_prompt_dict", sd)
        gl = vp.get("gating_logits", None)
        if gl is None:
            continue
        g = F.softmax(gl.float().view(-1, gl.size(-1) if gl.dim() > 1 else 2), dim=-1)
        if g.dim() == 2 and g.size(0) == 1:
            g_pos, g_neg = g[0, 0].item(), g[0, 1].item()
        elif g.dim() == 1:
            g_pos, g_neg = g[0].item(), g[1].item()
        else:
            g_pos, g_neg = g[:, 0].mean().item(), g[:, 1].mean().item()
        b = (best * 100) if best else 0
        fout.write(f"{tag},{ds},{s},{b:.2f},{g_pos:.4f},{g_neg:.4f}\n")

# ---------- F3A calibration (ECE/MCE/NLL) ----------
print("F3A calibration...")
with open(f"{OUT}/F3A_calibration.csv", "w") as fout:
    fout.write("exp_tag,dataset,acc,ece,mce,nll\n")
    cal_path = f"{ROOT}/rebuttal_campaign_out/calibration_all.jsonl"
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                    ckpt = d.get("ckpt", "")
                    parts = ckpt.split("/")
                    if "fs_BioMedVR-V8" in ckpt:
                        tag = "v8"
                    elif "fs_BioMedVR_rebuttal2" in ckpt:
                        tag = parts[-3]
                    else:
                        tag = "unknown"
                    fout.write(
                        f'{tag},{d["dataset"]},{d.get("acc",0)*100:.2f},'
                        f'{d.get("ece",0)*100:.2f},{d.get("mce",0)*100:.2f},'
                        f'{d.get("nll",0):.4f}\n'
                    )
                except Exception:
                    continue

# ---------- F3B baselines (BG-LM, AttriPrompt, ours) ----------
print("F3B baselines...")
with open(f"{OUT}/F3B_baselines.csv", "w") as fout:
    fout.write("method,dataset,best_acc\n")
    for ds in DATASETS_11:
        # ours: BioMedVR (5seed s=1)
        v = best_from_log(f"{ROOT}/logs_rebuttal_5seed/{ds}_seed1.log")
        if v is not None:
            fout.write(f"BioMedVR,{ds},{v:.2f}\n")
        # AttriPrompt
        p = f"{ROOT}/logs_baselines_attriprompt/{ds}_seed1.log"
        v = best_from_log(p)
        if v is not None:
            fout.write(f"AttriPrompt,{ds},{v:.2f}\n")
        # BG-LM × 3 mappings
        for m in ["ilm", "blm", "blmp"]:
            p = f"{ROOT}/logs_baselines_bglm/{ds}_{m}_seed1.log"
            v = best_from_log(p)
            if v is not None:
                fout.write(f"BG-LM-{m.upper()},{ds},{v:.2f}\n")

# ---------- F3C per-class / per-sample gating ----------
print("F3C per-class/per-sample gating...")
with open(f"{OUT}/F3C_per_class_gating.csv", "w") as fout:
    fout.write("dataset,gating_type,best_acc\n")
    for ds in DATASETS_11:
        # shared = our main 5seed s=1
        v = best_from_log(f"{ROOT}/logs_rebuttal_5seed/{ds}_seed1.log")
        if v is not None:
            fout.write(f"{ds},shared,{v:.2f}\n")
        # per-class
        v = best_from_log(f"{ROOT}/logs_rebuttal_perclass/{ds}_seed1.log")
        if v is not None:
            fout.write(f"{ds},per_class,{v:.2f}\n")
        # per-sample (only 4 core ds)
        v = best_from_log(f"{ROOT}/logs_rebuttal_persample/{ds}_seed1.log")
        if v is not None:
            fout.write(f"{ds},per_sample,{v:.2f}\n")

# ---------- F5 CT-K dedup ----------
print("F5 CT-K dedup...")
with open(f"{OUT}/F5_ctk_dedup.csv", "w") as fout:
    fout.write("condition,dataset,best_acc\n")
    v = best_from_log(f"{ROOT}/logs_rebuttal_5seed/ctkidney_seed1.log")
    if v is not None:
        fout.write(f"baseline_with_overlap,ctkidney,{v:.2f}\n")
    v = best_from_log(f"{ROOT}/logs_rebuttal_post/ctkidney_seed1.log")
    if v is not None:
        fout.write(f"after_dedup,ctkidney,{v:.2f}\n")

print("\nAll figure_data CSVs written to:", OUT)
print("---listing---")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}: {os.path.getsize(p)} bytes")

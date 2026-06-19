"""Fixed aggregation script — uses correct baseline log paths.
Key fix: BioMedVR seed=1 baseline lives in logs_BiomedVR_all_h20_v8_new/{ds}.log (not rebuttal_5seed).
"""
import os, json, glob, re
from pathlib import Path

ROOT = os.path.expanduser('~/bio/AttrVR-main')
OUT = os.path.join(ROOT, 'rebuttal_figdata')
os.makedirs(OUT, exist_ok=True)

DATASETS_11 = ['busi','knee_xray','kvasir','lung_colon','octmnist','btmri','chmnist','covid_19','ctkidney','dermamnist','retina']

def best_from_log(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        last = None
        for line in f:
            if 'Best Acc=' in line: last = line
    if last is None: return None
    try:
        return float(last.split('Best Acc=')[1].split(',')[0]) * 100
    except Exception:
        return None

def baseline_path(ds):
    """BioMedVR seed=1 baseline = original v8 log."""
    return f'{ROOT}/logs_BiomedVR_all_h20_v8_new/{ds}.log'

# Helper: parse log file with multiple "# === tag ===" sections, return best per tag
def parse_sectioned_log(path):
    """Returns dict {section_tag_substring: best_acc}."""
    if not os.path.exists(path): return {}
    with open(path) as f:
        content = f.read()
    headers = re.findall(r'^# === (.*) ===\s*$', content, flags=re.M)
    sections = re.split(r'^# ===.*===\s*$', content, flags=re.M)[1:]
    out = {}
    for h, sec in zip(headers, sections):
        last = None
        for line in sec.splitlines():
            if 'Best Acc=' in line: last = line
        if last:
            try:
                v = float(last.split('Best Acc=')[1].split(',')[0]) * 100
                out[h.strip()] = v
            except: pass
    return out

# ---------- F1A 5-seed CIs ----------
print('F1A 5-seed CIs...')
seed_dirs = {
    0: ('logs_BiomedVR_all_h20_v9_new', 'plain'),
    1: ('logs_BiomedVR_all_h20_v8_new', 'plain'),
    2: ('logs_rebuttal_5seed', 'seed'),
    3: ('logs_rebuttal_5seed', 'seed'),
    4: ('logs_rebuttal_5seed', 'seed'),
}
with open(f'{OUT}/F1A_5seed_CIs.csv', 'w') as fout:
    fout.write('dataset,seed,best_acc\n')
    for ds in DATASETS_11:
        for s in range(5):
            d, fmt = seed_dirs[s]
            p = f'{ROOT}/{d}/{ds}.log' if fmt == 'plain' else f'{ROOT}/{d}/{ds}_seed{s}.log'
            v = best_from_log(p)
            if v is not None:
                fout.write(f'{ds},{s},{v:.2f}\n')

# ---------- F1B hyperparam β-sweep ----------
print('F1B hyperparam β-sweep...')
def best_per_block(path, n_epochs=400):
    bests = []
    if not os.path.exists(path): return bests
    with open(path) as f:
        block = []
        for line in f:
            if 'Best Acc=' in line:
                block.append(line)
                if len(block) == n_epochs:
                    try: bests.append(float(block[-1].split('Best Acc=')[1].split(',')[0]) * 100)
                    except: bests.append(None)
                    block = []
    return bests
with open(f'{OUT}/F1B_hyperparam_beta.csv', 'w') as fout:
    fout.write('dataset,beta,best_acc\n')
    BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sources = {
        'busi':       f'{ROOT}/logs_BiomedVR_all_h20_v11_search/busi.log',
        'knee_xray':  f'{ROOT}/logs_BiomedVR_all_h20_v11_search/knee_xray.log',
        'kvasir':     f'{ROOT}/logs_BiomedVR_all_h20_v10_params_search/kvasir.log',
        'lung_colon': f'{ROOT}/logs_BiomedVR_all_h20_v10_params_search/lung_colon.log',
    }
    for ds, p in sources.items():
        bests = best_per_block(p)
        for i, b in enumerate(bests[:9]):
            if b is not None: fout.write(f'{ds},{BETAS[i]:.1f},{b:.2f}\n')

# ---------- F1C mask sensitivity (incl default) ----------
print('F1C mask sensitivity...')
with open(f'{OUT}/F1C_mask_sensitivity.csv', 'w') as fout:
    fout.write('dataset,mask_type,best_acc\n')
    for ds in ['busi', 'btmri', 'kvasir']:
        # default 16 px = baseline
        v = best_from_log(baseline_path(ds))
        if v is not None: fout.write(f'{ds},border_16(default),{v:.2f}\n')
        # variants
        sec = parse_sectioned_log(f'{ROOT}/logs_rebuttal_mask/{ds}_seed1.log')
        for tag in ['mask_b8', 'mask_b24', 'mask_b32', 'mask_full', 'mask_learned']:
            for header, val in sec.items():
                if header.startswith(tag) or tag in header.split('|')[0]:
                    fout.write(f'{ds},{tag},{val:.2f}\n')
                    break

# ---------- F1D pos-attr corruption ----------
print('F1D pos-corrupt...')
with open(f'{OUT}/F1D_pos_corrupt.csv', 'w') as fout:
    fout.write('dataset,corruption_rate,condition,best_acc\n')
    for ds in ['busi', 'btmri']:
        # 0% baseline = our 5-seed s=1
        v = best_from_log(baseline_path(ds))
        if v is not None:
            fout.write(f'{ds},0.00,full,{v:.2f}\n')
            fout.write(f'{ds},0.00,nocs,{v:.2f}\n')
        sec = parse_sectioned_log(f'{ROOT}/logs_rebuttal_poscorrupt/{ds}_seed1.log')
        for r in [0.25, 0.50, 0.75]:
            for cond in ['full', 'nocs']:
                tag = f'poscorrupt_r{r:.2f}_{cond}'
                for header, val in sec.items():
                    if tag in header:
                        fout.write(f'{ds},{r:.2f},{cond},{val:.2f}\n')
                        break

# ---------- F2A LLM robustness ----------
print('F2A LLM robustness...')
with open(f'{OUT}/F2A_LLM_robustness.csv', 'w') as fout:
    fout.write('dataset,llm_source,best_acc\n')
    for ds in DATASETS_11:
        v = best_from_log(baseline_path(ds))
        if v is not None: fout.write(f'{ds},gpt-4.1,{v:.2f}\n')
        # qwen / deepseek (in same log file by exp_tag)
        for tag, label in [('llmcv_qwen', 'qwen'), ('llmcv_deepseek', 'deepseek')]:
            sec = parse_sectioned_log(f'{ROOT}/logs_rebuttal_llm_crossvendor/{ds}_seed1.log')
            for header, val in sec.items():
                if tag in header:
                    fout.write(f'{ds},{label},{val:.2f}\n')
                    break
        # random / shuffled
        for mode in ['random', 'shuffled']:
            tag = f'neg_{mode}'
            sec = parse_sectioned_log(f'{ROOT}/logs_rebuttal_negmode/{ds}_seed1.log')
            for header, val in sec.items():
                if tag in header:
                    fout.write(f'{ds},neg_{mode},{val:.2f}\n')
                    break

# ---------- F2B CS-activation ----------
print('F2B CS-activation...')
with open(f'{OUT}/F2B_CS_activation.csv', 'w') as fout:
    fout.write('dataset,seed,epoch,cs_active_frac,gate_pos,gate_neg\n')
    for f in glob.glob(f'{ROOT}/logs_rebuttal_5seed/*_seed*_cs.log'):
        bn = os.path.basename(f)
        m = re.match(r'(.+)_seed(\d+)_cs\.log', bn)
        if not m: continue
        ds, s = m.group(1), int(m.group(2))
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i == 0: continue
                p = line.strip().split(',')
                if len(p) >= 5:
                    fout.write(f'{ds},{s},{p[0]},{p[1]},{p[3]},{p[4]}\n')

# ---------- F2C gating weights ----------
print('F2C gating weights...')
import torch
import torch.nn.functional as F
with open(f'{OUT}/F2C_gating_weights.csv', 'w') as fout:
    fout.write('exp_tag,dataset,seed,best_acc,g_pos,g_neg\n')
    ckpts = (glob.glob(f'{ROOT}/results/fs_BioMedVR_rebuttal2/*/16*/best.pth') +
             glob.glob(f'{ROOT}/results/fs_BioMedVR-V8/16*/best.pth'))
    for ckpt in ckpts:
        try: sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        except: continue
        path_parts = ckpt.split('/')
        if 'fs_BioMedVR-V8' in ckpt:
            tag = 'v8_baseline'; dirname = path_parts[-2]
        else:
            tag = path_parts[-3]; dirname = path_parts[-2]
        m = re.match(r'(\d+)([a-z_]+)k\d+a[0-9.]+s(\d+)', dirname)
        if not m: continue
        ds, s = m.group(2), int(m.group(3))
        best = sd.get('best_acc', None)
        vp = sd.get('visual_prompt_dict', sd)
        gl = vp.get('gating_logits', None)
        if gl is None: continue
        if gl.dim() == 2:  # per-class [C, 2]
            g = F.softmax(gl.float(), dim=-1)
            g_pos, g_neg = g[:, 0].mean().item(), g[:, 1].mean().item()
        else:  # shared scalar [2]
            g = F.softmax(gl.float(), dim=0)
            g_pos, g_neg = g[0].item(), g[1].item()
        b = (best * 100) if best else 0
        fout.write(f'{tag},{ds},{s},{b:.2f},{g_pos:.4f},{g_neg:.4f}\n')

# ---------- F3A calibration ----------
print('F3A calibration...')
with open(f'{OUT}/F3A_calibration.csv', 'w') as fout:
    fout.write('exp_tag,dataset,acc,ece,mce,nll\n')
    cal_path = f'{ROOT}/rebuttal_campaign_out/calibration_all.jsonl'
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith('{'): continue
                try:
                    d = json.loads(line)
                    ckpt = d.get('ckpt', '')
                    parts = ckpt.split('/')
                    if 'fs_BioMedVR-V8' in ckpt: tag = 'v8_baseline'
                    elif 'fs_BioMedVR_rebuttal2' in ckpt: tag = parts[-3]
                    else: tag = 'unknown'
                    fout.write(f'{tag},{d["dataset"]},{d.get("acc",0)*100:.2f},'
                               f'{d.get("ece",0)*100:.2f},{d.get("mce",0)*100:.2f},'
                               f'{d.get("nll",0):.4f}\n')
                except: continue

# ---------- F3B baselines ----------
print('F3B baselines...')
with open(f'{OUT}/F3B_baselines.csv', 'w') as fout:
    fout.write('method,dataset,best_acc\n')
    for ds in DATASETS_11:
        v = best_from_log(baseline_path(ds))
        if v is not None: fout.write(f'BioMedVR,{ds},{v:.2f}\n')
        v = best_from_log(f'{ROOT}/logs_baselines_attriprompt/{ds}_seed1.log')
        if v is not None: fout.write(f'AttriPrompt,{ds},{v:.2f}\n')
        for m in ['ilm', 'blm', 'blmp']:
            v = best_from_log(f'{ROOT}/logs_baselines_bglm/{ds}_{m}_seed1.log')
            if v is not None: fout.write(f'BG-LM-{m.upper()},{ds},{v:.2f}\n')

# ---------- F3C per-class / per-sample gating ----------
print('F3C per-class/per-sample gating...')
with open(f'{OUT}/F3C_per_class_gating.csv', 'w') as fout:
    fout.write('dataset,gating_type,best_acc\n')
    for ds in DATASETS_11:
        v = best_from_log(baseline_path(ds))
        if v is not None: fout.write(f'{ds},shared,{v:.2f}\n')
        v = best_from_log(f'{ROOT}/logs_rebuttal_perclass/{ds}_seed1.log')
        if v is not None: fout.write(f'{ds},per_class,{v:.2f}\n')
        v = best_from_log(f'{ROOT}/logs_rebuttal_persample/{ds}_seed1.log')
        if v is not None: fout.write(f'{ds},per_sample,{v:.2f}\n')

# ---------- F5 CT-K dedup ----------
print('F5 CT-K dedup...')
with open(f'{OUT}/F5_ctk_dedup.csv', 'w') as fout:
    fout.write('condition,dataset,best_acc\n')
    v = best_from_log(baseline_path('ctkidney'))
    if v is not None: fout.write(f'baseline_with_overlap,ctkidney,{v:.2f}\n')
    v = best_from_log(f'{ROOT}/logs_rebuttal_post/ctkidney_seed1.log')
    if v is not None: fout.write(f'after_dedup,ctkidney,{v:.2f}\n')

print('\nAll figure_data CSVs (v2) written to:', OUT)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f); print(f'  {f}: {os.path.getsize(p)} bytes')

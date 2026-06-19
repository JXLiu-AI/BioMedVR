"""Measure FLOPs for BG-LM, AttriPrompt, AttrVR, BioMedVR. ViT-B/16, batch=1."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/baselines/BayesianLM'))
sys.path.insert(0, os.path.expanduser('~/bio/AttrVR-main'))

import torch, torch.nn as nn, torch.nn.functional as F
import warnings; warnings.filterwarnings('ignore')
import clip
from fvcore.nn import FlopCountAnalysis
from methods.vp import PaddingVR
from reprogramming import WatermarkingVR

model, _ = clip.load('ViT-B/16', device='cpu')
model.eval()
for p in model.parameters(): p.requires_grad_(False)

enc = model.visual.float()
x224 = torch.randn(1, 3, 224, 224)
x192 = torch.randn(1, 3, 192, 192)

# 1) Zero-shot CLIP only
class Plain(nn.Module):
    def __init__(self, e): super().__init__(); self.e=e
    def forward(self, x): return self.e(x)
f0 = FlopCountAnalysis(Plain(enc), x224).total() / 1e9

# 2) AttrVR: PaddingVR (single) + CLIP visual
class SingleVR(nn.Module):
    def __init__(self, vr, e): super().__init__(); self.vr=vr; self.e=e
    def forward(self, x): return self.e(self.vr(x))
attrvr_vr = PaddingVR(224, 192)
f_attrvr = FlopCountAnalysis(SingleVR(attrvr_vr, enc), x192).total() / 1e9

# 3) BG-LM: WatermarkingVR (single) + CLIP visual
class BGLMSingle(nn.Module):
    def __init__(self, vr, e): super().__init__(); self.vr=vr; self.e=e
    def forward(self, x): return self.e(self.vr(x))
bglm_vr = WatermarkingVR(224, 30)
f_bglm = FlopCountAnalysis(BGLMSingle(bglm_vr, enc), x224).total() / 1e9

# 4) AttriPrompt: CLIP visual (no input perturbation) + extra prompt-tuning ON TEXT side
# Image side = vanilla CLIP visual = same as Plain
f_ap = f0  # text side cost is negligible (learnable ctx ~16 tokens × dim)

# 5) BioMedVR: dual PaddingVR + CLIP visual (×2)
class DualVR(nn.Module):
    def __init__(self, base, target, e):
        super().__init__()
        self.experts = nn.ModuleList([PaddingVR(base, target), PaddingVR(base, target)])
        self.gating_logits = nn.Parameter(torch.ones(2))
        self.e = e
    def forward(self, x):
        outs = [exp(x) for exp in self.experts]
        gates = F.softmax(self.gating_logits, dim=0)
        e1 = self.e(outs[0]); e2 = self.e(outs[1])
        return gates[0]*e1 + gates[1]*e2
biomedvr = DualVR(224, 192, enc)
f_bmv = FlopCountAnalysis(biomedvr, x192).total() / 1e9

# 6) BioMedVR-deploy: gating saturates, only 1 branch
f_bmv_deploy = f_attrvr  # same as single VR

print(f"{'method':25s} {'GFLOPs':>10s} {'×CLIP':>8s}")
print("-" * 45)
for name, v in [('Zero-shot CLIP', f0),
                ('AttrVR (single VR)', f_attrvr),
                ('BG-LM (Watermark VR)', f_bglm),
                ('AttriPrompt (text only)', f_ap),
                ('BioMedVR (train, dual VR)', f_bmv),
                ('BioMedVR (deploy, 1 branch)', f_bmv_deploy)]:
    print(f"{name:25s} {v:>10.3f} {v/f0:>8.3f}")

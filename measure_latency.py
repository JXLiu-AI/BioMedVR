import os
import sys
import time
import warnings

sys.path.insert(0, os.path.expanduser("~/baselines/BayesianLM"))
sys.path.insert(0, os.path.expanduser("~/bio/AttrVR-main"))
warnings.filterwarnings("ignore")
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.vp import PaddingVR
from reprogramming import WatermarkingVR

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

model, _ = clip.load("ViT-B/16", device=device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
enc = model.visual.float()

BATCH = 16
N_WARM = 20
N_ITER = 100


def bench(net, x):
    net.eval()
    with torch.no_grad():
        for _ in range(N_WARM):
            _ = net(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_ITER):
            _ = net(x)
        torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / N_ITER * 1000


x192 = torch.randn(BATCH, 3, 192, 192, device=device)
x224 = torch.randn(BATCH, 3, 224, 224, device=device)


class Plain(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.e = e

    def forward(self, x):
        return self.e(x)


class SingleVR(nn.Module):
    def __init__(self, vr, e):
        super().__init__()
        self.vr = vr
        self.e = e

    def forward(self, x):
        return self.e(self.vr(x))


class BGLMSingle(nn.Module):
    def __init__(self, vr, e, nc=11):
        super().__init__()
        self.vr = vr
        self.e = e
        self.lm = nn.Linear(512, nc, bias=False).to(device)

    def forward(self, x):
        emb = self.e(self.vr(x))
        return self.lm(emb)


class DualVR(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.experts = nn.ModuleList(
            [PaddingVR(224, 192).to(device), PaddingVR(224, 192).to(device)]
        )
        self.gating_logits = nn.Parameter(torch.ones(2, device=device))
        self.e = e

    def forward(self, x):
        outs = [exp(x) for exp in self.experts]
        return torch.stack([self.e(outs[0]), self.e(outs[1])]).sum(0)


zs = bench(Plain(enc).to(device), x224)
av = bench(SingleVR(PaddingVR(224, 192).to(device), enc).to(device), x192)
bglm = bench(BGLMSingle(WatermarkingVR(224, 30).to(device), enc).to(device), x224)
bmvr = bench(DualVR(enc).to(device), x192)

print(f"\nbatch_size={BATCH}, GPU latency over {N_ITER} iters (after {N_WARM} warmup)")
print(f"{'method':10s} {'lat/batch (ms)':>16s} {'lat/img (ms)':>14s} {'×CLIP':>8s}")
print("-" * 55)
for name, t in [("ZS-CLIP", zs), ("AttrVR", av), ("BG-LM", bglm), ("BMVR", bmvr)]:
    print(f"{name:10s} {t:>13.2f}    {t/BATCH:>10.3f}    {t/zs:>7.3f}")

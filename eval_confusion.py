"""Evaluate confusion matrix for given BioMedVR ckpt; save to npz.
Used to compare BioMedVR-full vs no-CS to show CS loss effect on inter-class confusion.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/bio/AttrVR-main"))

import clip
import numpy as np
import torch
import torch.nn as nn
from cfg import DOWNSTREAM_PATH
from datasets.build_loader import build_loader
from methods.vp import PaddingVR
from tools import clip_attr_classifier, convert_models_to_fp32, getLogits, set_seed
from torch.nn import functional as F
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Lambda, Resize, ToTensor


class MultiPromptVR(nn.Module):
    def __init__(self, base, target):
        super().__init__()
        self.experts = nn.ModuleList([PaddingVR(base, target), PaddingVR(base, target)])
        self.gating_logits = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, x):
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=0), F.softmax(self.gating_logits, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input_size", type=int, default=192)
    p.add_argument("--num_attr", type=int, default=20)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--shot", type=int, default=16)
    p.add_argument("--use_negative", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda:0")
    set_seed(args.seed)
    model, _ = clip.load("ViT-B/16")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    test_tf = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(args.input_size),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
        ]
    )
    _, testloader, classes = build_loader(
        args.dataset, DOWNSTREAM_PATH, test_tf, test_tf, batch_size=64, shot=args.shot
    )

    txt_des = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{args.dataset}_des.json", num_attr=args.num_attr
    )
    txt_dist = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{args.dataset}_dist.json", num_attr=args.num_attr
    )
    txt_conf = None
    if args.use_negative:
        cp = f"attributes/gpt3/gpt3/{args.dataset}_confuse.json"
        if os.path.exists(cp):
            txt_conf = clip_attr_classifier(classes, model, cp, num_attr=args.num_attr)

    vr = MultiPromptVR(224, args.input_size).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    vp_dict = sd.get("visual_prompt_dict", sd)
    vr.load_state_dict(vp_dict, strict=False)
    vr.eval()
    g = F.softmax(vr.gating_logits.detach(), dim=0).tolist()
    print(f"Loaded {args.ckpt}, gating={g}")

    n_cls = len(classes)
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    all_probs, all_labels = [], []
    n_correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outs, gates = vr(x)
            x_embs = [model.encode_image(outs[i]) for i in range(2)]
            x_embs = [e / e.norm(dim=-1, keepdim=True) for e in x_embs]
            scale = model.logit_scale.exp()
            pos = 0.5 * (
                getLogits(args.num_attr, scale, x_embs[0], txt_des, k=args.k)
                + getLogits(args.num_attr, scale, x_embs[0], txt_dist, k=args.k)
            )
            if txt_conf is not None:
                neg = getLogits(args.num_attr, scale, x_embs[1], txt_conf, k=args.k)
            else:
                neg = torch.zeros_like(pos)
            stack = torch.stack([pos, neg], dim=0)
            fused = (gates.view(-1, 1, 1) * stack).sum(0)
            probs = F.softmax(fused, dim=1)
            preds = fused.argmax(1)
            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            n_correct += (preds == y).sum().item()
            total += y.size(0)

    acc = n_correct / total
    np.savez(
        args.out,
        confusion=confusion,
        classes=classes,
        probs=np.concatenate(all_probs),
        labels=np.concatenate(all_labels),
        gating=np.array(g),
        acc=acc,
    )
    print(f"Acc: {acc*100:.2f}% | Confusion saved to {args.out}")


if __name__ == "__main__":
    main()

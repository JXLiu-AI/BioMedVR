"""Cross-task transfer: load source-trained δ⁺/δ⁻, evaluate on target dataset's test set
using target's text attributes (no further training)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import clip
import torch
import torch.nn as nn
from cfg import *
from datasets import *
from datasets.build_loader import build_loader
from methods.vp import PaddingVR
from tools import *
from torch.nn import functional as F
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Lambda, Resize, ToTensor


class MultiPromptVR(nn.Module):
    def __init__(self, base_size, target_size):
        super().__init__()
        self.experts = nn.ModuleList(
            [PaddingVR(base_size, target_size), PaddingVR(base_size, target_size)]
        )
        self.gating_logits = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, x):
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=0), F.softmax(self.gating_logits, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_ckpt", required=True)
    p.add_argument("--tgt_dataset", required=True)
    p.add_argument("--input_size", type=int, default=192)
    p.add_argument("--num_attr", type=int, default=20)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--shot", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda:0")
    set_seed(0)
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
    train_tf = test_tf
    _, testloader, classes = build_loader(
        args.tgt_dataset, DOWNSTREAM_PATH, train_tf, test_tf, batch_size=64, shot=args.shot
    )

    # Target text features
    txt_des = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{args.tgt_dataset}_des.json", num_attr=args.num_attr
    )
    txt_dist = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{args.tgt_dataset}_dist.json", num_attr=args.num_attr
    )
    conf_path = f"attributes/gpt3/gpt3/{args.tgt_dataset}_confuse.json"
    txt_conf = (
        clip_attr_classifier(classes, model, conf_path, num_attr=args.num_attr)
        if os.path.exists(conf_path)
        else None
    )

    # Load source VR pattern
    vr = MultiPromptVR(224, args.input_size).to(device)
    sd = torch.load(args.src_ckpt, map_location=device)
    vp_dict = sd.get("visual_prompt_dict", sd)
    vr.load_state_dict(vp_dict, strict=False)
    vr.eval()
    g = F.softmax(vr.gating_logits.detach(), dim=0).tolist()

    correct = total = 0
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
            correct += (fused.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(
        json.dumps(
            {
                "src_ckpt": args.src_ckpt,
                "tgt_dataset": args.tgt_dataset,
                "acc": acc,
                "n": total,
                "gate_pos": g[0],
                "gate_neg": g[1],
            }
        )
    )


if __name__ == "__main__":
    main()

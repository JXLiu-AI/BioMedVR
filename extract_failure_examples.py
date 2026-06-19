"""Find concrete failure cases on CT-Kidney: misclassified images + the neg attr that caused it."""

import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/bio/AttrVR-main"))

import random

import clip
import numpy as np
import torch
import torch.nn as nn
from cfg import DOWNSTREAM_PATH
from datasets.build_loader import build_loader
from methods.vp import PaddingVR
from PIL import Image
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
    device = torch.device("cuda:0")
    SEED = 2
    set_seed(SEED)
    DS = "ctkidney"
    INPUT_SIZE = 192

    model, _ = clip.load("ViT-B/16")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    test_tf = Compose(
        [
            Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(INPUT_SIZE),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
        ]
    )
    _, testloader, classes = build_loader(
        DS, DOWNSTREAM_PATH, test_tf, test_tf, batch_size=64, shot=16
    )
    print("Classes:", classes)

    # Load text features (re-init seed before sampling)
    set_seed(SEED)
    txt_des = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{DS}_des.json", num_attr=20
    )
    txt_dist = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{DS}_dist.json", num_attr=20
    )
    txt_conf = clip_attr_classifier(
        classes, model, f"attributes/gpt3/gpt3/{DS}_confuse.json", num_attr=20
    )

    # Also load raw confuse JSON for the actual attribute strings
    conf_strs = json.load(open(f"attributes/gpt3/gpt3/{DS}_confuse.json"))

    vr = MultiPromptVR(224, INPUT_SIZE).to(device)
    sd = torch.load(
        "results/fs_BioMedVR_rebuttal2/5seed_s2/16ctkidneyk3a0.5s2/best.pth",
        map_location=device,
        weights_only=False,
    )
    vr.load_state_dict(sd["visual_prompt_dict"], strict=False)
    vr.eval()

    # Run inference, collect all (true, pred, pos_logits, neg_logits, image_idx)
    all_records = []
    img_idx = 0
    with torch.no_grad():
        for x, y in testloader:
            x_dev, y_dev = x.to(device), y.to(device)
            outs, gates = vr(x_dev)
            x_embs = [model.encode_image(outs[i]) for i in range(2)]
            x_embs = [e / e.norm(dim=-1, keepdim=True) for e in x_embs]
            scale = model.logit_scale.exp()
            pos = 0.5 * (
                getLogits(20, scale, x_embs[0], txt_des, k=3)
                + getLogits(20, scale, x_embs[0], txt_dist, k=3)
            )
            neg = getLogits(20, scale, x_embs[1], txt_conf, k=3)
            stack = torch.stack([pos, neg], dim=0)
            fused = (gates.view(-1, 1, 1) * stack).sum(0)
            preds = fused.argmax(1)
            for i in range(x.size(0)):
                all_records.append(
                    {
                        "idx": img_idx,
                        "true": y_dev[i].item(),
                        "pred": preds[i].item(),
                        "pos": pos[i].cpu().numpy(),
                        "neg": neg[i].cpu().numpy(),
                        "fused": fused[i].cpu().numpy(),
                    }
                )
                img_idx += 1

    # Pick interesting failure cases:
    # tumor -> cyst (true class kidney_tumor predicted as cyst)
    target_pairs = [
        ("kidney_tumor", "cyst_kidney"),  # 0.30 in confusion mat
        ("kidney_stone", "normal_kidney"),  # 0.34
        ("kidney_tumor", "normal_kidney"),  # 0.34
    ]
    # Map classes
    cls2idx = {c: i for i, c in enumerate(classes)}

    # Read split for image paths
    split_path = f"MedicalData/CTKidney/split_liu_CTKidney.json"
    split = json.load(open(split_path))
    test_paths = [(rec[0], rec[1]) for rec in split["test"]]  # path, label_idx
    # Actually our loader probably reads in same order. Let me index by global position.

    selected = []
    for true_name, pred_name in target_pairs:
        t_idx = cls2idx[true_name]
        p_idx = cls2idx[pred_name]
        # find first record matching
        for r in all_records:
            if r["true"] == t_idx and r["pred"] == p_idx:
                # Find which confuse attr of pred class would attract this image
                # neg score per class — pick the class with high neg s
                neg_scores = r["neg"]
                top_neg_class = int(np.argmax(neg_scores))
                selected.append(
                    {
                        "global_idx": r["idx"],
                        "true_name": true_name,
                        "pred_name": pred_name,
                        "top_neg_class": classes[top_neg_class],
                        "pos_score_correct": float(r["pos"][t_idx]),
                        "pos_score_pred": float(r["pos"][p_idx]),
                        "neg_score_correct": float(r["neg"][t_idx]),
                        "neg_score_pred": float(r["neg"][p_idx]),
                        # representative attr from confuse[pred] (most likely to overlap with true)
                        "overlapping_attr": (
                            conf_strs[pred_name][0] if conf_strs.get(pred_name) else "N/A"
                        ),
                    }
                )
                break

    # Save image paths corresponding to these global_idx
    if len(test_paths) >= len(all_records):
        for s in selected:
            try:
                s["img_path"] = test_paths[s["global_idx"]][0]
            except Exception:
                s["img_path"] = None

    # Save to JSON for later figure render
    out_path = "rebuttal_figdata/confusion/ctk_failure_examples.json"
    json.dump(selected, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")
    for s in selected:
        print(
            f"  true={s['true_name']:15s} pred={s['pred_name']:15s} top_neg_cls={s['top_neg_class']:15s} idx={s['global_idx']}"
        )
        print(
            f"    pos[true]={s['pos_score_correct']:.3f} pos[pred]={s['pos_score_pred']:.3f}  neg[true]={s['neg_score_correct']:.3f} neg[pred]={s['neg_score_pred']:.3f}"
        )
        print(f"    img={s['img_path']}")
        print(f"    overlap attr (from conf[{s['pred_name']}]): {s['overlapping_attr'][:130]}...")


if __name__ == "__main__":
    main()

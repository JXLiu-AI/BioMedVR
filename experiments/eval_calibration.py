"""Post-hoc calibration eval for BioMedVR best.pth checkpoints.
Reads checkpoint, runs full test set, reports ECE/MCE/NLL/Acc.
"""
import sys, os, json, argparse, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop
from torchvision.transforms import InterpolationMode
import clip
from cfg import *
from tools import *
from datasets import *
from datasets.build_loader import build_loader
from methods.vp import PaddingVR


class MultiPromptVR(nn.Module):
    def __init__(self, base_size, target_size):
        super().__init__()
        self.experts = nn.ModuleList([PaddingVR(base_size, target_size), PaddingVR(base_size, target_size)])
        self.gating_logits = nn.Parameter(torch.ones(2, dtype=torch.float32))
    def forward(self, x):
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=0), F.softmax(self.gating_logits, dim=0)


def compute_ece(probs, labels, n_bins=15):
    confs, preds = probs.max(dim=1)
    accs = preds.eq(labels).float()
    ece = mce = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for lo, hi in zip(bin_lowers, bin_uppers):
        mask = (confs > lo) & (confs <= hi)
        n = mask.sum().item()
        if n > 0:
            avg_conf = confs[mask].mean().item()
            avg_acc = accs[mask].mean().item()
            gap = abs(avg_conf - avg_acc)
            ece += (n / labels.size(0)) * gap
            mce = max(mce, gap)
    return ece, mce


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--input_size', type=int, default=192)
    p.add_argument('--num_attr', type=int, default=20)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shot', type=int, default=16)
    p.add_argument('--use_negative', type=int, default=1, help='1 to use confuse attrs, 0 to use only positive')
    args = p.parse_args()

    device = torch.device('cuda:0')
    set_seed(args.seed)
    model, _ = clip.load('ViT-B/16')
    convert_models_to_fp32(model)
    model.eval(); model.requires_grad_(False)

    test_tf = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])
    train_tf = test_tf  # not used
    _, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_tf, test_tf, batch_size=64, shot=args.shot)

    txt_des = clip_attr_classifier(classes, model, f'attributes/gpt3/gpt3/{args.dataset}_des.json', num_attr=args.num_attr)
    txt_dist = clip_attr_classifier(classes, model, f'attributes/gpt3/gpt3/{args.dataset}_dist.json', num_attr=args.num_attr)
    conf_path = f'attributes/gpt3/gpt3/{args.dataset}_confuse.json'
    txt_conf = clip_attr_classifier(classes, model, conf_path, num_attr=args.num_attr) if os.path.exists(conf_path) and args.use_negative else None

    vr = MultiPromptVR(224, args.input_size).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    vp_dict = sd.get('visual_prompt_dict', sd)
    vr.load_state_dict(vp_dict, strict=False)
    vr.eval()
    print(f'Loaded {args.ckpt}, gating={F.softmax(vr.gating_logits.detach(), dim=0).tolist()}')

    all_probs = []
    all_labels = []
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outs, gates = vr(x)
            x_embs = [model.encode_image(outs[i]) for i in range(2)]
            x_embs = [e / e.norm(dim=-1, keepdim=True) for e in x_embs]
            scale = model.logit_scale.exp()
            pos = 0.5*(getLogits(args.num_attr, scale, x_embs[0], txt_des, k=args.k) +
                       getLogits(args.num_attr, scale, x_embs[0], txt_dist, k=args.k))
            if txt_conf is not None:
                neg = getLogits(args.num_attr, scale, x_embs[1], txt_conf, k=args.k)
            else:
                neg = torch.zeros_like(pos)
            stack = torch.stack([pos, neg], dim=0)
            fused = (gates.view(-1,1,1) * stack).sum(0)
            probs = F.softmax(fused, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            correct += (fused.argmax(1) == y).sum().item()
            total += y.size(0)
    P = torch.cat(all_probs); L = torch.cat(all_labels)
    acc = correct/total
    nll = F.nll_loss(torch.log(P.clamp(1e-12)), L).item()
    ece, mce = compute_ece(P, L, n_bins=15)
    print(json.dumps({'ckpt': args.ckpt, 'dataset': args.dataset, 'use_negative': args.use_negative,
                      'acc': acc, 'ece': ece, 'mce': mce, 'nll': nll, 'n': total}))

if __name__ == '__main__':
    main()

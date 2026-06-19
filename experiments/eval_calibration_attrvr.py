"""ECE eval for AttrVR ckpts (single PaddingVR + des+dist, no neg expert)."""
import sys, os, argparse, json
sys.path.insert(0, os.path.expanduser('~/bio/AttrVR-main'))

import torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, InterpolationMode
import clip
from cfg import DOWNSTREAM_PATH
from tools import set_seed, convert_models_to_fp32, clip_attr_classifier, getLogits
from datasets.build_loader import build_loader
from methods.vp import PaddingVR


def compute_ece(probs, labels, n_bins=15):
    confs, preds = probs.max(dim=1)
    accs = preds.eq(labels).float()
    ece = mce = 0.0
    bin_lowers = torch.linspace(0, 1, n_bins+1)[:-1]
    bin_uppers = torch.linspace(0, 1, n_bins+1)[1:]
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
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--shot', type=int, default=16)
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
    _, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, test_tf, test_tf, batch_size=64, shot=args.shot)
    txt_des = clip_attr_classifier(classes, model, f'attributes/gpt3/gpt3/{args.dataset}_des.json', num_attr=args.num_attr)
    txt_dist = clip_attr_classifier(classes, model, f'attributes/gpt3/gpt3/{args.dataset}_dist.json', num_attr=args.num_attr)

    vr = PaddingVR(224, args.input_size).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    # AttrVR ckpt structure: 'visual_prompt_dict' contains state of single PaddingVR
    vp_dict = sd.get('visual_prompt_dict', sd)
    vr.load_state_dict(vp_dict, strict=False)
    vr.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            x_emb = model.encode_image(vr(x))
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp()
            logits = 0.5*(getLogits(args.num_attr, scale, x_emb, txt_des, k=args.k) +
                          getLogits(args.num_attr, scale, x_emb, txt_dist, k=args.k))
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu()); all_labels.append(y.cpu())
    P = torch.cat(all_probs); L = torch.cat(all_labels)
    correct = (P.argmax(1) == L).sum().item()
    acc = correct / L.size(0)
    nll = F.nll_loss(torch.log(P.clamp(1e-12)), L).item()
    ece, mce = compute_ece(P, L, n_bins=15)
    print(json.dumps({'ckpt':args.ckpt,'dataset':args.dataset,'method':'AttrVR',
                      'acc':acc,'ece':ece,'mce':mce,'nll':nll,'n':L.size(0)}))

if __name__ == '__main__':
    main()

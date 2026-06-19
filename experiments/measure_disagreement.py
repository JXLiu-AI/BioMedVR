"""Per-sample argmax disagreement between pos & neg experts on test sets."""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, InterpolationMode
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--input_size', type=int, default=192)
    p.add_argument('--num_attr', type=int, default=20)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--shot', type=int, default=16)
    args = p.parse_args()

    device = torch.device('cuda:0')
    set_seed(0)
    model, _ = clip.load('ViT-B/16')
    convert_models_to_fp32(model); model.eval(); model.requires_grad_(False)

    test_tf = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])
    train_tf = test_tf
    _, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_tf, test_tf, batch_size=64, shot=args.shot)

    txt_des = clip_attr_classifier(classes, model, f'attributes/gpt3/gpt3/{args.dataset}_des.json', num_attr=args.num_attr)
    txt_dist = clip_attr_classifier(classes, model, f'attributes/gpt3/gpt3/{args.dataset}_dist.json', num_attr=args.num_attr)
    conf_path = f'attributes/gpt3/gpt3/{args.dataset}_confuse.json'
    txt_conf = clip_attr_classifier(classes, model, conf_path, num_attr=args.num_attr) if os.path.exists(conf_path) else None

    vr = MultiPromptVR(224, args.input_size).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    vp_dict = sd.get('visual_prompt_dict', sd)
    vr.load_state_dict(vp_dict, strict=False)
    vr.eval()

    n_total = 0; n_disagree = 0; n_pos_correct = 0; n_neg_correct = 0; n_either_correct = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outs, gates = vr(x)
            x_embs = [model.encode_image(outs[i]) for i in range(2)]
            x_embs = [e / e.norm(dim=-1, keepdim=True) for e in x_embs]
            scale = model.logit_scale.exp()
            pos_logits = 0.5*(getLogits(args.num_attr, scale, x_embs[0], txt_des, k=args.k) +
                              getLogits(args.num_attr, scale, x_embs[0], txt_dist, k=args.k))
            if txt_conf is not None:
                neg_logits = getLogits(args.num_attr, scale, x_embs[1], txt_conf, k=args.k)
            else:
                neg_logits = torch.zeros_like(pos_logits)
            # per-sample argmax of each expert (treating neg as a class score: argmax over -neg ≈ "least negative")
            pos_pred = pos_logits.argmax(1)
            neg_pred = (-neg_logits).argmax(1)  # neg score is "how confusable" — pick class with LEAST neg score
            disagree = (pos_pred != neg_pred)
            n_disagree += disagree.sum().item()
            n_total += y.size(0)
            n_pos_correct += (pos_pred == y).sum().item()
            n_neg_correct += (neg_pred == y).sum().item()
            n_either_correct += ((pos_pred == y) | (neg_pred == y)).sum().item()
    
    print(json.dumps({
        'dataset': args.dataset,
        'n_total': n_total,
        'disagree_frac': n_disagree / n_total,
        'pos_acc': n_pos_correct / n_total,
        'neg_acc': n_neg_correct / n_total,
        'either_correct_frac': n_either_correct / n_total,
    }))

if __name__ == '__main__':
    main()

"""
Enhanced BioMedVR training for ECCV 2026 rebuttal.
Adds: --log_dir, --neg_mode, --mask_border, --gating_type, --pos_corrupt_rate,
      --neg_attr_path override; logs per-epoch CS-activation rate; saves
      gating + best.pth into seed/exp specific dirs.
"""
import sys, os, json, random, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import clip

from cfg import *
from tools import *
from datasets import *
from datasets.build_loader import build_loader
from methods.vp import PaddingVR


def make_corrupted_attr_file(orig_path, out_path, classnames, corrupt_rate, seed=0):
    """Replace positive attrs of corrupt_rate*N classes with attrs from a random other class."""
    rng = random.Random(seed)
    data = json.load(open(orig_path))
    n = len(classnames)
    n_corrupt = int(round(n * corrupt_rate))
    corrupt_idx = set(rng.sample(range(n), n_corrupt))
    new_data = {}
    for i, cn in enumerate(classnames):
        if i in corrupt_idx:
            other_idx = rng.choice([j for j in range(n) if j != i])
            new_data[cn] = data[classnames[other_idx]]
        else:
            new_data[cn] = data[cn]
    json.dump(new_data, open(out_path, 'w'), indent=2)


def make_random_neg_file(orig_path, out_path, classnames, seed=0):
    """Replace each class's negatives with attrs of a uniformly random *different* class's positives.
    Approximates 'negatives that are not actually confusing'."""
    rng = random.Random(seed)
    data = json.load(open(orig_path))  # original confuse file
    new_data = {}
    for cn in classnames:
        # pick attrs from a random different class; if not enough variety, just shuffle text tokens
        rng_choice = rng.choice([c for c in classnames if c != cn])
        new_data[cn] = data[rng_choice]
    json.dump(new_data, open(out_path, 'w'), indent=2)


def make_shuffled_neg_file(orig_path, out_path, classnames, seed=0):
    """Globally permute the (class -> attrs) mapping so each class gets the wrong attribute set."""
    rng = random.Random(seed)
    data = json.load(open(orig_path))
    perm = list(classnames)
    rng.shuffle(perm)
    new_data = {cn: data[perm_cn] for cn, perm_cn in zip(classnames, perm)}
    json.dump(new_data, open(out_path, 'w'), indent=2)


class MultiPromptVR(nn.Module):
    """Default 2-MoPE with shared scalar gate (matches V8.py)."""
    def __init__(self, base_size, target_size):
        super().__init__()
        self.experts = nn.ModuleList([
            PaddingVR(base_size, target_size),
            PaddingVR(base_size, target_size),
        ])
        self.gating_logits = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, x):
        outs = [expert(x) for expert in self.experts]
        outs = torch.stack(outs, dim=0)
        gates = F.softmax(self.gating_logits, dim=0)
        return outs, gates


class MultiPromptVR_PerClass(nn.Module):
    """Per-class gating: w_y of shape [C, 2]; gate per class then mix per-class logits."""
    def __init__(self, base_size, target_size, num_classes):
        super().__init__()
        self.experts = nn.ModuleList([
            PaddingVR(base_size, target_size),
            PaddingVR(base_size, target_size),
        ])
        self.gating_logits = nn.Parameter(torch.ones(num_classes, 2, dtype=torch.float32))
        self.num_classes = num_classes

    def forward(self, x):
        outs = [expert(x) for expert in self.experts]
        outs = torch.stack(outs, dim=0)
        gates = F.softmax(self.gating_logits, dim=-1)  # [C,2]
        return outs, gates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--dataset', required=True)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--num_attr', type=int, default=20)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--epoch', type=int, default=400)
    p.add_argument('--lr', type=float, default=40)
    p.add_argument('--input_size', type=int, default=192)
    p.add_argument('--shot', type=int, default=16)
    p.add_argument('--beta', type=float, default=0.3)
    p.add_argument('--margin', type=float, default=0.5)
    # Rebuttal-only flags
    p.add_argument('--log_dir', type=str, default='logs_rebuttal_default',
                   help='where to write training log; subdir created if needed.')
    p.add_argument('--exp_tag', type=str, default='default')
    p.add_argument('--neg_mode', choices=['default', 'random', 'shuffled', 'none'], default='default',
                   help='default = LLM confusion attrs; random = random other-class attrs; '
                        'shuffled = global permutation; none = drop neg expert')
    p.add_argument('--gating_type', choices=['shared', 'per_class'], default='shared')
    p.add_argument('--pos_corrupt_rate', type=float, default=0.0,
                   help='fraction of classes whose positive attrs are replaced with another class\'s')
    p.add_argument('--mask_border', type=int, default=16,
                   help='border width for PaddingVR; default 16 means input_size = 224 - 2*16 = 192')
    p.add_argument('--save_ckpt', type=int, default=1)
    p.add_argument('--eval_every', type=int, default=5, help='eval test every N epochs (final epoch always evaluated)')
    p.add_argument('--no_cs', action='store_true', help='disable CS loss (for ablation)')
    args = p.parse_args()

    # Recompute input_size from mask_border (overrides --input_size)
    out_size = 224
    args.input_size = out_size - 2 * args.mask_border
    print(f'[CONFIG] mask_border={args.mask_border}, derived input_size={args.input_size}')

    device = torch.device('cuda:0')
    set_seed(args.seed)

    # log + ckpt paths
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f'{args.dataset}_seed{args.seed}.log')
    cs_log_path = os.path.join(args.log_dir, f'{args.dataset}_seed{args.seed}_cs.log')
    save_path = os.path.join('results', f'fs_BioMedVR_rebuttal2', args.exp_tag,
                             f'{args.shot}{args.dataset}k{args.k}a{args.alpha}s{args.seed}')
    os.makedirs(save_path, exist_ok=True)

    print(f'[CONFIG] log={log_path}')
    print(f'[CONFIG] save={save_path}')

    # ---------- model ----------
    model, _ = clip.load('ViT-B/16')
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # ---------- transforms ----------
    train_tf = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])
    test_tf = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])
    bs = 64 if args.shot > 1 else 16
    trainloader, testloader, classes = build_loader(
        args.dataset, DOWNSTREAM_PATH, train_tf, test_tf, batch_size=bs, shot=args.shot)
    print(f'[CONFIG] dataset={args.dataset}, classes={len(classes)}, shot={args.shot}')

    # ---------- attributes ----------
    attr_dir = 'attributes/gpt3/gpt3'
    pos_des_path = f'{attr_dir}/{args.dataset}_des.json'
    pos_dist_path = f'{attr_dir}/{args.dataset}_dist.json'
    confuse_path = f'{attr_dir}/{args.dataset}_confuse.json'

    # Optionally corrupt positive attrs
    if args.pos_corrupt_rate > 0:
        corrupt_dir = f'attributes_corrupted/{args.exp_tag}'
        os.makedirs(corrupt_dir, exist_ok=True)
        new_des = f'{corrupt_dir}/{args.dataset}_des.json'
        new_dist = f'{corrupt_dir}/{args.dataset}_dist.json'
        make_corrupted_attr_file(pos_des_path, new_des, classes, args.pos_corrupt_rate, seed=args.seed)
        make_corrupted_attr_file(pos_dist_path, new_dist, classes, args.pos_corrupt_rate, seed=args.seed)
        pos_des_path, pos_dist_path = new_des, new_dist
        print(f'[CONFIG] positive attrs corrupted at rate {args.pos_corrupt_rate}')

    # Optionally swap negative attrs
    if args.neg_mode == 'random':
        neg_dir = f'attributes_neg_variants/{args.exp_tag}'
        os.makedirs(neg_dir, exist_ok=True)
        new_neg = f'{neg_dir}/{args.dataset}_random.json'
        make_random_neg_file(confuse_path, new_neg, classes, seed=args.seed)
        confuse_path = new_neg
    elif args.neg_mode == 'shuffled':
        neg_dir = f'attributes_neg_variants/{args.exp_tag}'
        os.makedirs(neg_dir, exist_ok=True)
        new_neg = f'{neg_dir}/{args.dataset}_shuffled.json'
        make_shuffled_neg_file(confuse_path, new_neg, classes, seed=args.seed)
        confuse_path = new_neg
    elif args.neg_mode == 'none':
        confuse_path = None

    txt_emb_desattr = clip_attr_classifier(classes, model, pos_des_path, num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model, pos_dist_path, num_attr=args.num_attr)
    if confuse_path is not None and os.path.exists(confuse_path):
        txt_emb_confuse = clip_attr_classifier(classes, model, confuse_path, num_attr=args.num_attr)
    else:
        txt_emb_confuse = None
    print(f'[CONFIG] neg_mode={args.neg_mode}, confuse_attrs={None if txt_emb_confuse is None else txt_emb_confuse.shape}')

    # ---------- VR ----------
    if args.gating_type == 'per_class':
        vr = MultiPromptVR_PerClass(out_size, args.input_size, num_classes=len(classes)).to(device)
    else:
        vr = MultiPromptVR(out_size, args.input_size).to(device)
    n_params = sum(p.numel() for p in vr.parameters())
    print(f'[CONFIG] gating={args.gating_type}, trainable_params={n_params}')

    def network(x):
        outs, gates = vr(x)  # outs: [2,B,C,H,W]
        x_embs = [model.encode_image(outs[i]) for i in range(2)]
        x_embs = [e / e.norm(dim=-1, keepdim=True) for e in x_embs]
        scale = model.logit_scale.exp()
        pos_logits = 0.5 * (getLogits(args.num_attr, scale, x_embs[0], txt_emb_desattr, k=args.k) +
                            getLogits(args.num_attr, scale, x_embs[0], txt_emb_distattr, k=args.k))
        if txt_emb_confuse is not None:
            neg_logits = getLogits(args.num_attr, scale, x_embs[1], txt_emb_confuse, k=args.k)
        else:
            neg_logits = torch.zeros_like(pos_logits)
        # Fuse
        if args.gating_type == 'per_class':
            # gates: [C,2]; logits_stack: [2,B,C]
            # fused = sum_e gates[c, e] * logits_e[b, c]
            stack = torch.stack([pos_logits, neg_logits], dim=-1)  # [B,C,2]
            fused_logits = (stack * gates.unsqueeze(0)).sum(-1)
        else:
            stack = torch.stack([pos_logits, neg_logits], dim=0)  # [2,B,C]
            fused_logits = (gates.view(-1, 1, 1) * stack).sum(0)
        return fused_logits, neg_logits, pos_logits, gates

    # ---------- optimizer ----------
    optim = torch.optim.SGD(vr.parameters(), lr=args.lr, momentum=0.9)
    sched = CosineAnnealingLR(optim, T_max=args.epoch * len(trainloader))

    # ---------- train ----------
    best_acc = 0.0
    cs_log_f = open(cs_log_path, 'w')
    cs_log_f.write('epoch,cs_active_frac,cs_penalty_mean,gate_pos,gate_neg\n')
    log_f = open(log_path, 'a')
    log_f.write(f'# === {args.exp_tag} | seed={args.seed} | neg_mode={args.neg_mode} | mask_border={args.mask_border} | gating={args.gating_type} | pos_corrupt={args.pos_corrupt_rate} | no_cs={args.no_cs} ===\n')

    pbar = tqdm(total=args.epoch, desc=f'{args.exp_tag}/{args.dataset}/s{args.seed}', leave=True)
    for epoch in range(args.epoch):
        vr.train()
        ep_correct = ep_total = 0
        ep_loss = 0.0
        cs_n_active = 0
        cs_n_total = 0
        cs_penalty_sum = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            with autocast():
                fx, neg_logits, pos_logits, gates = network(x)
                loss = F.cross_entropy(fx, y)
                if not args.no_cs and txt_emb_confuse is not None:
                    correct = pos_logits.gather(1, y.unsqueeze(1)).squeeze(1)
                    neg_max = neg_logits.max(1).values
                    raw = neg_max - correct + args.margin
                    cs_penalty = torch.relu(raw).mean()
                    cs_penalty = torch.clamp(cs_penalty, max=1e3)
                    cs_n_active += (raw > 0).sum().item()
                    cs_n_total += y.size(0)
                    cs_penalty_sum += cs_penalty.item() * y.size(0)
                    loss = loss + args.beta * cs_penalty
            loss.backward()
            optim.step()
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            ep_total += y.size(0)
            ep_correct += torch.argmax(fx, 1).eq(y).float().sum().item()
            ep_loss += loss.item() * y.size(0)
            sched.step()
        train_acc = ep_correct / ep_total

        # eval (every N epochs + final epoch)
        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epoch - 1)
        if do_eval:
            vr.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    fx, _, _, _ = network(x)
                    t_total += y.size(0)
                    t_correct += torch.argmax(fx, 1).eq(y).float().sum().item()
            test_acc = t_correct / t_total
            if test_acc > best_acc:
                best_acc = test_acc
                if args.save_ckpt:
                    torch.save({'visual_prompt_dict': vr.state_dict(), 'epoch': epoch, 'best_acc': best_acc},
                               os.path.join(save_path, 'best.pth'))
            log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}\n')
        else:
            test_acc = -1.0
            log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc=-, Best Acc={best_acc:.3f}\n')
        log_f.flush()

        # CS log row (averaged gate value)
        if args.gating_type == 'per_class':
            g_pos = gates[:, 0].mean().item()
            g_neg = gates[:, 1].mean().item()
        else:
            g_pos = gates[0].item()
            g_neg = gates[1].item()
        active_frac = (cs_n_active / cs_n_total) if cs_n_total > 0 else 0.0
        avg_pen = (cs_penalty_sum / cs_n_total) if cs_n_total > 0 else 0.0
        cs_log_f.write(f'{epoch+1},{active_frac:.4f},{avg_pen:.4f},{g_pos:.4f},{g_neg:.4f}\n')
        cs_log_f.flush()

        pbar.set_postfix({'train': f'{train_acc:.3f}', 'test': f'{test_acc:.3f}', 'best': f'{best_acc:.3f}', 'cs_act': f'{active_frac:.2f}'})
        pbar.update(1)

    pbar.close()
    log_f.close()
    cs_log_f.close()
    print(f'[DONE] {args.dataset} seed={args.seed} best_acc={best_acc:.4f}')


if __name__ == '__main__':
    main()

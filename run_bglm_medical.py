"""BG-LM (BLM++) on our medical datasets using BG-LM's official mapping/reprogramming code.

Adapts:
- Our build_loader() for medical datasets (BUSI/BTMRI/Knee_xray/DermaMNIST etc., 16-shot, ViT-B/16)
- BG-LM's WatermarkingVR + label-mapping (ILM / BLM / BLM++)
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.expanduser('~/baselines/BayesianLM'))
sys.path.insert(0, os.path.expanduser('~/bio/AttrVR-main'))

from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import clip

# Our medical dataset infra
from cfg import DOWNSTREAM_PATH
from tools import set_seed, convert_models_to_fp32
from datasets.build_loader import build_loader

# BG-LM's modules
from reprogramming import WatermarkingVR
from mapping import (
    one2one_mappnig_matrix, blm_reweight_matrix, blmp_reweight_matrix,
    label_mapping_base, label_mapping_calculation
)
from data import DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES, get_saparate_text_embedding


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--dataset', required=True)
    p.add_argument('--mapping', choices=['ilm', 'blm', 'blmp'], default='blmp')
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=40)
    p.add_argument('--shot', type=int, default=16)
    p.add_argument('--input_size', type=int, default=224)
    p.add_argument('--log_dir', default='logs_baselines_bglm')
    p.add_argument('--save_path', default=None)
    p.add_argument('--eval_every', type=int, default=5)
    args = p.parse_args()

    device = torch.device('cuda:0')
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f'{args.dataset}_{args.mapping}_seed{args.seed}.log')
    if args.save_path is None:
        args.save_path = f'results/baseline_bglm/{args.mapping}_{args.dataset}_s{args.seed}'
    os.makedirs(args.save_path, exist_ok=True)

    model, _ = clip.load('ViT-B/16')
    convert_models_to_fp32(model)
    model.eval(); model.requires_grad_(False)

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
    trainloader, testloader, class_names = build_loader(
        args.dataset, DOWNSTREAM_PATH, train_tf, test_tf, batch_size=bs, shot=args.shot)
    print(f'[BG-LM] dataset={args.dataset} mapping={args.mapping} #cls={len(class_names)}')

    # Build text embeddings the BG-LM way (DEFAULT + ENSEMBLE templates)
    templates = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES
    txt_emb = torch.cat(get_saparate_text_embedding(class_names, templates, model)).to(device)
    print(f'  text embedding shape: {txt_emb.shape}  (num_templates × num_classes, dim)')

    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
        return logits

    # WatermarkingVR — BG-LM's input-side prompt (perimeter padding 30 px on 224)
    visual_prompt = WatermarkingVR(224, 30).to(device)
    optimizer = torch.optim.SGD(visual_prompt.parameters(), lr=args.lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch * len(trainloader))
    scaler = GradScaler()

    log_f = open(log_path, 'a')
    log_f.write(f'# BG-LM {args.mapping} {args.dataset} seed={args.seed}\n')
    best_acc = 0.0
    pbar = tqdm(total=args.epoch, desc=f'BG-LM/{args.mapping}/{args.dataset}', leave=True)

    for epoch in range(args.epoch):
        # Re-compute label mapping each epoch (BG-LM's algorithm)
        if args.mapping == 'ilm':
            mapping = one2one_mappnig_matrix(visual_prompt, network, trainloader)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping)
        elif args.mapping == 'blm':
            mapping = blm_reweight_matrix(visual_prompt, network, trainloader, lap=1)
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping)
        elif args.mapping == 'blmp':
            mapping = blmp_reweight_matrix(visual_prompt, network, trainloader, lap=1,
                                           k=int(len(class_names) * 0.15) or 1)
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping)

        visual_prompt.train()
        ep_correct = ep_total = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = label_mapping(network(visual_prompt(x)))
                loss = nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total += y.size(0)
            scheduler.step()
        train_acc = ep_correct / ep_total

        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epoch - 1)
        if do_eval:
            visual_prompt.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    logits = label_mapping(network(visual_prompt(x)))
                    t_correct += (logits.argmax(1) == y).sum().item()
                    t_total += y.size(0)
            test_acc = t_correct / t_total
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({'visual_prompt': visual_prompt.state_dict(),
                            'mapping_matrix': mapping, 'epoch': epoch, 'best_acc': best_acc},
                           os.path.join(args.save_path, 'best.pth'))
            log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}\n')
        else:
            log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc=-, Best Acc={best_acc:.3f}\n')
        log_f.flush()
        pbar.set_postfix({'train': f'{train_acc:.3f}', 'best': f'{best_acc:.3f}'})
        pbar.update(1)
    pbar.close()
    log_f.close()
    print(f'[DONE] BG-LM {args.mapping} {args.dataset} best={best_acc:.4f}')

if __name__ == '__main__':
    main()

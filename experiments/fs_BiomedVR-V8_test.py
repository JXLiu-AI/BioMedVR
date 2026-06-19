import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
from cfg import *
from tools import *
from datasets import *
from datasets.build_loader import build_loader
from methods.vp import PaddingVR

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101',
                                         'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101', 'resisc45', 'I',
                                         'busi', 'knee_xray', 'kvasir', 'lung_colon', 'octmnist', 'btmri', 'chmnist', 'covid_19', 'ctkidney', 'dermamnist', 'retina'],
               default='dtd')
    p.add_argument('--alpha', type=float, default=0.5) # alpha: the balance hyperparameter lambda
    p.add_argument('--num_attr', type=int, default=20) # number of attributes
    p.add_argument('--k', type=int, default=3) # k for knn attribute selection
    p.add_argument('--epoch', type=int, default=400)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    p.add_argument('--beta', type=float, default=0.1, help='confuse loss weight')
    p.add_argument('--margin', type=float, default=0.5, help='margin for confuse loss')
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_biomedvr"
    save_path = os.path.join(exp, str(args.shot) + args.dataset  + 'k' + str(args.k) + 'a' + str(args.alpha) + 's' + str(args.seed))

    # 加载 pretrained CLIP
    model, _ = clip.load("ViT-B/16")
    # model, _ = clip.load("RN50")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # 数据增强
    train_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    # 加载数据集和文本特征
    # if args.shot == 1: bs = 16
    # else: bs = 64
    if args.shot == 1: bs = 256
    else: bs = 512
    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=bs, shot=args.shot)

    # Loading Attributes and Text Embeddings
    txt_emb_desattr = clip_attr_classifier(classes, model, 'attributes/gpt3/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)
    # 加载混淆文本特征
    confuse_json = f'attributes/gpt3/gpt3/{args.dataset}_confuse.json'
    if os.path.exists(confuse_json):
        txt_emb_confuse = clip_attr_classifier(classes, model, confuse_json, num_attr=args.num_attr)
 #       txt_emb_confuse = None
    else:
        txt_emb_confuse = None

    # Visual Reprogramming
    class MultiPromptVR(nn.Module):
        """2专家VR：正类（des+dist）与负类（confuse），gating融合"""
        def __init__(self, base_size, target_size, device=None):
            super().__init__()
            self.experts = nn.ModuleList([
                PaddingVR(base_size, target_size),  # 正类
                PaddingVR(base_size, target_size),  # 负类
            ])
            self.gating_logits = nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.device = device

        def forward(self, x):
            outs = [expert(x) for expert in self.experts]
            outs = torch.stack(outs, dim=0)  # [2, B, C, H, W]
            gates = F.softmax(self.gating_logits, dim=0)  # [2]
            return outs, gates

    visual_reprogram = MultiPromptVR(224, args.input_size, device=device).to(device)

    def network(x):
        outs, gates = visual_reprogram(x)
        x_embs = [model.encode_image(outs[i]) for i in range(2)]
        x_embs = [x_emb / x_emb.norm(dim=-1, keepdim=True) for x_emb in x_embs]
        gates = gates.view(-1, 1, 1)
        fused_emb = (gates[0] * x_embs[0] + gates[1] * x_embs[1])
        fused_emb = fused_emb / fused_emb.norm(dim=-1, keepdim=True)
        exp = model.logit_scale.exp()
        pos_logits = 0.5 * (getLogits(args.num_attr, exp, x_embs[0], txt_emb_desattr, k=args.k) +
                            getLogits(args.num_attr, exp, x_embs[0], txt_emb_distattr, k=args.k))
        if txt_emb_confuse is not None:
            neg_logits = getLogits(args.num_attr, exp, x_embs[1], txt_emb_confuse, k=args.k)
        else:
            neg_logits = torch.zeros_like(pos_logits)
        logits_stack = torch.stack([pos_logits, neg_logits], dim=0)
        fused_logits = (gates.view(-1, 1, 1) * logits_stack).sum(dim=0)
        return fused_emb, fused_logits

    # 加载已训练好的 VR 模型并生成 t-SNE
    import torch
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoint_path = os.path.join(save_path, 'best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    visual_reprogram.load_state_dict(state_dict['visual_prompt_dict'])
    print(f"Loaded VR weights from {checkpoint_path}")
    print("Trainable VR parameters:")
    total_vr_params = 0
    for name, param in visual_reprogram.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            print(f"  {name:<24} shape={tuple(param.shape)}, numel={numel:,}")
            total_vr_params += numel
    print(f"Total trainable VR params: {total_vr_params:,}")

    visual_reprogram.eval()
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            fused_emb, _ = network(x)
            all_embeds.append(fused_emb.cpu().numpy())
            all_labels.append(y.numpy())
    all_embeds = np.concatenate(all_embeds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(all_embeds)

    plt.figure(figsize=(7, 5))
    cmap = plt.cm.get_cmap('Spectral', len(classes))
    colors = [cmap(i) for i in range(len(classes))]
    for idx, cname in enumerate(classes):
        mask = (all_labels == idx)
        if mask.any():
            plt.scatter(
                embeds_2d[mask, 0],
                embeds_2d[mask, 1],
                s=36,
                color=colors[idx],
                edgecolors='white',
                linewidths=0.6,
                alpha=0.85,
                label=cname,
            )
    plt.legend(fontsize=8, frameon=True)
    plt.title('t-SNE of VR Embeddings')
    plt.tight_layout()
    # out_path = os.path.join(save_path, 'tsne_vr.png')
    out_path = os.path.join(save_path, 'tsne_vr.svg')
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.savefig(out_path, dpi=300)
    print(f"t-SNE plot saved to {out_path}")

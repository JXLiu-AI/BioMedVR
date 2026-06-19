import argparse
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, CenterCrop
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from cfg import *
from tools import *
from datasets import *
from datasets import basic_template
from datasets.build_loader import build_loader
from methods.vp import WatermarkingVR

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101',
                                         'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101', 'resisc45', 'I',
                                         'busi', 'knee_xray', 'kvasir', 'lung_colon', 'octmnist', 'btmri', 'chmnist', 'covid_19', 'ctkidney', 'dermamnist', 'retina'],
           default='dtd')
    p.add_argument('--epoch', type=int, default=200) # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--pad', type=float, default=30) # default padding size for VR
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_vp"
    save_path = os.path.join(exp, str(args.shot) + args.dataset + 's' + str(args.seed))

    # Load pretrained CLIP
    model, _ = clip.load("ViT-B/16")
    # model, _ = clip.load("RN50")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # Data augmentation
    train_process = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    test_process = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Loading dataset
    if args.shot == 1: bs = 16
    else: bs = 64
    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=bs, shot=args.shot)

    # Loading Labels and Text Embeddings
    TEMPLATES = [basic_template]
    txt_emb = clip_classifier(classes, TEMPLATES, model)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        return logits

    # Visual Reprogramming
    visual_reprogram = WatermarkingVR(224, args.pad).to(device)

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
            vr_x = visual_reprogram(x)
            x_emb = model.encode_image(vr_x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            all_embeds.append(x_emb.cpu().numpy())
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
    out_path = os.path.join(save_path, 'tsne_vr.png')
    plt.savefig(out_path, dpi=300)
    print(f"t-SNE plot saved to {out_path}")
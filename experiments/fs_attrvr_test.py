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
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_attrvr"
    save_path = os.path.join(exp, str(args.shot) + args.dataset  + 'k' + str(args.k) + 'a' + str(args.alpha) + 's' + str(args.seed))

    # Load pretrained CLIP
    model, _ = clip.load("ViT-B/32")
    # model, _ = clip.load("RN50")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # Data augmentation
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

    # Loading dataset
    if args.shot == 1: bs = 16
    else: bs = 64
    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=bs, shot=args.shot)

    # Loading Attributes and Text Embeddings
    txt_emb_desattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        exp = model.logit_scale.exp()
        desattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_desattr, k=args.k)
        distattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_distattr, k=args.k)
        return desattr_logits * args.alpha + distattr_logits * (1 - args.alpha)

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)

    # 加载已训练好的 VR 模型并打印权重规模
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
    model.eval()

    # 收集测试集嵌入并可视化 t-SNE
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    all_embeds, all_labels = [], []
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            vr_x = visual_reprogram(x)
            fused_logits = network(vr_x)
            all_embeds.append(fused_logits.cpu().numpy())
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

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
from tool_copy import *
from datasets import *
# from datasets.build_loader import build_loader
from methods.vp import PaddingVR
import json  # 新增
from open_clip import create_model_from_pretrained, get_tokenizer
import torch

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
    p.add_argument('--input_size', type=int, default=224) # 224*224 images for CLIP ViT-B/16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    p.add_argument('--beta', type=float, default=0.1, help='confuse loss weight')
    p.add_argument('--margin', type=float, default=0.5, help='margin for confuse loss')
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_attrvr"
    save_path = os.path.join(exp, str(args.shot) + args.dataset  + 'k' + str(args.k) + 'a' + str(args.alpha) + 's' + str(args.seed))

    # 加载BioMedCLIP
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Data augmentation
    train_process = preprocess
    test_process = preprocess

    # Loading dataset
    if args.shot == 1: bs = 256
    else: bs = 512
    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=bs, shot=args.shot)

    # Loading Attributes and Text Embeddings
    txt_emb_desattr = clip_attr_classifier(classes, model.module, 'attributes/gpt3/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model.module, 'attributes/gpt3/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)
    confuse_json = f'attributes/gpt3/gpt3/{args.dataset}_confuse.json'

    # 预检查并动态设置混淆文本 num_attr
    txt_emb_confuse = None
    confuse_num_attr = None
    if os.path.exists(confuse_json):
        try:
            with open(confuse_json, 'r', encoding='utf-8') as f:
                confuse_data = json.load(f)
            per_class_counts = {}
            for c in classes:
                vals = confuse_data.get(c, None)
                if isinstance(vals, list):
                    cleaned = [s.strip() for s in vals if isinstance(s, str) and s.strip()]
                    per_class_counts[c] = len(cleaned)
                else:
                    per_class_counts[c] = 0
            missing = [c for c, n in per_class_counts.items() if n == 0]
            if len(missing) == 0:
                confuse_num_attr = min(args.num_attr, min(per_class_counts.values()))
                txt_emb_confuse = clip_attr_classifier(classes, model.module, confuse_json, num_attr=confuse_num_attr)
                print(f'[INFO] Confuse embeddings enabled. per_class_confuse_attrs={confuse_num_attr}')
            else:
                print(f'[WARN] {confuse_json} missing classes: {missing[:10]}{" ..." if len(missing) > 10 else ""}. Skip confuse embeddings.')
        except Exception as e:
            print(f'[WARN] Failed to load/parse {confuse_json}: {e}. Skip confuse embeddings.')

    # Zero-shot prediction
    total_num = 0
    true_num = 0
    confuse_margins = []
    progress_bar = tqdm(testloader, desc='Zero-shot Predict', leave=True)
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x_emb = model.module.encode_image(x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp = model.module.logit_scale.exp()
            desattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_desattr, k=args.k)
            distattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_distattr, k=args.k)
            attr_logits = desattr_logits * args.alpha + distattr_logits * (1 - args.alpha)
            if txt_emb_confuse is not None:
                n_conf = confuse_num_attr if confuse_num_attr is not None else args.num_attr
                confuse_logits = getLogits(n_conf, exp, x_emb, txt_emb_confuse, k=args.k)
                fx = attr_logits - 0.3 * confuse_logits
                correct_score = attr_logits.gather(1, y.unsqueeze(1)).squeeze(1)
                confuse_max = confuse_logits.max(1).values
                confuse_margins.extend((correct_score - confuse_max).cpu().numpy().tolist())
            else:
                fx = attr_logits
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        acc = true_num / total_num
        progress_bar.set_postfix({'Acc': acc}, refresh=False)
    progress_bar.close()
    print(f'Zero-shot Test Acc: {true_num / total_num:.4f}')

    # 保存到日志
    log_dir = 'logs_biomedclip_confuse'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{args.dataset}.log')
    with open(log_path, 'a') as f:
        f.write(f'Zero-shot Test Acc={true_num / total_num:.4f}\n')
        if confuse_margins:
            f.write(f'Confuse Margin Mean={sum(confuse_margins)/len(confuse_margins):.4f}\n')
            f.write(f'Confuse Margin List={confuse_margins}\n')
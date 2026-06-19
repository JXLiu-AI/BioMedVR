import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from cfg import *
from tool_copy import *
from datasets import *
# from datasets.build_loader import build_loader
from methods.vp import PaddingVR
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
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=224) # 224*224 images for CLIP ViT-B/16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    # # Load pretrained CLIP
    # model, _ = clip.load("ViT-B/16")
    # convert_models_to_fp32(model)
    # model.eval()
    # model.requires_grad_(False)

    # 加载BioMedCLIP
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Data augmentation
    test_process = preprocess

    # Loading dataset
    batch_size = 64
    _, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, test_process, test_process, batch_size=batch_size, shot=args.shot)

    # Loading Attributes and Text Embeddings
    txt_emb_desattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)

    # Loading原始文本特征
    # 只用 "a photo of {class}" 作为类别描述
    class_prompts = [f"a photo of {c}" for c in classes]
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    with torch.no_grad():
        txt_tokens = tokenizer(class_prompts).to(device)
        txt_emb = model.module.encode_text(txt_tokens)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    # Zero-shot prediction
    total_num = 0
    true_num = 0
    progress_bar = tqdm(testloader, desc='Zero-shot Predict', leave=True)
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x_emb = model.module.encode_image(x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            logits = (x_emb @ txt_emb.t()) * model.module.logit_scale.exp()
        total_num += y.size(0)
        true_num += torch.argmax(logits, 1).eq(y).float().sum().item()
        acc = true_num / total_num
        progress_bar.set_postfix({'Acc': acc}, refresh=False)
    progress_bar.close()
    print(f'Zero-shot Test Acc: {true_num / total_num:.4f}')

    # 保存到日志
    log_dir = 'log_biomedclip_raw'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{args.dataset}.log')
    with open(log_path, 'a') as f:
        f.write(f'Zero-shot Test Acc={true_num / total_num:.4f}\n')
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
import torch
from cfg import *
from tool_copy import *
from datasets import *
from datasets.build_loader import build_loader
from methods.vp import PaddingVR
from open_clip import create_model_from_pretrained, get_tokenizer
import torchvision

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
    p.add_argument('--lr', type=float, default=40) # the initial learning rate（建议从1开始）
    p.add_argument('--input_size', type=int, default=224) # 224*224 images with VR pattern frame size=16
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
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    # 冻结BioMedCLIP参数
    for param in model.parameters():
        param.requires_grad = False

    # 数据增强
    train_process = Compose([
        RandomResizedCrop(args.input_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        # 加强颜色扰动
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        preprocess.transforms[-1] if hasattr(preprocess, 'transforms') and hasattr(preprocess.transforms[-1], '__call__') else nn.Identity(),
    ])
    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        preprocess.transforms[-1] if hasattr(preprocess, 'transforms') and hasattr(preprocess.transforms[-1], '__call__') else nn.Identity(),
    ])

    # 加载数据集
    if args.shot == 1: bs = 32
    else: bs = 64
    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=bs, shot=args.shot)

    # 加载属性和文本特征
    txt_emb_desattr = clip_attr_classifier(classes, model, 'attributes/gpt3/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)
    confuse_json = f'attributes/gpt3/gpt3/{args.dataset}_confuse.json'
    if os.path.exists(confuse_json):
        txt_emb_confuse = clip_attr_classifier(classes, model, confuse_json, num_attr=args.num_attr)
    else:
        txt_emb_confuse = None

    # Visual Reprogramming
    # 只保留单专家VR
    visual_reprogram = PaddingVR(args.input_size, 224).to(device)

    # patch expert 的 mask
    if hasattr(visual_reprogram, "mask"):
        visual_reprogram.mask.data.fill_(1)

    def network(x):
        x_out = visual_reprogram(x)
        x_emb = model.module.encode_image(x_out)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        exp = model.module.logit_scale.exp()
        # desattr和distattr做logits，取加权平均
        desattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_desattr, k=args.k)
        distattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_distattr, k=args.k)
        return desattr_logits * args.alpha + distattr_logits * (1 - args.alpha)

    # 优化器
    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    t_max = args.epoch * len(trainloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # 训练
    visual_reprogram.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training', leave=True)

    for epoch in range(args.epoch):
        visual_reprogram.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                fx = network(x)
                loss = F.cross_entropy(fx, y, reduction='mean')
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(visual_reprogram.parameters(), max_norm=5.0)
            optimizer.step()
            model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            scheduler.step()
        train_acc = true_num / total_num

        # 测试
        visual_reprogram.eval()
        total_num = 0
        true_num = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                fx = network(x)
                total_num += y.size(0)
                true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Acc': train_acc, 'Test Acc': test_acc, 'Best Acc': best_acc}, refresh=False)

        # 日志保存
        log_dir = 'logs_BiomedVR_all_h20_v10_Bio'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{args.dataset}.log')
        with open(log_path, 'a') as f:
            f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}\n')

        # Save
        state_dict = {
            "visual_prompt_dict": visual_reprogram.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        progress_bar.update(1)

    progress_bar.close()

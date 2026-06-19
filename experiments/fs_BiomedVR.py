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
    p.add_argument('--epoch', type=int, default=400)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    p.add_argument('--beta', type=float, default=0.1, help='confuse loss weight')
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_attrvr"
    save_path = os.path.join(exp, str(args.shot) + args.dataset  + 'k' + str(args.k) + 'a' + str(args.alpha) + 's' + str(args.seed))

    # Load pretrained CLIP
    model, _ = clip.load("ViT-B/16")
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
    # if args.shot == 1: bs = 16
    # else: bs = 64
    if args.shot == 1: bs = 256
    else: bs = 512
    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=bs, shot=args.shot)

    # Loading Attributes and Text Embeddings
    txt_emb_desattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)
    # 加载混淆文本特征
    confuse_json = f'attributes/gpt3/{args.dataset}_confuse.json'
    if os.path.exists(confuse_json):
        txt_emb_confuse = clip_attr_classifier(classes, model, confuse_json, num_attr=args.num_attr)
        txt_emb_confuse = None
    else:
        txt_emb_confuse = None

    # Repurposing CLIP for Downstream Classification
    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        exp = model.logit_scale.exp()
        desattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_desattr, k=args.k)
        distattr_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_distattr, k=args.k)
        confuse_logits = None
        if txt_emb_confuse is not None:
            confuse_logits = getLogits(args.num_attr, exp, x_emb, txt_emb_confuse, k=args.k)
        return desattr_logits * args.alpha + distattr_logits * (1 - args.alpha), confuse_logits

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(trainloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train
    visual_reprogram.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training', leave=True)

    for epoch in range(args.epoch):
        total_num = 0
        true_num = 0
        loss_sum = 0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx, confuse_logits = network(visual_reprogram(x))
                loss = F.cross_entropy(fx, y, reduction='mean')
                # 混淆损失
                if confuse_logits is not None:
                    confuse_loss = confuse_logits.mean()
                    loss = loss - args.beta * confuse_loss
            loss.backward()
            optimizer.step()

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            scheduler.step()
        train_acc = true_num / total_num

        # Test
        visual_reprogram.eval()
        total_num = 0
        true_num = 0
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx, _ = network(visual_reprogram(x))
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Acc': train_acc, 'Test Acc': test_acc, 'Best Acc': best_acc}, refresh=False)

        # 日志保存
        log_dir = 'logs_BimedVR'
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

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
import time
import subprocess
import statistics
import torch.nn as nn

from cfg import *
from tools import *
from datasets.build_loader import build_loader
from methods.vp import PaddingVR
from datasets import basic_template

class EmbeddingAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64, scale=1.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = scale
    def forward(self, x):
        # x: (B, D)
        res = self.up(self.act(self.down(x)))
        return x + self.scale * res

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101',
                                         'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101', 'resisc45', 'I',
                                         'busi', 'knee_xray', 'kvasir', 'lung_colon', 'octmnist', 'btmri', 'chmnist', 'covid_19', 'ctkidney', 'dermamnist', 'retina'],
           default='dtd')
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=4) # few-shot training set
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_ar"
    save_path = os.path.join(exp, str(args.shot) + args.dataset + 's' + str(args.seed))

    # Load pretrained CLIP
    model, _ = clip.load("ViT-B/16")
    # model, _ = clip.load("RN50")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # Adapter for parameter-efficient finetuning
    # determine embedding dim dynamically from model parameters
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = device
    with torch.no_grad():
        # try to use model's expected input resolution (e.g., 224 for ViT)
        input_res = getattr(model, 'input_resolution', 224)
        dummy = torch.zeros(1, 3, input_res, input_res).to(param_device)
        try:
            emb = model.encode_image(dummy)
            emb_dim = emb.shape[-1]
        except RuntimeError:
            # fallback: try 224 explicitly, then infer from model attributes
            try:
                dummy = torch.zeros(1, 3, 224, 224).to(param_device)
                emb = model.encode_image(dummy)
                emb_dim = emb.shape[-1]
            except Exception:
                if hasattr(model, 'visual') and hasattr(model.visual, 'output_dim'):
                    emb_dim = model.visual.output_dim
                elif hasattr(model, 'ln_final') and hasattr(model.ln_final, 'normalized_shape'):
                    emb_dim = int(model.ln_final.normalized_shape[0])
                else:
                    raise
    adapter = EmbeddingAdapter(emb_dim, bottleneck=64, scale=1.0).to(device)


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

    # Loading Labels and Text Embeddings
    TEMPLATES = [basic_template]
    txt_emb = clip_classifier(classes, TEMPLATES, model)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        x_emb = model.encode_image(x)
        # pass through adapter for PEFT
        x_emb = adapter(x_emb)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        return logits

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)
    # Freeze visual_reprogram parameters for adapter-based PEFT
    for p in visual_reprogram.parameters():
        p.requires_grad = False

    # Print parameter statistics
    def _print_param_stats():
        total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in visual_reprogram.parameters()) + sum(p.numel() for p in adapter.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in visual_reprogram.parameters() if p.requires_grad) + sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for p in adapter.parameters())
        mb_total = total * 4 / 1024**2
        mb_train = trainable * 4 / 1024**2
        print(f'[Params] Total: {total} ({mb_total:.2f} MB) | Trainable: {trainable} ({mb_train:.2f} MB) | Adapter: {adapter_params} params')
    _print_param_stats()

    # Optimizer (train adapter parameters only for PEFT)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(trainloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train
    visual_reprogram.train()
    adapter.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training', leave=True)
    training_time_accum = 0.0

    for epoch in range(args.epoch):
        # training timing and gpu monitoring
        train_start = time.time()
        train_gpu_utils = []
        total_num = 0
        true_num = 0
        loss_sum = 0
        visual_reprogram.train()
        adapter.train()
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                fx = network(visual_reprogram(x))
                loss = F.cross_entropy(fx, y, reduction='mean')
            loss.backward()
            optimizer.step()
            # sample GPU utilization
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL
                ).decode('utf-8').strip().split('\n')[0]
                util, mem_used, mem_total = [int(x.strip()) for x in out.split(',')]
                train_gpu_utils.append(util)
            except Exception:
                pass

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            scheduler.step()
        train_time = time.time() - train_start
        train_acc = true_num / total_num
        avg_train_gpu = statistics.mean(train_gpu_utils) if len(train_gpu_utils)>0 else None
        training_time_accum += train_time

        # Test (inference timing and gpu monitoring)
        visual_reprogram.eval()
        total_num = 0
        true_num = 0
        test_gpu_utils = []
        test_start = time.time()
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(visual_reprogram(x))
            # sample GPU utilization
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL
                ).decode('utf-8').strip().split('\n')[0]
                util, mem_used, mem_total = [int(x.strip()) for x in out.split(',')]
                test_gpu_utils.append(util)
            except Exception:
                pass
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_time = time.time() - test_start
        test_acc = true_num / total_num
        avg_test_gpu = statistics.mean(test_gpu_utils) if len(test_gpu_utils)>0 else None

        progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Acc': train_acc, 'Test Acc': test_acc, 'Best Acc': best_acc,
                                  'Ttrain(s)': f'{train_time:.1f}', 'Ttest(s)': f'{test_time:.1f}'}, refresh=False)

        # 日志保存
        log_dir = 'logs_ar_ViT-B16_32shot_rebuttal'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{args.dataset}.log')
        with open(log_path, 'a') as f:
            f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}, '
                    f'TrainTime={train_time:.3f}, TestTime={test_time:.3f}, '
                    f'AvgGPUTrain={avg_train_gpu if avg_train_gpu is not None else "NA"}, '
                    f'AvgGPUTest={avg_test_gpu if avg_test_gpu is not None else "NA"}\n')

        # Save
        state_dict = {
            "visual_prompt_dict": visual_reprogram.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        progress_bar.update(1)

    progress_bar.close()
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

class LoRALinear(nn.Module):
    """LoRA for nn.Linear: keeps original weight frozen and adds a low-rank update W + alpha/r * (B @ A)
    Implemented as: y = x W^T + scaling * (x @ A^T @ B^T)
    """
    def __init__(self, orig_linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        # keep original linear as frozen parametrization
        self.orig = orig_linear
        self.orig.weight.requires_grad = False
        if self.orig.bias is not None:
            self.orig.bias.requires_grad = False
        # LoRA components
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.lora_down = nn.Linear(self.in_features, r, bias=False)
            self.lora_up = nn.Linear(r, self.out_features, bias=False)
            # initialize to zero so the pretrained behavior is preserved at start
            nn.init.zeros_(self.lora_down.weight)
            nn.init.zeros_(self.lora_up.weight)
            self.scaling = alpha / r
            # move LoRA params to the same device as the original weight
            device = self.orig.weight.device
            self.lora_down = self.lora_down.to(device)
            self.lora_up = self.lora_up.to(device)
        else:
            self.lora_down = None
            self.lora_up = None
            self.scaling = 0.0
    def forward(self, x):
        out = F.linear(x, self.orig.weight, self.orig.bias)
        if self.r > 0:
            # LoRA path: x -> down -> up
            delta = self.lora_up(self.lora_down(x)) * self.scaling
            out = out + delta
        return out

    # expose original weight and bias attributes for compatibility with code that accesses .weight/.bias
    @property
    def weight(self):
        return self.orig.weight

    @property
    def bias(self):
        return self.orig.bias


def apply_lora_to_module(module, r=4, alpha=1.0, whitelist=['attn', 'proj', 'mlp', 'q', 'k', 'v', 'fc']):
    """Recursively replace nn.Linear modules in `module` with LoRALinear if their qualified name matches whitelist.
    If whitelist is empty, replace all nn.Linear modules.
    """
    for name, child in list(module.named_children()):
        # determine whether to replace this child
        replace = False
        if isinstance(child, nn.Linear):
            if len(whitelist) == 0:
                replace = True
            else:
                for w in whitelist:
                    if w in name:
                        replace = True
                        break
        if replace:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            apply_lora_to_module(child, r=r, alpha=alpha, whitelist=whitelist)


def count_lora_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_lora_state_dict(module):
    """Collect state dicts of all LoRA modules for saving/reloading."""
    sd = {}
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            sd[name] = {
                'lora_down': None if child.lora_down is None else child.lora_down.state_dict(),
                'lora_up': None if child.lora_up is None else child.lora_up.state_dict(),
                'r': child.r,
                'alpha': child.alpha,
            }
    return sd

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
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=1.0)
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

    # LoRA injection for parameter-efficient finetuning
    # (lora params parsed earlier)

    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = device

    # Apply LoRA to the visual encoder if present, otherwise to whole model
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    if hasattr(model, 'visual'):
        apply_lora_to_module(model.visual, r=lora_r, alpha=lora_alpha, whitelist=['q', 'k', 'v', 'proj', 'mlp', 'fc', 'attn'])
    else:
        apply_lora_to_module(model, r=lora_r, alpha=lora_alpha, whitelist=['q', 'k', 'v', 'proj', 'mlp', 'fc', 'attn'])

    # Ensure all backbone params remain frozen, then enable LoRA params
    model.requires_grad_(False)
    lora_param_count = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            # enable LoRA trainable params
            for p in [m.lora_down.weight, m.lora_up.weight]:
                p.requires_grad = True
                lora_param_count += p.numel()
            # ensure module in train mode
            m.train()
    print(f'LoRA trainable params: {lora_param_count}')

    # additional parameter size summary
    lora_modules = [name for name, m in model.named_modules() if isinstance(m, LoRALinear) and m.r>0]
    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb_total = total_model_params * 4 / 1024**2
    mb_train = trainable_model_params * 4 / 1024**2
    print(f'[Params] Model total: {total_model_params} ({mb_total:.2f} MB) | Trainable: {trainable_model_params} ({mb_train:.2f} MB) | LoRA modules: {len(lora_modules)}')
    print('LoRA module names (sample):', lora_modules[:10])

    if lora_param_count == 0:
        print('Warning: No LoRA parameters found. Check whitelist or model structure.')
    # no separate adapter object; LoRA is applied inline into model layers


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
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        return logits

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)
    # Freeze visual_reprogram parameters for PEFT where applicable
    for p in visual_reprogram.parameters():
        p.requires_grad = False

    # Optimizer (train LoRA parameters only)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(lora_params, lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(trainloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train
    visual_reprogram.train()
    # set LoRA modules to train mode
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.train()
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
        # ensure LoRA modules are trainable
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.train()
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                fx = network(visual_reprogram(x))
                loss = F.cross_entropy(fx, y, reduction='mean')
            # backward/update only LoRA params
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
        log_dir = 'logs_ar_ViT-B16_32shot_rebuttal_lora'
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
            "lora_state_dict": get_lora_state_dict(model),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        progress_bar.update(1)

    progress_bar.close()
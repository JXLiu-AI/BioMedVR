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
    p.add_argument('--seed', type=int, default=1)
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
    p.add_argument('--beta', type=float, default=0.3, help='confuse loss weight')
    p.add_argument('--margin', type=float, default=0.5, help='margin for confuse loss')
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_BioMedVR-V8"
    save_path = os.path.join(exp, str(args.shot) + args.dataset  + 'k' + str(args.k) + 'a' + str(args.alpha) + 's' + str(args.seed))

    # Load pretrained CLIP
    model, _ = clip.load("ViT-B/16")
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
    # if args.shot == 1: bs = 256
    # else: bs = 512
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

    # 打印 VR 层参数数量和形状
    total_params = 0
    print("Trainable parameters in visual_reprogram:")
    for name, param in visual_reprogram.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={tuple(param.shape)}, numel={param.numel()}")
            total_params += param.numel()
    print(f"Total trainable parameters: {total_params}")

    def network(x):
        outs, gates = visual_reprogram(x)
        # outs: [2, B, C, H, W], gates: [2]
        x_embs = [model.encode_image(outs[i]) for i in range(2)]
        x_embs = [x_emb / x_emb.norm(dim=-1, keepdim=True) for x_emb in x_embs]
        exp = model.logit_scale.exp()
        # 正类专家：分别与des和dist做logits，取平均
        pos_logits = 0.5 * (getLogits(args.num_attr, exp, x_embs[0], txt_emb_desattr, k=args.k) +
                            getLogits(args.num_attr, exp, x_embs[0], txt_emb_distattr, k=args.k))
        # 负类专家：与confuse做logits
        if txt_emb_confuse is not None:
            neg_logits = getLogits(args.num_attr, exp, x_embs[1], txt_emb_confuse, k=args.k)
        else:
            neg_logits = torch.zeros_like(pos_logits)
        logits_stack = torch.stack([pos_logits, neg_logits], dim=0)  # [2, B, num_class]
        fused_logits = (gates.view(-1, 1, 1) * logits_stack).sum(dim=0)  # [B, num_class]
        # 返回fused_logits和neg_logits（用于混淆loss）
        return fused_logits, neg_logits

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
                fx, confuse_logits = network(x)
                loss = F.cross_entropy(fx, y, reduction='mean')
                # 混淆损失（margin 形式）：只有当混淆分数高于正确类一定阈值才惩罚
                if confuse_logits is not None:
                    # 正确类得分
                    correct = fx.gather(1, y.unsqueeze(1)).squeeze(1)
                    # 混淆分数取最大（对每个样本）
                    confuse_max = confuse_logits.max(1).values
                    # margin_loss：只有当 confuse_max - correct + margin > 0 时才有惩罚
                    confuse_penalty = torch.relu(confuse_max - correct + args.margin).mean()
                    # 可选：对惩罚值做上限，防止数值过大导致训练不稳定
                    confuse_penalty = torch.clamp(confuse_penalty, max=1e3)
                    # 使用负号来鼓励混淆（最小化 loss 会推动增大 confuse_penalty）
                    print(loss.item())
                    print("confuse_penalty:", confuse_penalty.item())
                    loss = loss + args.beta * confuse_penalty
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
                fx, _ = network(x)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        test_acc = true_num / total_num
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Acc': train_acc, 'Test Acc': test_acc, 'Best Acc': best_acc}, refresh=False)

        # 日志保存
        log_dir = 'logs_BiomedVR_all_h20_v10_params_search_rebuttal'
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

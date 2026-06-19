"""AttriPrompt minimal baseline (code not public).
Based on title "Class Attribute-Aware Prompt Tuning for Vision-Language Model" (Su et al. TIP 2026):
- CoOp-style learnable context tokens M × dim_text
- For each class y, prompt = [ctx_1 ... ctx_M] + [class_name_emb] + [attr_emb_y]
- attr_emb_y is the average of class y's GPT-generated descriptive attribute embeddings (frozen).

Trains the context tokens via cross-entropy on similarity scores.
Uses our medical dataset infra; ViT-B/16; 16-shot.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/bio/AttrVR-main"))

import json

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import DOWNSTREAM_PATH
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.build_loader import build_loader
from tools import convert_models_to_fp32, set_seed
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Lambda,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm import tqdm

_tokenizer = _Tokenizer()


class AttriPromptLearner(nn.Module):
    """Learn M context tokens + use per-class attribute bias.
    Inspired by AttriPrompt (TIP 2026). Approximates the official method.
    """

    def __init__(
        self,
        classnames,
        attribute_embeds_per_class,
        clip_model,
        n_ctx=16,
        ctx_init=None,
        attr_weight=0.5,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.attr_weight = attr_weight
        self.classnames = [c.replace("_", " ") for c in classnames]
        n_cls = len(self.classnames)

        # Token embedding from CLIP
        embedding = clip_model.token_embedding
        ctx_dim = embedding.weight.shape[1]
        dtype = embedding.weight.dtype

        # ctx_init: random init of context tokens
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Build prompt prefix tokens (SOS) and class-name tokens
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in self.classnames]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, 77]
        with torch.no_grad():
            embedded = embedding(tokenized.to(embedding.weight.device)).type(
                dtype
            )  # [n_cls, 77, ctx_dim]

        self.register_buffer("token_prefix", embedded[:, :1, :])  # [n_cls, 1, ctx_dim]  SOS
        self.register_buffer(
            "token_suffix", embedded[:, 1 + n_ctx :, :]
        )  # [n_cls, *, ctx_dim]  class+EOS
        self.register_buffer("tokenized", tokenized)  # [n_cls, 77]

        # Frozen class-attribute bias (avg of attribute embeddings)
        # attribute_embeds_per_class: [n_cls, n_attrs, ctx_dim]
        self.register_buffer(
            "attr_embed", attribute_embeds_per_class.mean(dim=1)
        )  # [n_cls, ctx_dim]

    def forward(self):
        # Concatenate ctx for each class: [n_cls, n_ctx, ctx_dim]
        ctx = self.ctx.unsqueeze(0).expand(self.token_prefix.size(0), -1, -1)
        prompts = torch.cat(
            [self.token_prefix, ctx, self.token_suffix], dim=1
        )  # [n_cls, 77, ctx_dim]
        return prompts  # logits computed externally


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(prompts.dtype)
        eot = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.size(0)), eot] @ self.text_projection
        return x  # [n_cls, embed_dim]


def get_attr_embeds(classnames, attr_files, clip_model, num_attr=20, device="cuda"):
    """Encode class attributes (des/dist/confuse merged) via CLIP text encoder.
    Returns tensor of shape [n_cls, n_attrs_per_class, embed_dim]."""
    import random

    all_classes_texts = []
    for cn in classnames:
        texts_for_class = []
        for af in attr_files:
            data = json.load(open(af))
            if cn not in data:
                continue
            attrs = [s for s in data[cn] if len(s) >= 20]
            if len(attrs) < num_attr:
                attrs = random.choices(attrs, k=num_attr) if attrs else []
            else:
                attrs = random.sample(attrs, num_attr)
            texts_for_class.extend(attrs)
        if not texts_for_class:
            texts_for_class = [cn] * num_attr
        all_classes_texts.append(texts_for_class)

    embeds = []
    with torch.no_grad():
        for texts in all_classes_texts:
            tok = clip.tokenize(texts).to(device)
            emb = clip_model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeds.append(emb)
    return torch.stack(embeds, dim=0)  # [n_cls, n_attrs, embed_dim]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dataset", required=True)
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--shot", type=int, default=16)
    p.add_argument("--n_ctx", type=int, default=16)
    p.add_argument("--num_attr", type=int, default=20)
    p.add_argument("--attr_weight", type=float, default=0.5)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--log_dir", default="logs_baselines_attriprompt")
    p.add_argument("--save_path", default=None)
    p.add_argument("--eval_every", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda:0")
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.dataset}_seed{args.seed}.log")
    if args.save_path is None:
        args.save_path = f"results/baseline_attriprompt/{args.dataset}_s{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)

    model, _ = clip.load("ViT-B/16")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    train_tf = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
        ]
    )
    test_tf = Compose(
        [
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(args.input_size),
            Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            ToTensor(),
        ]
    )
    bs = 64 if args.shot > 1 else 16
    trainloader, testloader, classes = build_loader(
        args.dataset, DOWNSTREAM_PATH, train_tf, test_tf, batch_size=bs, shot=args.shot
    )
    print(f"[AttriPrompt] dataset={args.dataset} #cls={len(classes)}")

    # Load class attributes (des + dist as attribute-aware prior)
    attr_files = [
        f"attributes/gpt3/gpt3/{args.dataset}_des.json",
        f"attributes/gpt3/gpt3/{args.dataset}_dist.json",
    ]
    attr_embeds = get_attr_embeds(
        classes, attr_files, model, num_attr=args.num_attr, device=device
    ).float()
    print(f"  attr_embeds: {attr_embeds.shape}")

    # Build prompt learner
    prompt_learner = (
        AttriPromptLearner(
            classes, attr_embeds, model, n_ctx=args.n_ctx, attr_weight=args.attr_weight
        )
        .to(device)
        .float()
    )
    text_encoder = TextEncoder(model).float().to(device)

    optimizer = torch.optim.SGD(prompt_learner.parameters(), lr=args.lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch * len(trainloader))
    scaler = GradScaler()

    # Build attr-bias projection (frozen)
    # attr_embed shape [n_cls, embed_dim_text]; we add weighted bias to text features
    attr_bias = prompt_learner.attr_embed.float()

    log_f = open(log_path, "a")
    log_f.write(f"# AttriPrompt {args.dataset} seed={args.seed}\n")
    best_acc = 0.0
    pbar = tqdm(total=args.epoch, desc=f"AttriPrompt/{args.dataset}", leave=True)

    def text_features():
        prompts = prompt_learner()
        text_feats = text_encoder(prompts, prompt_learner.tokenized)
        # Add attribute-aware bias
        text_feats = text_feats + args.attr_weight * attr_bias
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats

    for epoch in range(args.epoch):
        prompt_learner.train()
        ep_correct = ep_total = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                text_feats = text_features()
                with torch.no_grad():
                    img_feats = model.encode_image(x)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * img_feats @ text_feats.t()
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total += y.size(0)
            scheduler.step()
        train_acc = ep_correct / ep_total

        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epoch - 1)
        if do_eval:
            prompt_learner.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                text_feats = text_features()
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    img_feats = model.encode_image(x)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    logits = model.logit_scale.exp() * img_feats @ text_feats.t()
                    t_correct += (logits.argmax(1) == y).sum().item()
                    t_total += y.size(0)
            test_acc = t_correct / t_total
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    {
                        "prompt_learner": prompt_learner.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc,
                    },
                    os.path.join(args.save_path, "best.pth"),
                )
            log_f.write(
                f"Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}\n"
            )
        else:
            log_f.write(
                f"Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc=-, Best Acc={best_acc:.3f}\n"
            )
        log_f.flush()
        pbar.set_postfix({"train": f"{train_acc:.3f}", "best": f"{best_acc:.3f}"})
        pbar.update(1)
    pbar.close()
    log_f.close()
    print(f"[DONE] AttriPrompt {args.dataset} best={best_acc:.4f}")


if __name__ == "__main__":
    main()

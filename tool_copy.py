from datasets import build_dataset, build_data_loader
import torch
import random
import numpy as np
from open_clip import get_tokenizer
import open_clip
import json

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def clip_classifier(classnames, template, clip_model, tokenizer=None):
    '''
    Text encoder for label-based classification

    params:
    classnames: class name of the label space
    template: text prompts
    clip_model: the pretrained CLIP/open_clip
    tokenizer: open_clip tokenizer (可选)
    '''
    device = next(clip_model.parameters()).device
    if tokenizer is None:
        tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = tokenizer(texts).to(device)
            # 自动兼容 DataParallel
            model = clip_model.module if hasattr(clip_model, "module") else clip_model
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def clip_attr_classifier(classnames, clip_model, dir, num_attr, tokenizer=None):
    '''
    Text encoder for attribute-based classification.

    params:
    classnames: class name of the label space
    clip_model: the pretrained CLIP/open_clip
    dir: attribute path
    num_attr: 'm' in the paper, the attribute number
    tokenizer: open_clip tokenizer (可选)
    '''
    data = json.load(open(dir, 'r'))
    device = next(clip_model.parameters()).device
    if tokenizer is None:
        tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            filtered_sentences = [sentence for sentence in data[classname] if len(sentence) >= 20]
            if len(filtered_sentences) == 0:
                print(classname)
                raise ValueError("No valid attributes have been generated")
            if num_attr < len(filtered_sentences):
                texts = random.sample(filtered_sentences, num_attr)
            else:
                texts = random.choices(filtered_sentences, k=num_attr)
            texts = tokenizer(texts).to(device)
            # 自动兼容 DataParallel
            model = clip_model.module if hasattr(clip_model, "module") else clip_model
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def getLogits(num_attr, exp, x_emb, t_emb, k=3):
    '''
    Return the similarity logits output

    params:
    num_attr: 'm' in the paper, the attribute number
    exp: CLIP logit_scale
    x_emb: image embeddings
    t_emb: text embeddings
    '''
    fea_dim = t_emb.shape[-1]
    t_emb = t_emb.permute(2, 1, 0).reshape(fea_dim, -1)
    res = exp * x_emb @ t_emb
    bs = res.shape[0]
    res = res.reshape(bs, -1, num_attr)
    res, _ = torch.sort(res, dim=-1, descending=True)
    logits = torch.mean(res[:, :, :k], dim=2)
    return logits

def build_loader(dataset_name, root_path, train_preprocess=None, test_preprocess=None, batch_size=64, shot=16, seed=0):
    '''
    Retuen the loader of downstream tasks.

    params:
    dataset_name: downstream dataset name
    root_path: the path of dataset
    train_preprocess/test_preprocess: the data argumentation performed on samples
    batch_size: training batch size
    shot: the available number of samples per class
    seed: the random seed
    '''
    dataset = build_dataset(dataset_name, root_path, shot, seed)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=True, tfm=train_preprocess, shuffle=True)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, tfm=test_preprocess, shuffle=False)
    return train_loader, test_loader, dataset.classnames
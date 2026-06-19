import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .busi import BUSI
from .knee_xray import KneeXray
from .kvasir import Kvasir
from .lung_colon import LungColon
from .octmnist import OCTMNIST
from .btmri import BTMRI
from .chmnist import CHMNIST
from .covid_19 import COVID19
from .ctkidney import CTKidney
from .dermamnist import DermaMNIST
from .retina import RETINA

class SimpleImageDataset(Dataset):
    def __init__(self, data, image_root, transform=None):
        self.data = data
        self.image_root = image_root
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        datum = self.data[idx]
        img_path = datum.impath if hasattr(datum, 'impath') else datum[0]
        label = datum.label if hasattr(datum, 'label') else datum[1]
        # 修正路径拼接，避免重复
        if os.path.isabs(img_path):
            img_full_path = img_path
        elif img_path.startswith(self.image_root):
            img_full_path = img_path
        else:
            img_full_path = os.path.join(self.image_root, img_path)
        img = Image.open(img_full_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def build_loader(name, root, train_transform, test_transform, batch_size=64, shot=16):
    dataset_map = {
        'busi': BUSI,
        'knee_xray': KneeXray,
        'kvasir': Kvasir,
        'lung_colon': LungColon,
        'octmnist': OCTMNIST,
        'btmri': BTMRI,
        'chmnist': CHMNIST,
        'covid_19': COVID19,
        'ctkidney': CTKidney,
        'dermamnist': DermaMNIST,
        'retina': RETINA
    }
    if name in dataset_map:
        dataset = dataset_map[name](root, shot, seed=0)
        image_root = dataset.image_dir if hasattr(dataset, 'image_dir') else dataset.dataset_dir
        train_set = SimpleImageDataset(dataset.train_x, image_root, transform=train_transform)
        test_set = SimpleImageDataset(dataset.test, image_root, transform=test_transform)
        classes = dataset.classnames if hasattr(dataset, 'classnames') else list(set([x[2] for x in dataset.train_x+dataset.test]))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, classes
    raise NotImplementedError(f"Dataset {name} not supported.")
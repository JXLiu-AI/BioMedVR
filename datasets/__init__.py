from .btmri import BTMRI
from .busi import BUSI
from .caltech101 import Caltech101
from .chmnist import CHMNIST
from .covid_19 import COVID19
from .ctkidney import CTKidney
from .dermamnist import DermaMNIST
from .dtd import DescribableTextures
from .eurosat import EuroSAT
from .fgvc_aircraft import FGVCAircraft
from .food101 import Food101
from .imagenet import ImageNet
from .knee_xray import KneeXray
from .kvasir import Kvasir
from .lung_colon import LungColon
from .octmnist import OCTMNIST
from .oxford_flowers import OxfordFlowers
from .oxford_pets import OxfordPets
from .resisc45 import Resisc45
from .retina import RETINA
from .stanford_cars import StanfordCars
from .sun397 import SUN397
from .ucf101 import UCF101
from .utils import *

basic_template = "This is a photo of {}."
dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "resisc45": Resisc45,
    "imagenet": ImageNet,
    "busi": BUSI,
    "knee_xray": KneeXray,
    "kvasir": Kvasir,
    "lung_colon": LungColon,
    "octmnist": OCTMNIST,
    "btmri": BTMRI,
    "chmnist": CHMNIST,
    "covid_19": COVID19,
    "ctkidney": CTKidney,
    "dermamnist": DermaMNIST,
    "retina": RETINA,
}


def build_dataset(dataset, root, shot=16, seed=0):
    return dataset_list[dataset](root, shot, seed)

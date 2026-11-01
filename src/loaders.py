"""PyTorch Dataset + DataLoaders on top of src/data.py.
All images are decoded once into RAM. Augmentation is applied to the TRAIN set only.
"""
from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data import CAT_BREEDS, get_splits, load_images

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # just normalisation constants


def eval_transform():
    """Validation / test / predict: no augmentation (images are already resized by src/data.py)."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_transform(img_size: int, augment: bool = True):
    if not augment:
        return eval_transform()
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # mild: colour matters for breeds
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CatDataset(Dataset):
    def __init__(self, items, img_size, transform):
        self.paths = [p for p, _ in items]
        self.labels = [y for _, y in items]
        self.images = load_images(items, img_size)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.labels[i]


def get_train_val_loaders(root="data", batch_size=32, img_size=128, augment=True, seed=42,
                          val_frac=0.2, download=True):
    """Train + validation loaders. The test images are never decoded here."""
    train, val, _ = get_splits(root, val_frac, seed, download)
    g = torch.Generator()
    g.manual_seed(seed)  # fixed shuffling order
    train_loader = DataLoader(CatDataset(train, img_size, train_transform(img_size, augment)),
                              batch_size=batch_size, shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(CatDataset(val, img_size, eval_transform()), batch_size=batch_size)
    return train_loader, val_loader, list(CAT_BREEDS)


def get_test_loader(root="data", batch_size=64, img_size=128, seed=42, val_frac=0.2, download=False):
    """Test loader (used only by evaluate.py). Returns (loader, list of image paths)."""
    _, _, test = get_splits(root, val_frac, seed, download)
    ds = CatDataset(test, img_size, eval_transform())
    return DataLoader(ds, batch_size=batch_size), ds.paths

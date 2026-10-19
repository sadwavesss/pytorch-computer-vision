"""BUSI (Breast Ultrasound Images) dataset: preparation, loading and augmentation.

The raw dataset has `benign/`, `malignant/` and `normal/` folders with images and one or more
`*_mask*.png` files per image. It is converted to a binary task: lesion (benign or malignant) vs background.
"""

import random
import shutil
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CATEGORIES = ("benign", "malignant", "normal")


def find_busi_root(path: Path) -> Path:
    for candidate in [path, *path.rglob("*")]:
        if candidate.is_dir() and all((candidate / c).is_dir() for c in CATEGORIES):
            return candidate
    raise FileNotFoundError(f"BUSI folders {CATEGORIES} not found under {path}")


def prepare_busi(raw_dir: Path, out_dir: Path, val_fraction: float = 0.2, seed: int = 42) -> Path:
    """Merge multiple masks per image, binarize them and split images into train/val folders."""
    raw_dir = find_busi_root(Path(raw_dir))
    if out_dir.exists():
        shutil.rmtree(out_dir)

    samples = []
    for category in CATEGORIES:
        for image_path in sorted((raw_dir / category).glob("*.png")):
            if "_mask" in image_path.stem:
                continue
            mask_paths = sorted(raw_dir.joinpath(category).glob(f"{image_path.stem}_mask*.png"))
            if mask_paths:
                samples.append((image_path, mask_paths, category))

    random.Random(seed).shuffle(samples)
    n_val = max(1, int(len(samples) * val_fraction))

    for i, (image_path, mask_paths, category) in enumerate(samples):
        split = "val" if i < n_val else "train"
        name = f"{i:04d}_{category}.png"
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)

        shutil.copy(image_path, out_dir / split / "images" / name)
        mask = np.maximum.reduce([np.array(Image.open(p).convert("L")) for p in mask_paths])
        Image.fromarray(((mask > 127) & (category != "normal")).astype(np.uint8)).save(out_dir / split / "masks" / name)

    print(f"BUSI prepared in {out_dir}: {len(samples) - n_val} train / {n_val} val images")
    return out_dir


def get_transforms(train: bool, size: int = 256) -> A.Compose:
    if not train:
        return A.Compose([A.Resize(size, size), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


class SegmentationDataset(Dataset):
    def __init__(self, split_dir: Path, transform: A.Compose):
        self.images = sorted((Path(split_dir) / "images").glob("*.png"))
        self.masks_dir = Path(split_dir) / "masks"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks_dir / self.images[idx].name)).astype(np.int64)
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()

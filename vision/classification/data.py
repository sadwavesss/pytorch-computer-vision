import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

TRANSFORMS = {
    "basic": T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]),
    "improved": T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ]),
    "test": T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]),
}


def get_loaders(root: str, augmentation: str, batch_size: int = 128, num_workers: int = 2, seed: int = 42):
    """CIFAR-10 loaders with an 80/20 train/validation split; validation and test are not augmented."""
    train_full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=TRANSFORMS[augmentation])
    val_full = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=TRANSFORMS["test"])
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=TRANSFORMS["test"])

    indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    split = int(0.8 * len(indices))
    train_set, val_set = Subset(train_full, indices[:split]), Subset(val_full, indices[split:])

    loader_args = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **loader_args),
        DataLoader(val_set, shuffle=False, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=images.device).view(-1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=images.device).view(-1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)

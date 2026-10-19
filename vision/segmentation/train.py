"""Train U-Net for breast lesion segmentation on BUSI.

Example:
    python -m vision.segmentation.train --download
    python -m vision.segmentation.train --raw-data path/to/Dataset_BUSI_with_GT
"""

import argparse
import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from vision.segmentation.data import IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset, get_transforms, prepare_busi
from vision.segmentation.metrics import CombinedLoss, SegmentationMeter
from vision.segmentation.unet import UNet
from vision.utils import count_parameters, get_device, save_json, seed_everything

NUM_CLASSES = 2
KAGGLE_DATASET = "sabahesaraki/breast-ultrasound-images-dataset"


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    meter, total_loss = SegmentationMeter(NUM_CLASSES), 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            total_loss += criterion(logits, masks).item()
        meter.update(logits.argmax(1).cpu(), masks.cpu())
    return {"loss": total_loss / len(loader), **meter.compute()}


@torch.no_grad()
def save_predictions(model, dataset, device, path: Path, n: int = 4) -> None:
    model.eval()
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    # Prefer images that actually contain a lesion
    indices = [i for i in range(len(dataset)) if dataset[i][1].any()][:n]

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    for row, idx in zip(axes, indices):
        image, mask = dataset[idx]
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu()
        panels = [((image * std + mean).clamp(0, 1).permute(1, 2, 0), "Image"), (mask, "Ground truth"), (pred, "Prediction")]
        for ax, (data, title) in zip(row, panels):
            ax.imshow(data, cmap=None if title == "Image" else "gray")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(history: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set(title="Loss (CE + Dice)", xlabel="Epoch")
    ax2.plot(history["val_lesion_iou"], label="lesion IoU")
    ax2.plot(history["val_lesion_dice"], label="lesion Dice")
    ax2.set(title="Validation metrics", xlabel="Epoch")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="download BUSI from Kaggle via kagglehub")
    source.add_argument("--raw-data", type=Path, help="path to the extracted BUSI dataset")
    parser.add_argument("--data", type=Path, default=Path("data/busi"), help="where the prepared split is stored")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("results/segmentation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    if args.download:
        import kagglehub

        args.raw_data = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    prepare_busi(args.raw_data, args.data, seed=args.seed)

    train_set = SegmentationDataset(args.data / "train", get_transforms(train=True, size=args.img_size))
    val_set = SegmentationDataset(args.data / "val", get_transforms(train=False, size=args.img_size))
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = UNet(num_classes=NUM_CLASSES).to(device)
    criterion = CombinedLoss(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "val_lesion_iou": [], "val_lesion_dice": []}
    best, best_state = None, None
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(metrics["mean_iou"])
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(metrics["loss"])
        history["val_lesion_iou"].append(metrics["iou_per_class"][1])
        history["val_lesion_dice"].append(metrics["dice_per_class"][1])
        print(
            f"epoch {epoch + 1}/{args.epochs}  train loss {history['train_loss'][-1]:.4f}  "
            f"val mIoU {metrics['mean_iou']:.4f}  lesion IoU {metrics['iou_per_class'][1]:.4f}  "
            f"lesion Dice {metrics['dice_per_class'][1]:.4f}"
        )
        if best is None or metrics["mean_iou"] > best["mean_iou"]:
            best, best_state = {**metrics, "epoch": epoch + 1}, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_history(history, args.out / "training_curves.png")
    save_predictions(model, val_set, device, args.out / "predictions.png")
    summary = {
        "dataset": {"train_images": len(train_set), "val_images": len(val_set), "img_size": args.img_size},
        "parameters": count_parameters(model),
        "epochs": args.epochs,
        "train_minutes": (time.time() - start) / 60,
        "best_val": {
            "epoch": best["epoch"],
            "mean_iou": best["mean_iou"],
            "mean_dice": best["mean_dice"],
            "lesion_iou": best["iou_per_class"][1],
            "lesion_dice": best["dice_per_class"][1],
        },
    }
    save_json({**summary, "history": history}, args.out / "summary.json")
    print(f"Best epoch {best['epoch']}: mIoU {best['mean_iou']:.4f}, lesion Dice {best['dice_per_class'][1]:.4f}")


if __name__ == "__main__":
    main()

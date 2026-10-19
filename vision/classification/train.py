"""Train CIFAR-10 classifiers and evaluate them (and their ensemble) on the test set.

Example:
    python -m vision.classification.train --experiments cnn improved_cnn resnet --ensemble improved_cnn resnet
"""

import argparse
import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from vision.classification.augment import cutmix, mixed_loss, mixup
from vision.classification.data import CLASSES, get_loaders
from vision.classification.models import MODELS
from vision.utils import count_parameters, get_device, save_json, seed_everything


@dataclass
class Experiment:
    model: str
    augmentation: str
    optimizer: str
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    scheduler: str | None = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0


EXPERIMENTS = {
    "cnn": Experiment("cnn", "basic", "adamw", lr=1e-3, weight_decay=1e-4, epochs=30, patience=8),
    "improved_cnn": Experiment(
        "improved_cnn", "improved", "adamw", lr=1e-3, weight_decay=1e-4, epochs=40, patience=10,
        scheduler="cosine", label_smoothing=0.1, mixup_alpha=0.2,
    ),
    "resnet": Experiment(
        "resnet", "improved", "sgd", lr=0.1, weight_decay=5e-4, epochs=40, patience=10,
        scheduler="onecycle", label_smoothing=0.1, mixup_alpha=1.0, cutmix_alpha=1.0,
    ),
}


def build_optimizer(exp: Experiment, model: nn.Module):
    if exp.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=exp.lr, momentum=0.9, weight_decay=exp.weight_decay, nesterov=True)
    return torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)


def build_scheduler(exp: Experiment, optimizer, steps_per_epoch: int):
    if exp.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp.epochs, eta_min=1e-6)
    if exp.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=exp.lr, epochs=exp.epochs, steps_per_epoch=steps_per_epoch
        )
    return None


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probabilities, labels) for the whole loader."""
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        probs.append(logits.float().softmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train(exp: Experiment, loaders, device) -> tuple[nn.Module, dict]:
    train_loader, val_loader, _ = loaders
    model = MODELS[exp.model]().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=exp.label_smoothing)
    optimizer = build_optimizer(exp, model)
    scheduler = build_scheduler(exp, optimizer, len(train_loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "val_accuracy": [], "lr": []}
    best_acc, best_state, stale_epochs = 0.0, None, 0
    start = time.time()

    for epoch in range(exp.epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Each batch gets Mixup or CutMix with probability 0.5 (if enabled)
            mix = None
            if exp.mixup_alpha and np.random.rand() < 0.5:
                mix = mixup(images, targets, exp.mixup_alpha)
            elif exp.cutmix_alpha and np.random.rand() < 0.5:
                mix = cutmix(images, targets, exp.cutmix_alpha)

            with torch.autocast(device.type, enabled=device.type == "cuda"):
                if mix:
                    images, y_a, y_b, lam = mix
                    loss = mixed_loss(criterion, model(images), y_a, y_b, lam)
                else:
                    loss = criterion(model(images), targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            running_loss += loss.item()

        history["lr"].append(optimizer.param_groups[0]["lr"])
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        probs, labels = predict(model, val_loader, device)
        val_acc = float((probs.argmax(1) == labels).mean() * 100)
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_accuracy"].append(val_acc)
        print(f"[{exp.model}] epoch {epoch + 1}/{exp.epochs}  loss {history['train_loss'][-1]:.4f}  val acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc, best_state, stale_epochs = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
            if stale_epochs >= exp.patience:
                print(f"[{exp.model}] early stopping after {epoch + 1} epochs")
                break

    model.load_state_dict(best_state)
    history.update(best_val_accuracy=best_acc, epochs_trained=len(history["val_accuracy"]), train_minutes=(time.time() - start) / 60)
    return model, history


def evaluate(name: str, probs: np.ndarray, labels: np.ndarray, out_dir: Path) -> dict:
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds)
    per_class = {cls: float(cm[i, i] / cm[i].sum() * 100) for i, cls in enumerate(CLASSES)}

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set(title=f"{name}: confusion matrix (test)", xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=120)
    plt.close(fig)

    return {"test_accuracy": float((preds == labels).mean() * 100), "per_class_accuracy": per_class}


def plot_history(histories: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, history in histories.items():
        ax1.plot(history["train_loss"], label=name)
        ax2.plot(history["val_accuracy"], label=name)
    ax1.set(title="Train loss", xlabel="Epoch")
    ax2.set(title="Validation accuracy, %", xlabel="Epoch")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--ensemble", nargs="*", default=[], help="experiments whose probabilities are averaged")
    parser.add_argument("--epochs", type=int, help="override the number of epochs (e.g. for a quick check)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="results/classification")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not set(args.ensemble) <= set(args.experiments):
        parser.error("--ensemble members must be listed in --experiments")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    histories, summary, test_probs = {}, {}, {}
    for name in args.experiments:
        seed_everything(args.seed)
        exp = copy.deepcopy(EXPERIMENTS[name])
        if args.epochs:
            exp.epochs = args.epochs
        loaders = get_loaders(args.data, exp.augmentation, args.batch_size, args.num_workers, args.seed)

        model, history = train(exp, loaders, device)
        test_probs[name], labels = predict(model, loaders[2], device)
        histories[name] = history
        summary[name] = {
            "config": asdict(exp),
            "parameters": count_parameters(model),
            **{k: v for k, v in history.items() if not isinstance(v, list)},
            **evaluate(name, test_probs[name], labels, out_dir),
        }
        save_json({**summary[name], "history": history}, out_dir / f"{name}.json")
        print(f"[{name}] test accuracy {summary[name]['test_accuracy']:.2f}%")

    if len(args.ensemble) > 1:
        probs = np.mean([test_probs[name] for name in args.ensemble], axis=0)
        summary["ensemble"] = {"members": args.ensemble, **evaluate("ensemble", probs, labels, out_dir)}
        print(f"[ensemble] test accuracy {summary['ensemble']['test_accuracy']:.2f}%")

    plot_history(histories, out_dir)
    save_json(summary, out_dir / "summary.json")
    for name, result in summary.items():
        print(f"{name:>14}: {result['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()

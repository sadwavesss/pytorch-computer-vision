import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = logits.softmax(dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * target_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        return 1.0 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    """0.5 * cross-entropy + 0.5 * soft Dice."""

    def __init__(self, num_classes: int, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class SegmentationMeter:
    """Accumulates per-class intersection and union over a whole dataset."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes)
        self.pred_area = torch.zeros(num_classes)
        self.target_area = torch.zeros(num_classes)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for cls in range(self.num_classes):
            p, t = preds == cls, target == cls
            self.intersection[cls] += (p & t).sum().item()
            self.pred_area[cls] += p.sum().item()
            self.target_area[cls] += t.sum().item()

    def compute(self) -> dict:
        union = self.pred_area + self.target_area - self.intersection
        iou = self.intersection / union.clamp(min=1)
        dice = 2 * self.intersection / (self.pred_area + self.target_area).clamp(min=1)
        return {
            "iou_per_class": iou.tolist(),
            "dice_per_class": dice.tolist(),
            "mean_iou": iou.mean().item(),
            "mean_dice": dice.mean().item(),
        }

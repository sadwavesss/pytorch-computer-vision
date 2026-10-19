import numpy as np
import torch


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[2], x.shape[3]
    cut_h, cut_w = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)

    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # Recompute lambda from the actual pasted area (the box is clipped at the borders)
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, y, y[index], lam


def mixed_loss(criterion, outputs, y_a, y_b, lam):
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

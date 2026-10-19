"""Neural style transfer (Gatys et al.) with VGG19 features and L-BFGS.

Example:
    python -m vision.style_transfer --content assets/content.jpg --style assets/style.jpg --betas 1e3 1e5 1e7
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from vision.utils import get_device, save_json, seed_everything

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Indices of layers in torchvision's vgg19().features
STYLE_LAYERS = {"conv1_1": 0, "conv2_1": 5, "conv3_1": 10, "conv4_1": 19, "conv5_1": 28}
STYLE_WEIGHTS = {"conv1_1": 1.0, "conv2_1": 0.8, "conv3_1": 0.5, "conv4_1": 0.3, "conv5_1": 0.1}
CONTENT_LAYER = ("conv4_2", 21)


def load_image(path: Path, size: int, device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def to_image(tensor: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor.detach()[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def load_vgg(device) -> torch.nn.Module:
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[: max(STYLE_LAYERS.values()) + 1]
    return vgg.to(device).eval().requires_grad_(False)


def extract_features(image: torch.Tensor, vgg: torch.nn.Module) -> dict:
    wanted = {idx: name for name, idx in [*STYLE_LAYERS.items(), CONTENT_LAYER]}
    features, x = {}, image
    for idx, layer in enumerate(vgg):
        x = layer(x)
        if idx in wanted:
            features[wanted[idx]] = x
    return features


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    flat = features.view(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


def style_transfer(content, style, vgg, steps: int = 300, alpha: float = 1.0, beta: float = 1e5, init: str = "content"):
    """Optimize the generated image so it keeps the content features and matches the style Gram matrices."""
    generated = (content.clone() if init == "content" else torch.randn_like(content)).requires_grad_(True)
    with torch.no_grad():
        content_target = extract_features(content, vgg)[CONTENT_LAYER[0]]
        style_targets = {name: gram_matrix(f) for name, f in extract_features(style, vgg).items() if name in STYLE_LAYERS}

    optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)
    mean = torch.tensor(IMAGENET_MEAN, device=content.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=content.device).view(1, 3, 1, 1)
    history = {"content": [], "style": []}
    losses = {}

    def closure():
        optimizer.zero_grad()
        features = extract_features(generated, vgg)
        losses["content"] = F.mse_loss(features[CONTENT_LAYER[0]], content_target)
        losses["style"] = sum(
            STYLE_WEIGHTS[name] * F.mse_loss(gram_matrix(features[name]), target) for name, target in style_targets.items()
        )
        total = alpha * losses["content"] + beta * losses["style"]
        total.backward()
        return total

    for _ in range(steps):
        optimizer.step(closure)
        with torch.no_grad():
            # Keep pixels inside the valid [0, 1] range in the unnormalized space
            generated.copy_(((generated * std + mean).clamp(0, 1) - mean) / std)
        history["content"].append(losses["content"].item())
        history["style"].append(losses["style"].item())
    return generated, history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[1e5], help="style weights to compare")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("results/style_transfer"))
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()
    vgg = load_vgg(device)
    content, style = load_image(args.content, args.size, device), load_image(args.style, args.size, device)

    results = {}
    fig, axes = plt.subplots(1, len(args.betas) + 2, figsize=(4 * (len(args.betas) + 2), 4))
    axes[0].imshow(to_image(content))
    axes[0].set_title("Content")
    axes[1].imshow(to_image(style))
    axes[1].set_title("Style")
    for ax, beta in zip(axes[2:], args.betas):
        generated, history = style_transfer(content, style, vgg, steps=args.steps, beta=beta)
        ax.imshow(to_image(generated))
        ax.set_title(f"beta = {beta:.0e}")
        results[f"{beta:.0e}"] = {"final_content_loss": history["content"][-1], "final_style_loss": history["style"][-1]}
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "style_transfer.png", dpi=110)
    save_json({"steps": args.steps, "size": args.size, "results": results}, args.out / "summary.json")
    print(f"Saved {args.out / 'style_transfer.png'}")


if __name__ == "__main__":
    main()

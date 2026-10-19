"""Run the classical filters on an image and save a comparison figure.

Example:
    python -m vision.classical.demo --image assets/street.jpg
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vision.classical.filters import edge_detection, equalize_histogram, gaussian_blur, median_filter


def add_salt_and_pepper(image: np.ndarray, amount: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    mask = rng.random(image.shape)
    noisy[mask < amount / 2] = 0
    noisy[mask > 1 - amount / 2] = 255
    return noisy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/classical"))
    args = parser.parse_args()

    color = cv2.imread(str(args.image))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    noisy = add_salt_and_pepper(gray)

    panels = [
        (cv2.cvtColor(color, cv2.COLOR_BGR2RGB), "Original"),
        (cv2.cvtColor(equalize_histogram(color), cv2.COLOR_BGR2RGB), "Histogram equalization"),
        (noisy, "Salt & pepper noise"),
        (median_filter(noisy, 3), "Median filter 3x3"),
        (gaussian_blur(gray, 5, 1.0), "Gaussian blur 5x5"),
        (edge_detection(gaussian_blur(gray, 5, 1.0)), "Sobel + double threshold"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "filters.png", dpi=100)
    print(f"Saved {args.out / 'filters.png'}")


if __name__ == "__main__":
    main()

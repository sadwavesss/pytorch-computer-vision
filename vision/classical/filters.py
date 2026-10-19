"""Classical image processing implemented from scratch with NumPy (OpenCV is used only for color conversion)."""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter for a grayscale image; borders are handled by replicating edge pixels."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    windows = sliding_window_view(np.pad(image, pad, mode="edge"), (kernel_size, kernel_size))
    return np.median(windows, axis=(-2, -1)).astype(image.dtype)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D correlation with edge padding; returns float32 of the same size as the input."""
    pad = kernel.shape[0] // 2
    windows = sliding_window_view(np.pad(image.astype(np.float32), pad, mode="edge"), kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel.astype(np.float32))


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    return np.clip(np.rint(convolve2d(image, gaussian_kernel(kernel_size, sigma))), 0, 255).astype(np.uint8)


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T


def edge_detection(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Simplified Canny: Sobel gradient magnitude and double thresholding (strong = 255, weak = 100)."""
    magnitude = np.hypot(convolve2d(image, SOBEL_X), convolve2d(image, SOBEL_Y))
    magnitude = magnitude / max(magnitude.max(), 1e-8) * 255
    edges = np.zeros(image.shape, dtype=np.uint8)
    edges[(magnitude >= low) & (magnitude <= high)] = 100
    edges[magnitude > high] = 255
    return edges


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Histogram equalization; color images are equalized on the luma (Y) channel."""
    if image.ndim == 3:
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = equalize_histogram(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    hist = np.bincount(image.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255).clip(0, 255).astype(np.uint8)
    return lut[image]


def anaglyph(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    """Red-cyan stereo anaglyph: red channel from the left view, green and blue from the right one."""
    if left_rgb.shape != right_rgb.shape:
        raise ValueError("left and right images must have the same size")
    return np.dstack([left_rgb[:, :, 0], right_rgb[:, :, 1], right_rgb[:, :, 2]])

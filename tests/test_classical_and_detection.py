import cv2
import numpy as np
import pytest

from vision.classical.filters import (
    anaglyph,
    edge_detection,
    equalize_histogram,
    gaussian_blur,
    gaussian_kernel,
    median_filter,
)
from vision.detection.replace_object import label_detections, paste_with_alpha


@pytest.fixture
def gray():
    return np.random.default_rng(0).integers(0, 256, (40, 50), dtype=np.uint8)


def test_median_filter_matches_opencv(gray):
    assert np.array_equal(median_filter(gray, 3), cv2.medianBlur(gray, 3))


def test_median_filter_rejects_even_kernel(gray):
    with pytest.raises(ValueError):
        median_filter(gray, 4)


def test_gaussian_kernel_is_normalized_and_symmetric():
    kernel = gaussian_kernel(5, 1.0)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel.T) and np.allclose(kernel, kernel[::-1])


def test_gaussian_blur_matches_opencv(gray):
    expected = cv2.GaussianBlur(gray, (5, 5), 1.0, borderType=cv2.BORDER_REPLICATE)
    assert np.abs(gaussian_blur(gray, 5, 1.0).astype(int) - expected).max() <= 1


def test_edge_detection_finds_vertical_step_in_both_directions():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 255
    edges = edge_detection(image)
    assert edges[:, 9:11].min() == 255
    assert edges[:, :8].max() == 0 and edges[:, 12:].max() == 0
    assert edge_detection(255 - image)[:, 9:11].min() == 255  # dark-to-light and light-to-dark


def test_equalize_histogram_matches_opencv(gray):
    low_contrast = (gray // 4 + 60).astype(np.uint8)
    assert np.abs(equalize_histogram(low_contrast).astype(int) - cv2.equalizeHist(low_contrast)).max() <= 1


def test_equalize_histogram_keeps_color_shape():
    color = np.random.default_rng(1).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    assert equalize_histogram(color).shape == color.shape


def test_anaglyph_takes_red_from_left_and_green_blue_from_right():
    left, right = np.full((4, 4, 3), 10, np.uint8), np.full((4, 4, 3), 200, np.uint8)
    result = anaglyph(left, right)
    assert (result[..., 0] == 10).all() and (result[..., 1:] == 200).all()
    with pytest.raises(ValueError):
        anaglyph(left, right[:2])


def test_label_detections_numbers_repeated_classes():
    labeled = label_detections([[0, 0, 1, 1]] * 3, ["car", "person", "car"])
    assert [label for label, _ in labeled] == ["car", "person", "car_2"]


def test_paste_with_alpha_blends_transparent_overlay():
    background = np.zeros((10, 10, 3), np.uint8)
    overlay = np.zeros((4, 4, 4), np.uint8)
    overlay[..., :3], overlay[..., 3] = 255, 128
    result = paste_with_alpha(background, overlay, (2, 2, 6, 6))
    assert result[3, 3, 0] == pytest.approx(128, abs=1)
    assert result[0, 0].sum() == 0

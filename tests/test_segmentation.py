import numpy as np
import pytest
import torch
from PIL import Image

from vision.segmentation.metrics import DiceLoss, SegmentationMeter


def test_meter_computes_iou_and_dice_over_dataset():
    meter = SegmentationMeter(num_classes=2)
    target = torch.tensor([[0, 0, 1, 1]])
    meter.update(torch.tensor([[0, 1, 1, 1]]), target)
    result = meter.compute()
    assert result["iou_per_class"] == pytest.approx([1 / 2, 2 / 3])
    assert result["dice_per_class"] == pytest.approx([2 / 3, 4 / 5])


def test_dice_loss_is_near_zero_for_perfect_prediction():
    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.stack([(target == 0).float(), (target == 1).float()], dim=1) * 50
    assert DiceLoss(num_classes=2)(logits, target).item() < 1e-3


def _save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_prepare_busi_merges_masks_and_splits(tmp_path):
    pytest.importorskip("albumentations")
    from vision.segmentation.data import SegmentationDataset, get_transforms, prepare_busi

    raw = tmp_path / "Dataset_BUSI_with_GT"
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    left, right = np.zeros((16, 16), np.uint8), np.zeros((16, 16), np.uint8)
    left[:, :4], right[:, -4:] = 255, 255
    for category in ("benign", "malignant", "normal"):
        for i in range(1, 4):
            _save(raw / category / f"{category} ({i}).png", image)
            _save(raw / category / f"{category} ({i})_mask.png", left)
    _save(raw / "benign" / "benign (1)_mask_1.png", right)

    out = prepare_busi(tmp_path, tmp_path / "prepared", val_fraction=0.2)

    train, val = sorted((out / "train/images").glob("*.png")), sorted((out / "val/images").glob("*.png"))
    assert len(train) + len(val) == 9 and len(val) == 1
    masks = {p.name: np.array(Image.open(p)) for p in (out / "train/masks").glob("*.png")} | {
        p.name: np.array(Image.open(p)) for p in (out / "val/masks").glob("*.png")
    }
    assert all(m.max() == 0 for name, m in masks.items() if "normal" in name)
    assert all(set(np.unique(m)) == {0, 1} for name, m in masks.items() if "normal" not in name)
    assert any(m[:, -4:].all() and m[:, :4].all() for m in masks.values())  # benign (1): two masks merged

    dataset = SegmentationDataset(out / "train", get_transforms(train=True, size=32))
    image_t, mask_t = dataset[0]
    assert image_t.shape == (3, 32, 32) and mask_t.shape == (32, 32) and mask_t.dtype == torch.long

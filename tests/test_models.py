import pytest
import torch

from vision.classification.augment import cutmix, mixed_loss, mixup
from vision.classification.models import MODELS
from vision.segmentation.unet import UNet


@pytest.mark.parametrize("name", list(MODELS))
def test_classifier_output_shape(name):
    model = MODELS[name]().eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_unet_keeps_spatial_size():
    model = UNet(num_classes=2, base=8).eval()
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_mixup_keeps_shapes_and_lambda_range():
    x, y = torch.randn(8, 3, 32, 32), torch.arange(8)
    mixed, y_a, y_b, lam = mixup(x, y, alpha=1.0)
    assert mixed.shape == x.shape
    assert torch.equal(y_a, y) and sorted(y_b.tolist()) == list(range(8))
    assert 0 <= lam <= 1


def test_cutmix_lambda_matches_unchanged_area():
    torch.manual_seed(0)
    x, y = torch.zeros(4, 1, 32, 32), torch.arange(4)
    x += torch.arange(4).view(4, 1, 1, 1).float()
    mixed, _, y_b, lam = cutmix(x, y, alpha=1.0)
    kept = (mixed[0] == x[0]).float().mean().item()
    if y_b[0] != 0:
        assert lam == pytest.approx(kept)
    assert torch.equal(x[0], torch.zeros(1, 32, 32))  # input is not modified in place


def test_mixed_loss_equals_plain_loss_when_labels_match():
    criterion = torch.nn.CrossEntropyLoss()
    logits, y = torch.randn(4, 10), torch.tensor([1, 2, 3, 4])
    assert mixed_loss(criterion, logits, y, y, 0.3).item() == pytest.approx(criterion(logits, y).item())

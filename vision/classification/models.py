import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    return [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]


class CNN(nn.Module):
    """Baseline: three conv-bn-relu-pool blocks and a two-layer classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        blocks = []
        for in_ch, out_ch in [(3, 64), (64, 128), (128, 256)]:
            blocks += [*conv_bn_relu(in_ch, out_ch), nn.MaxPool2d(2), nn.Dropout2d(0.25)]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """VGG-style: two convolutions per block and a three-layer classifier with batch norm."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        blocks = []
        for in_ch, out_ch in [(3, 64), (64, 128), (128, 256)]:
            blocks += [*conv_bn_relu(in_ch, out_ch), *conv_bn_relu(out_ch, out_ch), nn.MaxPool2d(2), nn.Dropout2d(0.25)]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """ResNet with three stages of two residual blocks (64-128-256 channels) for 32x32 inputs."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResidualBlock(64, 64), ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


MODELS = {"cnn": CNN, "improved_cnn": ImprovedCNN, "resnet": ResNetCIFAR}

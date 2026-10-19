import torch
import torch.nn as nn


def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net with four down/up levels (64 -> 1024 channels) and skip connections."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base: int = 64):
        super().__init__()
        widths = [base * 2**i for i in range(5)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for width in widths[:-1]:
            self.encoders.append(double_conv(prev, width))
            prev = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(widths[-2], widths[-1])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2))
            self.decoders.append(double_conv(width * 2, width))
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upsample(x), skip], dim=1))
        return self.head(x)

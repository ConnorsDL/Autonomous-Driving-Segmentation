"""Model definitions for semantic segmentation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


class UNet(nn.Module):
    """A lightweight UNet implementation suitable for mid-sized datasets."""

    def __init__(self, in_channels: int, num_classes: int, features: Tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(current_channels, feature))
            current_channels = feature

        bottleneck_channels = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_channels)

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(bottleneck_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
            bottleneck_channels = feature

        self.output_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        out = x
        for encoder_block in self.encoder:
            out = encoder_block(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)

        skip_connections = list(reversed(skip_connections))
        for idx, (up, decoder_block) in enumerate(zip(self.up_transpose, self.decoder)):
            out = up(out)
            skip = skip_connections[idx]
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = decoder_block(out)

        return self.output_conv(out)


def build_model(config: Dict) -> UNet:
    """Factory that builds a UNet model from a configuration dictionary."""

    in_channels = config.get("in_channels", 3)
    num_classes = config["num_classes"]
    features = tuple(config.get("features", (64, 128, 256, 512)))
    return UNet(in_channels=in_channels, num_classes=num_classes, features=features)

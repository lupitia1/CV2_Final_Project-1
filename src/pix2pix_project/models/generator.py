import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, apply_batchnorm: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not apply_batchnorm),
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, apply_dropout: bool = False) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64) -> None:
        super().__init__()
        feature_sizes = [features, features * 2, features * 4, features * 8, features * 8, features * 8]

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(in_channels, feature_sizes[0], apply_batchnorm=False),
                DownBlock(feature_sizes[0], feature_sizes[1]),
                DownBlock(feature_sizes[1], feature_sizes[2]),
                DownBlock(feature_sizes[2], feature_sizes[3]),
                DownBlock(feature_sizes[3], feature_sizes[4]),
                DownBlock(feature_sizes[4], feature_sizes[5]),
            ]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_sizes[5], feature_sizes[5], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.up_blocks = nn.ModuleList(
            [
                UpBlock(feature_sizes[5], feature_sizes[5], apply_dropout=True),
                UpBlock(feature_sizes[5] * 2, feature_sizes[4], apply_dropout=True),
                UpBlock(feature_sizes[4] * 2, feature_sizes[3], apply_dropout=True),
                UpBlock(feature_sizes[3] * 2, feature_sizes[2]),
                UpBlock(feature_sizes[2] * 2, feature_sizes[1]),
                UpBlock(feature_sizes[1] * 2, feature_sizes[0]),
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(feature_sizes[0] * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def _resize_skip(self, skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] == target.shape[-2:]:
            return skip
        return F.interpolate(skip, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        current = x
        for down_block in self.down_blocks:
            current = down_block(current)
            skips.append(current)

        current = self.bottleneck(current)

        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            current = up_block(current)
            skip = self._resize_skip(skip, current)
            current = torch.cat([current, skip], dim=1)

        output = self.final(current)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return output

import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, *, apply_batchnorm: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=not apply_batchnorm),
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DiscriminatorBlock(in_channels, features, stride=2, apply_batchnorm=False),
            DiscriminatorBlock(features, features * 2, stride=2),
            DiscriminatorBlock(features * 2, features * 4, stride=2),
            DiscriminatorBlock(features * 4, features * 8, stride=1),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

import torch
import torch.nn as nn


class TinyGenerator(nn.Module):
    """Minimal placeholder generator. Replace with a U-Net for baseline."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

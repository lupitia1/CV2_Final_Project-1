from dataclasses import dataclass

import torch
import torch.nn as nn

from .discriminator import PatchDiscriminator
from .generator import UNetGenerator


def build_generator(in_channels: int = 3, out_channels: int = 3, features: int = 64) -> nn.Module:
    return UNetGenerator(in_channels=in_channels, out_channels=out_channels, features=features)


def build_discriminator(in_channels: int = 6, features: int = 64) -> nn.Module:
    return PatchDiscriminator(in_channels=in_channels, features=features)


@dataclass
class Pix2PixModel:
    generator: nn.Module
    discriminator: nn.Module

    def to(self, device: str) -> "Pix2PixModel":
        self.generator.to(device)
        self.discriminator.to(device)
        return self

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

import torch.nn as nn


def gan_loss() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def pixel_loss() -> nn.Module:
    return nn.L1Loss()

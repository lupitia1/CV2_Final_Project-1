from pathlib import Path

import torch
from torch import optim

from pix2pix_project.models.pix2pix import build_discriminator, build_generator
from pix2pix_project.utils.io import ensure_dir
from pix2pix_project.utils.seed import set_seed


def run_training(cfg):
    set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"

    generator = build_generator(
        in_channels=cfg["data"]["channels_in"],
        out_channels=cfg["data"]["channels_out"],
        features=cfg["model"]["generator_features"],
    ).to(device)
    discriminator = build_discriminator(
        in_channels=cfg["data"]["channels_in"] + cfg["data"]["channels_out"],
        features=cfg["model"]["discriminator_features"],
    ).to(device)

    optim_g = optim.Adam(generator.parameters(), lr=cfg["training"]["lr"], betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]))
    optim_d = optim.Adam(discriminator.parameters(), lr=cfg["training"]["lr"], betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]))

    ensure_dir(Path(cfg["logging"]["checkpoint_dir"]))
    print("Training skeleton initialized.")
    print("TODO: add dataloaders, losses, training loop, validation, and checkpoint selection.")

    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict()}, Path(cfg["logging"]["checkpoint_dir"]) / "last.pt")
    _ = (optim_g, optim_d)

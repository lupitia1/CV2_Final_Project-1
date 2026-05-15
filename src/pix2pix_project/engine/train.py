from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pix2pix_project.data.dataset import PairedImageDataset
from pix2pix_project.models.pix2pix import build_discriminator, build_generator
from pix2pix_project.utils.io import ensure_dir
from pix2pix_project.utils.seed import set_seed


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in VALID_EXTS]


def _split_paths(paths: List[Path], split_cfg: Dict[str, float], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(paths), generator=g).tolist()
    shuffled = [paths[i] for i in perm]

    n_total = len(shuffled)
    n_train = int(n_total * split_cfg["train"])
    n_val = int(n_total * split_cfg["val"])
    n_test = n_total - n_train - n_val

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train : n_train + n_val]
    test_paths = shuffled[n_train + n_val : n_train + n_val + n_test]
    return train_paths, val_paths, test_paths


def _build_datasets(cfg) -> Tuple[PairedImageDataset, PairedImageDataset]:
    root = Path(cfg["data"]["root_dir"])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    size = tuple(cfg["data"]["image_size"])
    tensor_tf = T.Compose([T.Resize(size), T.ToTensor()])

    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.exists() and val_dir.exists():
        train_paths = _collect_images(train_dir)
        val_paths = _collect_images(val_dir)
    else:
        all_paths = _collect_images(root)
        if not all_paths:
            raise ValueError(f"No image files found under dataset root: {root}")
        train_paths, val_paths, _ = _split_paths(all_paths, cfg["data"]["split"], cfg["project"]["seed"])

    if len(train_paths) == 0 or len(val_paths) == 0:
        raise ValueError("Train/validation split is empty. Check dataset path and split configuration.")

    train_ds = PairedImageDataset(train_paths, transform_in=tensor_tf, transform_out=tensor_tf)
    val_ds = PairedImageDataset(val_paths, transform_in=tensor_tf, transform_out=tensor_tf)
    return train_ds, val_ds


def _evaluate_val(generator: nn.Module, val_loader: DataLoader, l1_loss: nn.Module, device: str) -> float:
    generator.eval()
    total_l1 = 0.0
    total_count = 0
    with torch.no_grad():
        for labels, reals in val_loader:
            labels = labels.to(device)
            reals = reals.to(device)
            fakes = generator(labels)
            loss = l1_loss(fakes, reals)
            bs = labels.size(0)
            total_l1 += loss.item() * bs
            total_count += bs
    return total_l1 / max(total_count, 1)


def run_training(cfg):
    set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"

    train_ds, val_ds = _build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

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

    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    lambda_l1 = cfg["training"]["lambda_l1"]

    ensure_dir(Path(cfg["logging"]["checkpoint_dir"]))
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])

    best_val = float("inf")
    num_epochs = cfg["training"]["num_epochs"]
    log_every = cfg["logging"]["log_every"]

    print(f"Starting training on {device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g = 0.0
        epoch_d = 0.0

        for step, (labels, reals) in enumerate(train_loader, start=1):
            labels = labels.to(device)
            reals = reals.to(device)

            # Discriminator step
            with torch.no_grad():
                fakes_detached = generator(labels)
            pred_real = discriminator(torch.cat([labels, reals], dim=1))
            pred_fake = discriminator(torch.cat([labels, fakes_detached], dim=1))
            loss_d_real = gan_loss(pred_real, torch.ones_like(pred_real))
            loss_d_fake = gan_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            optim_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optim_d.step()

            # Generator step
            fakes = generator(labels)
            pred_fake_for_g = discriminator(torch.cat([labels, fakes], dim=1))
            loss_g_gan = gan_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = l1_loss(fakes, reals)
            loss_g = loss_g_gan + lambda_l1 * loss_g_l1

            optim_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optim_g.step()

            epoch_d += loss_d.item()
            epoch_g += loss_g.item()

            if step % log_every == 0 or step == len(train_loader):
                print(
                    f"Epoch [{epoch}/{num_epochs}] Step [{step}/{len(train_loader)}] "
                    f"D: {loss_d.item():.4f} | G: {loss_g.item():.4f} "
                    f"(GAN: {loss_g_gan.item():.4f}, L1: {loss_g_l1.item():.4f})"
                )

        mean_d = epoch_d / max(len(train_loader), 1)
        mean_g = epoch_g / max(len(train_loader), 1)
        val_l1 = _evaluate_val(generator, val_loader, l1_loss, device)
        print(f"Epoch [{epoch}/{num_epochs}] done | Train D: {mean_d:.4f} | Train G: {mean_g:.4f} | Val L1: {val_l1:.4f}")

        last_path = checkpoint_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optim_g.state_dict(),
                "optimizer_d": optim_d.state_dict(),
                "val_l1": val_l1,
                "config": cfg,
            },
            last_path,
        )

        if val_l1 < best_val:
            best_val = val_l1
            best_path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optim_g.state_dict(),
                    "optimizer_d": optim_d.state_dict(),
                    "val_l1": val_l1,
                    "config": cfg,
                },
                best_path,
            )
            print(f"New best checkpoint at epoch {epoch}: {best_path} (Val L1: {val_l1:.4f})")

    print(f"Training finished. Best Val L1: {best_val:.4f}")

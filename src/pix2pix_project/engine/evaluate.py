import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from pix2pix_project.data.dataset import PairedImageDataset
from pix2pix_project.models.pix2pix import build_generator


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in VALID_EXTS]


def _split_paths(paths: List[Path], split_cfg: Dict[str, float], seed: int) -> List[Path]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(paths), generator=g).tolist()
    shuffled = [paths[i] for i in perm]

    n_total = len(shuffled)
    n_train = int(n_total * split_cfg["train"])
    n_val = int(n_total * split_cfg["val"])
    n_test = n_total - n_train - n_val
    test_paths = shuffled[n_train + n_val : n_train + n_val + n_test]
    return test_paths


def _build_test_dataset(cfg) -> PairedImageDataset:
    root = Path(cfg["data"]["root_dir"])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    size = tuple(cfg["data"]["image_size"])
    tensor_tf = T.Compose([T.Resize(size), T.ToTensor()])

    test_dir = root / "test"
    if test_dir.exists():
        test_paths = _collect_images(test_dir)
    else:
        all_paths = _collect_images(root)
        if not all_paths:
            raise ValueError(f"No image files found under dataset root: {root}")
        test_paths = _split_paths(all_paths, cfg["data"]["split"], cfg["project"]["seed"])

    if len(test_paths) == 0:
        raise ValueError("Test split is empty. Check dataset path and split configuration.")

    return PairedImageDataset(test_paths, transform_in=tensor_tf, transform_out=tensor_tf)


def _load_generator(cfg, checkpoint_path: str, device: str) -> torch.nn.Module:
    generator = build_generator(
        in_channels=cfg["data"]["channels_in"],
        out_channels=cfg["data"]["channels_out"],
        features=cfg["model"]["generator_features"],
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        generator.load_state_dict(state["generator"])
    else:
        generator.load_state_dict(state)
    generator.eval()
    return generator


def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def run_evaluation(cfg, checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"
    test_ds = _build_test_dataset(cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    generator = _load_generator(cfg, checkpoint_path, device)

    mae_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    with torch.no_grad():
        for labels, reals in test_loader:
            labels = labels.to(device)
            reals = reals.to(device)
            fakes = generator(labels).clamp(0.0, 1.0)

            batch_size = labels.size(0)
            for i in range(batch_size):
                pred_np = _to_numpy_image(fakes[i])
                real_np = _to_numpy_image(reals[i])

                mae_sum += float(np.mean(np.abs(pred_np - real_np)))
                psnr_sum += float(peak_signal_noise_ratio(real_np, pred_np, data_range=1.0))
                ssim_sum += float(structural_similarity(real_np, pred_np, data_range=1.0, channel_axis=2))
                count += 1

    if count == 0:
        raise RuntimeError("No test samples were evaluated.")

    metrics = {
        "checkpoint": str(checkpoint_path),
        "num_samples": count,
        "mae": mae_sum / count,
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
    }

    out_dir = Path(cfg.get("logging", {}).get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation complete on {count} test samples.")
    print(f"MAE: {metrics['mae']:.6f} | PSNR: {metrics['psnr']:.4f} | SSIM: {metrics['ssim']:.4f}")
    print(f"Metrics file: {out_file}")

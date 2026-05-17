import argparse
from pathlib import Path
import zipfile
import shutil


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def unzip_dataset(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def collect_images(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in VALID_EXTS]


def maybe_resolve_dataset_root(out_dir: Path, dataset_name: str) -> Path:
    direct = out_dir / dataset_name
    if direct.exists():
        return direct
    return out_dir


def has_explicit_splits(dataset_root: Path) -> bool:
    return (dataset_root / "train").exists() and (dataset_root / "val").exists() and (dataset_root / "test").exists()


def create_splits(
    dataset_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    all_images = [p for p in collect_images(dataset_root) if p.parent == dataset_root]
    if not all_images:
        raise ValueError(f"No paired image files found directly under dataset root: {dataset_root}")

    total = len(all_images)
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")

    import random

    rng = random.Random(seed)
    shuffled = all_images[:]
    rng.shuffle(shuffled)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train : n_train + n_val]
    test_files = shuffled[n_train + n_val : n_train + n_val + n_test]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in split_map.items():
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = split_dir / src.name
            shutil.copy2(src, dst)

    print(
        f"Created explicit splits in {dataset_root}: "
        f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for Pix2Pix project")
    parser.add_argument("--zip", type=Path, required=True, help="Path to dataset zip")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="maps", help="Dataset folder name under output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic train/val/test split")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Only unzip dataset and skip explicit train/val/test folder creation",
    )
    args = parser.parse_args()

    unzip_dataset(args.zip, args.out)
    print(f"Dataset extracted to: {args.out}")

    dataset_root = maybe_resolve_dataset_root(args.out, args.dataset_name)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Could not locate dataset root: {dataset_root}")

    if args.skip_split:
        print("Skipping split creation (--skip-split enabled).")
        return

    if has_explicit_splits(dataset_root):
        print(f"Explicit splits already exist under: {dataset_root}")
        return

    create_splits(
        dataset_root=dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

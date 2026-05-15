import argparse
from pathlib import Path
import zipfile


def unzip_dataset(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for Pix2Pix project")
    parser.add_argument("--zip", type=Path, required=True, help="Path to dataset zip")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Output directory")
    args = parser.parse_args()

    unzip_dataset(args.zip, args.out)
    print(f"Dataset extracted to: {args.out}")


if __name__ == "__main__":
    main()

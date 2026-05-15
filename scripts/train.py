import argparse
import sys
from pathlib import Path

# Ensure `src/` is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pix2pix_project.config import load_config
from pix2pix_project.engine.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pix2Pix model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()

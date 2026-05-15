import argparse

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

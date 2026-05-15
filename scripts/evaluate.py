import argparse

from pix2pix_project.config import load_config
from pix2pix_project.engine.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pix2Pix model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg, args.checkpoint)


if __name__ == "__main__":
    main()

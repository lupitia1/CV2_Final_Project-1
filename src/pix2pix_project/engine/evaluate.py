import json
from pathlib import Path


def run_evaluation(cfg, checkpoint_path: str):
    _ = (cfg, checkpoint_path)
    metrics = {
        "psnr": None,
        "ssim": None,
        "mae": None,
    }
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation skeleton complete. Metrics file: {out_file}")
    print("TODO: load checkpoint, run inference on test split, and compute real metrics.")

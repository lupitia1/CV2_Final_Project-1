import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image
import torch
import numpy as np

from pix2pix_project.models.pix2pix import build_generator
from pix2pix_project.utils.io import ensure_dir


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in VALID_EXTS]
    if not images:
        raise ValueError(f"No images found in folder: {path}")
    return images


def split_paired_image(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Split pix2pix-style paired image: left=real image, right=label map.
    Returns (label_map, real_image).
    """
    w, h = img.size
    if w % 2 != 0:
        raise ValueError("Paired image width must be even.")
    real = img.crop((0, 0, w // 2, h))
    label = img.crop((w // 2, 0, w, h))
    return label, real


def image_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_side_by_side(label: Image.Image, pred: Image.Image, real: Optional[Image.Image] = None) -> Image.Image:
    panels = [label, pred] if real is None else [label, pred, real]
    widths, heights = zip(*(p.size for p in panels))
    canvas = Image.new("RGB", (sum(widths), max(heights)))

    x_off = 0
    for panel in panels:
        canvas.paste(panel, (x_off, 0))
        x_off += panel.size[0]
    return canvas


def load_generator(checkpoint: Path, device: str) -> torch.nn.Module:
    gen = build_generator().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        gen.load_state_dict(state["generator"])
    else:
        gen.load_state_dict(state)
    gen.eval()
    return gen


def output_path_for_image(input_root: Path, image_path: Path, output_path: Path, input_is_dir: bool) -> Path:
    if input_is_dir:
        rel = image_path.relative_to(input_root)
        return output_path / rel
    if output_path.suffix:
        return output_path
    return output_path / f"{image_path.stem}_demo.png"


def run_inference_on_image(gen: torch.nn.Module, image_path: Path, device: str, paired_input: bool) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
    image = Image.open(image_path).convert("RGB")
    if paired_input:
        label, real = split_paired_image(image)
    else:
        label, real = image, None

    tensor = image_to_tensor(label, device)
    with torch.no_grad():
        pred = gen(tensor).clamp(0, 1)

    pred_img = tensor_to_image(pred)
    return label, pred_img, real


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI demo inference for Pix2Pix")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Path to a single image or a folder of images")
    parser.add_argument("--output", type=Path, required=True, help="Output image path (single input) or output folder (directory input)")
    parser.add_argument(
        "--paired-input",
        action="store_true",
        help="Use pix2pix paired format where each file is [real | label] and inference uses the label half",
    )
    parser.add_argument(
        "--save-generated-only",
        action="store_true",
        help="Save only generated output instead of side-by-side visualization",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = load_generator(args.checkpoint, device)

    input_is_dir = args.input.is_dir()
    if input_is_dir:
        ensure_dir(args.output)
    else:
        ensure_dir(args.output.parent if args.output.suffix else args.output)

    image_paths = list_images(args.input)
    for image_path in image_paths:
        label_img, pred_img, real_img = run_inference_on_image(
            gen=gen,
            image_path=image_path,
            device=device,
            paired_input=args.paired_input,
        )
        if args.save_generated_only:
            output_image = pred_img
        else:
            output_image = build_side_by_side(label_img, pred_img, real_img)

        out_path = output_path_for_image(
            input_root=args.input,
            image_path=image_path,
            output_path=args.output,
            input_is_dir=input_is_dir,
        )
        ensure_dir(out_path.parent)
        output_image.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

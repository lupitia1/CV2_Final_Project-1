from pathlib import Path
import random
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """
    Expects each sample as a single image where left half is RGB target and
    right half is input label map (same format as pix2pix datasets).
    """

    def __init__(
        self,
        image_paths: List[Path],
        transform_in: Optional[Callable] = None,
        transform_out: Optional[Callable] = None,
        hflip_prob: float = 0.0,
    ) -> None:
        self.image_paths = image_paths
        self.transform_in = transform_in
        self.transform_out = transform_out
        self.hflip_prob = float(hflip_prob)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = img.size
        real = img.crop((0, 0, w // 2, h))
        label = img.crop((w // 2, 0, w, h))

        if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
            real = real.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform_in is not None:
            label = self.transform_in(label)
        if self.transform_out is not None:
            real = self.transform_out(real)
        return label, real

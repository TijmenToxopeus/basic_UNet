from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils.reproducibility import make_generator, seed_everything


@dataclass
class ShapesDatasetConfig:
    num_samples: int = 1000
    image_size: int = 64
    mode: str = "binary"  # "binary" or "multiclass"
    seed: int = 42
    noise_std: float = 0.08
    min_radius: int = 6
    max_radius: int = 16
    min_square: int = 8
    max_square: int = 20


def _draw_circle(mask: np.ndarray, cx: int, cy: int, r: int, value: int) -> None:
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = value


def _draw_square(mask: np.ndarray, cx: int, cy: int, size: int, value: int) -> None:
    h, w = mask.shape
    half = size // 2
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    mask[y0:y1, x0:x1] = value


class SyntheticShapesDataset(Dataset):
    """
    On-the-fly synthetic segmentation dataset.

    - binary: one foreground class
    - multiclass: two foreground classes
    """
    def __init__(self, cfg: ShapesDatasetConfig):
        self.cfg = cfg
        self.num_samples = int(cfg.num_samples)
        self.image_size = int(cfg.image_size)
        self.mode = str(cfg.mode).lower()
        if self.mode not in {"binary", "multiclass"}:
            raise ValueError("mode must be 'binary' or 'multiclass'")

    def __len__(self) -> int:
        return self.num_samples

    def _sample_center(self, rng: np.random.Generator) -> Tuple[int, int]:
        margin = max(self.cfg.max_radius, self.cfg.max_square) + 2
        cx = int(rng.integers(margin, self.image_size - margin))
        cy = int(rng.integers(margin, self.image_size - margin))
        return cx, cy

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.cfg.seed + idx)
        h = w = self.image_size

        image = rng.normal(loc=0.1, scale=self.cfg.noise_std, size=(h, w)).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.int64)

        # Draw shapes
        cx, cy = self._sample_center(rng)
        if rng.random() < 0.5:
            r = int(rng.integers(self.cfg.min_radius, self.cfg.max_radius + 1))
            _draw_circle(mask, cx, cy, r, value=1)
        else:
            size = int(rng.integers(self.cfg.min_square, self.cfg.max_square + 1))
            _draw_square(mask, cx, cy, size, value=1)

        if self.mode == "multiclass":
            cx2, cy2 = self._sample_center(rng)
            if rng.random() < 0.5:
                r2 = int(rng.integers(self.cfg.min_radius, self.cfg.max_radius + 1))
                _draw_circle(mask, cx2, cy2, r2, value=2)
            else:
                size2 = int(rng.integers(self.cfg.min_square, self.cfg.max_square + 1))
                _draw_square(mask, cx2, cy2, size2, value=2)

        # Increase foreground intensity
        image = np.clip(image, 0.0, 1.0)
        image[mask > 0] = np.clip(image[mask > 0] + 0.75, 0.0, 1.0)

        img_t = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
        mask_t = torch.from_numpy(mask).long()                # [H, W]
        return img_t, mask_t


def get_synthetic_loaders(
    *,
    cfg: ShapesDatasetConfig,
    batch_size: int = 16,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    deterministic: bool = False,
):
    seed_everything(cfg.seed, deterministic=deterministic)

    dataset = SyntheticShapesDataset(cfg)
    n_val = int(round(len(dataset) * float(val_ratio)))
    n_train = len(dataset) - n_val

    g = make_generator(cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

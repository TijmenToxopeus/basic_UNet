from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

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
    fg_classes: int = 2  # number of foreground classes for 'multiclass' mode


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


def _apply_class_intensity(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Assign distinct grayscale ranges per class label.
    Background remains as-is (noise).
    """
    ranges = {
        1: (0.45, 0.60),
        2: (0.65, 0.80),
        3: (0.85, 0.95),
    }
    out = image.copy()
    present = np.unique(mask)
    for cls_id in present:
        if cls_id == 0:
            continue
        lo, hi = ranges.get(int(cls_id), (0.5, 0.9))
        coords = (mask == cls_id)
        if coords.any():
            out[coords] = rng.uniform(low=lo, high=hi, size=int(coords.sum()))
    return out


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

        self.fg_classes = 1 if self.mode == "binary" else max(2, int(getattr(cfg, "fg_classes", 2)))

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

        # helper: place a shape for a given class id with limited overlap
        def _place_shape_cls(cls_id: int, min_bg_ratio: float = 0.6, attempts: int = 10) -> None:
            last_pixels = None
            for _ in range(attempts):
                cx, cy = self._sample_center(rng)
                temp = np.zeros_like(mask)
                if rng.random() < 0.5:
                    r = int(rng.integers(self.cfg.min_radius, self.cfg.max_radius + 1))
                    _draw_circle(temp, cx, cy, r, value=1)
                else:
                    size = int(rng.integers(self.cfg.min_square, self.cfg.max_square + 1))
                    _draw_square(temp, cx, cy, size, value=1)
                pixels = temp > 0
                last_pixels = pixels
                # fraction of new shape pixels that lie on background
                bg_ratio = float((mask[pixels] == 0).sum()) / float(pixels.sum())
                if bg_ratio >= min_bg_ratio:
                    mask[pixels] = cls_id
                    return
            # fallback: apply last candidate even if overlap is high
            if last_pixels is not None:
                mask[last_pixels] = cls_id

        # Draw shapes ensuring each class has pixels
        _place_shape_cls(1)
        if self.mode == "multiclass":
            for cls_id in range(2, self.fg_classes + 1):
                _place_shape_cls(cls_id)

        # Apply per-class intensity ranges
        image = np.clip(image, 0.0, 1.0)
        image = _apply_class_intensity(image, mask, rng)

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


@dataclass
class MNISTConfig:
    root: str = "./data"
    train: bool = True
    download: bool = True
    seed: int = 42


@dataclass
class FashionMNISTConfig:
    root: str = "./data"
    train: bool = True
    download: bool = True
    seed: int = 42


class MNISTSegmentationWrapper(Dataset):
    """
    Wraps MNIST as a segmentation dataset.
    - Input: grayscale image [1, 28, 28]
    - Target: binary mask [28, 28] (digit vs background)
    """
    def __init__(self, mnist_dataset: datasets.MNIST, threshold: float = 0.1):
        self.mnist_dataset = mnist_dataset
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def __getitem__(self, idx: int):
        img, label = self.mnist_dataset[idx]
        # img is PIL Image, convert to tensor and normalize
        img_t = transforms.ToTensor()(img)  # [1, 28, 28], range [0, 1]
        
        # Create binary mask: foreground where pixel > threshold
        mask_t = (img_t.squeeze(0) > self.threshold).long()  # [28, 28]
        
        return img_t, mask_t


def get_mnist_loaders(
    *,
    cfg: MNISTConfig,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    deterministic: bool = False,
    mask_threshold: float = 0.1,
):
    """
    Load MNIST as segmentation task (digit vs background).
    Returns train and val loaders with grayscale images.
    """
    seed_everything(cfg.seed, deterministic=deterministic)

    # Load MNIST training set
    mnist_train = datasets.MNIST(
        root=cfg.root,
        train=True,
        download=cfg.download,
        transform=None,
    )
    
    # Wrap as segmentation dataset
    dataset = MNISTSegmentationWrapper(mnist_train, threshold=mask_threshold)
    
    # Split into train/val
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


def get_fashion_mnist_loaders(
    *,
    cfg: FashionMNISTConfig,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    deterministic: bool = False,
    mask_threshold: float = 0.1,
):
    """
    Load Fashion-MNIST as segmentation task (object vs background).
    """
    seed_everything(cfg.seed, deterministic=deterministic)
    fashion_train = datasets.FashionMNIST(
        root=cfg.root,
        train=True,
        download=cfg.download,
        transform=None,
    )
    dataset = MNISTSegmentationWrapper(fashion_train, threshold=mask_threshold)
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

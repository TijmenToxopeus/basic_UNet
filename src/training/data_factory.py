# src/training/data_factory.py
from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader

from src.training.data_loader import SegmentationDataset, _collect_img_lbl_pairs


def build_eval_loader(
    img_dir,
    lbl_dir,
    *,
    batch_size: int,
    num_slices_per_volume: Optional[int],
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, int]:
    """
    Builds an evaluation DataLoader from image/label folders.
    Returns (loader, num_batches).

    Notes:
    - Uses the same SegmentationDataset as training (augment=False).
    - Keeps all eval-specific DataLoader knobs in one place.
    """
    pairs = _collect_img_lbl_pairs(img_dir, lbl_dir)

    dataset = SegmentationDataset(
        img_lbl_pairs=pairs,
        augment=False,
        num_slices_per_volume=num_slices_per_volume,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, len(loader)

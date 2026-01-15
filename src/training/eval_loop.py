# src/training/eval_loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    # overall
    mean_dice_all: float
    std_dice_all: float
    mean_iou_all: float
    std_iou_all: float

    # foreground (classes 1..C-1)
    mean_dice_fg: float
    std_dice_fg: float
    mean_iou_fg: float
    std_iou_fg: float

    # per class arrays aligned with class_names
    class_dice_mean: List[float]
    class_dice_std: List[float]
    class_iou_mean: List[float]
    class_iou_std: List[float]

    # vram peak during evaluation
    vram_peak_mb: float


def _mean_std(arr: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    dice_fn: Callable[..., object],
    iou_fn: Callable[..., object],
    vram_track: bool = True,
) -> Tuple[EvalResult, Dict[str, List[float]]]:
    """
    Runs evaluation and returns:
      - aggregated EvalResult
      - raw lists (useful if you want histograms later)
    Assumes dice_fn/iou_fn support per_class=True and return list/np/torch-like.
    """
    model.eval()

    class_dice_values: List[List[float]] = [[] for _ in range(num_classes)]
    class_iou_values: List[List[float]] = [[] for _ in range(num_classes)]
    fg_dice_values: List[float] = []
    fg_iou_values: List[float] = []
    all_dice_values: List[float] = []
    all_iou_values: List[float] = []

    if vram_track and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)
        preds = model(imgs)

        dice_list = dice_fn(preds, masks, num_classes=num_classes, per_class=True)
        iou_list = iou_fn(preds, masks, num_classes=num_classes, per_class=True)

        # ensure python floats
        dice_list = [float(x) for x in dice_list]
        iou_list = [float(x) for x in iou_list]

        for c in range(num_classes):
            class_dice_values[c].append(dice_list[c])
            class_iou_values[c].append(iou_list[c])

        fg_ids = list(range(1, num_classes))
        fg_dice_values.append(float(np.mean([dice_list[c] for c in fg_ids])))
        fg_iou_values.append(float(np.mean([iou_list[c] for c in fg_ids])))

        all_dice_values.append(float(np.mean(dice_list)))
        all_iou_values.append(float(np.mean(iou_list)))

    mean_dice_all, std_dice_all = _mean_std(all_dice_values)
    mean_iou_all, std_iou_all = _mean_std(all_iou_values)
    mean_dice_fg, std_dice_fg = _mean_std(fg_dice_values)
    mean_iou_fg, std_iou_fg = _mean_std(fg_iou_values)

    class_dice_mean, class_dice_std = [], []
    class_iou_mean, class_iou_std = [], []

    for c in range(num_classes):
        m, s = _mean_std(class_dice_values[c])
        class_dice_mean.append(m)
        class_dice_std.append(s)
        m, s = _mean_std(class_iou_values[c])
        class_iou_mean.append(m)
        class_iou_std.append(s)

    if vram_track and device.type == "cuda":
        vram_peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    else:
        vram_peak_mb = float("nan")

    result = EvalResult(
        mean_dice_all=mean_dice_all,
        std_dice_all=std_dice_all,
        mean_iou_all=mean_iou_all,
        std_iou_all=std_iou_all,
        mean_dice_fg=mean_dice_fg,
        std_dice_fg=std_dice_fg,
        mean_iou_fg=mean_iou_fg,
        std_iou_fg=std_iou_fg,
        class_dice_mean=class_dice_mean,
        class_dice_std=class_dice_std,
        class_iou_mean=class_iou_mean,
        class_iou_std=class_iou_std,
        vram_peak_mb=vram_peak_mb,
    )

    raw = {
        "all_dice": all_dice_values,
        "all_iou": all_iou_values,
        "fg_dice": fg_dice_values,
        "fg_iou": fg_iou_values,
    }
    return result, raw

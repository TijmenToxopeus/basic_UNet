# src/training/loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EpochTrainResult:
    loss_mean: float
    loss_std: float
    vram_epoch_max_mb: float  # nan if not cuda


@dataclass
class EpochValResult:
    dice_mean: float
    dice_std: float
    iou_mean: float
    iou_std: float


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    log_fn: Optional[Callable[[Dict], None]] = None,
    log_interval: int = 20,
    track_vram: bool = True,
    pbar_desc: Optional[str] = None,
) -> EpochTrainResult:
    """
    Trains model for one epoch and returns mean/std of batch losses.
    - `log_fn`: e.g. wandb.log (pass None to disable)
    - VRAM tracking: logs current/max occasionally; returns epoch peak.
    """
    model.train()
    losses: List[float] = []

    is_cuda = (device.type == "cuda")
    if track_vram and is_cuda:
        # Peak stats are "since last reset"
        torch.cuda.reset_peak_memory_stats(device)

    # tqdm should stay in the caller if you want full control, but keeping it here is ok.
    # We'll avoid importing tqdm here; caller can wrap loader if desired.
    for step, (imgs, masks) in enumerate(loader, start=1):
        imgs = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().item()))

        if log_fn is not None and track_vram and is_cuda and (step % max(1, log_interval) == 0):
            current_vram = torch.cuda.memory_allocated(device) / (1024**2)
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
            log_fn({
                "epoch": epoch,
                "step": step,
                "vram_current_mb": current_vram,
                "vram_peak_so_far_mb": peak_vram,
            })

    loss_mean = float(np.mean(losses)) if losses else float("nan")
    loss_std = float(np.std(losses)) if losses else float("nan")

    if track_vram and is_cuda:
        vram_epoch_max_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    else:
        vram_epoch_max_mb = float("nan")

    return EpochTrainResult(
        loss_mean=loss_mean,
        loss_std=loss_std,
        vram_epoch_max_mb=vram_epoch_max_mb,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    out_ch: int,
    dice_fn: Callable[[torch.Tensor, torch.Tensor, int], float],
    iou_fn: Callable[[torch.Tensor, torch.Tensor, int], float],
) -> EpochValResult:
    """
    Runs validation and returns mean/std for Dice and IoU across batches.
    `dice_fn`/`iou_fn` should return float or 0-d tensor (we cast to float).
    """
    model.eval()
    dices: List[float] = []
    ious: List[float] = []

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)
        preds = model(imgs)

        dices.append(float(dice_fn(preds, masks, num_classes=out_ch)))
        ious.append(float(iou_fn(preds, masks, num_classes=out_ch)))

    return EpochValResult(
        dice_mean=float(np.mean(dices)) if dices else float("nan"),
        dice_std=float(np.std(dices)) if dices else float("nan"),
        iou_mean=float(np.mean(ious)) if ious else float("nan"),
        iou_std=float(np.std(ious)) if ious else float("nan"),
    )

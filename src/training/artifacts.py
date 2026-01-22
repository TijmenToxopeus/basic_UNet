# src/training/artifacts.py
from __future__ import annotations

import json
import os
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.training.metrics import compute_model_flops, compute_inference_time


@torch.no_grad()
def profile_model(model: torch.nn.Module, *, in_ch: int, input_hw=(256, 256)) -> Dict[str, float]:
    model.eval()
    flops, params_count = compute_model_flops(model, (1, in_ch, input_hw[0], input_hw[1]))
    infer_ms = compute_inference_time(model, (1, in_ch, input_hw[0], input_hw[1]))
    return {
        "params_m": float(params_count / 1e6),
        "flops_g": float(flops / 1e9),
        "inference_ms": float(infer_ms),
        "params_count": float(params_count),
        "flops": float(flops),
    }


def save_training_curves(metrics_log: Dict[str, list], save_dir: str, phase: str) -> str:
    epochs_arr = metrics_log["epoch"]
    plt.figure(figsize=(8, 6))

    mean = np.array(metrics_log["train_loss_mean"])
    std = np.array(metrics_log["train_loss_std"])
    plt.plot(epochs_arr, mean, label="Train Loss")
    plt.fill_between(epochs_arr, mean - std, mean + std, alpha=0.2)

    mean = np.array(metrics_log["val_dice_mean"])
    std = np.array(metrics_log["val_dice_std"])
    plt.plot(epochs_arr, mean, label="Val Dice")
    plt.fill_between(epochs_arr, mean - std, mean + std, alpha=0.2)

    mean = np.array(metrics_log["val_iou_mean"])
    std = np.array(metrics_log["val_iou_std"])
    plt.plot(epochs_arr, mean, label="Val IoU")
    plt.fill_between(epochs_arr, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Progress ({phase})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def save_metrics_per_epoch_json(
    metrics_log: Dict[str, list],
    save_dir: str,
    filename: str = "metrics_per_epoch.json",
) -> str:
    """
    Save per-epoch training/validation metrics to JSON.

    This file is intended for post-hoc analysis of training dynamics
    (e.g. retraining after pruning, checkpoint vs reinitialization).
    """
    path = os.path.join(save_dir, filename)
    return save_json(metrics_log, path)


def save_json(obj: Dict[str, Any], path: str) -> str:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
    return path

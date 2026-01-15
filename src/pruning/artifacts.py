# src/pruning/artifacts.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ParamStats:
    original_params: int
    pruned_params: int
    reduction_percent: float


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def compute_param_stats(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
) -> ParamStats:
    orig = count_params(original_model)
    pruned = count_params(pruned_model)
    reduction = 100.0 * (1.0 - pruned / orig) if orig > 0 else float("nan")
    return ParamStats(original_params=orig, pruned_params=pruned, reduction_percent=float(reduction))

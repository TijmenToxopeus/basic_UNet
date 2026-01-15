# src/training/logging.py
from __future__ import annotations
from typing import Any, Callable, Dict, MutableMapping, Optional

def log_epoch(
    metrics_log: MutableMapping[str, list],
    epoch_metrics: Dict[str, Any],
    *,
    wandb_log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    Logs metrics to W&B (optional) and appends matching keys into metrics_log.
    Only keys that already exist in metrics_log are appended (prevents typos creating new keys).
    """
    if wandb_log_fn is not None:
        wandb_log_fn(epoch_metrics)

    for k, v in epoch_metrics.items():
        if k in metrics_log:
            metrics_log[k].append(v)

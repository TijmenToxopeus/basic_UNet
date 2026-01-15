from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class PruneOutput:
    masks: Dict[str, Any]                 # whatever your rebuild expects
    method: str                           # "l1_norm" | "correlation"
    extra: Dict[str, Any]                 # optional: df stats, threshold used, etc.


class BasePruningMethod:
    """Common interface for all pruning methods."""

    name: str  # override in subclasses

    def compute_masks(
        self,
        model: torch.nn.Module,
        *,
        cfg: Dict[str, Any],
        seed: int,
        deterministic: bool,
        device: torch.device,
    ) -> PruneOutput:
        raise NotImplementedError

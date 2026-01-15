from __future__ import annotations

from typing import Any, Dict

import torch

from .base import BasePruningMethod, PruneOutput
from src.pruning.model_inspect import (
    model_to_dataframe_with_l1,
    compute_l1_norms,
    compute_l1_stats,
    get_pruning_masks_blockwise,
)


class L1NormPruning(BasePruningMethod):
    name = "l1_norm"

    def compute_masks(
        self,
        model: torch.nn.Module,
        *,
        cfg: Dict[str, Any],
        seed: int,
        deterministic: bool,
        device: torch.device,
    ) -> PruneOutput:
        pruning_cfg = cfg["pruning"]
        block_ratios = pruning_cfg.get("ratios", {}).get("block_ratios", {})
        default_ratio = pruning_cfg.get("ratios", {}).get("default", 0.25)

        norms = compute_l1_norms(model)
        l1_stats = compute_l1_stats(norms)
        df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)

        masks = get_pruning_masks_blockwise(
            model,
            norms,
            block_ratios=block_ratios,
            default_ratio=default_ratio,
            seed=seed,
            deterministic=deterministic,
        )

        return PruneOutput(
            masks=masks,
            method=self.name,
            extra={
                "block_ratios": block_ratios,
                "default_ratio": default_ratio,
                "l1_df": df,  # can be logged to wandb by orchestrator
            },
        )

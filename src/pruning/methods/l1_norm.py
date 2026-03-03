from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json

import torch

from .base import BasePruningMethod, PruneOutput
from src.pruning.importance_pruning import (
    model_to_dataframe_with_importance,
    compute_l1_norms,
    compute_importance_stats,
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
        l1_stats = compute_importance_stats(norms, label="L1")
        df = model_to_dataframe_with_importance(model, l1_stats, remove_nan_layers=True, mean_stat_col="Mean L1")

        masks = get_pruning_masks_blockwise(
            model,
            norms,
            block_ratios=block_ratios,
            default_ratio=default_ratio,
            seed=seed,
            deterministic=deterministic,
        )

        save_dir = pruning_cfg.get("save_masks_dir")
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(masks, save_path / "l1_masks.pt")
            meta = {
                "default_ratio": default_ratio,
                "block_ratios": block_ratios,
                "layers": {
                    name: {"kept": int(mask.sum().item()), "total": int(mask.numel())}
                    for name, mask in masks.items()
                },
            }
            (save_path / "l1_masks_meta.json").write_text(json.dumps(meta, indent=2))

        return PruneOutput(
            masks=masks,
            method=self.name,
            extra={
                "block_ratios": block_ratios,
                "default_ratio": default_ratio,
                "l1_df": df,  # can be logged to wandb by orchestrator
            },
        )

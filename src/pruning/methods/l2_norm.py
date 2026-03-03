from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json

import torch

from .base import BasePruningMethod, PruneOutput
from src.pruning.importance_pruning import (
    compute_l2_norms,
    get_pruning_masks_blockwise,
)


class L2NormPruning(BasePruningMethod):
    name = "l2_norm"

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

        norms = compute_l2_norms(model)
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
            torch.save(masks, save_path / "l2_masks.pt")
            meta = {
                "default_ratio": default_ratio,
                "block_ratios": block_ratios,
                "layers": {
                    name: {"kept": int(mask.sum().item()), "total": int(mask.numel())}
                    for name, mask in masks.items()
                },
            }
            (save_path / "l2_masks_meta.json").write_text(json.dumps(meta, indent=2))

        return PruneOutput(
            masks=masks,
            method=self.name,
            extra={
                "block_ratios": block_ratios,
                "default_ratio": default_ratio,
            },
        )

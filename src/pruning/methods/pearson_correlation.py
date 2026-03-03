from __future__ import annotations

from typing import Any, Dict

import torch

from .base import BasePruningMethod, PruneOutput
from src.pruning.redundancy_pruning import get_redundancy_masks, load_random_slices_acdc


class PearsonCorrelationPruning(BasePruningMethod):
    name = "pearson_correlation"

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

        # IMPORTANT: use paths from cfg/paths; no hardcoded dataset path.
        # Use train_dir from cfg like you already do elsewhere.
        img_dir = cfg["train"]["paths"]["train_dir"]

        num_samples = int(pruning_cfg.get("num_slices", 20))
        threshold = float(pruning_cfg.get("threshold", 0.90))
        batch_size = int(pruning_cfg.get("batch_size", 4))

        example_slices = load_random_slices_acdc(
            img_dir,
            num_slices=num_samples,
            seed=seed,
        )

        save_dir = pruning_cfg.get("save_masks_dir")
        masks = get_redundancy_masks(
            model=model,
            example_slices=example_slices,
            block_ratios=block_ratios,
            similarity_metric="pearson",
            default_ratio=default_ratio,
            threshold=threshold,
            batch_size=batch_size,
            plot=False,
            save_dir=save_dir,
        )

        return PruneOutput(
            masks=masks,
            method=self.name,
            extra={
                "block_ratios": block_ratios,
                "default_ratio": default_ratio,
                "threshold": threshold,
                "num_slices": num_samples,
                "batch_size": batch_size,
                "similarity_metric": "pearson",
            },
        )


# Backward-compatible alias
SimilarFeaturePruning = PearsonCorrelationPruning

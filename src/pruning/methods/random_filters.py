from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json

import numpy as np
import torch

from .base import BasePruningMethod, PruneOutput
from src.pruning.importance_pruning import _infer_block_from_layer_name, _should_skip_for_pruning


class RandomFilterPruning(BasePruningMethod):
    name = "random_filters"

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

        rng = np.random.default_rng(seed)
        layer_lookup = dict(model.named_modules())
        masks: Dict[str, torch.Tensor] = {}

        for name, layer in model.named_modules():
            if not isinstance(layer, torch.nn.Conv2d):
                continue

            num_out = int(layer.out_channels)
            block = _infer_block_from_layer_name(name)
            ratio = float(block_ratios.get(block, default_ratio))

            if _should_skip_for_pruning(name, layer):
                masks[name] = torch.ones(num_out, dtype=torch.bool)
                continue

            ratio = max(0.0, min(1.0, ratio))
            k_prune = int(np.floor(num_out * ratio))
            k_prune = max(0, min(num_out, k_prune))

            mask = torch.ones(num_out, dtype=torch.bool)
            if k_prune > 0:
                prune_idx = rng.choice(num_out, size=k_prune, replace=False)
                mask[torch.as_tensor(prune_idx, dtype=torch.long)] = False

            kept = int(mask.sum().item())
            print(
                f"Block {block:15s} | Layer {name:25s} | ratio={ratio:.2f} | "
                f"pruned {k_prune}/{num_out} | kept {kept}/{num_out}"
            )
            masks[name] = mask

        save_dir = pruning_cfg.get("save_masks_dir")
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(masks, save_path / "random_masks.pt")
            meta = {
                "default_ratio": default_ratio,
                "block_ratios": block_ratios,
                "seed": seed,
                "layers": {
                    name: {"kept": int(mask.sum().item()), "total": int(mask.numel())}
                    for name, mask in masks.items()
                },
            }
            (save_path / "random_masks_meta.json").write_text(json.dumps(meta, indent=2))

        return PruneOutput(
            masks=masks,
            method=self.name,
            extra={
                "block_ratios": block_ratios,
                "default_ratio": default_ratio,
                "seed": seed,
            },
        )

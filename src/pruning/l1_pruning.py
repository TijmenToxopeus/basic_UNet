# """
# Pruning pipeline for basic UNet
# --------------------------------
# Loads a trained baseline model, applies structured L1-based pruning,
# rebuilds the pruned model, and saves both the weights and architecture metadata.
# """

# import os
# import json
# import torch
# import pandas as pd
# import wandb  # ✅ added

# from src.models.unet import UNet
# from src.pruning.model_inspect import (
#     model_to_dataframe_with_l1,
#     compute_l1_norms,
#     compute_l1_stats,
#     get_pruning_masks_blockwise,
# )
# from src.pruning.rebuild import rebuild_pruned_unet
# from src.utils.config import load_config
# from src.utils.paths import get_paths


# # ------------------------------------------------------------
# # MAIN PRUNING PIPELINE
# # ------------------------------------------------------------
# def run_pruning(cfg=None):
#     # ============================================================
#     # --- LOAD CONFIGURATION ---
#     # ============================================================
#     if cfg is None:
#         cfg, config_path = load_config(return_path=True)
#     else:
#         config_path = None

#     paths = get_paths(cfg, config_path)
#     pruning_cfg = cfg["pruning"]
#     model_cfg = cfg["train"]["model"]

#     exp_name = cfg["experiment"]["experiment_name"]
#     model_name = cfg["experiment"]["model_name"]
#     print(f"✂️ Starting L1-based structured pruning for {model_name}")
#     print(paths)

#     block_ratios = pruning_cfg.get("ratios", {}).get("block_ratios", {})
#     default_ratio = pruning_cfg.get("ratios", {}).get("default", 0.25)

#     # ============================================================
#     # --- INIT WANDB RUN ---
#     # ============================================================
#     wandb.init(
#         project="unet-pruning",
#         group=exp_name,
#         job_type="pruning",
#         name=f"{exp_name}_pruning",
#         config=cfg,
#         dir=str(paths.base_dir),
#     )

#     # ============================================================
#     # --- LOAD BASELINE MODEL ---
#     # ============================================================
#     baseline_ckpt = paths.baseline_ckpt
#     if not baseline_ckpt.exists():
#         raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

#     in_ch = model_cfg["in_channels"]
#     out_ch = model_cfg["out_channels"]
#     enc_features = model_cfg["features"]

#     device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     print(f"📦 Loading baseline model from: {baseline_ckpt}")

#     model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#     state = torch.load(baseline_ckpt, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # ============================================================
#     # --- COMPUTE L1 NORMS ---
#     # ============================================================
#     print("📊 Computing L1 norms for all Conv layers...")
#     norms = compute_l1_norms(model)
#     l1_stats = compute_l1_stats(norms)
#     df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
#     pd.set_option("display.max_rows", None)
#     print("✅ L1 statistics computed.\n")

#     # Log table of L1 stats to W&B
#     wandb.log({"l1_norms": wandb.Table(dataframe=df)})

#     # ============================================================
#     # --- GENERATE MASKS ---
#     # ============================================================
#     print("✂️ Generating pruning masks...")
#     masks = get_pruning_masks_blockwise(model, norms, block_ratios=block_ratios, default_ratio=default_ratio)
#     print("✅ Pruning masks generated.\n")

#     # ============================================================
#     # --- REBUILD PRUNED MODEL ---
#     # ============================================================
#     paths.ensure_dir(paths.pruned_model_dir)
#     pruned_model = rebuild_pruned_unet(model, masks, save_path=paths.pruned_model)

#     # ============================================================
#     # --- PARAMETER REDUCTION SUMMARY ---
#     # ============================================================
#     orig_params = sum(p.numel() for p in model.parameters())
#     pruned_params = sum(p.numel() for p in pruned_model.parameters())
#     reduction = 100 * (1 - pruned_params / orig_params)

#     meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#     print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")

#     # Log summary metrics to W&B
#     wandb.log({
#         "orig_params": orig_params,
#         "pruned_params": pruned_params,
#         "reduction_percent": reduction,
#         "default_ratio": default_ratio,
#     })

#     # ============================================================
#     # --- SAVE SUMMARY JSON ---
#     # ============================================================
#     summary = {
#         "experiment": exp_name,
#         "model_name": model_name,
#         "block_ratios": block_ratios,
#         "default_ratio": default_ratio,
#         "orig_params": int(orig_params),
#         "pruned_params": int(pruned_params),
#         "reduction_percent": float(reduction),
#         "baseline_ckpt": str(baseline_ckpt),
#         "pruned_model": str(paths.pruned_model),
#         "meta_path": str(meta_path),
#     }

#     summary_path = paths.pruned_model_dir / "pruning_summary.json"
#     with open(summary_path, "w") as f:
#         json.dump(summary, f, indent=4)

#     wandb.save(str(summary_path))
#     wandb.save(str(paths.pruned_model))
#     wandb.finish()

#     print(f"💾 Summary saved to {summary_path}")
#     print("✅ Pruning complete.\n")


# if __name__ == "__main__":
#     run_pruning()


"""
Pruning pipeline for basic UNet
--------------------------------
Loads a trained baseline model, applies structured L1-based pruning,
rebuilds the pruned model, and saves both the weights and architecture metadata.
"""

import os
import json
import torch
import pandas as pd
import wandb

from src.models.unet import UNet
from src.pruning.model_inspect import (
    model_to_dataframe_with_l1,
    compute_l1_norms,
    compute_l1_stats,
    get_pruning_masks_blockwise,
)
from src.pruning.rebuild import rebuild_pruned_unet
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb


# ------------------------------------------------------------
# MAIN PRUNING PIPELINE
# ------------------------------------------------------------
def run_pruning(cfg=None):
    # ============================================================
    # --- LOAD CONFIGURATION ---
    # ============================================================
    if cfg is None:
        cfg, config_path = load_config(return_path=True)
    else:
        config_path = None

    paths = get_paths(cfg, config_path)
    pruning_cfg = cfg["pruning"]
    model_cfg = cfg["train"]["model"]

    exp_name = cfg["experiment"]["experiment_name"]
    model_name = cfg["experiment"]["model_name"]
    print(f"✂️ Starting L1-based structured pruning for {model_name}")
    print(paths)

    block_ratios = pruning_cfg.get("ratios", {}).get("block_ratios", {})
    default_ratio = pruning_cfg.get("ratios", {}).get("default", 0.25)

    # ============================================================
    # --- INIT WANDB RUN ---
    # ============================================================
    run = setup_wandb(cfg, job_type="pruning")

    # ============================================================
    # --- LOAD BASELINE MODEL ---
    # ============================================================
    baseline_ckpt = paths.baseline_ckpt
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

    in_ch = model_cfg["in_channels"]
    out_ch = model_cfg["out_channels"]
    enc_features = model_cfg["features"]

    device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"📦 Loading baseline model from: {baseline_ckpt}")

    model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
    state = torch.load(baseline_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ============================================================
    # --- COMPUTE L1 NORMS ---
    # ============================================================
    print("📊 Computing L1 norms for all Conv layers...")
    norms = compute_l1_norms(model)
    l1_stats = compute_l1_stats(norms)
    df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
    pd.set_option("display.max_rows", None)
    print("✅ L1 statistics computed.\n")

    # Log L1 norm table to W&B
    wandb.log({"l1_norms": wandb.Table(dataframe=df)})

    # ============================================================
    # --- GENERATE MASKS ---
    # ============================================================
    print("✂️ Generating pruning masks...")
    masks = get_pruning_masks_blockwise(model, norms, block_ratios=block_ratios, default_ratio=default_ratio)
    print("✅ Pruning masks generated.\n")

    # ============================================================
    # --- REBUILD PRUNED MODEL ---
    # ============================================================
    paths.ensure_dir(paths.pruned_model_dir)
    pruned_model = rebuild_pruned_unet(model, masks, save_path=paths.pruned_model)

    # ============================================================
    # --- PARAMETER REDUCTION SUMMARY ---
    # ============================================================
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = 100 * (1 - pruned_params / orig_params)

    meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
    print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")

    # Log parameter stats to W&B
    wandb.log({
        "orig_params": orig_params,
        "pruned_params": pruned_params,
        "reduction_percent": reduction,
        "default_ratio": default_ratio,
        **{f"ratio_{k}": v for k, v in block_ratios.items()}
    })

    # ============================================================
    # --- SAVE SUMMARY JSON ---
    # ============================================================
    summary = {
        "experiment": exp_name,
        "model_name": model_name,
        "block_ratios": block_ratios,
        "default_ratio": default_ratio,
        "orig_params": int(orig_params),
        "pruned_params": int(pruned_params),
        "reduction_percent": float(reduction),
        "baseline_ckpt": str(baseline_ckpt),
        "pruned_model": str(paths.pruned_model),
        "meta_path": str(meta_path),
    }

    summary_path = paths.pruned_model_dir / "pruning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    wandb.save(str(summary_path))
    wandb.save(str(paths.pruned_model))

    run.finish()

    print(f"💾 Summary saved to {summary_path}")
    print("✅ Pruning complete.\n")


if __name__ == "__main__":
    run_pruning()

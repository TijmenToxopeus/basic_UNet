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


    # ============================================================
    # --- OPTIONAL: REINITIALIZE PRUNED MODEL ---
    # ============================================================
    if pruning_cfg.get("reinitialize_weights") == "rewind":
        print("🔄 Reinitializing pruned model with rewind weights...")
        rewind_ckpt = paths.rewind_ckpt

        if not rewind_ckpt.exists():
            raise FileNotFoundError(f"❌ Rewind checkpoint not found at {rewind_ckpt}")
        
        rewind_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        state = torch.load(rewind_ckpt, map_location=device)
        rewind_model.load_state_dict(state)
        rewind_model.eval()
        pruned_model = rebuild_pruned_unet(rewind_model, masks, save_path=paths.pruned_model)

    else:
        print("🔄 Reinitializing pruned model with current weights...")
        pruned_model = rebuild_pruned_unet(model, masks, save_path=paths.pruned_model)

    # ============================================================
    # --- OPTIONAL: REINITIALIZE PRUNED MODEL ---
    # ============================================================
    if pruning_cfg.get("reinitialize_weights") == "random":
        print("🔄 Reinitializing pruned model with random weights...")

        # ---- Compute global pre-reinit stats ----
        before_params = []
        for m in pruned_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                before_params.append(m.weight.detach().cpu().flatten())
        before_params = torch.cat(before_params)
        before_mean = before_params.mean().item()
        before_std = before_params.std().item()

        print(f"📊 BEFORE reinit: mean={before_mean:.6f}, std={before_std:.6f}")

        # ---- Apply initialization ----
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        pruned_model.apply(init_weights)

        # ---- Compute global post-reinit stats ----
        after_params = []
        for m in pruned_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                after_params.append(m.weight.detach().cpu().flatten())
        after_params = torch.cat(after_params)
        after_mean = after_params.mean().item()
        after_std = after_params.std().item()

        print(f"📊 AFTER reinit:  mean={after_mean:.6f}, std={after_std:.6f}")

        # ---- Check success ----
        print("🔍 Reinit verification:")
        print(f"Mean changed by {abs(after_mean - before_mean):.6f}")
        print(f"Std  changed by {abs(after_std - before_std):.6f}")
        print("✅ Reinitialized weights.\n")

        # Log to W&B
        wandb.log({
            "reinit_mean_before": before_mean,
            "reinit_std_before": before_std,
            "reinit_mean_after": after_mean,
            "reinit_std_after": after_std,
        })

    # ============================================================
    # --- PARAMETER REDUCTION SUMMARY ---
    # ============================================================
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = 100 * (1 - pruned_params / orig_params)

    meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
    print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")

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
        "reinitialized": pruning_cfg.get("reinitialize", False),
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

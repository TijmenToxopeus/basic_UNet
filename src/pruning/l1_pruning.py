"""
Pruning pipeline for basic UNet
--------------------------------
Loads a trained baseline model, applies structured L1-based pruning,
rebuilds the pruned model, and saves both the weights and architecture metadata.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torchvision.transforms as T
import wandb
import copy

from src.models.unet import UNet
from src.pruning.model_inspect import (
    model_to_dataframe_with_l1,
    compute_l1_norms,
    compute_l1_stats,
    get_pruning_masks_blockwise,
)
from src.pruning.similar_feature_pruning import get_redundancy_masks, load_random_slices_acdc 
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

    wandb.log({"l1_norms": wandb.Table(dataframe=df)})

    # ============================================================
    # --- GENERATE MASKS ---
    # ============================================================
    print("✂️ Generating pruning masks...")
    method = pruning_cfg.get("method", "l1_norm")

    print(f"✂️ Using pruning method: {method}")

    if method == "l1_norm":
        # ---- L1-based pruning ----
        print("📊 Computing L1 norms for all Conv layers...")
        norms = compute_l1_norms(model)
        l1_stats = compute_l1_stats(norms)
        df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
        wandb.log({"l1_norms": wandb.Table(dataframe=df)})

        print("✂️ Generating L1-based pruning masks...")
        masks = get_pruning_masks_blockwise(
            model,
            norms,
            block_ratios=block_ratios,
            default_ratio=default_ratio
        )

    elif method == "correlation":
        print("🔍 Running correlation-based redundancy pruning...")
        
        # -------------------------
        # Load MULTIPLE slices
        # -------------------------
        img_dir = "/mnt/hdd/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr/"
        num_samples = pruning_cfg.get("num_slices", 20)
        print(f"num slices: {num_samples}")

        example_slices = load_random_slices_acdc(img_dir, num_slices=num_samples, seed=42)

        print(f"Loaded {len(example_slices)} example slices for correlation pruning.")

        threshold = pruning_cfg.get("threshold", 0.9)
        batch_size = pruning_cfg.get("batch_size", 4)

        # -------------------------
        # Run correlation pruning
        # -------------------------
        masks = get_redundancy_masks(
            model=model,
            example_slices=example_slices,
            block_ratios=block_ratios,
            threshold=threshold,
            batch_size=batch_size,
            plot=False
        )
    else:
        raise ValueError(f"❌ Unknown pruning method '{method}'. Choose 'l1' or 'correlation'.")


    # ============================================================
    # --- REBUILD PRUNED MODEL ---
    # ============================================================
    paths.ensure_dir(paths.pruned_model_dir)

    print("🔧 Rebuilding pruned model...")
    if pruning_cfg.get("reinitialize_weights") == "rewind":
        print("🔄 Reinitializing pruned model with rewind weights...")
        rewind_ckpt = paths.rewind_ckpt
        if not rewind_ckpt.exists():
            raise FileNotFoundError(f"❌ Rewind checkpoint not found at {rewind_ckpt}")
        else:
            print(f"Loading rewind weights from {rewind_ckpt}")
        
        rewind_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        rewind_model.load_state_dict(torch.load(rewind_ckpt, map_location=device))
        rewind_model.eval()
        pruned_model = rebuild_pruned_unet(rewind_model, masks, save_path=paths.pruned_model)

    else:        
        pruned_model = rebuild_pruned_unet(model, masks, save_path=paths.pruned_model)

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

            def init_weights(m):
                # Conv layers
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

                # Transposed conv layers (decoder upsampling)
                elif isinstance(m, torch.nn.ConvTranspose2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

                # Linear layers (if your model has any)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

                # BatchNorm layers
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            pruned_model.apply(init_weights)

            save_path=paths.pruned_model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(pruned_model.state_dict(), save_path)
            print(f"💾 Saved pruned model to {save_path}")

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
        else:
            print("🔄 Using current weights for rebuild...")

    # PARAM SUMMARY, SAVE, ETC...
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = 100 * (1 - pruned_params / orig_params)

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
    }

    summary_path = paths.pruned_model_dir / "pruning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    wandb.save(str(summary_path))
    # wandb.save(str(paths.pruned_model))

    run.finish()
    print(f"💾 Summary saved to {summary_path}")
    print("✅ Pruning complete.\n")


if __name__ == "__main__":
    run_pruning()

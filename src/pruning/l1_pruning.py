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
from src.pruning.weight_initialization import apply_finetune_mode
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb


def run_pruning(cfg=None):
    # ============================================================
    # --- LOAD CONFIG & PATHS ---
    # ============================================================
    if cfg is None:
        cfg, config_path = load_config(return_path=True)
    else:
        config_path = None

    paths = get_paths(cfg, config_path)
    pruning_cfg = cfg["pruning"]
    finetune_cfg = pruning_cfg["finetune"]
    model_cfg = cfg["train"]["model"]

    exp_name = cfg["experiment"]["experiment_name"]
    model_name = cfg["experiment"]["model_name"]

    finetune_mode = finetune_cfg["mode"]
    device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    block_ratios = pruning_cfg["ratios"]["block_ratios"]
    default_ratio = pruning_cfg["ratios"]["default"]

    print(f"✂️ Starting L1-based pruning for {model_name}")
    print(f"🔧 Finetune mode: {finetune_mode}")
    print(paths)

    run = setup_wandb(cfg, job_type="pruning")


    # ============================================================
    # STEP 1 — LOAD FINAL BASELINE MODEL
    # ============================================================
    baseline_ckpt = paths.baseline_ckpt
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

    in_ch = model_cfg["in_channels"]
    out_ch = model_cfg["out_channels"]
    enc_features = model_cfg["features"]

    print(f"📦 Loading baseline model from: {baseline_ckpt}")
    baseline_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
    baseline_state = torch.load(baseline_ckpt, map_location=device)
    baseline_model.load_state_dict(baseline_state)
    baseline_model.eval()


    # ============================================================
    # STEP 2 — COMPUTE L1 NORMS + MASKS
    # ============================================================
    print("📊 Computing L1 norms for Conv layers...")
    norms = compute_l1_norms(baseline_model)
    l1_stats = compute_l1_stats(norms)
    df = model_to_dataframe_with_l1(baseline_model, l1_stats, remove_nan_layers=True)

    wandb.log({"l1_norms": wandb.Table(dataframe=df)})

    print("✂️ Generating pruning masks...")
    masks = get_pruning_masks_blockwise(baseline_model, norms, block_ratios=block_ratios, default_ratio=default_ratio)
    print("✅ Masks generated.\n")


    # ============================================================
    # STEP 3 — REBUILD PRUNED MODEL (baseline weights sliced)
    # ============================================================
    print("🏗 Rebuilding pruned UNet...")
    paths.ensure_dir(paths.pruned_model_dir)
    # pruned_model = rebuild_pruned_unet(baseline_model, masks, save_path=paths.pruned_model)

        # 1) Rebuild architecture with pruned channels + copy baseline weights
    pruned_model = rebuild_pruned_unet(
        model=baseline_model,
        masks=masks,
        save_path=None
    )

    # 2) Apply finetune mode (current / random / rewind)
    pruned_model = apply_finetune_mode(
        pruned_model=pruned_model,
        finetune_cfg=cfg["pruning"]["finetune"],
        masks=masks,
        baseline_model=baseline_model,
        device=device,
        paths=paths,
    )
       
       

    # 3) NOW save model, after initialization is correct
    torch.save(pruned_model.state_dict(), paths.pruned_model)


    # ============================================================
    # STEP 4 — APPLY FINETUNE MODE TO THE PRUNED MODEL
    # ============================================================
    # print(f"🔧 Applying finetune mode: {finetune_mode}")
    # pruned_model = apply_finetune_mode(
    #     pruned_model=pruned_model,
    #     finetune_cfg=cfg["pruning"]["finetune"],
    #     masks=masks,
    #     baseline_model=baseline_model,   # the unpruned baseline model
    #     device=device,
    #     paths=paths,
    # )

    wandb.log({
        "finetune/mode": finetune_mode,
        "finetune/rewind_checkpoint": str(finetune_cfg.get("rewind_checkpoint", "AUTO")),
    })


    # ============================================================
    # STEP 5 — PARAMETER STATS
    # ============================================================
    orig_params = sum(p.numel() for p in baseline_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = 100 * (1 - pruned_params / orig_params)

    print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")

    wandb.log({
        "orig_params": orig_params,
        "pruned_params": pruned_params,
        "reduction_percent": reduction,
        "default_ratio": default_ratio,
        **{f"ratio_{k}": v for k, v in block_ratios.items()}
    })


    # ============================================================
    # STEP 6 — SAVE METADATA (meta JSON)
    # ============================================================
    meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")

    meta = {
        "enc_features": pruned_model.enc_features,
        "dec_features": pruned_model.dec_features,
        "bottleneck_out": pruned_model.bottleneck_out_channels,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    wandb.save(str(meta_path))


    # ============================================================
    # STEP 7 — SAVE PRUNING SUMMARY JSON
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
        "finetune_mode": finetune_mode,
    }

    summary_path = paths.pruned_model_dir / "pruning_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    wandb.save(str(summary_path))
    wandb.save(str(paths.pruned_model))

    run.finish()

    print(f"💾 Summary saved to {summary_path}")
    print("✅ Pruning complete.\n")







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
#     run = setup_wandb(cfg, job_type="pruning")

#     finetune_mode = cfg["pruning"]["finetune"]["mode"]

#     if finetune_mode == "current":
#         baseline_ckpt = paths.baseline_ckpt
#         if not baseline_ckpt.exists():
#             raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

#         in_ch = model_cfg["in_channels"]
#         out_ch = model_cfg["out_channels"]
#         enc_features = model_cfg["features"]

#         device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#         print(f"📦 Loading baseline model from: {baseline_ckpt}")

#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         state = torch.load(baseline_ckpt, map_location=device)
#         model.load_state_dict(state)
#         model.eval()

#     elif finetune_mode == "random":
#         baseline_ckpt = None
#         in_ch = model_cfg["in_channels"]
#         out_ch = model_cfg["out_channels"]
#         enc_features = model_cfg["features"]

#         device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#         print(f"📦 Loading random model")

#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         model.eval()

        
#     elif finetune_mode == "rewind":
#         baseline_ckpt = r"/media/ttoxopeus/basic_UNet/results/UNet_ACDC/exp22_try_new_finetune_modes/baseline/training/epoch_10.pth"

#         in_ch = model_cfg["in_channels"]
#         out_ch = model_cfg["out_channels"]
#         enc_features = model_cfg["features"]

#         device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#         print(f"📦 Loading rewind model from: {baseline_ckpt}")

#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         state = torch.load(baseline_ckpt, map_location=device)
#         model.load_state_dict(state)
#         model.eval()

#     else:
#         raise ValueError(f"Unknown finetune mode: {finetune_mode}")







#     # # ============================================================
#     # # --- LOAD BASELINE MODEL ---
#     # # ============================================================
#     # baseline_ckpt = paths.baseline_ckpt
#     # if not baseline_ckpt.exists():
#     #     raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

#     # in_ch = model_cfg["in_channels"]
#     # out_ch = model_cfg["out_channels"]
#     # enc_features = model_cfg["features"]

#     # device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     # print(f"📦 Loading baseline model from: {baseline_ckpt}")

#     # model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#     # state = torch.load(baseline_ckpt, map_location=device)
#     # model.load_state_dict(state)
#     # model.eval()



#     # # ============================================================
#     # # --- LOAD REWIND MODEL ---
#     # # ============================================================
#     # rewind_ckpt = r"/media/ttoxopeus/basic_UNet/results/UNet_ACDC/exp22_try_new_finetune_modes/baseline/training/epoch_10.pth"
#     # if not rewind_ckpt.exists():
#     #     raise FileNotFoundError(f"❌ Rewind checkpoint not found at {rewind_ckpt}")

#     # in_ch = model_cfg["in_channels"]
#     # out_ch = model_cfg["out_channels"]
#     # enc_features = model_cfg["features"]

#     # device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     # print(f"📦 Loading rewind model from: {rewind_ckpt}")

#     # rewind_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#     # state = torch.load(rewind_ckpt, map_location=device)
#     # rewind_model.load_state_dict(state)
#     # rewind_model.eval()

#     # ============================================================
#     # --- COMPUTE L1 NORMS ---
#     # ============================================================
#     print("📊 Computing L1 norms for all Conv layers...")
#     norms = compute_l1_norms(model)
#     l1_stats = compute_l1_stats(norms)
#     df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
#     pd.set_option("display.max_rows", None)
#     print("✅ L1 statistics computed.\n")

#     # Log L1 norm table to W&B
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
#     # # --- APPLY INITIALIZATION MODE ---
#     # pruned_model = apply_finetune_mode(pruned_model, cfg, device, paths)

#     # finetune_mode = cfg["pruning"]["finetune"]["mode"]
#     # rewind_ckpt = cfg["pruning"]["finetune"].get("rewind_checkpoint", None)

#     # wandb.log({
#     #     "finetune/mode": finetune_mode,
#     #     "finetune/rewind_checkpoint": str(rewind_ckpt) if rewind_ckpt else "AUTO",
#     # })

#     # ============================================================
#     # --- PARAMETER REDUCTION SUMMARY ---
#     # ============================================================
#     orig_params = sum(p.numel() for p in model.parameters())
#     pruned_params = sum(p.numel() for p in pruned_model.parameters())
#     reduction = 100 * (1 - pruned_params / orig_params)

#     meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#     print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")

#     # Log parameter stats to W&B
#     wandb.log({
#         "orig_params": orig_params,
#         "pruned_params": pruned_params,
#         "reduction_percent": reduction,
#         "default_ratio": default_ratio,
#         **{f"ratio_{k}": v for k, v in block_ratios.items()}
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

#     run.finish()

#     print(f"💾 Summary saved to {summary_path}")
#     print("✅ Pruning complete.\n")


# if __name__ == "__main__":
#     run_pruning()

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


# ------------------------------------------------------------
# MAIN PRUNING PIPELINE
# ------------------------------------------------------------
# def run_pruning():
#     # ============================================================
#     # --- LOAD CONFIGURATION ---
#     # ============================================================
#     cfg, config_path = load_config(return_path=True)
#     paths = get_paths(cfg, config_path)

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

    print(f"✂️ Starting L1-based structured pruning for {cfg['experiment']['model_name']}")
    print(paths)

    block_ratios = pruning_cfg.get("ratios", {}).get("block_ratios", {})
    default_ratio = pruning_cfg.get("ratios", {}).get("default", 0.25)

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
    print("✅ Baseline model loaded.\n")

    # ============================================================
    # --- COMPUTE L1 NORMS ---
    # ============================================================
    print("📊 Computing L1 norms for all Conv layers...")
    norms = compute_l1_norms(model)
    l1_stats = compute_l1_stats(norms)

    df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
    pd.set_option("display.max_rows", None)
    print(df[["Layer", "Out Ch", "Mean L1", "Min L1", "Max L1"]].head(10))
    print("✅ L1 statistics computed.\n")

    # ============================================================
    # --- GENERATE MASKS ---
    # ============================================================
    print("✂️ Generating pruning masks...")
    masks = get_pruning_masks_blockwise(model, norms, block_ratios=block_ratios, default_ratio=default_ratio)
    print("✅ Pruning masks generated.\n")

    # ============================================================
    # --- REBUILD PRUNED MODEL ---
    # ============================================================
    # os.makedirs(paths.pruned_dir, exist_ok=True)
    paths.ensure_dir(paths.pruned_model_dir)
    pruned_model = rebuild_pruned_unet(model, masks, save_path=paths.pruned_model)
    print(f"\n✅ Pruned model rebuilt and saved to {paths.pruned_model}")

    # ============================================================
    # --- PARAMETER REDUCTION SUMMARY ---
    # ============================================================
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = 100 * (1 - pruned_params / orig_params)

    meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
    print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")
    print(f"🧾 Metadata saved to: {meta_path}\n")

    # ============================================================
    # --- SAVE SUMMARY JSON ---
    # ============================================================
    summary = {
        "experiment": cfg["experiment"]["experiment_name"],
        "model_name": cfg["experiment"]["model_name"],
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

    print(f"💾 Pruning summary saved to {summary_path}\n")

    print("🎯 Next steps:")
    print("   ▶ Retrain:   python -m src.training.train    # with phase='retraining'")
    print("   ▶ Evaluate:  python -m src.evaluation.eval  # with phase='pruned_evaluation'")


if __name__ == "__main__":
    run_pruning()

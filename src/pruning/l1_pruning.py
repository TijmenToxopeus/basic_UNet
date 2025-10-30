"""
Pruning pipeline for basic UNet
--------------------------------
Loads a trained baseline model, applies structured L1-based pruning,
rebuilds the pruned model, and saves both the weights and architecture metadata.
"""

import os
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


# ============================================================
# --- USER CONFIGURATION ---
# ============================================================

BASELINE_CKPT = "/media/ttoxopeus/basic_UNet/results/UNet_ACDC/exp2_larger_UNet/baseline/training/final_model.pth"
SAVE_PATH = "/media/ttoxopeus/basic_UNet/results/UNet_ACDC/exp2_larger_UNet/pruned/pruned_model.pth"

in_ch = 1
out_ch = 4
enc_features = [64, 128, 256, 512, 1024]

block_ratios = {
    "encoders.0": 0.0,
    "encoders.1": 0.0,
    "encoders.2": 0.1,
    "encoders.3": 0.2,
    "encoders.4": 0.3,
    "bottleneck": 0.3,
    "decoders.1": 0.3,
    "decoders.3": 0.2,
    "decoders.5": 0.1,
    "decoders.7": 0.0,
    "decoders.9": 0.0,
}
default_ratio = 0.25


# ============================================================
# --- STEP 1: Load trained baseline ---
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"📦 Loading baseline model from: {BASELINE_CKPT}")
model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
state = torch.load(BASELINE_CKPT, map_location=device)
model.load_state_dict(state)
model.eval()
print("✅ Baseline model loaded.\n")


# ============================================================
# --- STEP 2: Compute L1 norms & stats ---
# ============================================================

print("📊 Computing L1 norms for all Conv layers...")
norms = compute_l1_norms(model)
l1_stats = compute_l1_stats(norms)

df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
pd.set_option("display.max_rows", None)
print(df[["Layer", "Out Ch", "Mean L1", "Min L1", "Max L1"]].head(10))
print("✅ L1 statistics computed.\n")


# ============================================================
# --- STEP 3: Generate pruning masks ---
# ============================================================

masks = get_pruning_masks_blockwise(model, norms, block_ratios=block_ratios, default_ratio=default_ratio)
print("✅ Pruning masks generated.\n")


# ============================================================
# --- STEP 4: Rebuild and save pruned model ---
# ============================================================

save_dir = os.path.dirname(SAVE_PATH)
os.makedirs(save_dir, exist_ok=True)

pruned_model = rebuild_pruned_unet(model, masks, save_path=SAVE_PATH)
print("\n✅ Pruned model successfully rebuilt and saved!\n")


# ============================================================
# --- STEP 5: Parameter reduction summary ---
# ============================================================

orig_params = sum(p.numel() for p in model.parameters())
pruned_params = sum(p.numel() for p in pruned_model.parameters())
reduction = 100 * (1 - pruned_params / orig_params)

print(f"📉 Parameter reduction: {reduction:.2f}% ({orig_params/1e6:.2f}M → {pruned_params/1e6:.2f}M)")
print(f"💾 Pruned model saved to: {SAVE_PATH}")
print(f"🧾 Metadata saved to: {SAVE_PATH.replace('.pth', '_meta.json')}\n")

print("🎯 You can now retrain using:")
print("   python train.py  # with subfolder='pruned'")
print("or evaluate using:")
print("   python evaluate.py  # with subfolder='pruned'")
import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  

# --- Project imports ---
from src.models.unet import UNet
from src.training.data_loader import SegmentationDataset
from src.training.metrics import dice_score, iou_score, compute_model_flops, compute_inference_time
from src.pruning.rebuild import (
    build_pruned_unet,
    plot_unet_schematic,
    load_full_pruned_model
)
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb


# ------------------------------------------------------------
# EVALUATION PIPELINE (with VRAM + STD)
# ------------------------------------------------------------
def evaluate(cfg=None, debug=False):

    # ============================================================
    # --- LOAD CONFIG & PATHS ---
    # ============================================================
    if cfg is None:
        cfg, config_path = load_config(return_path=True)
    else:
        config_path = None

    paths = get_paths(cfg, config_path)

    exp_cfg  = cfg["experiment"]
    eval_cfg = cfg["evaluation"]

    phase = eval_cfg["phase"].lower()
    valid_phases = [
        "baseline_evaluation",
        "pruned_evaluation",
        "retrained_pruned_evaluation",
    ]
    if phase not in valid_phases:
        raise ValueError(f"❌ Invalid phase '{phase}'. Must be: {', '.join(valid_phases)}")

    print(f"🔍 Starting evaluation: {phase} for {exp_cfg['experiment_name']}")
    print(paths)

    # ============================================================
    # --- INIT WANDB ---
    # ============================================================
    run = setup_wandb(cfg, job_type=phase)

    # ============================================================
    # --- CONFIG PARAMETERS ---
    # ============================================================
    device      = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    num_slices  = eval_cfg["num_slices_per_volume"]
    batch_size  = eval_cfg["batch_size"]
    num_visuals = eval_cfg["num_visuals"]

    in_ch  = cfg["train"]["model"]["in_channels"]
    out_ch = cfg["train"]["model"]["out_channels"]

    img_dir  = paths.eval_dir
    lbl_dir  = paths.eval_label_dir
    save_dir = paths.eval_save_dir
    paths.ensure_dir(save_dir)

    # ============================================================
    # --- MODEL LOADING ---
    # ============================================================
    if phase == "baseline_evaluation":
        print("🧠 Loading BASELINE model...")

        enc = cfg["train"]["model"]["features"]
        model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc).to(device)

        model_ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"
        print(f"📂 Loading baseline checkpoint: {model_ckpt}")

        if not model_ckpt.exists():
            raise FileNotFoundError(f"❌ Baseline checkpoint not found: {model_ckpt}")

        state = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state)

    else:
        meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"❌ Meta file missing: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        enc_features = meta["enc_features"]
        dec_features = meta["dec_features"]
        bottleneck   = meta["bottleneck_out"]

        if phase == "pruned_evaluation":
            print("🧠 Loading PRUNED (not retrained) model...")
            model_ckpt = paths.pruned_model
        else:
            print("🧠 Loading RETRAINED pruned model...")
            model_ckpt = paths.retrain_pruned_dir / "final_model.pth"

        model = load_full_pruned_model(
            meta      = meta,
            ckpt_path = model_ckpt,
            in_ch     = in_ch,
            out_ch    = out_ch,
            device    = device
        )

    # Ensure checkpoint exists
    if not model_ckpt.exists():
        raise FileNotFoundError(f"❌ Checkpoint not found: {model_ckpt}")

    print(f"📂 Loaded checkpoint: {model_ckpt}")
    model.eval()

    # ============================================================
    # --- PROFILE MODEL ---
    # ============================================================
    print("\n📊 Profiling model (Params, FLOPs, Inference)...")
    input_shape = (1, in_ch, 256, 256)

    with torch.no_grad():
        flops, params = compute_model_flops(model, input_shape)
        infer_ms = compute_inference_time(model, input_shape)

    print(f"🧮 Params: {params/1e6:.2f}M")
    print(f"⚙️ FLOPs:  {flops/1e9:.2f} GFLOPs")
    print(f"⚡ Inference time: {infer_ms:.2f} ms")

    # ============================================================
    # --- LOAD TEST DATASET ---
    # ============================================================
    test_dataset = SegmentationDataset(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        augment=False,
        num_slices_per_volume=num_slices
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Loaded {len(test_loader)} evaluation batches.")

    visual_indices = random.sample(
        range(len(test_loader)),
        k=min(num_visuals, len(test_loader))
    )

    # ============================================================
    # --- DEBUG MODEL STRUCTURE ---
    # ============================================================
    if debug:
        print("\n🔎 Conv layer shapes:")
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                print(f"{name:30s} {tuple(layer.weight.shape)}")

        try:
            plot_unet_schematic(
                enc_features if phase != "baseline_evaluation" else cfg["train"]["model"]["features"],
                dec_features if phase != "baseline_evaluation" else cfg["train"]["model"]["features"][::-1],
                bottleneck   if phase != "baseline_evaluation" else cfg["train"]["model"]["features"][-1] * 2,
                in_ch, out_ch,
                title=f"{phase} U-Net"
            )
        except Exception as e:
            print(f"(⚠️ Could not plot schematic: {e})")

    # ============================================================
    # --- METRIC STORAGE ---
    # ============================================================
    num_classes = out_ch

    class_dice_values = [[] for _ in range(num_classes)]
    class_iou_values  = [[] for _ in range(num_classes)]

    fg_dice_values = []
    fg_iou_values  = []

    all_dice_values = []
    all_iou_values  = []

    vis_dir = os.path.join(save_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    # Reset VRAM peak counter
    torch.cuda.reset_peak_memory_stats(device)

    # ============================================================
    # --- EVALUATION LOOP ---
    # ============================================================
    print("🚀 Running evaluation...")

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):

            imgs  = imgs.to(device)
            masks = masks.to(device, dtype=torch.long)

            preds = model(imgs)

            # Per-class scores
            dice_list = dice_score(preds, masks, num_classes=num_classes, per_class=True)
            iou_list  = iou_score(preds, masks, num_classes=num_classes, per_class=True)

            for c in range(num_classes):
                class_dice_values[c].append(dice_list[c])
                class_iou_values[c].append(iou_list[c])

            # Foreground only
            fg_ids = list(range(1, num_classes))
            fg_dice_values.append(np.mean([dice_list[c] for c in fg_ids]))
            fg_iou_values.append(np.mean([iou_list[c] for c in fg_ids]))

            # Overall
            all_dice_values.append(np.mean(dice_list))
            all_iou_values.append(np.mean(iou_list))

            # Visual logging
            if idx in visual_indices:
                img_path = save_visual(imgs[0], masks[0], preds[0], vis_dir, idx)
                wandb.log({"sample_prediction": wandb.Image(img_path)})

    # ============================================================
    # --- COMPUTE AVERAGES + STANDARD DEVIATIONS ---
    # ============================================================
    def mean_std(arr):
        arr = np.array(arr)
        return float(arr.mean()), float(arr.std())

    avg_dice_all, std_dice_all = mean_std(all_dice_values)
    avg_iou_all,  std_iou_all  = mean_std(all_iou_values)

    avg_dice_fg, std_dice_fg = mean_std(fg_dice_values)
    avg_iou_fg,  std_iou_fg  = mean_std(fg_iou_values)

    class_names = ["Background", "RV", "Myocardium", "LV"]

    avg_class_dice = []
    std_class_dice = []
    avg_class_iou  = []
    std_class_iou  = []

    for c in range(num_classes):
        m, s = mean_std(class_dice_values[c])
        avg_class_dice.append(m)
        std_class_dice.append(s)

        m, s = mean_std(class_iou_values[c])
        avg_class_iou.append(m)
        std_class_iou.append(s)

    # ============================================================
    # --- VRAM USAGE (full evaluation peak) ---
    # ============================================================
    vram_peak = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    wandb.log({"vram_eval_peak_mb": vram_peak})

    # ============================================================
    # --- PRINT RESULTS ---
    # ============================================================
    print("\n✅ Evaluation complete!")
    print(f"📊 Mean Dice (foreground): {avg_dice_fg:.4f} ± {std_dice_fg:.4f}")
    print(f"📊 Mean IoU  (foreground): {avg_iou_fg:.4f} ± {std_iou_fg:.4f}")
    print(f"💾 VRAM peak during evaluation: {vram_peak:.1f} MB")

    for name, d, s in zip(class_names, avg_class_dice, std_class_dice):
        print(f"{name:12s} Dice={d:.4f} ± {s:.4f}")

    # ============================================================
    # --- SAVE JSON METRICS ---
    # ============================================================
    metrics = {
        "phase": phase,
        "params_m": params / 1e6,
        "flops_g": flops / 1e9,
        "inference_ms": infer_ms,

        "vram_peak_mb": vram_peak,

        "mean_dice_fg": avg_dice_fg,
        "std_dice_fg":  std_dice_fg,

        "mean_iou_fg": avg_iou_fg,
        "std_iou_fg":  std_iou_fg,

        "per_class": {
            name: {
                "dice_mean": float(dm),
                "dice_std":  float(ds),
                "iou_mean":  float(im),
                "iou_std":   float(is_),
            }
            for name, dm, ds, im, is_ in zip(class_names, avg_class_dice, std_class_dice, avg_class_iou, std_class_iou)
        }
    }

    metrics_path = os.path.join(save_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # save_pruning_summary(cfg, metrics, save_dir) # data for comoparison plots

    wandb.log({
        "mean_dice_fg": avg_dice_fg,
        "std_dice_fg":  std_dice_fg,
        "mean_iou_fg":  avg_iou_fg,
        "std_iou_fg":   std_iou_fg,
        "vram_eval_peak_mb": vram_peak,
        **{f"class_dice_mean/{c}": m for c, m in zip(class_names, avg_class_dice)},
        **{f"class_dice_std/{c}":  s for c, s in zip(class_names, std_class_dice)},
        **{f"class_iou_mean/{c}":  m for c, m in zip(class_names, avg_class_iou)},
        **{f"class_iou_std/{c}":   s for c, s in zip(class_names, std_class_iou)},
        "params_m": params / 1e6,
        "flops_g": flops / 1e9,
        "inference_ms": infer_ms,
    })

    wandb.save(metrics_path)
    print(f"💾 Metrics saved to: {metrics_path}")
    print(f"🖼️ Visualizations saved to: {vis_dir}")

    run.finish()


# ------------------------------------------------------------
# Visualization Helper
# ------------------------------------------------------------
def save_visual(img, mask, pred, save_dir, idx):
    """Save Input / Ground Truth / Prediction triplet."""
    img  = img.cpu().squeeze().numpy()
    mask = mask.cpu().numpy()
    pred = torch.argmax(pred, dim=0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("Input")
    axs[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=3); axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap="nipy_spectral", vmin=0, vmax=3); axs[2].set_title("Prediction")
    for a in axs: a.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, f"sample_{idx}.png")
    plt.savefig(path)
    plt.close(fig)
    return path

# def save_pruning_summary(cfg, metrics, save_dir):
#     """
#     Saves a lightweight JSON file containing:
#     - layer name pruned
#     - pruning ratio
#     - dice score
#     - phase (pruned or retrained)
#     """

#     # Extract layer + ratio from config
#     prune_dict = cfg["pruning"]["ratios"]["block_ratios"]

#     # The layer being pruned is the one that is > 0
#     pruned_layers = [(k, v) for k, v in prune_dict.items() if v > 0]
#     if len(pruned_layers) == 0:
#         layer_name  = "none"
#         prune_ratio = 0.0
#     else:
#         layer_name, prune_ratio = pruned_layers[0]   # Only one layer varies in your sweep

#     # Dice score for your plots (foreground dice)
#     dice = metrics["mean_dice_fg"]

#     # Phase determines filename
#     phase = metrics["phase"]

#     if phase == "pruned_evaluation":
#         filename = "summary_pruned.json"
#     elif phase == "retrained_pruned_evaluation":
#         filename = "summary_retrained.json"
#     else:
#         filename = "summary_baseline.json"

#     summary = {
#         "phase": phase,
#         "layer": layer_name,
#         "ratio": prune_ratio,
#         "dice_fg": float(dice),
#         "dice_all": float(metrics["per_class"]["LV"]["dice_mean"]),  # example if needed
#         "params_m": metrics["params_m"],
#         "flops_g": metrics["flops_g"],
#         "inference_ms": metrics["inference_ms"],
#     }

#     save_path = os.path.join(save_dir, filename)

#     with open(save_path, "w") as f:
#         json.dump(summary, f, indent=4)

#     print(f"💾 Saved pruning summary → {save_path}")


if __name__ == "__main__":
    evaluate(debug=True)

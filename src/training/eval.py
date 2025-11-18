# import os
# import json
# import random
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import wandb  

# # --- Project imports ---
# from src.models.unet import UNet
# from src.training.data_loader import SegmentationDataset
# from src.training.metrics import dice_score, iou_score, compute_model_flops, compute_inference_time
# from src.pruning.rebuild import build_pruned_unet, plot_unet_schematic, load_full_pruned_model
# from src.utils.config import load_config
# from src.utils.paths import get_paths
# from src.utils.wandb_utils import setup_wandb


# # ------------------------------------------------------------
# # EVALUATION PIPELINE
# # ------------------------------------------------------------
# def evaluate(cfg=None, debug=False):
#     # ============================================================
#     # --- LOAD CONFIG & PATHS ---
#     # ============================================================
#     if cfg is None:
#         cfg, config_path = load_config(return_path=True)
#     else:
#         config_path = None

#     paths = get_paths(cfg, config_path)

#     exp_cfg = cfg["experiment"]
#     eval_cfg = cfg["evaluation"]

#     phase = eval_cfg["phase"].lower()
#     valid_phases = ["baseline_evaluation", "pruned_evaluation", "retrained_pruned_evaluation"]
#     if phase not in valid_phases:
#         raise ValueError(f"❌ Invalid phase '{phase}'. Must be one of: {', '.join(valid_phases)}")

#     print(f"🔍 Starting evaluation phase: {phase} for {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
#     print(paths)

#     # ============================================================
#     # --- WANDB INITIALIZATION ---
#     # ============================================================
#     run = setup_wandb(cfg, job_type="evaluation")

#     # ============================================================
#     # --- CONFIG PARAMETERS ---
#     # ============================================================
#     device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     num_slices = eval_cfg["num_slices_per_volume"]
#     batch_size = eval_cfg["batch_size"]
#     num_visuals = eval_cfg["num_visuals"]

#     in_ch = cfg["train"]["model"]["in_channels"]
#     out_ch = cfg["train"]["model"]["out_channels"]

#     # Dataset and save paths
#     img_dir = paths.eval_dir
#     lbl_dir = paths.eval_label_dir
#     save_dir = paths.eval_save_dir
#     paths.ensure_dir(save_dir)

#     # ============================================================
#     # --- MODEL LOADING ---
#     # ============================================================
#     if phase == "baseline_evaluation":
#         print(f"🧠 Loading baseline UNet with features {cfg['train']['model']['features']}")
#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=cfg["train"]["model"]["features"]).to(device)
#         model_ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"

#     elif phase == "pruned_evaluation":
#         meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#         if not meta_path.exists():
#             raise FileNotFoundError(f"❌ Metadata file missing: {meta_path}")
#         with open(meta_path, "r") as f:
#             meta = json.load(f)
#         enc_features, dec_features, bottleneck_out = (
#             meta["enc_features"], meta["dec_features"], meta["bottleneck_out"]
#         )

#         print(f"🧠 Loading pruned model: enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
#         base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
#         model_ckpt = paths.pruned_model

#     elif phase == "retrained_pruned_evaluation":
#         meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#         if not meta_path.exists():
#             raise FileNotFoundError(f"❌ Metadata file missing: {meta_path}")
#         with open(meta_path, "r") as f:
#             meta = json.load(f)
#         enc_features, dec_features, bottleneck_out = (
#             meta["enc_features"], meta["dec_features"], meta["bottleneck_out"]
#         )

#         print(f"🧠 Loading retrained pruned model: enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
#         base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
#         model_ckpt = paths.retrain_pruned_dir / "final_model.pth"

#     # --- Load model weights ---
#     if not model_ckpt.exists():
#         raise FileNotFoundError(f"❌ Model checkpoint not found at {model_ckpt}")

#     print(f"📂 Loading checkpoint: {model_ckpt}")
#     state = torch.load(model_ckpt, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # ============================================================
#     # --- MODEL PROFILING (FLOPs, Params, Inference Time) ---
#     # ============================================================
#     print("\n📊 Profiling model FLOPs, params & inference speed...")

#     # Input shape for profiling (2D UNet on ACDC slices)
#     input_shape = (1, in_ch, 256, 256)

#     with torch.no_grad():
#         flops, params = compute_model_flops(model, input_shape)
#         infer_ms = compute_inference_time(model, input_shape)

#     print(f"🧮 Params: {params/1e6:.2f} M")
#     print(f"⚙️ FLOPs:  {flops/1e9:.2f} GFLOPs")
#     print(f"⚡ Inference time: {infer_ms:.2f} ms")

#     # ============================================================
#     # --- DATASET ---
#     # ============================================================
#     test_dataset = SegmentationDataset(
#         img_dir=img_dir,
#         lbl_dir=lbl_dir,
#         augment=False,
#         num_slices_per_volume=num_slices
#     )
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     print(f"✅ Loaded {len(test_loader)} test batches.")

#     visual_indices = random.sample(range(len(test_loader)), k=min(num_visuals, len(test_loader)))

#     # ============================================================
#     # --- DEBUG INFO ---
#     # ============================================================
#     if debug:
#         print("\n🔎 Model Conv2d layer shapes:")
#         for name, layer in model.named_modules():
#             if isinstance(layer, nn.Conv2d):
#                 print(f"{name:<30s} weight={tuple(layer.weight.shape)}")

#         try:
#             plot_unet_schematic(
#                 enc_features if phase != "baseline_evaluation" else cfg["train"]["model"]["features"],
#                 dec_features if phase != "baseline_evaluation" else cfg["train"]["model"]["features"][::-1],
#                 bottleneck_out if phase != "baseline_evaluation" else cfg["train"]["model"]["features"][-1] * 2,
#                 in_ch, out_ch, title=f"{phase.replace('_', ' ').title()} U-Net"
#             )
#         except Exception as e:
#             print(f"(⚠️ Could not plot schematic: {e})")

#     # ============================================================
#     # --- METRIC INITIALIZATION ---
#     # ============================================================
#     num_classes = out_ch
#     class_dice = [0.0] * num_classes
#     class_iou = [0.0] * num_classes
#     total_dice, total_iou = 0.0, 0.0
#     total_dice_fg, total_iou_fg = 0.0, 0.0
#     num_samples = 0

#     vis_dir = os.path.join(save_dir, "predictions")
#     os.makedirs(vis_dir, exist_ok=True)

#     # ============================================================
#     # --- EVALUATION LOOP ---
#     # ============================================================
#     print("🚀 Running evaluation...")
#     with torch.no_grad():
#         for i, (imgs, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
#             imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
#             preds = model(imgs)

#             dice_list = dice_score(preds, masks, num_classes=num_classes, per_class=True)
#             iou_list  = iou_score(preds, masks, num_classes=num_classes, per_class=True)

#             # Foreground-only (exclude background class 0)
#             foreground_classes = list(range(1, num_classes))
#             total_dice_fg += sum(dice_list[c] for c in foreground_classes) / len(foreground_classes)
#             total_iou_fg  += sum(iou_list[c]  for c in foreground_classes) / len(foreground_classes)

#             # Per-class accumulation
#             for c in range(num_classes):
#                 class_dice[c] += dice_list[c]
#                 class_iou[c]  += iou_list[c]

#             # Mean including background
#             total_dice += sum(dice_list) / num_classes
#             total_iou  += sum(iou_list)  / num_classes
#             num_samples += 1

#             # Optional visualization
#             if i in visual_indices:
#                 img_path = save_visual(imgs[0], masks[0], preds[0], vis_dir, i)
#                 wandb.log({"sample_prediction": wandb.Image(img_path)})

#     # ============================================================
#     # --- METRICS OUTPUT ---
#     # ============================================================
#     avg_dice_all = total_dice / num_samples
#     avg_iou_all  = total_iou / num_samples
#     avg_dice_fg  = total_dice_fg / num_samples
#     avg_iou_fg   = total_iou_fg / num_samples

#     avg_class_dice = [d / num_samples for d in class_dice]
#     avg_class_iou  = [i / num_samples for i in class_iou]
#     class_names = ["Background", "RV", "Myocardium", "LV"]

#     print("\n✅ Evaluation complete!")
#     # print(f"📊 Mean Dice (all classes): {avg_dice_all:.4f}")
#     # print(f"📊 Mean IoU  (all classes): {avg_iou_all:.4f}")
#     print(f"📊 Mean Dice (foreground):  {avg_dice_fg:.4f}")
#     print(f"📊 Mean IoU  (foreground):  {avg_iou_fg:.4f}\n")

#     for name, d, i in zip(class_names, avg_class_dice, avg_class_iou):
#         print(f"{name:12s} Dice={d:.4f}  IoU={i:.4f}")

#     metrics = {
#         "phase": phase,
#         "params_m": params / 1e6,
#         "flops_g": flops / 1e9,
#         "inference_ms": infer_ms,
#         "mean_dice_all": float(avg_dice_all),
#         "mean_iou_all":  float(avg_iou_all),
#         "mean_dice_fg":  float(avg_dice_fg),
#         "mean_iou_fg":   float(avg_iou_fg),
#         "per_class": {
#             name: {"dice": float(d), "iou": float(i)}
#             for name, d, i in zip(class_names, avg_class_dice, avg_class_iou)
#         }
#     }

#     metrics_path = os.path.join(save_dir, "eval_metrics.json")
#     with open(metrics_path, "w") as f:
#         json.dump(metrics, f, indent=4)

#     # ✅ Log metrics to W&B
#     wandb.log({
#         # "mean_dice_all": avg_dice_all,
#         # "mean_iou_all":  avg_iou_all,
#         "mean_dice_fg":  avg_dice_fg,
#         "mean_iou_fg":   avg_iou_fg,
#         **{f"dice_{name}": d for name, d in zip(class_names, avg_class_dice)},
#         **{f"iou_{name}": i for name, i in zip(class_names, avg_class_iou)},
#         "params_m": params / 1e6,
#         "flops_g": flops / 1e9,
#         "inference_ms": infer_ms,
#     })
#     wandb.save(metrics_path)

#     print(f"\n💾 Metrics saved to {metrics_path}")
#     print(f"🖼️ Visualizations saved to {vis_dir}")

#     run.finish()


# # ------------------------------------------------------------
# # Visualization Helper
# # ------------------------------------------------------------
# def save_visual(img, mask, pred, save_dir, idx):
#     """Save Input / Ground Truth / Prediction triplet."""
#     img = img.cpu().squeeze().numpy()
#     mask = mask.cpu().numpy()
#     pred = torch.argmax(pred, dim=0).cpu().numpy()

#     fig, axs = plt.subplots(1, 3, figsize=(10, 3))
#     axs[0].imshow(img, cmap="gray")
#     axs[0].set_title("Input")
#     axs[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=3)
#     axs[1].set_title("Ground Truth")
#     axs[2].imshow(pred, cmap="nipy_spectral", vmin=0, vmax=3)
#     axs[2].set_title("Prediction")
#     for a in axs:
#         a.axis("off")
#     plt.tight_layout()
#     path = os.path.join(save_dir, f"sample_{idx}.png")
#     plt.savefig(path)
#     plt.close(fig)
#     return path


# if __name__ == "__main__":
#     evaluate(debug=True)



import os
import json
import random
import torch
import torch.nn as nn
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
# EVALUATION PIPELINE
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
    valid_phases = ["baseline_evaluation", "pruned_evaluation", "retrained_pruned_evaluation"]
    if phase not in valid_phases:
        raise ValueError(f"❌ Invalid phase '{phase}'. Must be one of: {', '.join(valid_phases)}")

    print(f"🔍 Starting evaluation: {phase} for {exp_cfg['experiment_name']}")
    print(paths)

    # ============================================================
    # --- INIT WANDB ---
    # ============================================================
    run = setup_wandb(cfg, job_type="evaluation")

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
    # --- UNIFIED MODEL LOADING ---
    # ============================================================
    if phase == "baseline_evaluation":

        print("🧠 Loading BASELINE model...")
        enc = cfg["train"]["model"]["features"]
        model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc).to(device)

        # Baseline checkpoint
        model_ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"

    else:
        # Shared logic: pruned model metadata
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

        else:  # retrained_pruned_evaluation
            print("🧠 Loading RETRAINED pruned model...")
            model_ckpt = paths.retrain_pruned_dir / "final_model.pth"

        # Load full pruned architecture + weights
        model = load_full_pruned_model(
            meta      = meta,
            ckpt_path = model_ckpt,
            in_ch     = in_ch,
            out_ch    = out_ch,
            device    = device
        )

    # --- Validate checkpoint exists ---
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
    # --- DEBUG: PRINT MODEL STRUCTURE ---
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
    num_classes     = out_ch
    class_dice      = [0.0] * num_classes
    class_iou       = [0.0] * num_classes
    total_dice      = 0.0
    total_iou       = 0.0
    total_fg_dice   = 0.0
    total_fg_iou    = 0.0
    num_samples     = 0

    vis_dir = os.path.join(save_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    # ============================================================
    # --- EVALUATION LOOP ---
    # ============================================================
    print("🚀 Running evaluation...")

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):

            imgs  = imgs.to(device)
            masks = masks.to(device, dtype=torch.long)

            preds = model(imgs)

            dice_list = dice_score(preds, masks, num_classes=num_classes, per_class=True)
            iou_list  = iou_score(preds, masks, num_classes=num_classes, per_class=True)

            # Foreground-only metrics
            fg = list(range(1, num_classes))
            total_fg_dice += sum(dice_list[c] for c in fg) / len(fg)
            total_fg_iou  += sum(iou_list[c]  for c in fg) / len(fg)

            # Per-class
            for c in range(num_classes):
                class_dice[c] += dice_list[c]
                class_iou[c]  += iou_list[c]

            total_dice += sum(dice_list) / num_classes
            total_iou  += sum(iou_list)  / num_classes
            num_samples += 1

            # Visualizations
            if idx in visual_indices:
                img_path = save_visual(imgs[0], masks[0], preds[0], vis_dir, idx)
                wandb.log({"sample_prediction": wandb.Image(img_path)})

    # ============================================================
    # --- METRIC OUTPUT ---
    # ============================================================
    avg_dice_all = total_dice / num_samples
    avg_iou_all  = total_iou  / num_samples
    avg_dice_fg  = total_fg_dice / num_samples
    avg_iou_fg   = total_fg_iou  / num_samples

    avg_class_dice = [d / num_samples for d in class_dice]
    avg_class_iou  = [i / num_samples for i in class_iou]
    class_names    = ["Background", "RV", "Myocardium", "LV"]

    print("\n✅ Evaluation complete!")
    print(f"📊 Mean Dice (foreground): {avg_dice_fg:.4f}")
    print(f"📊 Mean IoU  (foreground): {avg_iou_fg:.4f}")

    for name, d, i in zip(class_names, avg_class_dice, avg_class_iou):
        print(f"{name:12s} Dice={d:.4f}  IoU={i:.4f}")

    # Save JSON
    metrics = {
        "phase": phase,
        "params_m": params / 1e6,
        "flops_g": flops / 1e9,
        "inference_ms": infer_ms,
        "mean_dice_fg": float(avg_dice_fg),
        "mean_iou_fg":  float(avg_iou_fg),
        "per_class": {
            name: {"dice": float(d), "iou": float(i)}
            for name, d, i in zip(class_names, avg_class_dice, avg_class_iou)
        }
    }

    metrics_path = os.path.join(save_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    wandb.log({
        "mean_dice_fg": avg_dice_fg,
        "mean_iou_fg": avg_iou_fg,
        **{f"dice_{c}": d for c, d in zip(class_names, avg_class_dice)},
        **{f"iou_{c}":  i for c, i in zip(class_names, avg_class_iou)},
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


if __name__ == "__main__":
    evaluate(debug=True)

# import os
# import json
# import random
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from datetime import datetime
# import matplotlib.pyplot as plt

# from src.models.unet import UNet
# from src.training.data_loader import SegmentationDataset
# from src.training.metrics import dice_score, iou_score
# from src.pruning.rebuild import build_pruned_unet, plot_unet_schematic


# # ------------------------------------------------------------
# # Helper: Create folder hierarchy like results/.../baseline/evaluation
# # ------------------------------------------------------------
# def make_save_dirs(model_name, save_root="results", run_name=None, subfolder=None, phase=None):
#     """
#     Create folder structure like:
#     results/UNet_ACDC/<timestamp>/<subfolder>/<phase>/
#     Example: results/UNet_ACDC/2025-10-27_11-08/baseline/evaluation/
#     """
#     model_root = os.path.join(save_root, model_name)
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
#     run_name = run_name or timestamp

#     path_parts = [model_root, run_name]
#     if subfolder:
#         path_parts.append(subfolder)
#     if phase:
#         path_parts.append(phase)

#     save_dir = os.path.join(*path_parts)
#     os.makedirs(save_dir, exist_ok=True)
#     return save_dir


# def evaluate(debug=True):
#     # ============================================================
#     # --- USER CONFIGURATION SECTION ---
#     # ============================================================

#     model_checkpoint = "/media/ttoxopeus/basic_UNet/results/UNet_ACDC/exp4_larger_UNet_all_slices/pruned/retraining/final_model.pth"
#     save_root = "results"
#     model_name = "UNet_ACDC"
#     run_name = "exp4_larger_UNet_all_slices"
#     subfolder = "pruned"
#     phase = "retrained_evaluation"

#     img_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTs"
#     lbl_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTs"
#     num_slices_per_volume = 4
#     batch_size = 1


#     if subfolder == "pruned":
#         # meta_path = model_checkpoint.replace(".pth", "_meta.json")
#         meta_path = "/media/ttoxopeus/basic_UNet/results/UNet_ACDC/exp4_larger_UNet_all_slices/pruned/0_0_10_20_30_30_30_20_10_0_0/pruned_model_meta.json"
#         with open(meta_path, "r") as f:
#             meta = json.load(f)
#         enc_features = meta["enc_features"]
#         dec_features = meta["dec_features"]
#         bottleneck_out = meta["bottleneck_out"]
#     else:
#         enc_features = [64, 128, 256, 512, 1024] # Need to make this dynamic as well
#         dec_features = enc_features[::-1]
#         bottleneck_out = 2048

#     in_ch = 1
#     out_ch = 4

#     num_visuals = 3
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ============================================================
#     # --- EVALUATION SCRIPT ---
#     # ============================================================

#     save_dir = make_save_dirs(model_name, save_root, run_name, subfolder, phase)
#     print(f"🔍 Evaluating model from: {model_checkpoint}")
#     print(f"📂 Saving results to: {save_dir}")

#     test_dataset = SegmentationDataset(
#         img_dir=img_dir,
#         lbl_dir=lbl_dir,
#         augment=False,
#         num_slices_per_volume=num_slices_per_volume
#     )
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     print(f"✅ Loaded {len(test_loader)} test batches.")

#     visual_indices = random.sample(range(len(test_loader)), k=min(num_visuals, len(test_loader)))

#     # --- Model ---
#     model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#     if subfolder == "pruned":
#         model = build_pruned_unet(model, enc_features, dec_features, bottleneck_out).to(device)

#     # Need to load a saved pruned model instead of pruning it here

#     state = torch.load(model_checkpoint, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # ============================================================
#     # --- DEBUG INFORMATION ---
#     # ============================================================
#     if debug:
#         print("\n🔎 Model Conv2d layer shapes:")
#         for name, layer in model.named_modules():
#             if isinstance(layer, nn.Conv2d):
#                 print(f"{name:<30s} weight={tuple(layer.weight.shape)}")

#         try:
#             # Optional visualization (comment out if not needed)
#             plot_unet_schematic(enc_features, dec_features, bottleneck_out, in_ch, out_ch,
#                                 title="Pruned U-Net Channel Schematic")
#         except Exception as e:
#             print(f"(⚠️ Could not plot schematic: {e})")

#     # ============================================================
#     # --- METRIC INITIALIZATION ---
#     # ============================================================
#     num_classes = out_ch
#     class_dice = [0.0] * num_classes
#     class_iou = [0.0] * num_classes
#     total_dice, total_iou = 0.0, 0.0
#     num_samples = 0
#     vis_save_dir = os.path.join(save_dir, "predictions")
#     os.makedirs(vis_save_dir, exist_ok=True)

#     # ============================================================
#     # --- EVALUATION LOOP ---
#     # ============================================================
#     print("🚀 Running evaluation on test set...")

#     with torch.no_grad():
#         for i, (imgs, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
#             imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)

#             try:
#                 preds = model(imgs)

#             except RuntimeError as e:
#                 print("\n❌ Forward pass failed!")
#                 print(f"Batch idx: {i}")
#                 print(f"Error message: {e}")
#                 print(f"Input shape: {imgs.shape}")

#                 # --- register temporary forward hooks for debugging ---
#                 print("\n🔍 Layer-by-layer channel trace:")
#                 hooks = []
#                 def hook_fn(layer, inp, out, name):
#                     if isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
#                         print(f"{name:<30s} | in={inp[0].shape[1]:>4} → out={out.shape[1]:>4}")

#                 for name, layer in model.named_modules():
#                     if isinstance(layer, nn.Conv2d):
#                         hooks.append(layer.register_forward_hook(
#                             lambda m, i, o, n=name: hook_fn(m, i, o, n)
#                         ))

#                 try:
#                     _ = model(imgs)
#                 except Exception as inner_e:
#                     print(f"⚠️ Second forward still failed: {inner_e}")

#                 # Remove hooks
#                 for h in hooks:
#                     h.remove()

#                 raise e  # re-raise for visibility if needed

#             # --- Metrics ---
#             dice_list = dice_score(preds, masks, num_classes=num_classes, per_class=True)
#             iou_list = iou_score(preds, masks, num_classes=num_classes, per_class=True)

#             for c in range(num_classes):
#                 class_dice[c] += dice_list[c]
#                 class_iou[c] += iou_list[c]

#             total_dice += sum(dice_list) / num_classes
#             total_iou += sum(iou_list) / num_classes
#             num_samples += 1

#             if i in visual_indices:
#                 save_prediction_visual(imgs[0], masks[0], preds[0], vis_save_dir, i)

#     # ============================================================
#     # --- METRIC OUTPUT ---
#     # ============================================================
#     avg_dice = total_dice / num_samples
#     avg_iou = total_iou / num_samples
#     avg_class_dice = [d / num_samples for d in class_dice]
#     avg_class_iou = [i / num_samples for i in class_iou]

#     class_names = ["Background", "RV", "Myocardium", "LV"]

#     print("\n✅ Evaluation complete!")
#     print(f"📊 Mean Dice: {avg_dice:.4f}")
#     print(f"📊 Mean IoU:  {avg_iou:.4f}")
#     print("───────────────────────────────")
#     for name, d, i in zip(class_names, avg_class_dice, avg_class_iou):
#         print(f"{name:12s}  Dice={d:.4f}  IoU={i:.4f}")
#     print("───────────────────────────────")

#     metrics = {
#         "mean_dice": float(avg_dice),
#         "mean_iou": float(avg_iou),
#         "per_class": {
#             name: {"dice": float(d), "iou": float(i)}
#             for name, d, i in zip(class_names, avg_class_dice, avg_class_iou)
#         }
#     }

#     with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=4)

#     print(f"💾 Metrics saved to {os.path.join(save_dir, 'eval_metrics.json')}")

# # ------------------------------------------------------------
# # Visualization
# # ------------------------------------------------------------
# def save_prediction_visual(img, mask, pred, save_dir, idx):
#     """Save a 3-panel comparison (Input / GT / Prediction)."""
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
#     plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
#     plt.close(fig)


# if __name__ == "__main__":
#     evaluate()


import os
import json
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet
from src.training.data_loader import SegmentationDataset
from src.training.metrics import dice_score, iou_score
from src.pruning.rebuild import build_pruned_unet, plot_unet_schematic
from src.utils.config import load_config
from src.utils.paths import get_paths


# ------------------------------------------------------------
# EVALUATION PIPELINE
# ------------------------------------------------------------
def evaluate(debug=False):
    # ============================================================
    # --- LOAD CONFIG & PATHS ---
    # ============================================================
    cfg, config_path = load_config(return_path=True)
    paths = get_paths(cfg, config_path)

    exp_cfg = cfg["experiment"]
    eval_cfg = cfg["evaluation"]

    target = eval_cfg.get("target", "baseline").lower()
    print(f"🔍 Evaluating {target.upper()} model for {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
    print(paths)

    # ============================================================
    # --- CONFIG PARAMETERS ---
    # ============================================================
    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    num_slices = eval_cfg["num_slices_per_volume"]
    batch_size = eval_cfg["batch_size"]
    num_visuals = eval_cfg["num_visuals"]

    in_ch = cfg["train"]["model"]["in_channels"]
    out_ch = cfg["train"]["model"]["out_channels"]

    # Dataset and save paths
    img_dir = paths.eval_dir
    lbl_dir = paths.eval_label_dir
    save_dir = paths.eval_save_dir  # e.g., baseline/evaluation, retrain/.../evaluation
    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # --- MODEL LOADING ---
    # ============================================================
    if target == "baseline":
        print(f"🧠 Loading baseline UNet with features {cfg['train']['model']['features']}")
        model = UNet(
            in_ch=in_ch,
            out_ch=out_ch,
            enc_features=cfg["train"]["model"]["features"]
        ).to(device)
        # model checkpoint now inside baseline/training/
        model_ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"

    elif target == "pruned":
        meta_path = paths.pruned_model.with_name(paths.pruned_model.name.replace(".pth", "_meta.json"))
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected pruning metadata file at {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        enc_features = meta["enc_features"]
        dec_features = meta["dec_features"]
        bottleneck_out = meta["bottleneck_out"]

        print(f"🧠 Loading pruned model: enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
        base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
        model_ckpt = paths.pruned_model

    elif target == "retrain":
        meta_path = paths.pruned_model.with_name(paths.pruned_model.name.replace(".pth", "_meta.json"))
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected pruning metadata file at {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        enc_features = meta["enc_features"]
        dec_features = meta["dec_features"]
        bottleneck_out = meta["bottleneck_out"]

        print(f"🧠 Loading retrained model (enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out})")
        base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
        # retrained model is stored under retrain/.../training/
        model_ckpt = paths.train_save_dir / "final_model.pth"

    else:
        raise ValueError(f"❌ Unknown evaluation target: {target}")

    # Load model weights
    if not model_ckpt.exists():
        raise FileNotFoundError(f"❌ Model checkpoint not found at {model_ckpt}")
    print(f"📂 Loading checkpoint: {model_ckpt}")
    state = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ============================================================
    # --- DATASET ---
    # ============================================================
    test_dataset = SegmentationDataset(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        augment=False,
        num_slices_per_volume=num_slices
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"✅ Loaded {len(test_loader)} test batches.")

    visual_indices = random.sample(range(len(test_loader)), k=min(num_visuals, len(test_loader)))

    # ============================================================
    # --- DEBUG INFO ---
    # ============================================================
    if debug:
        print("\n🔎 Model Conv2d layer shapes:")
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                print(f"{name:<30s} weight={tuple(layer.weight.shape)}")

        try:
            plot_unet_schematic(
                enc_features if target != "baseline" else cfg["train"]["model"]["features"],
                dec_features if target != "baseline" else cfg["train"]["model"]["features"][::-1],
                bottleneck_out if target != "baseline" else cfg["train"]["model"]["features"][-1]*2,
                in_ch, out_ch, title=f"{target.capitalize()} U-Net Architecture"
            )
        except Exception as e:
            print(f"(⚠️ Could not plot schematic: {e})")

    # ============================================================
    # --- METRIC INITIALIZATION ---
    # ============================================================
    num_classes = out_ch
    class_dice = [0.0] * num_classes
    class_iou = [0.0] * num_classes
    total_dice, total_iou = 0.0, 0.0
    num_samples = 0

    vis_dir = os.path.join(save_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    # ============================================================
    # --- EVALUATION LOOP ---
    # ============================================================
    print("🚀 Running evaluation...")
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
            preds = model(imgs)

            dice_list = dice_score(preds, masks, num_classes=num_classes, per_class=True)
            iou_list = iou_score(preds, masks, num_classes=num_classes, per_class=True)

            for c in range(num_classes):
                class_dice[c] += dice_list[c]
                class_iou[c] += iou_list[c]

            total_dice += sum(dice_list) / num_classes
            total_iou += sum(iou_list) / num_classes
            num_samples += 1

            if i in visual_indices:
                save_visual(imgs[0], masks[0], preds[0], vis_dir, i)

    # ============================================================
    # --- METRICS OUTPUT ---
    # ============================================================
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_class_dice = [d / num_samples for d in class_dice]
    avg_class_iou = [i / num_samples for i in class_iou]
    class_names = ["Background", "RV", "Myocardium", "LV"]

    print("\n✅ Evaluation complete!")
    print(f"📊 Mean Dice: {avg_dice:.4f}")
    print(f"📊 Mean IoU:  {avg_iou:.4f}")
    for name, d, i in zip(class_names, avg_class_dice, avg_class_iou):
        print(f"{name:12s} Dice={d:.4f}  IoU={i:.4f}")

    metrics = {
        "target": target,
        "mean_dice": float(avg_dice),
        "mean_iou": float(avg_iou),
        "per_class": {
            name: {"dice": float(d), "iou": float(i)}
            for name, d, i in zip(class_names, avg_class_dice, avg_class_iou)
        }
    }

    metrics_path = os.path.join(save_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Metrics saved to {metrics_path}")
    print(f"🖼️ Sample visualizations saved to {vis_dir}")


# ------------------------------------------------------------
# Visualization Helper
# ------------------------------------------------------------
def save_visual(img, mask, pred, save_dir, idx):
    """Save Input / Ground Truth / Prediction triplet."""
    img = img.cpu().squeeze().numpy()
    mask = mask.cpu().numpy()
    pred = torch.argmax(pred, dim=0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Input")
    axs[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=3)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap="nipy_spectral", vmin=0, vmax=3)
    axs[2].set_title("Prediction")
    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
    plt.close(fig)


if __name__ == "__main__":
    evaluate(debug=True)


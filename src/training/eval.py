import os
import json
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.unet import UNet
from src.training.data_loader import SegmentationDataset
from src.training.metrics import dice_score, iou_score


# ------------------------------------------------------------
# Helper: Create folder hierarchy like results/.../baseline/evaluation
# ------------------------------------------------------------
def make_save_dirs(model_name, save_root="results", run_name=None, subfolder=None, phase=None):
    """
    Create folder structure like:
    results/UNet_ACDC/<timestamp>/<subfolder>/<phase>/
    Example: results/UNet_ACDC/2025-10-27_11-08/baseline/evaluation/
    """
    model_root = os.path.join(save_root, model_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = run_name or timestamp

    path_parts = [model_root, run_name]
    if subfolder:
        path_parts.append(subfolder)
    if phase:
        path_parts.append(phase)

    save_dir = os.path.join(*path_parts)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate():
    # ============================================================
    # --- USER CONFIGURATION SECTION ---
    # ============================================================

    # --- Paths ---
    model_checkpoint = "results/UNet_ACDC/exp1/baseline/training/final_model.pth"
    save_root = "results"
    model_name = "UNet_ACDC"
    run_name = "exp1"  # match your training folder timestamp
    subfolder = "baseline"
    phase = "evaluation"

    # --- Dataset ---
    img_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTs"
    lbl_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTs"
    num_slices_per_volume = 30
    batch_size = 1  # typically 1 for evaluation

    # --- Model configuration ---
    in_ch = 1
    out_ch = 4
    features = [64, 128, 256, 512]

    # --- Evaluation options ---
    num_visuals = 3  # number of random prediction examples to save
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # --- EVALUATION SCRIPT ---
    # ============================================================

    save_dir = make_save_dirs(model_name, save_root, run_name, subfolder, phase)
    print(f"🔍 Evaluating model from: {model_checkpoint}")
    print(f"📂 Saving results to: {save_dir}")

    # --- Dataset ---
    test_dataset = SegmentationDataset(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        augment=False,
        num_slices_per_volume=num_slices_per_volume
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"✅ Loaded {len(test_loader)} test batches.")

    # --- Random slice selection ---
    total_samples = len(test_loader)
    visual_indices = random.sample(range(total_samples), k=min(num_visuals, total_samples))

    # --- Model ---
    model = UNet(in_ch=in_ch, out_ch=out_ch, features=features).to(device)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Metrics accumulators ---
    num_classes = out_ch
    class_dice = [0.0] * num_classes
    class_iou = [0.0] * num_classes
    total_dice, total_iou = 0.0, 0.0
    num_samples = 0

    vis_save_dir = os.path.join(save_dir, "predictions")
    os.makedirs(vis_save_dir, exist_ok=True)

    print("🚀 Running evaluation on test set...")
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
            preds = model(imgs)

            # Per-class metrics
            dice_list = dice_score(preds, masks, num_classes=num_classes, per_class=True)
            iou_list = iou_score(preds, masks, num_classes=num_classes, per_class=True)

            for c in range(num_classes):
                class_dice[c] += dice_list[c]
                class_iou[c] += iou_list[c]

            total_dice += sum(dice_list) / num_classes
            total_iou += sum(iou_list) / num_classes
            num_samples += 1

            # Save visualization for random slices
            if i in visual_indices:
                save_prediction_visual(imgs[0], masks[0], preds[0], vis_save_dir, i)

    # --- Compute averages ---
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_class_dice = [d / num_samples for d in class_dice]
    avg_class_iou = [i / num_samples for i in class_iou]

    # --- Display results ---
    class_names = ["Background", "RV", "Myocardium", "LV"]

    print("\n✅ Evaluation complete!")
    print(f"📊 Mean Dice: {avg_dice:.4f}")
    print(f"📊 Mean IoU:  {avg_iou:.4f}")
    print("───────────────────────────────")
    for name, d, i in zip(class_names, avg_class_dice, avg_class_iou):
        print(f"{name:12s}  Dice={d:.4f}  IoU={i:.4f}")
    print("───────────────────────────────")

    # --- Save metrics ---
    metrics = {
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
    print(f"💾 Saved metrics to {metrics_path}")

    # --- Save short summary ---
    summary = {
        "model_name": model_name,
        "run_name": run_name,
        "subfolder": subfolder,
        "phase": phase,
        "checkpoint": os.path.basename(model_checkpoint),
        "mean_dice": float(avg_dice),
        "mean_iou": float(avg_iou),
    }

    summary_path = os.path.join(save_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"🧾 Saved evaluation summary to {summary_path}")



# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def save_prediction_visual(img, mask, pred, save_dir, idx):
    """Save a 3-panel comparison (Input / GT / Prediction)."""
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
    evaluate()

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb  

# --- Project imports ---
from src.models.unet import UNet
from src.training.loss import get_loss_function
from src.training.metrics import dice_score, iou_score, compute_model_flops, compute_inference_time
from src.training.data_loader import get_train_val_loaders, summarize_torchio_pipeline
from src.pruning.rebuild import build_pruned_unet, load_full_pruned_model
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb


# ------------------------------------------------------------
# TRAINING PIPELINE (with STD + VRAM tracking)
# ------------------------------------------------------------
def train_model(cfg=None):

    # ============================================================
    # --- LOAD CONFIGURATION ---
    # ============================================================
    if cfg is None:
        cfg, config_path = load_config(return_path=True)
    else:
        config_path = None

    paths = get_paths(cfg, config_path)
    paths.save_config_snapshot()

    exp_cfg   = cfg["experiment"]
    train_cfg = cfg["train"]

    phase = train_cfg["phase"].lower()
    valid_phases = ["training", "retraining"]
    if phase not in valid_phases:
        raise ValueError(f"❌ Invalid training phase '{phase}'. Must be: {valid_phases}")

    print(f"🚀 Starting {phase} for experiment {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")


    # ============================================================
    # --- SETUP HYPERPARAMETERS ---
    # ============================================================
    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    params      = train_cfg["parameters"]
    model_cfg   = train_cfg["model"]
    batch_size  = train_cfg["batch_size"]
    val_ratio   = train_cfg["val_ratio"]
    num_slices  = train_cfg["num_slices_per_volume"]

    lr          = float(params["learning_rate"])
    epochs      = params["num_epochs"]
    save_int    = params["save_interval"]
    loss_name   = params["loss_function"]

    in_ch  = model_cfg["in_channels"]
    out_ch = model_cfg["out_channels"]


    # ============================================================
    # --- INITIALIZE WANDB ---
    # ============================================================
    run = setup_wandb(cfg, job_type=phase)
    wandb.watch_called = False


    # ============================================================
    # --- BUILD MODEL ---
    # ============================================================
    if phase == "retraining":
        meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"❌ Expected pruning metadata not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        print("🧠 Loading FULL pruned model (for retraining)...")

        model = load_full_pruned_model(
            meta          = meta,
            ckpt_path     = paths.pruned_model,
            in_ch         = in_ch,
            out_ch        = out_ch,
            device        = device
        )

    else:
        model = UNet(
            in_ch=in_ch,
            out_ch=out_ch,
            enc_features=model_cfg["features"]
        ).to(device)

    wandb.watch(model, log="all")


    # ============================================================
    # --- DATA & OPTIMIZATION SETUP ---
    # ============================================================
    save_dir = paths.train_save_dir
    paths.ensure_dir(save_dir)

    train_loader, val_loader, augmentation_summary = get_train_val_loaders(
        img_dir=paths.train_dir,
        lbl_dir=paths.label_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        shuffle=True,
        num_slices_per_volume=num_slices
    )

    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    criterion = get_loss_function(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # ============================================================
    # --- METRICS LOGGING STRUCTURE (with STD + VRAM) ---
    # ============================================================
    metrics_log = {
        "epoch": [],
        "train_loss_mean": [],
        "train_loss_std": [],
        "val_dice_mean": [],
        "val_dice_std": [],
        "val_iou_mean": [],
        "val_iou_std": [],
        "lr": [],
        "vram_max": []           # NEW
    }


    # ============================================================
    # --- TRAINING LOOP ---
    # ============================================================
    print(f"\n🚦 Training for {epochs} epochs...\n")

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_losses = []

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):

            imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)

            optimizer.zero_grad()

            preds = model(imgs)
            loss = criterion(preds, masks)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # ------------------------------------
            # VRAM usage tracking (per iteration)
            # ------------------------------------
            current_vram = torch.cuda.memory_allocated(device) / (1024**2)
            max_vram_iter = torch.cuda.max_memory_allocated(device) / (1024**2)

            wandb.log({
                "vram_current_mb": current_vram,
                "vram_max_iter_mb": max_vram_iter
            })

        # Epoch training stats
        train_loss_mean = float(np.mean(train_losses))
        train_loss_std  = float(np.std(train_losses))


        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_dices = []
        val_ious = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
                preds = model(imgs)

                val_dices.append(dice_score(preds, masks, num_classes=out_ch))
                val_ious.append(iou_score(preds, masks, num_classes=out_ch))

        val_dice_mean = float(np.mean(val_dices))
        val_dice_std  = float(np.std(val_dices))
        val_iou_mean  = float(np.mean(val_ious))
        val_iou_std   = float(np.std(val_ious))

        curr_lr = optimizer.param_groups[0]["lr"]

        # ------------------------------------
        # VRAM usage for the full epoch
        # ------------------------------------
        epoch_vram_max = torch.cuda.max_memory_allocated(device) / (1024**2)
        metrics_log["vram_max"].append(epoch_vram_max)

        wandb.log({
            "vram_epoch_max_mb": epoch_vram_max
        })

        torch.cuda.reset_peak_memory_stats(device)

        print(
            f"📈 Epoch {epoch+1:02d}: "
            f"Loss={train_loss_mean:.4f}±{train_loss_std:.4f}, "
            f"Dice={val_dice_mean:.4f}±{val_dice_std:.4f}, "
            f"IoU={val_iou_mean:.4f}±{val_iou_std:.4f}, "
            f"VRAM={epoch_vram_max:.1f}MB, "
            f"LR={curr_lr:.6e}"
        )


        # -------------------------
        # Logging
        # -------------------------
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_mean": train_loss_mean,
            "train_loss_std": train_loss_std,
            "val_dice_mean": val_dice_mean,
            "val_dice_std": val_dice_std,
            "val_iou_mean": val_iou_mean,
            "val_iou_std": val_iou_std,
            "lr": curr_lr,
        })

        metrics_log["epoch"].append(epoch + 1)
        metrics_log["train_loss_mean"].append(train_loss_mean)
        metrics_log["train_loss_std"].append(train_loss_std)
        metrics_log["val_dice_mean"].append(val_dice_mean)
        metrics_log["val_dice_std"].append(val_dice_std)
        metrics_log["val_iou_mean"].append(val_iou_mean)
        metrics_log["val_iou_std"].append(val_iou_std)
        metrics_log["lr"].append(curr_lr)


        # -------------------------
        # Save checkpoint
        # -------------------------
        if (epoch + 1) == save_int:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            #wandb.save(ckpt_path)



    # ============================================================
    # --- SAVE FINAL MODEL ---
    # ============================================================
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Saved final model: {final_model_path}")
    #wandb.save(final_model_path)


    # ============================================================
    # --- MODEL PROFILING ---
    # ============================================================
    model.eval()
    with torch.no_grad():
        flops, params = compute_model_flops(model, (1, in_ch, 256, 256))
        infer_ms     = compute_inference_time(model, (1, in_ch, 256, 256))

    wandb.log({
        "params_m": params / 1e6,
        "flops_g": flops / 1e9,
        "inference_ms": infer_ms,
    })


    # ============================================================
    # --- SAVE TRAINING CURVES (with std & VRAM)
    # ============================================================
    epochs_arr = metrics_log["epoch"]

    plt.figure(figsize=(8, 6))

    # Train loss
    mean = np.array(metrics_log["train_loss_mean"])
    std  = np.array(metrics_log["train_loss_std"])
    plt.plot(epochs_arr, mean, label="Train Loss", color="blue")
    plt.fill_between(epochs_arr, mean-std, mean+std, color="blue", alpha=0.2)

    # Val Dice
    mean = np.array(metrics_log["val_dice_mean"])
    std  = np.array(metrics_log["val_dice_std"])
    plt.plot(epochs_arr, mean, label="Val Dice", color="orange")
    plt.fill_between(epochs_arr, mean-std, mean+std, color="orange", alpha=0.2)

    # Val IoU
    mean = np.array(metrics_log["val_iou_mean"])
    std  = np.array(metrics_log["val_iou_std"])
    plt.plot(epochs_arr, mean, label="Val IoU", color="green")
    plt.fill_between(epochs_arr, mean-std, mean+std, color="green", alpha=0.2)


    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Progress ({phase})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    wandb.log({"training_curve": wandb.Image(plot_path)})


    # ============================================================
    # --- SAVE TRAINING SUMMARY ---
    # ============================================================
    summary = {
        "model_name":  exp_cfg["model_name"],
        "experiment":  exp_cfg["experiment_name"],
        "phase":       phase,
        "epochs":      epochs,
        "learning_rate": lr,
        "batch_size":  batch_size,
        "device":      str(device),
        "params_m":    params / 1e6,
        "flops_g":     flops / 1e9,
        "inference_ms": infer_ms,
        "vram_mb_last_epoch": metrics_log["vram_max"][-1],
        "final_train_loss": float(metrics_log["train_loss_mean"][-1]),
        "final_val_dice":  float(metrics_log["val_dice_mean"][-1]),
        "final_val_iou":   float(metrics_log["val_iou_mean"][-1]),
        "augmentation": {
            "library": "torchio",
            "transforms": augmentation_summary if augmentation_summary else "none"
        }
    }

    with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Training complete.")
    run.finish()



if __name__ == "__main__":
    train_model()

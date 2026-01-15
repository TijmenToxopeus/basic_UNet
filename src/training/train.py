# import os
# import json
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
# import wandb

# # --- Project imports ---
# from src.models.unet import UNet
# from src.training.loss import get_loss_function
# from src.training.loop import train_one_epoch, validate
# from src.training.metrics import dice_score, iou_score, compute_model_flops, compute_inference_time
# from src.training.data_loader import get_train_val_loaders, summarize_torchio_pipeline
# from src.pruning.rebuild import build_pruned_unet, load_full_pruned_model
# from src.utils.config import load_config
# from src.utils.paths import get_paths
# from src.utils.wandb_utils import setup_wandb

# # --- NEW: reproducibility ---
# from src.utils.reproducibility import seed_everything


# # ------------------------------------------------------------
# # TRAINING PIPELINE (with STD + VRAM tracking)
# # ------------------------------------------------------------
# def train_model(cfg=None):

#     # ============================================================
#     # --- LOAD CONFIGURATION ---
#     # ============================================================
#     if cfg is None:
#         cfg, config_path = load_config(return_path=True)
#     else:
#         config_path = None

#     # ============================================================
#     # --- SEEDING (must be before model/dataloader creation) ---
#     # ============================================================
#     exp_cfg = cfg["experiment"]
#     seed = exp_cfg.get("seed", 42)
#     deterministic = exp_cfg.get("deterministic", False)
#     seed_everything(seed, deterministic=deterministic)

#     # After seeding, create paths and snapshot config
#     paths = get_paths(cfg, config_path)
#     paths.save_config_snapshot()

#     train_cfg = cfg["train"]

#     phase = train_cfg["phase"].lower()
#     valid_phases = ["training", "retraining"]
#     if phase not in valid_phases:
#         raise ValueError(f"❌ Invalid training phase '{phase}'. Must be: {valid_phases}")

#     print(f"🚀 Starting {phase} for experiment {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
#     print(f"🔁 Seed = {seed} | Deterministic = {deterministic}")

#     # ============================================================
#     # --- SETUP HYPERPARAMETERS ---
#     # ============================================================
#     device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

#     params      = train_cfg["parameters"]
#     model_cfg   = train_cfg["model"]
#     batch_size  = train_cfg["batch_size"]
#     val_ratio   = train_cfg["val_ratio"]
#     num_slices  = train_cfg["num_slices_per_volume"]

#     lr          = float(params["learning_rate"])
#     epochs      = params["num_epochs"]
#     save_int    = params["save_interval"]
#     loss_name   = params["loss_function"]

#     in_ch  = model_cfg["in_channels"]
#     out_ch = model_cfg["out_channels"]

#     # ============================================================
#     # --- INITIALIZE WANDB ---
#     # ============================================================
#     run = setup_wandb(cfg, job_type=phase)
#     wandb.watch_called = False

#     # ============================================================
#     # --- BUILD MODEL ---
#     # ============================================================
#     if phase == "retraining":
#         meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#         if not meta_path.exists():
#             raise FileNotFoundError(f"❌ Expected pruning metadata not found: {meta_path}")

#         with open(meta_path, "r") as f:
#             meta = json.load(f)

#         print("🧠 Loading FULL pruned model (for retraining)...")

#         model = load_full_pruned_model(
#             meta          = meta,
#             ckpt_path     = paths.pruned_model,
#             in_ch         = in_ch,
#             out_ch        = out_ch,
#             device        = device
#         )
#     else:
#         model = UNet(
#             in_ch=in_ch,
#             out_ch=out_ch,
#             enc_features=model_cfg["features"]
#         ).to(device)

#     wandb.watch(model, log="all")

#     # ============================================================
#     # --- DATA & OPTIMIZATION SETUP ---
#     # ============================================================
#     save_dir = paths.train_save_dir
#     paths.ensure_dir(save_dir)

#     train_loader, val_loader, augmentation_summary = get_train_val_loaders(
#         img_dir=paths.train_dir,
#         lbl_dir=paths.label_dir,
#         batch_size=batch_size,
#         val_ratio=val_ratio,
#         shuffle=True,
#         seed=seed,  # NEW: ensure split/shuffle uses same seed
#         num_slices_per_volume=num_slices
#     )

#     print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

#     criterion = get_loss_function(loss_name)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # ============================================================
#     # --- METRICS LOGGING STRUCTURE (with STD + VRAM) ---
#     # ============================================================
#     metrics_log = {
#         "epoch": [],
#         "train_loss_mean": [],
#         "train_loss_std": [],
#         "val_dice_mean": [],
#         "val_dice_std": [],
#         "val_iou_mean": [],
#         "val_iou_std": [],
#         "lr": [],
#         "vram_max": []
#     }

#     # ============================================================
#     # --- TRAINING LOOP ---
#     # ============================================================
#     print(f"\n🚦 Training for {epochs} epochs...\n")

#     for epoch in range(epochs):
#         train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

#         train_res = train_one_epoch(
#             model=model,
#             loader=train_iter,
#             optimizer=optimizer,
#             criterion=criterion,
#             device=device,
#             epoch=epoch + 1,
#             epochs=epochs,
#             log_fn=wandb.log,
#             log_interval=20,
#             track_vram=True,
#         )

#         train_loss_mean = train_res.loss_mean
#         train_loss_std  = train_res.loss_std
#         epoch_vram_max   = train_res.vram_epoch_max_mb

#         val_res = validate(
#             model=model,
#             loader=val_loader,
#             device=device,
#             out_ch=out_ch,
#             dice_fn=dice_score,
#             iou_fn=iou_score,
#         )

#         val_dice_mean = val_res.dice_mean
#         val_dice_std  = val_res.dice_std
#         val_iou_mean  = val_res.iou_mean
#         val_iou_std   = val_res.iou_std

#         curr_lr = optimizer.param_groups[0]["lr"]

#         # log epoch VRAM peak (from helper)
#         metrics_log["vram_max"].append(epoch_vram_max)
#         if device.type == "cuda":
#             wandb.log({"vram_epoch_max_mb": epoch_vram_max})

#         print(
#             f"📈 Epoch {epoch+1:02d}: "
#             f"Loss={train_loss_mean:.4f}±{train_loss_std:.4f}, "
#             f"Dice={val_dice_mean:.4f}±{val_dice_std:.4f}, "
#             f"IoU={val_iou_mean:.4f}±{val_iou_std:.4f}, "
#             f"VRAM={epoch_vram_max:.1f}MB, "
#             f"LR={curr_lr:.6e}"
#         )

#         wandb.log({
#             "epoch": epoch + 1,
#             "train_loss_mean": train_loss_mean,
#             "train_loss_std": train_loss_std,
#             "val_dice_mean": val_dice_mean,
#             "val_dice_std": val_dice_std,
#             "val_iou_mean": val_iou_mean,
#             "val_iou_std": val_iou_std,
#             "lr": curr_lr,
#         })

#         metrics_log["epoch"].append(epoch + 1)
#         metrics_log["train_loss_mean"].append(train_loss_mean)
#         metrics_log["train_loss_std"].append(train_loss_std)
#         metrics_log["val_dice_mean"].append(val_dice_mean)
#         metrics_log["val_dice_std"].append(val_dice_std)
#         metrics_log["val_iou_mean"].append(val_iou_mean)
#         metrics_log["val_iou_std"].append(val_iou_std)
#         metrics_log["lr"].append(curr_lr)

#         # Save checkpoint every interval
#         if (epoch + 1) % save_int == 0:
#             ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
#             torch.save(model.state_dict(), ckpt_path)

#     # ============================================================
#     # --- SAVE FINAL MODEL ---
#     # ============================================================
#     final_model_path = os.path.join(save_dir, "final_model.pth")
#     torch.save(model.state_dict(), final_model_path)
#     print(f"💾 Saved final model: {final_model_path}")

#     # ============================================================
#     # --- MODEL PROFILING ---
#     # ============================================================
#     model.eval()
#     with torch.no_grad():
#         flops, params = compute_model_flops(model, (1, in_ch, 256, 256))
#         infer_ms     = compute_inference_time(model, (1, in_ch, 256, 256))

#     wandb.log({
#         "params_m": params / 1e6,
#         "flops_g": flops / 1e9,
#         "inference_ms": infer_ms,
#     })

#     # ============================================================
#     # --- SAVE TRAINING CURVES (with std & VRAM) ---
#     # ============================================================
#     epochs_arr = metrics_log["epoch"]
#     plt.figure(figsize=(8, 6))

#     mean = np.array(metrics_log["train_loss_mean"])
#     std  = np.array(metrics_log["train_loss_std"])
#     plt.plot(epochs_arr, mean, label="Train Loss", color="blue")
#     plt.fill_between(epochs_arr, mean-std, mean+std, color="blue", alpha=0.2)

#     mean = np.array(metrics_log["val_dice_mean"])
#     std  = np.array(metrics_log["val_dice_std"])
#     plt.plot(epochs_arr, mean, label="Val Dice", color="orange")
#     plt.fill_between(epochs_arr, mean-std, mean+std, color="orange", alpha=0.2)

#     mean = np.array(metrics_log["val_iou_mean"])
#     std  = np.array(metrics_log["val_iou_std"])
#     plt.plot(epochs_arr, mean, label="Val IoU", color="green")
#     plt.fill_between(epochs_arr, mean-std, mean+std, color="green", alpha=0.2)

#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.title(f"Training Progress ({phase})")
#     plt.legend()
#     plt.grid(True)

#     plot_path = os.path.join(save_dir, "training_curves.png")
#     plt.savefig(plot_path)
#     plt.close()
#     wandb.log({"training_curve": wandb.Image(plot_path)})

#     # ============================================================
#     # --- SAVE TRAINING SUMMARY ---
#     # ============================================================
#     summary = {
#         "model_name":  exp_cfg["model_name"],
#         "experiment":  exp_cfg["experiment_name"],
#         "phase":       phase,
#         "seed":        seed,                 # NEW
#         "deterministic": deterministic,       # NEW
#         "epochs":      epochs,
#         "learning_rate": lr,
#         "batch_size":  batch_size,
#         "device":      str(device),
#         "params_m":    params / 1e6,
#         "flops_g":     flops / 1e9,
#         "inference_ms": infer_ms,
#         "vram_mb_last_epoch": metrics_log["vram_max"][-1],
#         "final_train_loss": float(metrics_log["train_loss_mean"][-1]),
#         "final_val_dice":  float(metrics_log["val_dice_mean"][-1]),
#         "final_val_iou":   float(metrics_log["val_iou_mean"][-1]),
#         "augmentation": {
#             "library": "torchio",
#             "transforms": augmentation_summary if augmentation_summary else "none"
#         }
#     }

#     with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
#         json.dump(summary, f, indent=4)

#     print("✅ Training complete.")
#     run.finish()


# if __name__ == "__main__":
#     train_model()



import os
import json
import torch
import numpy as np
from tqdm import tqdm
import wandb

# --- Project imports ---
from src.models.unet import UNet
from src.training.loss import get_loss_function
from src.training.loop import train_one_epoch, validate
from src.training.metrics import dice_score, iou_score
from src.training.data_loader import get_train_val_loaders
from src.pruning.rebuild import load_full_pruned_model
from src.training.logging import log_epoch
from src.training.artifacts import profile_model, save_training_curves, save_json

from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb
from src.utils.reproducibility import seed_everything


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

    # ============================================================
    # --- SEEDING (must be before model/dataloader creation) ---
    # ============================================================
    exp_cfg = cfg["experiment"]
    seed = exp_cfg.get("seed", 42)
    deterministic = exp_cfg.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    # After seeding, create paths and snapshot config
    paths = get_paths(cfg, config_path)
    paths.save_config_snapshot()

    train_cfg = cfg["train"]

    phase = train_cfg["phase"].lower()
    valid_phases = ["training", "retraining"]
    if phase not in valid_phases:
        raise ValueError(f"❌ Invalid training phase '{phase}'. Must be: {valid_phases}")

    print(f"🚀 Starting {phase} for experiment {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
    print(f"🔁 Seed = {seed} | Deterministic = {deterministic}")

    # ============================================================
    # --- SETUP HYPERPARAMETERS ---
    # ============================================================
    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    params = train_cfg["parameters"]
    model_cfg = train_cfg["model"]
    batch_size = train_cfg["batch_size"]
    val_ratio = train_cfg["val_ratio"]
    num_slices = train_cfg["num_slices_per_volume"]

    lr = float(params["learning_rate"])
    epochs = int(params["num_epochs"])
    save_int = int(params["save_interval"])  # "rewind epoch" checkpoint
    loss_name = params["loss_function"]

    in_ch = int(model_cfg["in_channels"])
    out_ch = int(model_cfg["out_channels"])

    # ============================================================
    # --- INITIALIZE WANDB ---
    # ============================================================
    run = setup_wandb(cfg, job_type=phase)

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
            meta=meta,
            ckpt_path=paths.pruned_model,
            in_ch=in_ch,
            out_ch=out_ch,
            device=device,
        )
    else:
        model = UNet(
            in_ch=in_ch,
            out_ch=out_ch,
            enc_features=model_cfg["features"],
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
        seed=seed,
        num_slices_per_volume=num_slices,
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
        "vram_max": [],
    }

    # ============================================================
    # --- TRAINING LOOP ---
    # ============================================================
    print(f"\n🚦 Training for {epochs} epochs...\n")

    for epoch in range(epochs):
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

        train_res = train_one_epoch(
            model=model,
            loader=train_iter,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            epochs=epochs,
            log_fn=wandb.log,
            log_interval=20,
            track_vram=True,
        )

        val_res = validate(
            model=model,
            loader=val_loader,
            device=device,
            out_ch=out_ch,
            dice_fn=dice_score,
            iou_fn=iou_score,
        )

        curr_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss_mean": train_res.loss_mean,
            "train_loss_std": train_res.loss_std,
            "val_dice_mean": val_res.dice_mean,
            "val_dice_std": val_res.dice_std,
            "val_iou_mean": val_res.iou_mean,
            "val_iou_std": val_res.iou_std,
            "lr": curr_lr,
            "vram_max": train_res.vram_epoch_max_mb,
        }

        # Optional dedicated W&B key for readability
        if device.type == "cuda":
            wandb.log({"vram_epoch_max_mb": train_res.vram_epoch_max_mb})

        print(
            f"📈 Epoch {epoch+1:02d}: "
            f"Loss={epoch_metrics['train_loss_mean']:.4f}±{epoch_metrics['train_loss_std']:.4f}, "
            f"Dice={epoch_metrics['val_dice_mean']:.4f}±{epoch_metrics['val_dice_std']:.4f}, "
            f"IoU={epoch_metrics['val_iou_mean']:.4f}±{epoch_metrics['val_iou_std']:.4f}, "
            f"VRAM={epoch_metrics['vram_max']:.1f}MB, "
            f"LR={epoch_metrics['lr']:.6e}"
        )

        # One call logs to W&B and appends locally (only keys that exist in metrics_log)
        log_epoch(metrics_log, epoch_metrics, wandb_log_fn=wandb.log)

        # Save exactly one checkpoint at a specific epoch (e.g. for rewind pruning)
        if (epoch + 1) == save_int:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            if not os.path.exists(ckpt_path):  # avoid overwriting on reruns
                torch.save(model.state_dict(), ckpt_path)

    # ============================================================
    # --- SAVE FINAL MODEL ---
    # ============================================================
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Saved final model: {final_model_path}")

    # ============================================================
    # --- MODEL PROFILING ---
    # ============================================================
    prof = profile_model(model, in_ch=in_ch)
    wandb.log({"params_m": prof["params_m"], "flops_g": prof["flops_g"], "inference_ms": prof["inference_ms"]})

    # ============================================================
    # --- SAVE TRAINING CURVES ---
    # ============================================================
    plot_path = save_training_curves(metrics_log, save_dir, phase)
    wandb.log({"training_curve": wandb.Image(plot_path)})

    # ============================================================
    # --- SAVE TRAINING SUMMARY ---
    # ============================================================
    summary = {
        "model_name": exp_cfg["model_name"],
        "experiment": exp_cfg["experiment_name"],
        "phase": phase,
        "seed": seed,
        "deterministic": deterministic,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "device": str(device),
        "params_m": prof["params_m"],
        "flops_g": prof["flops_g"],
        "inference_ms": prof["inference_ms"],
        "vram_mb_last_epoch": metrics_log["vram_max"][-1] if metrics_log["vram_max"] else float("nan"),
        "final_train_loss": float(metrics_log["train_loss_mean"][-1]) if metrics_log["train_loss_mean"] else float("nan"),
        "final_val_dice": float(metrics_log["val_dice_mean"][-1]) if metrics_log["val_dice_mean"] else float("nan"),
        "final_val_iou": float(metrics_log["val_iou_mean"][-1]) if metrics_log["val_iou_mean"] else float("nan"),
        "augmentation": {
            "library": "torchio",
            "transforms": augmentation_summary if augmentation_summary else "none",
        },
    }

    save_json(summary, os.path.join(save_dir, "train_summary.json"))

    print("✅ Training complete.")
    run.finish()


if __name__ == "__main__":
    train_model()




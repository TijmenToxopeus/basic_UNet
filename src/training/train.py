# import os
# import json
# import torch
# import numpy as np
# from tqdm import tqdm
# import wandb

# # --- Project imports ---
# from src.models.unet import UNet
# from src.training.loss import get_loss_function
# from src.training.training_loop import train_one_epoch, validate
# from src.training.metrics import dice_score, iou_score
# from src.training.data_loader import get_train_val_loaders
# from src.pruning.rebuild import load_full_pruned_model
# from src.training.logging import log_epoch
# from src.training.artifacts import profile_model, save_training_curves, save_json, save_metrics_per_epoch_json

# from src.utils.config import load_config
# from src.utils.paths import get_paths
# from src.utils.wandb_utils import setup_wandb
# from src.utils.reproducibility import seed_everything
# from src.utils.run_summary import base_run_info, attach_profile, write_json



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

#     params = train_cfg["parameters"]
#     model_cfg = train_cfg["model"]
#     batch_size = train_cfg["batch_size"]
#     val_ratio = train_cfg["val_ratio"]
#     num_slices = train_cfg["num_slices_per_volume"]

#     lr = float(params["learning_rate"])
#     epochs = int(params["num_epochs"])
#     save_int = int(params["save_interval"])  # "rewind epoch" checkpoint
#     loss_name = params["loss_function"]

#     in_ch = int(model_cfg["in_channels"])
#     out_ch = int(model_cfg["out_channels"])

#     # ============================================================
#     # --- INITIALIZE WANDB ---
#     # ============================================================
#     # run = setup_wandb(cfg, job_type=phase)

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
#             meta=meta,
#             ckpt_path=paths.pruned_model,
#             in_ch=in_ch,
#             out_ch=out_ch,
#             device=device,
#         )
#     else:
#         model = UNet(
#             in_ch=in_ch,
#             out_ch=out_ch,
#             enc_features=model_cfg["features"],
#         ).to(device)

#     # wandb.watch(model, log="all")

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
#         seed=seed,
#         num_slices_per_volume=num_slices,
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
#         "vram_max": [],
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
#             # log_fn=wandb.log,
#             log_interval=20,
#             track_vram=True,
#         )

#         val_res = validate(
#             model=model,
#             loader=val_loader,
#             device=device,
#             out_ch=out_ch,
#             dice_fn=dice_score,
#             iou_fn=iou_score,
#         )

#         curr_lr = optimizer.param_groups[0]["lr"]

#         epoch_metrics = {
#             "epoch": epoch + 1,
#             "train_loss_mean": train_res.loss_mean,
#             "train_loss_std": train_res.loss_std,
#             "val_dice_mean": val_res.dice_mean,
#             "val_dice_std": val_res.dice_std,
#             "val_iou_mean": val_res.iou_mean,
#             "val_iou_std": val_res.iou_std,
#             "lr": curr_lr,
#             "vram_max": train_res.vram_epoch_max_mb,
#         }

#         # Optional dedicated W&B key for readability
#         # if device.type == "cuda":
#         #     wandb.log({"vram_epoch_max_mb": train_res.vram_epoch_max_mb})

#         print(
#             f"📈 Epoch {epoch+1:02d}: "
#             f"Loss={epoch_metrics['train_loss_mean']:.4f}±{epoch_metrics['train_loss_std']:.4f}, "
#             f"Dice={epoch_metrics['val_dice_mean']:.4f}±{epoch_metrics['val_dice_std']:.4f}, "
#             f"IoU={epoch_metrics['val_iou_mean']:.4f}±{epoch_metrics['val_iou_std']:.4f}, "
#             f"VRAM={epoch_metrics['vram_max']:.1f}MB, "
#             f"LR={epoch_metrics['lr']:.6e}"
#         )

#         # One call logs to W&B and appends locally (only keys that exist in metrics_log)
#         # log_epoch(metrics_log, epoch_metrics, wandb_log_fn=wandb.log)
#         log_epoch(metrics_log, epoch_metrics)

#         # Save exactly one checkpoint at a specific epoch (e.g. for rewind pruning)
#         if (epoch + 1) == save_int:
#             ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
#             if not os.path.exists(ckpt_path):  # avoid overwriting on reruns
#                 torch.save(model.state_dict(), ckpt_path)

#         # # Save checkpoints every save_int epochs
#         # if (epoch + 1) % save_int == 0:
#         #     ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
#         #     if not os.path.exists(ckpt_path):  # avoid overwriting on reruns
#         #         torch.save(model.state_dict(), ckpt_path)


#     # ============================================================
#     # --- SAVE FINAL MODEL ---
#     # ============================================================
#     final_model_path = os.path.join(save_dir, "final_model.pth")
#     torch.save(model.state_dict(), final_model_path)
#     print(f"💾 Saved final model: {final_model_path}")

#     # ============================================================
#     # --- MODEL PROFILING ---
#     # ============================================================
#     prof = profile_model(model, in_ch=in_ch)
#     # wandb.log(
#     #     {
#     #         "params_m": prof["params_m"],
#     #         "flops_g": prof["flops_g"],
#     #         "inference_ms": prof["inference_ms"],
#     #     }
#     # )

#     # ============================================================
#     # --- SAVE TRAINING CURVES ---
#     # ============================================================
#     plot_path = save_training_curves(metrics_log, save_dir, phase)
#     # wandb.log({"training_curve": wandb.Image(plot_path)})
#     metrics_path = save_metrics_per_epoch_json(metrics_log, save_dir)
#     # wandb.save(str(metrics_path))

#     # ============================================================
#     # --- SAVE RUN SUMMARY (shared schema) ---
#     # ============================================================
#     summary = base_run_info(cfg, stage="train")
#     summary["train"] = {
#         "phase": phase,
#         "epochs": epochs,
#         "learning_rate": lr,
#         "batch_size": batch_size,
#         "device": str(device),
#         "final": {
#             "train_loss": float(metrics_log["train_loss_mean"][-1]) if metrics_log["train_loss_mean"] else float("nan"),
#             "val_dice": float(metrics_log["val_dice_mean"][-1]) if metrics_log["val_dice_mean"] else float("nan"),
#             "val_iou": float(metrics_log["val_iou_mean"][-1]) if metrics_log["val_iou_mean"] else float("nan"),
#             "vram_epoch_peak_mb": float(metrics_log["vram_max"][-1]) if metrics_log["vram_max"] else float("nan"),
#         },
#         "artifacts": {
#             "final_model": final_model_path,
#             "training_curve": plot_path,
#         },
#         "augmentation": {
#             "library": "torchio",
#             "transforms": augmentation_summary if augmentation_summary else "none",
#         },
#     }

#     attach_profile(summary, prof)

#     summary_path = write_json(os.path.join(save_dir, "run_summary.json"), summary)
#     # wandb.save(str(summary_path))

#     # (optional) keep your old filename for backwards compatibility
#     # save_json(summary, os.path.join(save_dir, "train_summary.json"))

#     print("✅ Training complete.")
#     # run.finish()


# if __name__ == "__main__":
#     train_model()



import os
import json
import time

import torch
import numpy as np
from tqdm import tqdm
import wandb

# --- Project imports ---
from src.models.unet import UNet
from src.training.loss import get_loss_function
from src.training.training_loop import train_one_epoch, validate
from src.training.metrics import dice_score, iou_score
from src.training.data_loader import get_train_val_loaders
from src.pruning.rebuild import load_full_pruned_model
from src.training.logging import log_epoch
from src.training.artifacts import (
    profile_model,
    save_training_curves,
    save_json,
    save_metrics_per_epoch_json,
)

from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb
from src.utils.reproducibility import seed_everything
from src.utils.run_summary import base_run_info, attach_profile, write_json


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
    # run = setup_wandb(cfg, job_type=phase)

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

    # wandb.watch(model, log="all")

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
    # --- BEST CHECKPOINT TRACKING (overall best by val_dice_mean) ---
    # ============================================================
    best_val_dice = -1.0
    best_epoch = None
    final_model_path = os.path.join(save_dir, "final_model.pth")  # keep same name as before

    # ============================================================
    # --- TRAINING LOOP ---
    # ============================================================
    print(f"\n🚦 Training for {epochs} epochs...\n")
    t0 = time.perf_counter()

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
            # log_fn=wandb.log,
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

        print(
            f"📈 Epoch {epoch+1:02d}: "
            f"Loss={epoch_metrics['train_loss_mean']:.4f}±{epoch_metrics['train_loss_std']:.4f}, "
            f"Dice={epoch_metrics['val_dice_mean']:.4f}±{epoch_metrics['val_dice_std']:.4f}, "
            f"IoU={epoch_metrics['val_iou_mean']:.4f}±{epoch_metrics['val_iou_std']:.4f}, "
            f"VRAM={epoch_metrics['vram_max']:.1f}MB, "
            f"LR={epoch_metrics['lr']:.6e}"
        )

        # One call logs to W&B and appends locally (only keys that exist in metrics_log)
        # log_epoch(metrics_log, epoch_metrics, wandb_log_fn=wandb.log)
        log_epoch(metrics_log, epoch_metrics)

        # Save exactly one checkpoint at a specific epoch (e.g. for rewind pruning)
        if (epoch + 1) == save_int:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            if not os.path.exists(ckpt_path):  # avoid overwriting on reruns
                torch.save(model.state_dict(), ckpt_path)

        # --- Save BEST overall model checkpoint under the SAME name as before ---
        # This will overwrite "final_model.pth" whenever a new best appears.
        val_dice = float(epoch_metrics["val_dice_mean"])
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), final_model_path)

    t1 = time.perf_counter()
    total_train_time_s = t1 - t0

    # If for some reason no "best" was saved (e.g., NaNs), fall back to saving the last state
    if not os.path.exists(final_model_path):
        torch.save(model.state_dict(), final_model_path)

    print(f"💾 Saved best model (named final_model.pth): {final_model_path}")
    if best_epoch is not None:
        print(f"🏆 Best val Dice = {best_val_dice:.4f} at epoch {best_epoch}")
    print(
        f"⏱️ Total training time: {total_train_time_s:.1f}s "
        f"({total_train_time_s/60.0:.2f} min, {total_train_time_s/3600.0:.2f} h)"
    )

    # ============================================================
    # --- MODEL PROFILING ---
    # ============================================================
    prof = profile_model(model, in_ch=in_ch)
    # wandb.log(
    #     {
    #         "params_m": prof["params_m"],
    #         "flops_g": prof["flops_g"],
    #         "inference_ms": prof["inference_ms"],
    #     }
    # )

    # ============================================================
    # --- SAVE TRAINING CURVES ---
    # ============================================================
    plot_path = save_training_curves(metrics_log, save_dir, phase)
    # wandb.log({"training_curve": wandb.Image(plot_path)})
    metrics_path = save_metrics_per_epoch_json(metrics_log, save_dir)
    # wandb.save(str(metrics_path))

    # ============================================================
    # --- SAVE RUN SUMMARY (shared schema) ---
    # ============================================================
    summary = base_run_info(cfg, stage="train")
    summary["train"] = {
        "phase": phase,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "device": str(device),
        "time": {
            "total_seconds": float(total_train_time_s),
            "total_minutes": float(total_train_time_s / 60.0),
            "total_hours": float(total_train_time_s / 3600.0),
        },
        "best": {
            "val_dice": float(best_val_dice),
            "epoch": int(best_epoch) if best_epoch is not None else None,
        },
        "final": {
            "train_loss": float(metrics_log["train_loss_mean"][-1]) if metrics_log["train_loss_mean"] else float("nan"),
            "val_dice": float(metrics_log["val_dice_mean"][-1]) if metrics_log["val_dice_mean"] else float("nan"),
            "val_iou": float(metrics_log["val_iou_mean"][-1]) if metrics_log["val_iou_mean"] else float("nan"),
            "vram_epoch_peak_mb": float(metrics_log["vram_max"][-1]) if metrics_log["vram_max"] else float("nan"),
        },
        "artifacts": {
            "final_model": final_model_path,  # now actually best-overall
            "training_curve": plot_path,
            "metrics_per_epoch": str(metrics_path),
        },
        "augmentation": {
            "library": "torchio",
            "transforms": augmentation_summary if augmentation_summary else "none",
        },
    }

    attach_profile(summary, prof)

    summary_path = write_json(os.path.join(save_dir, "run_summary.json"), summary)
    # wandb.save(str(summary_path))

    # (optional) keep your old filename for backwards compatibility
    # save_json(summary, os.path.join(save_dir, "train_summary.json"))

    print("✅ Training complete.")
    # run.finish()


if __name__ == "__main__":
    train_model()

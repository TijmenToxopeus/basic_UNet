# import os
# import json
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
# import wandb  


# # --- Project imports ---
# from src.models.unet import UNet
# from src.training.loss import get_loss_function
# from src.training.metrics import dice_score, iou_score
# from src.training.data_loader import get_train_val_loaders
# from src.pruning.rebuild import build_pruned_unet
# from src.utils.config import load_config
# from src.utils.paths import get_paths
# from src.utils.wandb_utils import setup_wandb


# # ------------------------------------------------------------
# # TRAINING PIPELINE
# # ------------------------------------------------------------
# def train_model(cfg=None):
#     # ============================================================
#     # --- LOAD CONFIGURATION ---
#     # ============================================================
#     if cfg is None:
#         cfg, config_path = load_config(return_path=True)
#     else:
#         config_path = None

#     paths = get_paths(cfg, config_path)
#     paths.save_config_snapshot()

#     exp_cfg = cfg["experiment"]
#     train_cfg = cfg["train"]

#     phase = train_cfg["phase"].lower()
#     valid_phases = ["training", "retraining"]
#     if phase not in valid_phases:
#         raise ValueError(f"❌ Invalid training phase '{phase}'. Must be one of: {valid_phases}")

#     print(f"🚀 Starting {phase} for experiment {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
#     print(paths)

#     # ============================================================
#     # --- SETUP HYPERPARAMETERS ---
#     # ============================================================
#     device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

#     model_cfg = train_cfg["model"]
#     params = train_cfg["parameters"]
#     batch_size = train_cfg["batch_size"]
#     val_ratio = train_cfg["val_ratio"]
#     num_slices_per_volume = train_cfg["num_slices_per_volume"]

#     lr = float(params["learning_rate"])
#     epochs = params["num_epochs"]
#     save_interval = params["save_interval"]
#     loss_fn_name = params["loss_function"]

#     in_ch = model_cfg["in_channels"]
#     out_ch = model_cfg["out_channels"]

#     # ============================================================
#     # --- INITIALIZE WANDB ---
#     # ============================================================
#     wandb.init(
#         project="unet-pruning",
#         group=exp_cfg["experiment_name"],
#         job_type=phase,
#         name=f"{exp_cfg['experiment_name']}_{phase}",
#         config=cfg,
#         dir=str(paths.base_dir),
#     )

#     # ============================================================
#     # --- BUILD MODEL ---
#     # ============================================================
#     if phase == "retraining":
#         meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
#         if not meta_path.exists():
#             raise FileNotFoundError(f"❌ Expected pruning metadata not found: {meta_path}")

#         with open(meta_path, "r") as f:
#             meta = json.load(f)

#         enc_features = meta["enc_features"]
#         dec_features = meta["dec_features"]
#         bottleneck_out = meta["bottleneck_out"]

#         print(f"🧠 Rebuilding pruned UNet → enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
#         base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
#     else:
#         print(f"🧠 Building baseline UNet → features {model_cfg['features']}")
#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=model_cfg["features"]).to(device)

#     # ============================================================
#     # --- DATA & OPTIMIZATION SETUP ---
#     # ============================================================
#     save_dir = paths.train_save_dir
#     paths.ensure_dir(save_dir)
#     print(f"📂 Saving training outputs to: {save_dir}")

#     train_loader, val_loader = get_train_val_loaders(
#         img_dir=paths.train_dir,
#         lbl_dir=paths.label_dir,
#         batch_size=batch_size,
#         val_ratio=val_ratio,
#         shuffle=True,
#         num_slices_per_volume=num_slices_per_volume
#     )
#     print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

#     criterion = get_loss_function(loss_fn_name)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

#     metrics_log = {"epoch": [], "train_loss": [], "val_dice": [], "val_iou": [], "lr": []}

#     # ============================================================
#     # --- TRAINING LOOP ---
#     # ============================================================
#     print(f"\n🚦 Training for {epochs} epochs...\n")
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0

#         for batch_idx, (imgs, masks) in tqdm(
#             enumerate(train_loader),
#             total=len(train_loader),
#             desc=f"Epoch {epoch+1}/{epochs}",
#             ncols=100,
#         ):
#             imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
#             optimizer.zero_grad()
#             preds = model(imgs)
#             loss = criterion(preds, masks)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(train_loader)

#         # --- Validation ---
#         model.eval()
#         val_dice, val_iou = 0.0, 0.0
#         with torch.no_grad():
#             for imgs, masks in val_loader:
#                 imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
#                 preds = model(imgs)
#                 val_dice += dice_score(preds, masks, num_classes=out_ch)
#                 val_iou += iou_score(preds, masks, num_classes=out_ch)

#         val_dice /= len(val_loader)
#         val_iou /= len(val_loader)

#         scheduler.step(val_dice)
#         current_lr = optimizer.param_groups[0]["lr"]

#         print(f"📈 Epoch {epoch+1:02d}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}, LR={current_lr:.6e}")

#         metrics_log["epoch"].append(epoch + 1)
#         metrics_log["train_loss"].append(avg_loss)
#         metrics_log["val_dice"].append(val_dice)
#         metrics_log["val_iou"].append(val_iou)
#         metrics_log["lr"].append(current_lr)

#         # ✅ Log to W&B
#         wandb.log({
#             "epoch": epoch + 1,
#             "train_loss": avg_loss,
#             "val_dice": val_dice,
#             "val_iou": val_iou,
#             "lr": current_lr,
#         })

#         # --- Save checkpoint ---
#         if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
#             ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"💾 Saved checkpoint: {ckpt_path}")
#             wandb.save(ckpt_path)

#     # ============================================================
#     # --- SAVE METRICS & PLOTS ---
#     # ============================================================
#     final_model_path = os.path.join(save_dir, "final_model.pth")
#     torch.save(model.state_dict(), final_model_path)
#     print(f"💾 Saved final model: {final_model_path}")
#     wandb.save(final_model_path)

#     with open(os.path.join(save_dir, "metrics.json"), "w") as f:
#         json.dump(metrics_log, f, indent=4)

#     plt.figure()
#     plt.plot(metrics_log["epoch"], metrics_log["train_loss"], label="Train Loss")
#     plt.plot(metrics_log["epoch"], metrics_log["val_dice"], label="Val Dice")
#     plt.plot(metrics_log["epoch"], metrics_log["val_iou"], label="Val IoU")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.title(f"Training Progress ({phase})")
#     plt.legend()
#     plt.grid(True)
#     plot_path = os.path.join(save_dir, "training_curves.png")
#     plt.savefig(plot_path)
#     plt.close()
#     wandb.log({"training_curve": wandb.Image(plot_path)})

#     summary = {
#         "model_name": exp_cfg["model_name"],
#         "experiment": exp_cfg["experiment_name"],
#         "phase": phase,
#         "epochs": epochs,
#         "learning_rate": lr,
#         "batch_size": batch_size,
#         "device": str(device),
#         "final_train_loss": float(metrics_log["train_loss"][-1]),
#         "final_val_dice": float(metrics_log["val_dice"][-1]),
#         "final_val_iou": float(metrics_log["val_iou"][-1]),
#     }
#     with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
#         json.dump(summary, f, indent=4)

#     print("✅ Training complete.")
#     wandb.finish()


# if __name__ == "__main__":
#     train_model()


import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb  

# --- Project imports ---
from src.models.unet import UNet
from src.training.loss import get_loss_function
from src.training.metrics import dice_score, iou_score
from src.training.data_loader import get_train_val_loaders
from src.pruning.rebuild import build_pruned_unet
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb


# ------------------------------------------------------------
# TRAINING PIPELINE
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

    exp_cfg = cfg["experiment"]
    train_cfg = cfg["train"]

    phase = train_cfg["phase"].lower()
    valid_phases = ["training", "retraining"]
    if phase not in valid_phases:
        raise ValueError(f"❌ Invalid training phase '{phase}'. Must be one of: {valid_phases}")

    print(f"🚀 Starting {phase} for experiment {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
    print(paths)

    # ============================================================
    # --- SETUP HYPERPARAMETERS ---
    # ============================================================
    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model_cfg = train_cfg["model"]
    params = train_cfg["parameters"]
    batch_size = train_cfg["batch_size"]
    val_ratio = train_cfg["val_ratio"]
    num_slices_per_volume = train_cfg["num_slices_per_volume"]

    lr = float(params["learning_rate"])
    epochs = params["num_epochs"]
    save_interval = params["save_interval"]
    loss_fn_name = params["loss_function"]

    in_ch = model_cfg["in_channels"]
    out_ch = model_cfg["out_channels"]

    # ============================================================
    # --- INITIALIZE WANDB ---
    # ============================================================
    run = setup_wandb(cfg, job_type="training")
    wandb.watch_called = False  # ensure no duplicate hooks

    # ============================================================
    # --- BUILD MODEL ---
    # ============================================================
    if phase == "retraining":
        meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"❌ Expected pruning metadata not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        enc_features = meta["enc_features"]
        dec_features = meta["dec_features"]
        bottleneck_out = meta["bottleneck_out"]

        print(f"🧠 Rebuilding pruned UNet → enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
        base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
    else:
        print(f"🧠 Building baseline UNet → features {model_cfg['features']}")
        model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=model_cfg["features"]).to(device)

    wandb.watch(model, log="all")

    # ============================================================
    # --- DATA & OPTIMIZATION SETUP ---
    # ============================================================
    save_dir = paths.train_save_dir
    paths.ensure_dir(save_dir)
    print(f"📂 Saving training outputs to: {save_dir}")

    train_loader, val_loader = get_train_val_loaders(
        img_dir=paths.train_dir,
        lbl_dir=paths.label_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        shuffle=True,
        num_slices_per_volume=num_slices_per_volume
    )
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    criterion = get_loss_function(loss_fn_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    metrics_log = {"epoch": [], "train_loss": [], "val_dice": [], "val_iou": [], "lr": []}

    # ============================================================
    # --- TRAINING LOOP ---
    # ============================================================
    print(f"\n🚦 Training for {epochs} epochs...\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (imgs, masks) in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=100,
        ):
            imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
                preds = model(imgs)
                val_dice += dice_score(preds, masks, num_classes=out_ch)
                val_iou += iou_score(preds, masks, num_classes=out_ch)

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"📈 Epoch {epoch+1:02d}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}, LR={current_lr:.6e}")

        metrics_log["epoch"].append(epoch + 1)
        metrics_log["train_loss"].append(avg_loss)
        metrics_log["val_dice"].append(val_dice)
        metrics_log["val_iou"].append(val_iou)
        metrics_log["lr"].append(current_lr)

        # ✅ Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "lr": current_lr,
        })

        # --- Save checkpoint ---
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")
            wandb.save(ckpt_path)

    # ============================================================
    # --- SAVE METRICS & PLOTS ---
    # ============================================================
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Saved final model: {final_model_path}")
    wandb.save(final_model_path)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    plt.figure()
    plt.plot(metrics_log["epoch"], metrics_log["train_loss"], label="Train Loss")
    plt.plot(metrics_log["epoch"], metrics_log["val_dice"], label="Val Dice")
    plt.plot(metrics_log["epoch"], metrics_log["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Progress ({phase})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    wandb.log({"training_curve": wandb.Image(plot_path)})

    summary = {
        "model_name": exp_cfg["model_name"],
        "experiment": exp_cfg["experiment_name"],
        "phase": phase,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "device": str(device),
        "final_train_loss": float(metrics_log["train_loss"][-1]),
        "final_val_dice": float(metrics_log["val_dice"][-1]),
        "final_val_iou": float(metrics_log["val_iou"][-1]),
    }
    with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Training complete.")
    run.finish()


if __name__ == "__main__":
    train_model()

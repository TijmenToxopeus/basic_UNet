# import os
# import json
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from datetime import datetime
# from tqdm import tqdm

# from src.models.unet import UNet
# from src.training.loss import get_loss_function
# from src.training.metrics import dice_score, iou_score
# from src.training.data_loader import get_train_val_loaders
# from src.pruning.rebuild import build_pruned_unet  


# def make_save_dirs(model_name, save_root="results", run_name=None, subfolder=None, phase=None):
#     """
#     Create folder structure like:
#     results/UNet_ACDC/<timestamp>/<subfolder>/<phase>/
#     Example: results/UNet_ACDC/2025-10-27_11-08/baseline/training/
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

# def train_model():
#     # ============================================================
#     # --- USER CONFIGURATION SECTION ---
#     # ============================================================
#     save_root = "results"
#     model_name = "UNet_ACDC"
#     run_name = "exp4_larger_UNet_all_slices"
#     subfolder = "pruned"  # ✅ can also be "pruned"
#     phase = "retraining"

#     train_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr"
#     label_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTr"
#     val_ratio = 0.2
#     batch_size = 8
#     num_slices_per_volume = None  # use all slices

#     in_ch = 1
#     out_ch = 4
#     features = [64, 128, 256, 512, 1024]  # default encoder features

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     lr = 5e-4
#     epochs = 5
#     loss_fn_name = "ce"
#     save_interval = 25

#     # ============================================================
#     # --- BUILD MODEL ---
#     # ============================================================

#     if subfolder == "pruned":
#         model_checkpoint = f"{save_root}/{model_name}/{run_name}/pruned/0_0_10_20_30_30_30_20_10_0_0/pruned_model.pth"
#         meta_path = model_checkpoint.replace(".pth", "_meta.json")
#         with open(meta_path, "r") as f:
#             meta = json.load(f)

#         enc_features = meta["enc_features"]
#         dec_features = meta["dec_features"]
#         bottleneck_out = meta["bottleneck_out"]

#         print(f"🧠 Rebuilding pruned UNet: enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
#         base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
#     else:
#         print(f"🧠 Building baseline UNet: in_ch={in_ch}, out_ch={out_ch}, features={features}")
#         model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=features).to(device)

#     # ============================================================
#     # --- DATA & OPTIMIZATION SETUP ---
#     # ============================================================
#     save_dir = make_save_dirs(model_name, save_root, run_name, subfolder, phase)
#     print(f"📂 Saving run to {save_dir}")

#     print("🚀 Loading dataset...")
#     train_loader, val_loader = get_train_val_loaders(
#         img_dir=train_dir,
#         lbl_dir=label_dir,
#         batch_size=batch_size,
#         val_ratio=val_ratio,
#         shuffle=True,
#         num_slices_per_volume=num_slices_per_volume
#     )
#     print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

#     criterion = get_loss_function(loss_fn_name)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # 🔧 Reduce LR by factor 0.5 if validation Dice stops improving for 4 epochs
#     scheduler = ReduceLROnPlateau(
#         optimizer, mode='max',  # "max" because higher Dice is better
#         factor=0.5, patience=5
#     )

#     metrics_log = {"epoch": [], "train_loss": [], "val_dice": [], "val_iou": [], "lr": []}

#     # ============================================================
#     # --- TRAINING LOOP ---
#     # ============================================================
#     print(f"\n🚦 Starting training for {epochs} epochs...\n")
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0

#         progress_bar = tqdm(
#             enumerate(train_loader),
#             total=len(train_loader),
#             desc=f"Epoch {epoch+1}/{epochs}",
#             ncols=100
#         )

#         for batch_idx, (imgs, masks) in progress_bar:
#             imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)

#             optimizer.zero_grad()
#             preds = model(imgs)
#             loss = criterion(preds, masks)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#             progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

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

#         val_dice = (val_dice / len(val_loader)).item() if isinstance(val_dice, torch.Tensor) else val_dice / len(val_loader)
#         val_iou = (val_iou / len(val_loader)).item() if isinstance(val_iou, torch.Tensor) else val_iou / len(val_loader)

#         # 🔧 Step scheduler based on validation Dice
#         scheduler.step(val_dice)
#         current_lr = optimizer.param_groups[0]['lr']

#         print(f"📈 Epoch {epoch+1:02d}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}, LR={current_lr:.6e}")

#         metrics_log["epoch"].append(epoch + 1)
#         metrics_log["train_loss"].append(avg_loss)
#         metrics_log["val_dice"].append(val_dice)
#         metrics_log["val_iou"].append(val_iou)
#         metrics_log["lr"].append(current_lr)

#         # --- Save checkpoint ---
#         if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
#             ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"💾 Saved checkpoint to {ckpt_path}")

#     # ============================================================
#     # --- SAVE METRICS, PLOTS & SUMMARY ---
#     # ============================================================
#     torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
#     with open(os.path.join(save_dir, "metrics.json"), "w") as f:
#         json.dump(metrics_log, f, indent=4)
#     print("📊 Metrics saved.")

#     # --- Plot metrics ---
#     plt.figure()
#     plt.plot(metrics_log["epoch"], metrics_log["train_loss"], label="Train Loss")
#     plt.plot(metrics_log["epoch"], metrics_log["val_dice"], label="Val Dice")
#     plt.plot(metrics_log["epoch"], metrics_log["val_iou"], label="Val IoU")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.title("Training Progress")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, "training_curves.png"))
#     plt.close()

#     summary = {
#         "model_name": model_name,
#         "run_name": run_name,
#         "subfolder": subfolder,
#         "phase": phase,
#         "epochs": epochs,
#         "learning_rate": lr,
#         "batch_size": batch_size,
#         "device": str(device),
#         "final_train_loss": float(metrics_log["train_loss"][-1]),
#         "final_val_dice": float(metrics_log["val_dice"][-1]),
#         "final_val_iou": float(metrics_log["val_iou"][-1]),
#     }

#     summary_path = os.path.join(save_dir, "train_summary.json")
#     with open(summary_path, "w") as f:
#         json.dump(summary, f, indent=4)

#     print("✅ Training complete.")


# if __name__ == "__main__":
#     train_model()



import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import tqdm

# --- Project imports ---
from src.models.unet import UNet
from src.training.loss import get_loss_function
from src.training.metrics import dice_score, iou_score
from src.training.data_loader import get_train_val_loaders
from src.pruning.rebuild import build_pruned_unet
from src.utils.config import load_config
from src.utils.paths import get_paths


def train_model():
    # ============================================================
    # --- LOAD CONFIGURATION ---
    # ============================================================
    cfg, config_path = load_config(return_path=True)
    paths = get_paths(cfg, config_path)
    paths.save_config_snapshot()

    exp_cfg = cfg["experiment"]
    train_cfg = cfg["train"]

    print(f"🚀 Starting experiment {exp_cfg['experiment_name']} ({exp_cfg['model_name']})")
    print(paths)

    # ============================================================
    # --- SETUP HYPERPARAMETERS ---
    # ============================================================
    model_name = exp_cfg["model_name"]
    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # training params
    model_cfg = train_cfg["model"]
    params = train_cfg["parameters"]
    batch_size = train_cfg["batch_size"]
    val_ratio = train_cfg["val_ratio"]
    num_slices_per_volume = train_cfg["num_slices_per_volume"]

    lr = float(params["learning_rate"])
    epochs = params["num_epochs"]
    save_interval = params["save_interval"]
    loss_fn_name = params["loss_function"]

    # ============================================================
    # --- BUILD MODEL ---
    # ============================================================
    subfolder = train_cfg["paths"]["subfolder"]
    phase = train_cfg["phase"]

    if "pruned" in subfolder or "retrain" in phase.lower():
        # Load pruned meta info
        meta_path = paths.pruned_model.with_name(paths.pruned_model.name.replace(".pth", "_meta.json"))
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected pruning metadata: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        enc_features = meta["enc_features"]
        dec_features = meta["dec_features"]
        bottleneck_out = meta["bottleneck_out"]

        print(f"🧠 Rebuilding pruned UNet: enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
        base_model = UNet(
            in_ch=model_cfg["in_channels"],
            out_ch=model_cfg["out_channels"],
            enc_features=enc_features
        ).to(device)
        model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
    else:
        print(f"🧠 Building baseline UNet with features {model_cfg['features']}")
        model = UNet(
            in_ch=model_cfg["in_channels"],
            out_ch=model_cfg["out_channels"],
            enc_features=model_cfg["features"]
        ).to(device)

    # ============================================================
    # --- DATA & OPTIMIZATION SETUP ---
    # ============================================================
    save_dir = paths.train_save_dir
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
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

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
            ncols=100
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
                val_dice += dice_score(preds, masks, num_classes=model_cfg["out_channels"])
                val_iou += iou_score(preds, masks, num_classes=model_cfg["out_channels"])

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"📈 Epoch {epoch+1:02d}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}, LR={current_lr:.6e}")

        metrics_log["epoch"].append(epoch + 1)
        metrics_log["train_loss"].append(avg_loss)
        metrics_log["val_dice"].append(val_dice)
        metrics_log["val_iou"].append(val_iou)
        metrics_log["lr"].append(current_lr)

        # --- Save checkpoint ---
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    # ============================================================
    # --- SAVE METRICS & PLOTS ---
    # ============================================================
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    plt.figure()
    plt.plot(metrics_log["epoch"], metrics_log["train_loss"], label="Train Loss")
    plt.plot(metrics_log["epoch"], metrics_log["val_dice"], label="Val Dice")
    plt.plot(metrics_log["epoch"], metrics_log["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

    summary = {
        "model_name": model_name,
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


if __name__ == "__main__":
    train_model()

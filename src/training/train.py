# import os
# import json
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from datetime import datetime
# from tqdm import tqdm

# from src.models.unet import UNet
# from src.training.loss import get_loss_function
# from src.training.metrics import dice_score, iou_score
# from src.training.data_loader import get_train_val_loaders


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

#     # --- Experiment settings ---
#     save_root = "results"
#     model_name = "UNet_ACDC"
#     run_name = "exp1"  # leave None to auto-name with timestamp
#     subfolder = "baseline"
#     phase = "training" 

#     # --- Data settings ---
#     train_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr"
#     label_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTr"
#     val_ratio = 0.2
#     batch_size = 8
#     num_slices_per_volume = 30

#     # --- Model settings ---
#     in_ch = 1
#     out_ch = 4
#     features = [64, 128, 256, 512]

#     # --- Training settings ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     lr = 1e-4
#     epochs = 10
#     loss_fn_name = "ce"

#     # --- Logging ---
#     save_interval = 10  # save checkpoint every N epochs

#     # ============================================================
#     # --- TRAINING SCRIPT ---
#     # ============================================================

#     save_dir = make_save_dirs(model_name, save_root, run_name, subfolder, phase)
#     print(f"📂 Saving run to {save_dir}")

#     # --- Load data ---
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

#     # --- Build model ---
#     print(f"🧠 Building UNet: in_ch={in_ch}, out_ch={out_ch}, features={features}")
#     model = UNet(in_ch=in_ch, out_ch=out_ch, features=features).to(device)

#     # --- Loss and optimizer ---
#     criterion = get_loss_function(loss_fn_name)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # --- Metric logging ---
#     metrics_log = {"epoch": [], "train_loss": [], "val_dice": [], "val_iou": []}

#     # --- Training loop ---
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
#             imgs = imgs.to(device)
#             masks = masks.to(device, dtype=torch.long)

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
#                 imgs = imgs.to(device)
#                 masks = masks.to(device, dtype=torch.long)
#                 preds = model(imgs)
#                 val_dice += dice_score(preds, masks, num_classes=out_ch)
#                 val_iou  += iou_score(preds, masks, num_classes=out_ch)

#         val_dice = (val_dice / len(val_loader)).item() if isinstance(val_dice, torch.Tensor) else val_dice / len(val_loader)
#         val_iou  = (val_iou / len(val_loader)).item() if isinstance(val_iou, torch.Tensor) else val_iou / len(val_loader)

#         print(f"📈 Epoch {epoch+1:02d}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")

#         # --- Log metrics ---
#         metrics_log["epoch"].append(epoch + 1)
#         metrics_log["train_loss"].append(avg_loss)
#         metrics_log["val_dice"].append(val_dice)
#         metrics_log["val_iou"].append(val_iou)

#         # --- Save checkpoint ---
#         if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
#             ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"💾 Saved checkpoint to {ckpt_path}")

#     # --- Final saves ---
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


#     # --- Final summary save ---
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
from datetime import datetime
from tqdm import tqdm

from src.models.unet import UNet
from src.training.loss import get_loss_function
from src.training.metrics import dice_score, iou_score
from src.training.data_loader import get_train_val_loaders
from src.pruning.rebuild import build_pruned_unet  


def make_save_dirs(model_name, save_root="results", run_name=None, subfolder=None, phase=None):
    """
    Create folder structure like:
    results/UNet_ACDC/<timestamp>/<subfolder>/<phase>/
    Example: results/UNet_ACDC/2025-10-27_11-08/baseline/training/
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


def train_model():
    # ============================================================
    # --- USER CONFIGURATION SECTION ---
    # ============================================================
    save_root = "results"
    model_name = "UNet_ACDC"
    run_name = "exp1"
    subfolder = "pruned"  # ✅ can also be "pruned"
    phase = "retraining"

    train_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr"
    label_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTr"
    val_ratio = 0.2
    batch_size = 8
    num_slices_per_volume = 30

    in_ch = 1
    out_ch = 4
    features = [64, 128, 256, 512]  # default encoder features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    epochs = 10
    loss_fn_name = "ce"
    save_interval = 10

    # ============================================================
    # --- BUILD MODEL ---
    # ============================================================

    # ✅ NEW: automatically detect if we’re training a pruned model
    if subfolder == "pruned":
        model_checkpoint = f"{save_root}/{model_name}/{run_name}/pruned/pruned_model.pth"
        meta_path = model_checkpoint.replace(".pth", "_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        enc_features = meta["enc_features"]
        dec_features = meta["dec_features"]
        bottleneck_out = meta["bottleneck_out"]

        print(f"🧠 Rebuilding pruned UNet: enc={enc_features}, dec={dec_features}, bottleneck={bottleneck_out}")
        base_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        model = build_pruned_unet(base_model, enc_features, dec_features, bottleneck_out).to(device)
    else:
        print(f"🧠 Building baseline UNet: in_ch={in_ch}, out_ch={out_ch}, features={features}")
        model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=features).to(device)

    # ============================================================
    # --- DATA & OPTIMIZATION SETUP ---
    # ============================================================
    save_dir = make_save_dirs(model_name, save_root, run_name, subfolder, phase)
    print(f"📂 Saving run to {save_dir}")

    print("🚀 Loading dataset...")
    train_loader, val_loader = get_train_val_loaders(
        img_dir=train_dir,
        lbl_dir=label_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        shuffle=True,
        num_slices_per_volume=num_slices_per_volume
    )
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    criterion = get_loss_function(loss_fn_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_log = {"epoch": [], "train_loss": [], "val_dice": [], "val_iou": []}

    # ============================================================
    # --- TRAINING LOOP ---
    # ============================================================
    print(f"\n🚦 Starting training for {epochs} epochs...\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=100
        )

        for batch_idx, (imgs, masks) in progress_bar:
            imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

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

        val_dice = (val_dice / len(val_loader)).item() if isinstance(val_dice, torch.Tensor) else val_dice / len(val_loader)
        val_iou = (val_iou / len(val_loader)).item() if isinstance(val_iou, torch.Tensor) else val_iou / len(val_loader)

        print(f"📈 Epoch {epoch+1:02d}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")

        metrics_log["epoch"].append(epoch + 1)
        metrics_log["train_loss"].append(avg_loss)
        metrics_log["val_dice"].append(val_dice)
        metrics_log["val_iou"].append(val_iou)

        # --- Save checkpoint ---
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint to {ckpt_path}")

    # ============================================================
    # --- SAVE METRICS, PLOTS & SUMMARY ---
    # ============================================================
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)
    print("📊 Metrics saved.")

    # --- Plot metrics ---
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
        "run_name": run_name,
        "subfolder": subfolder,
        "phase": phase,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "device": str(device),
        "final_train_loss": float(metrics_log["train_loss"][-1]),
        "final_val_dice": float(metrics_log["val_dice"][-1]),
        "final_val_iou": float(metrics_log["val_iou"][-1]),
    }

    summary_path = os.path.join(save_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Training complete.")


if __name__ == "__main__":
    train_model()

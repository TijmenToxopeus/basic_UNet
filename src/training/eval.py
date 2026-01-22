import os
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

# --- Project imports ---
from src.models.unet import UNet
from src.training.data_factory import build_eval_loader
from src.training.metrics import dice_score, iou_score
from src.pruning.rebuild import plot_unet_schematic, load_full_pruned_model
from src.training.eval_loop import run_evaluation
from src.training.artifacts import profile_model, save_json
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb
from src.utils.reproducibility import seed_everything
from src.utils.run_summary import base_run_info, attach_profile, write_json



# ------------------------------------------------------------
# Visualization Helper
# ------------------------------------------------------------
def save_visual(img, mask, pred, save_dir, idx):
    """Save Input / Ground Truth / Prediction triplet."""
    img = img.cpu().squeeze().numpy()
    mask = mask.cpu().numpy()
    pred = torch.argmax(pred, dim=0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("Input")
    axs[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=3); axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap="nipy_spectral", vmin=0, vmax=3); axs[2].set_title("Prediction")
    for a in axs:
        a.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, f"sample_{idx}.png")
    plt.savefig(path)
    plt.close(fig)
    return path


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

    exp_cfg = cfg["experiment"]
    eval_cfg = cfg["evaluation"]

    # ============================================================
    # --- SEEDING (must be early) ---
    # ============================================================
    seed = exp_cfg.get("seed", 42)
    deterministic = exp_cfg.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    phase = eval_cfg["phase"].lower()
    valid_phases = ["baseline_evaluation", "pruned_evaluation", "retrained_pruned_evaluation"]
    if phase not in valid_phases:
        raise ValueError(f"❌ Invalid phase '{phase}'. Must be: {', '.join(valid_phases)}")

    print(f"🔍 Starting evaluation: {phase} for {exp_cfg['experiment_name']}")
    print(f"🔁 Seed = {seed} | Deterministic = {deterministic}")

    # ============================================================
    # --- INIT WANDB ---
    # ============================================================
    # run = setup_wandb(cfg, job_type=phase)

    # ============================================================
    # --- CONFIG PARAMETERS ---
    # ============================================================
    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    num_slices = eval_cfg["num_slices_per_volume"]
    batch_size = eval_cfg["batch_size"]
    num_visuals = eval_cfg["num_visuals"]

    in_ch = cfg["train"]["model"]["in_channels"]
    out_ch = cfg["train"]["model"]["out_channels"]

    img_dir = paths.eval_dir
    lbl_dir = paths.eval_label_dir
    save_dir = paths.eval_save_dir
    paths.ensure_dir(save_dir)

    # ============================================================
    # --- MODEL LOADING ---
    # ============================================================
    model_ckpt = None
    enc_features = dec_features = bottleneck = None

    if phase == "baseline_evaluation":
        print("🧠 Loading BASELINE model...")
        enc = cfg["train"]["model"]["features"]
        model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc).to(device)

        model_ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"
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

        enc_features = meta.get("enc_features")
        dec_features = meta.get("dec_features")
        bottleneck = meta.get("bottleneck_out")

        if phase == "pruned_evaluation":
            print("🧠 Loading PRUNED (not retrained) model...")
            model_ckpt = paths.pruned_model
        else:
            print("🧠 Loading RETRAINED pruned model...")
            model_ckpt = paths.retrain_pruned_dir / "final_model.pth"

        if not model_ckpt.exists():
            raise FileNotFoundError(f"❌ Checkpoint not found: {model_ckpt}")

        model = load_full_pruned_model(
            meta=meta,
            ckpt_path=model_ckpt,
            in_ch=in_ch,
            out_ch=out_ch,
            device=device,
        )

    print(f"📂 Loaded checkpoint: {model_ckpt}")
    model.eval()

    # ============================================================
    # --- PROFILE MODEL ---
    # ============================================================
    prof = profile_model(model, in_ch=in_ch)
    print("\n📊 Profiling model (Params, FLOPs, Inference)...")
    print(f"🧮 Params: {prof['params_m']:.2f}M")
    print(f"⚙️ FLOPs:  {prof['flops_g']:.2f} GFLOPs")
    print(f"⚡ Inference time: {prof['inference_ms']:.2f} ms")

    # ============================================================
    # --- LOAD TEST DATASET ---
    # ============================================================
    test_loader, n_batches = build_eval_loader(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        batch_size=batch_size,
        num_slices_per_volume=num_slices,
        shuffle=False,
        # optionally expose these via config later
        num_workers=0,
        pin_memory=True,
    )

    print(f"✅ Loaded {n_batches} evaluation batches.")


    # Seeded selection of visualization indices (batch indices)
    rng = random.Random(seed)
    visual_indices = set(
        rng.sample(range(len(test_loader)), k=min(num_visuals, len(test_loader)))
    )

    vis_dir = os.path.join(save_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    # ============================================================
    # --- DEBUG MODEL STRUCTURE ---
    # ============================================================
    if debug:
        try:
            if phase == "baseline_evaluation":
                enc = cfg["train"]["model"]["features"]
                dec = enc[::-1]
                bott = enc[-1] * 2
            else:
                enc = enc_features
                dec = dec_features
                bott = bottleneck

            plot_unet_schematic(
                enc, dec, bott, in_ch, out_ch, title=f"{phase} U-Net"
            )
        except Exception as e:
            print(f"(⚠️ Could not plot schematic: {e})")

    # ============================================================
    # --- EVALUATION LOOP (metrics) ---
    # ============================================================
    print("🚀 Running evaluation...")
    eval_res, _raw = run_evaluation(
        model=model,
        loader=tqdm(test_loader, desc="Evaluating"),
        device=device,
        num_classes=out_ch,
        dice_fn=dice_score,
        iou_fn=iou_score,
        vram_track=True,
    )

    # wandb.log({"vram_eval_peak_mb": eval_res.vram_peak_mb})

    # ============================================================
    # --- VISUAL LOGGING (separate pass; avoids mixing concerns) ---
    # ============================================================
    # If you want, you can integrate this into the eval loop with a callback,
    # but keeping it separate keeps run_evaluation pure.
    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(test_loader):
            if idx not in visual_indices:
                continue
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.long)
            preds = model(imgs)
            img_path = save_visual(imgs[0], masks[0], preds[0], vis_dir, idx)
            # wandb.log({"sample_prediction": wandb.Image(img_path)})

    # ============================================================
    # --- PRINT RESULTS ---
    # ============================================================
    class_names = ["Background", "RV", "Myocardium", "LV"]

    print("\n✅ Evaluation complete!")
    print(f"📊 Mean Dice (foreground): {eval_res.mean_dice_fg:.4f} ± {eval_res.std_dice_fg:.4f}")
    print(f"📊 Mean IoU  (foreground): {eval_res.mean_iou_fg:.4f} ± {eval_res.std_iou_fg:.4f}")
    print(f"💾 VRAM peak during evaluation: {eval_res.vram_peak_mb:.1f} MB")

    for name, d, s in zip(class_names, eval_res.class_dice_mean, eval_res.class_dice_std):
        print(f"{name:12s} Dice={d:.4f} ± {s:.4f}")

    # ============================================================
    # --- SAVE RUN SUMMARY (shared schema) ---
    # ============================================================
    from src.utils.run_summary import base_run_info, attach_profile, write_json

    summary = base_run_info(cfg, stage="eval")
    summary["eval"] = {
        "phase": phase,
        "checkpoint": str(model_ckpt),
        "vram_peak_mb": float(eval_res.vram_peak_mb),
        "foreground": {
            "dice_mean": float(eval_res.mean_dice_fg),
            "dice_std": float(eval_res.std_dice_fg),
            "iou_mean": float(eval_res.mean_iou_fg),
            "iou_std": float(eval_res.std_iou_fg),
        },
        "per_class": {
            name: {
                "dice_mean": float(dm),
                "dice_std": float(ds),
                "iou_mean": float(im),
                "iou_std": float(is_),
            }
            for name, dm, ds, im, is_ in zip(
                class_names,
                eval_res.class_dice_mean,
                eval_res.class_dice_std,
                eval_res.class_iou_mean,
                eval_res.class_iou_std,
            )
        },
        "artifacts": {
            "predictions_dir": vis_dir,
        },
    }

    attach_profile(summary, prof)

    summary_path = write_json(os.path.join(save_dir, "run_summary.json"), summary)
    # wandb.save(str(summary_path))

    # ============================================================
    # --- WANDB METRICS LOGGING (flat keys for dashboards) ---
    # ============================================================
    # wandb.log(
    #     {
    #         "mean_dice_fg": float(eval_res.mean_dice_fg),
    #         "std_dice_fg": float(eval_res.std_dice_fg),
    #         "mean_iou_fg": float(eval_res.mean_iou_fg),
    #         "std_iou_fg": float(eval_res.std_iou_fg),
    #         "vram_eval_peak_mb": float(eval_res.vram_peak_mb),
    #         **{f"class_dice_mean/{c}": float(m) for c, m in zip(class_names, eval_res.class_dice_mean)},
    #         **{f"class_dice_std/{c}": float(s) for c, s in zip(class_names, eval_res.class_dice_std)},
    #         **{f"class_iou_mean/{c}": float(m) for c, m in zip(class_names, eval_res.class_iou_mean)},
    #         **{f"class_iou_std/{c}": float(s) for c, s in zip(class_names, eval_res.class_iou_std)},
    #         "params_m": float(prof["params_m"]),
    #         "flops_g": float(prof["flops_g"]),
    #         "inference_ms": float(prof["inference_ms"]),
    #     }
    # )

    print(f"💾 Summary saved to: {summary_path}")
    print(f"🖼️ Visualizations saved to: {vis_dir}")

    # run.finish()



if __name__ == "__main__":
    evaluate(debug=True)

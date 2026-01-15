# """
# Pruning pipeline for basic UNet
# --------------------------------
# Loads a trained baseline model, applies structured L1-based pruning,
# rebuilds the pruned model, and saves both the weights and architecture metadata.
# """

# import os
# import json
# import random
# import torch
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import torchvision.transforms as T
# import wandb
# import copy

# from src.models.unet import UNet
# from src.pruning.model_inspect import (
#     model_to_dataframe_with_l1,
#     compute_l1_norms,
#     compute_l1_stats,
#     get_pruning_masks_blockwise,
# )
# from src.pruning.similar_feature_pruning import get_redundancy_masks, load_random_slices_acdc
# from src.pruning.rebuild import rebuild_pruned_unet
# from src.utils.config import load_config
# from src.utils.paths import get_paths
# from src.utils.wandb_utils import setup_wandb

# # --- NEW: reproducibility ---
# from src.utils.reproducibility import seed_everything


# # ------------------------------------------------------------
# # MAIN PRUNING PIPELINE
# # ------------------------------------------------------------
# def run_pruning(cfg=None):
#     # ============================================================
#     # --- LOAD CONFIGURATION ---
#     # ============================================================
#     if cfg is None:
#         cfg, config_path = load_config(return_path=True)
#     else:
#         config_path = None

#     # ============================================================
#     # --- SEEDING (must be early) ---
#     # ============================================================
#     exp_cfg = cfg["experiment"]
#     seed = exp_cfg.get("seed", 42)
#     deterministic = exp_cfg.get("deterministic", False)
#     seed_everything(seed, deterministic=deterministic)

#     paths = get_paths(cfg, config_path)
#     pruning_cfg = cfg["pruning"]
#     model_cfg = cfg["train"]["model"]

#     exp_name = cfg["experiment"]["experiment_name"]
#     model_name = cfg["experiment"]["model_name"]
#     print(f"✂️ Starting L1-based structured pruning for {model_name}")
#     print(f"🔁 Seed = {seed} | Deterministic = {deterministic}")
#     print(paths)

#     block_ratios = pruning_cfg.get("ratios", {}).get("block_ratios", {})
#     default_ratio = pruning_cfg.get("ratios", {}).get("default", 0.25)

#     # ============================================================
#     # --- INIT WANDB RUN ---
#     # ============================================================
#     run = setup_wandb(cfg, job_type="pruning")

#     # ============================================================
#     # --- LOAD BASELINE MODEL ---
#     # ============================================================
#     baseline_ckpt = paths.baseline_ckpt
#     if not baseline_ckpt.exists():
#         raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

#     in_ch = model_cfg["in_channels"]
#     out_ch = model_cfg["out_channels"]
#     enc_features = model_cfg["features"]

#     device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     print(f"📦 Loading baseline model from: {baseline_ckpt}")

#     model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#     state = torch.load(baseline_ckpt, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # ============================================================
#     # --- COMPUTE L1 NORMS ---
#     # ============================================================
#     print("📊 Computing L1 norms for all Conv layers...")
#     norms = compute_l1_norms(model)
#     l1_stats = compute_l1_stats(norms)
#     df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
#     pd.set_option("display.max_rows", None)
#     print("✅ L1 statistics computed.\n")

#     wandb.log({"l1_norms": wandb.Table(dataframe=df)})

#     # ============================================================
#     # --- GENERATE MASKS ---
#     # ============================================================
#     print("✂️ Generating pruning masks...")
#     method = pruning_cfg.get("method", "l1_norm")
#     print(f"✂️ Using pruning method: {method}")

#     if method == "l1_norm":
#         # ---- L1-based pruning ----
#         print("📊 Computing L1 norms for all Conv layers...")
#         norms = compute_l1_norms(model)
#         l1_stats = compute_l1_stats(norms)
#         df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)
#         wandb.log({"l1_norms": wandb.Table(dataframe=df)})

#         print("✂️ Generating L1-based pruning masks...")
#         masks = get_pruning_masks_blockwise(
#             model,
#             norms,
#             block_ratios=block_ratios,
#             default_ratio=default_ratio,
#             seed=seed, 
#             deterministic=deterministic
#         )

#     elif method == "correlation":
#         print("🔍 Running correlation-based redundancy pruning...")

#         # -------------------------
#         # Load MULTIPLE slices
#         # -------------------------
#         img_dir = "/mnt/hdd/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr/"
#         num_samples = pruning_cfg.get("num_slices", 20)
#         print(f"num slices: {num_samples}")

#         # Use config seed (not hardcoded)
#         example_slices = load_random_slices_acdc(
#             img_dir,
#             num_slices=num_samples,
#             seed=seed,
#             deterministic=deterministic
#         )

#         print(f"Loaded {len(example_slices)} example slices for correlation pruning.")

#         threshold = pruning_cfg.get("threshold", 0.9)
#         batch_size = pruning_cfg.get("batch_size", 4)

#         # -------------------------
#         # Run correlation pruning
#         # -------------------------
#         masks = get_redundancy_masks(
#             model=model,
#             example_slices=example_slices,
#             block_ratios=block_ratios,
#             threshold=threshold,
#             batch_size=batch_size,
#             plot=False
#         )
#     else:
#         raise ValueError(f"❌ Unknown pruning method '{method}'. Choose 'l1_norm' or 'correlation'.")

#     # ============================================================
#     # --- REBUILD PRUNED MODEL ---
#     # ============================================================
#     paths.ensure_dir(paths.pruned_model_dir)

#     print("🔧 Rebuilding pruned model...")
#     if pruning_cfg.get("reinitialize_weights") == "rewind":
#         print("🔄 Reinitializing pruned model with rewind weights...")
#         rewind_ckpt = paths.rewind_ckpt
#         if not rewind_ckpt.exists():
#             raise FileNotFoundError(f"❌ Rewind checkpoint not found at {rewind_ckpt}")
#         else:
#             print(f"Loading rewind weights from {rewind_ckpt}")

#         rewind_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#         rewind_model.load_state_dict(torch.load(rewind_ckpt, map_location=device))
#         rewind_model.eval()
#         pruned_model = rebuild_pruned_unet(rewind_model, masks, save_path=paths.pruned_model, seed=seed, deterministic=deterministic)

#     else:
#         pruned_model = rebuild_pruned_unet(model, masks, save_path=paths.pruned_model, seed=seed, deterministic=deterministic)

#         if pruning_cfg.get("reinitialize_weights") == "random":
#             print("🔄 Reinitializing pruned model with random weights...")

#             # ---- Compute global pre-reinit stats ----
#             before_params = []
#             for m in pruned_model.modules():
#                 if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
#                     before_params.append(m.weight.detach().cpu().flatten())
#             before_params = torch.cat(before_params)
#             before_mean = before_params.mean().item()
#             before_std = before_params.std().item()

#             print(f"📊 BEFORE reinit: mean={before_mean:.6f}, std={before_std:.6f}")

#             def init_weights(m):
#                 if isinstance(m, torch.nn.Conv2d):
#                     torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)

#                 elif isinstance(m, torch.nn.ConvTranspose2d):
#                     torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)

#                 elif isinstance(m, torch.nn.Linear):
#                     torch.nn.init.kaiming_normal_(m.weight)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)

#                 elif isinstance(m, torch.nn.BatchNorm2d):
#                     torch.nn.init.ones_(m.weight)
#                     torch.nn.init.zeros_(m.bias)

#             pruned_model.apply(init_weights)

#             save_path = paths.pruned_model
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             torch.save(pruned_model.state_dict(), save_path)
#             print(f"💾 Saved pruned model to {save_path}")

#             # ---- Compute global post-reinit stats ----
#             after_params = []
#             for m in pruned_model.modules():
#                 if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
#                     after_params.append(m.weight.detach().cpu().flatten())
#             after_params = torch.cat(after_params)
#             after_mean = after_params.mean().item()
#             after_std = after_params.std().item()

#             print(f"📊 AFTER reinit:  mean={after_mean:.6f}, std={after_std:.6f}")

#             print("🔍 Reinit verification:")
#             print(f"Mean changed by {abs(after_mean - before_mean):.6f}")
#             print(f"Std  changed by {abs(after_std - before_std):.6f}")
#             print("✅ Reinitialized weights.\n")

#             wandb.log({
#                 "reinit_mean_before": before_mean,
#                 "reinit_std_before": before_std,
#                 "reinit_mean_after": after_mean,
#                 "reinit_std_after": after_std,
#             })
#         else:
#             print("🔄 Using current weights for rebuild...")

#     # ============================================================
#     # --- PARAM SUMMARY, SAVE, ETC... ---
#     # ============================================================
#     orig_params = sum(p.numel() for p in model.parameters())
#     pruned_params = sum(p.numel() for p in pruned_model.parameters())
#     reduction = 100 * (1 - pruned_params / orig_params)

#     summary = {
#         "experiment": exp_name,
#         "model_name": model_name,
#         "seed": seed,
#         "deterministic": deterministic,
#         "method": method,
#         "block_ratios": block_ratios,
#         "default_ratio": default_ratio,
#         "orig_params": int(orig_params),
#         "pruned_params": int(pruned_params),
#         "reduction_percent": float(reduction),
#         "baseline_ckpt": str(baseline_ckpt),
#         "pruned_model": str(paths.pruned_model),
#     }

#     summary_path = paths.pruned_model_dir / "pruning_summary.json"
#     with open(summary_path, "w") as f:
#         json.dump(summary, f, indent=4)

#     wandb.save(str(summary_path))

#     run.finish()
#     print(f"💾 Summary saved to {summary_path}")
#     print("✅ Pruning complete.\n")


# if __name__ == "__main__":
#     run_pruning()


"""
Pruning pipeline for basic UNet
--------------------------------
Loads a trained baseline model, applies structured pruning (L1-norm or correlation),
rebuilds the pruned model, and saves both the weights and architecture metadata.
Also writes a shared run_summary.json compatible with train/eval summaries.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import wandb

from src.models.unet import UNet
from src.pruning.model_inspect import (
    model_to_dataframe_with_l1,
    compute_l1_norms,
    compute_l1_stats,
    get_pruning_masks_blockwise,
)
from src.pruning.similar_feature_pruning import (
    get_redundancy_masks,
    load_random_slices_acdc,
)
from src.pruning.rebuild import rebuild_pruned_unet
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb
from src.utils.reproducibility import seed_everything

# --- NEW: shared run summary schema ---
from src.utils.run_summary import base_run_info, write_json


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _init_weights_kaiming(m: torch.nn.Module) -> None:
    """Kaiming init for conv/linear, sane defaults for BN."""
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


def _collect_weight_stats(model: torch.nn.Module) -> tuple[float, float]:
    """Global mean/std across Conv2d/Linear weights (for reinit verification)."""
    params = []
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            if getattr(mod, "weight", None) is not None:
                params.append(mod.weight.detach().cpu().flatten())
    if not params:
        return float("nan"), float("nan")
    flat = torch.cat(params)
    return float(flat.mean().item()), float(flat.std().item())


# ------------------------------------------------------------
# MAIN PRUNING PIPELINE
# ------------------------------------------------------------
def run_pruning(cfg=None):
    # ============================================================
    # --- LOAD CONFIGURATION ---
    # ============================================================
    if cfg is None:
        cfg, config_path = load_config(return_path=True)
    else:
        config_path = None

    # ============================================================
    # --- SEEDING (must be early) ---
    # ============================================================
    exp_cfg = cfg["experiment"]
    seed = exp_cfg.get("seed", 42)
    deterministic = exp_cfg.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    paths = get_paths(cfg, config_path)
    pruning_cfg = cfg["pruning"]
    model_cfg = cfg["train"]["model"]

    exp_name = exp_cfg["experiment_name"]
    model_name = exp_cfg["model_name"]

    print(f"✂️ Starting pruning for {model_name}")
    print(f"🔁 Seed = {seed} | Deterministic = {deterministic}")

    block_ratios = pruning_cfg.get("ratios", {}).get("block_ratios", {})
    default_ratio = pruning_cfg.get("ratios", {}).get("default", 0.25)

    method = pruning_cfg.get("method", "l1_norm")
    if method not in ("l1_norm", "correlation"):
        raise ValueError(f"❌ Unknown pruning method '{method}'. Choose 'l1_norm' or 'correlation'.")

    # ============================================================
    # --- INIT WANDB RUN ---
    # ============================================================
    run = setup_wandb(cfg, job_type="pruning")

    # ============================================================
    # --- LOAD BASELINE MODEL ---
    # ============================================================
    baseline_ckpt = paths.baseline_ckpt
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"❌ Baseline checkpoint not found at {baseline_ckpt}")

    in_ch = int(model_cfg["in_channels"])
    out_ch = int(model_cfg["out_channels"])
    enc_features = model_cfg["features"]

    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"📦 Loading baseline model from: {baseline_ckpt}")

    model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
    state = torch.load(baseline_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ============================================================
    # --- GENERATE MASKS ---
    # ============================================================
    print(f"✂️ Using pruning method: {method}")

    l1_stats = None
    df = None

    if method == "l1_norm":
        # ---- L1-based pruning ----
        print("📊 Computing L1 norms for all Conv layers...")
        norms = compute_l1_norms(model)
        l1_stats = compute_l1_stats(norms)
        df = model_to_dataframe_with_l1(model, l1_stats, remove_nan_layers=True)

        # Nice for debugging; not required
        pd.set_option("display.max_rows", None)
        print("✅ L1 statistics computed.\n")

        wandb.log({"l1_norms": wandb.Table(dataframe=df)})

        print("✂️ Generating L1-based pruning masks...")
        masks = get_pruning_masks_blockwise(
            model=model,
            norms=norms,
            block_ratios=block_ratios,
            default_ratio=default_ratio,
            seed=seed,
            deterministic=deterministic,
        )

    else:
        # ---- Correlation-based redundancy pruning ----
        print("🔍 Running correlation-based redundancy pruning...")

        # IMPORTANT: use your configured train_dir (no hardcoded path)
        img_dir = str(paths.train_dir)

        num_samples = int(pruning_cfg.get("num_slices", 20))
        threshold = float(pruning_cfg.get("threshold", 0.9))
        batch_size = int(pruning_cfg.get("batch_size", 4))

        print(f"🧪 Using {num_samples} slices from: {img_dir}")
        example_slices = load_random_slices_acdc(
            img_dir,
            num_slices=num_samples,
            seed=seed,
            deterministic=deterministic,
        )
        print(f"Loaded {len(example_slices)} example slices for correlation pruning.")
        print(f"threshold={threshold} | batch_size={batch_size}")

        masks = get_redundancy_masks(
            model=model,
            example_slices=example_slices,
            block_ratios=block_ratios,
            threshold=threshold,
            batch_size=batch_size,
            plot=False,
        )

    # ============================================================
    # --- REBUILD PRUNED MODEL ---
    # ============================================================
    paths.ensure_dir(paths.pruned_model_dir)
    print("🔧 Rebuilding pruned model...")

    reinit_mode = pruning_cfg.get("reinitialize_weights", None)

    reinit_stats = None  # optionally filled for random mode

    if reinit_mode == "rewind":
        print("🔄 Reinitializing pruned model with rewind weights...")
        rewind_ckpt = paths.rewind_ckpt
        if rewind_ckpt is None or not rewind_ckpt.exists():
            raise FileNotFoundError(f"❌ Rewind checkpoint not found at {rewind_ckpt}")
        print(f"📦 Loading rewind weights from: {rewind_ckpt}")

        rewind_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
        rewind_model.load_state_dict(torch.load(rewind_ckpt, map_location=device))
        rewind_model.eval()

        pruned_model = rebuild_pruned_unet(
            rewind_model,
            masks,
            save_path=paths.pruned_model,
            seed=seed,
            deterministic=deterministic,
        )

    else:
        pruned_model = rebuild_pruned_unet(
            model,
            masks,
            save_path=paths.pruned_model,
            seed=seed,
            deterministic=deterministic,
        )

        if reinit_mode == "random":
            print("🔄 Reinitializing pruned model with random weights...")

            before_mean, before_std = _collect_weight_stats(pruned_model)
            print(f"📊 BEFORE reinit: mean={before_mean:.6f}, std={before_std:.6f}")

            pruned_model.apply(_init_weights_kaiming)

            # Save reinitialized weights
            save_path = paths.pruned_model
            os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
            torch.save(pruned_model.state_dict(), save_path)
            print(f"💾 Saved pruned model (random reinit) to {save_path}")

            after_mean, after_std = _collect_weight_stats(pruned_model)
            print(f"📊 AFTER reinit:  mean={after_mean:.6f}, std={after_std:.6f}")

            print("🔍 Reinit verification:")
            print(f"Mean changed by {abs(after_mean - before_mean):.6f}")
            print(f"Std  changed by {abs(after_std - before_std):.6f}")
            print("✅ Reinitialized weights.\n")

            reinit_stats = {
                "mean_before": float(before_mean),
                "std_before": float(before_std),
                "mean_after": float(after_mean),
                "std_after": float(after_std),
            }

            wandb.log(
                {
                    "reinit_mean_before": before_mean,
                    "reinit_std_before": before_std,
                    "reinit_mean_after": after_mean,
                    "reinit_std_after": after_std,
                }
            )
        else:
            print("🔄 Using current weights for rebuild...")

    # ============================================================
    # --- PARAM SUMMARY ---
    # ============================================================
    orig_params = int(sum(p.numel() for p in model.parameters()))
    pruned_params = int(sum(p.numel() for p in pruned_model.parameters()))
    reduction = float(100 * (1 - pruned_params / max(1, orig_params)))

    # ============================================================
    # --- SAVE RUN SUMMARY (shared schema) ---
    # ============================================================
    summary = base_run_info(cfg, stage="prune")

    summary["prune"] = {
        "method": method,
        "threshold": float(pruning_cfg.get("threshold")) if method == "correlation" else None,
        "num_slices": int(pruning_cfg.get("num_slices", 0)) if method == "correlation" else None,
        "batch_size": int(pruning_cfg.get("batch_size", 0)) if method == "correlation" else None,
        "ratios": {
            "default": float(default_ratio),
            "block_ratios": {k: float(v) for k, v in block_ratios.items()},
        },
        "reinitialize_weights": reinit_mode,
        "params": {
            "orig_params": orig_params,
            "pruned_params": pruned_params,
            "reduction_percent": reduction,
        },
        "checkpoints": {
            "baseline_ckpt": str(baseline_ckpt),
            "rewind_ckpt": str(paths.rewind_ckpt) if getattr(paths, "rewind_ckpt", None) else None,
        },
        "artifacts": {
            "pruned_model": str(paths.pruned_model),
            "meta_json": str(paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")),
        },
    }

    if reinit_stats is not None:
        summary["prune"]["random_reinit_stats"] = reinit_stats

    # Save consistent summary name everywhere
    summary_path = write_json(paths.pruned_model_dir / "run_summary.json", summary)
    wandb.save(str(summary_path))

    # Optional: keep a legacy pruning_summary.json (contains only prune payload)
    # legacy_path = write_json(paths.pruned_model_dir / "pruning_summary.json", summary["prune"])
    # wandb.save(str(legacy_path))

    run.finish()
    print(f"💾 Summary saved to {summary_path}")
    print("✅ Pruning complete.\n")


if __name__ == "__main__":
    run_pruning()

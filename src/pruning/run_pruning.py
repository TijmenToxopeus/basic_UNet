# src/pruning/run_pruning.py
from __future__ import annotations

import json
import torch
import wandb

from src.models.unet import UNet
from src.pruning.methods import get_method
from src.pruning.rebuild import rebuild_pruned_unet
from src.pruning.reinit import random_reinitialize, load_rewind_model
from src.pruning.artifacts import compute_param_stats
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.wandb_utils import setup_wandb
from src.utils.reproducibility import seed_everything
from src.utils.run_summary import base_run_info, write_json


def run_pruning(cfg=None):

    # ----------------------------
    # Load config + seed
    # ----------------------------
    if cfg is None:
        cfg, config_path = load_config(return_path=True)
    else:
        config_path = None

    exp_cfg = cfg["experiment"]
    pruning_cfg = cfg["pruning"]
    model_cfg = cfg["train"]["model"]

    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    paths = get_paths(cfg, config_path)

    device = torch.device(exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    print(f"✂️ Starting pruning for {exp_cfg.get('model_name')}")
    print(f"🔁 Seed = {seed} | Deterministic = {deterministic}")

    # ----------------------------
    # W&B
    # ----------------------------
    # run = setup_wandb(cfg, job_type="pruning")

    # ----------------------------
    # Load baseline model
    # ----------------------------
    baseline_ckpt = paths.baseline_ckpt
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"❌ Baseline checkpoint not found: {baseline_ckpt}")

    in_ch = int(model_cfg["in_channels"])
    out_ch = int(model_cfg["out_channels"])
    enc_features = model_cfg["features"]

    baseline_model = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
    baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    baseline_model.eval()

    print(f"📦 Loaded baseline model: {baseline_ckpt}")

    # ----------------------------
    # Select pruning method (l1_norm / similar_feature)
    # ----------------------------
    method_name = pruning_cfg.get("method", "l1_norm")
    pruner = get_method(method_name)
    print(f"✂️ Using pruning method: {method_name}")

    # Compute masks (method is responsible for any method-specific data loading)
    prune_out = pruner.compute_masks(
        model=baseline_model,
        cfg=cfg,
        seed=seed,
        deterministic=deterministic,
        device=device,
    )
    masks = prune_out.masks

    # Optional method logs
    # if "l1_df" in prune_out.extra:
    #     wandb.log({"l1_norms": wandb.Table(dataframe=prune_out.extra["l1_df"])})

    # ----------------------------
    # Rebuild pruned model
    # ----------------------------
    paths.ensure_dir(paths.pruned_model_dir)

    reinit_mode = pruning_cfg.get("reinitialize_weights", None)

    if reinit_mode == "rewind":
        print('Loading rewind weights...')
        source_model = load_rewind_model(
            rewind_ckpt=paths.rewind_ckpt,
            model_ctor=lambda: UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features),
            device=device,
        )
        used_rewind_ckpt = str(paths.rewind_ckpt)
    else:
        source_model = baseline_model
        used_rewind_ckpt = None

    pruned_model = rebuild_pruned_unet(
        source_model,
        masks,
        save_path=paths.pruned_model,
        seed=seed,
        deterministic=deterministic,
    )

    # Random reinit after rebuild
    reinit_stats = None
    if reinit_mode == "random":
        print("🔄 Reinitializing pruned model with random weights...")
        reinit_stats = random_reinitialize(pruned_model)
        torch.save(pruned_model.state_dict(), paths.pruned_model)
        # wandb.log(
        #     {
        #         "reinit_mean_before": reinit_stats.mean_before,
        #         "reinit_std_before": reinit_stats.std_before,
        #         "reinit_mean_after": reinit_stats.mean_after,
        #         "reinit_std_after": reinit_stats.std_after,
        #     }
        # )

    # ----------------------------
    # Param stats
    # ----------------------------
    pstats = compute_param_stats(baseline_model, pruned_model)

    # ----------------------------
    # Summary JSON (standardized)
    # ----------------------------
    summary = base_run_info(cfg, stage="prune")

    # keep method-provided config details if they exist
    block_ratios = prune_out.extra.get("block_ratios", pruning_cfg.get("ratios", {}).get("block_ratios", {}))
    default_ratio = prune_out.extra.get("default_ratio", pruning_cfg.get("ratios", {}).get("default", None))
    threshold = prune_out.extra.get("threshold", pruning_cfg.get("threshold", None))

    meta_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_meta.json")
    resize_log_path = paths.pruned_model.with_name(paths.pruned_model.stem + "_resize_log.json")

    summary["prune"] = {
        "method": prune_out.method,  # should match l1_norm / similar_feature
        "reinitialize_weights": reinit_mode,
        "seed": seed,
        "deterministic": deterministic,
        "ratios": {
            "block_ratios": block_ratios,
            "default": default_ratio,
        },
        "threshold": threshold,
        "params": {
            "original": pstats.original_params,
            "pruned": pstats.pruned_params,
            "reduction_percent": pstats.reduction_percent,
        },
        "checkpoints": {
            "baseline_ckpt": str(baseline_ckpt),
            "rewind_ckpt": used_rewind_ckpt,
        },
        "artifacts": {
            "pruned_model": str(paths.pruned_model),
            "meta_json": str(meta_path),
            "resize_log": str(resize_log_path) if resize_log_path.exists() else None,
        },
    }

    if reinit_stats is not None:
        summary["prune"]["reinit_stats"] = {
            "mean_before": reinit_stats.mean_before,
            "std_before": reinit_stats.std_before,
            "mean_after": reinit_stats.mean_after,
            "std_after": reinit_stats.std_after,
        }

    summary_path = write_json(paths.pruned_model_dir / "run_summary.json", summary)
    # wandb.save(str(summary_path))

    # keep your old pruning_summary.json if you still want it, but optional:
    # write_json(paths.pruned_model_dir / "pruning_summary.json", summary["prune"])

    # run.finish()
    print(f"💾 Saved: {summary_path}")
    print("✅ Pruning complete.")


if __name__ == "__main__":
    run_pruning()

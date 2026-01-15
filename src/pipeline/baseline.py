"""
Pipeline for training and evaluating the baseline UNet model.
This version keeps all configuration changes in memory (does not modify config.yaml on disk).
"""

import os
import yaml
from copy import deepcopy

from src.training.train import train_model
from src.training.eval import evaluate
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.reproducibility import seed_everything


def run_baseline_pipeline():
    print("\n🚀 Starting BASELINE training + evaluation pipeline...\n")

    # ============================================================
    # --- LOAD CONFIG ---
    # ============================================================
    cfg, cfg_path = load_config(return_path=True)
    baseline_cfg = deepcopy(cfg)
    exp_cfg = cfg["experiment"]
    seed = exp_cfg.get("seed", 42)
    deterministic = exp_cfg.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    # ------------------------------------------------------------
    # 1️⃣ Train baseline model
    # ------------------------------------------------------------
    print("\n🏋️ Training baseline model...\n")

    baseline_cfg["train"]["phase"] = "training"
    baseline_cfg["train"]["paths"]["subfolder"] = "baseline"

    train_model(cfg=baseline_cfg)

    # ------------------------------------------------------------
    # 2️⃣ Evaluate baseline model
    # ------------------------------------------------------------
    baseline_cfg["evaluation"]["phase"] = "baseline_evaluation"
    evaluate(cfg=baseline_cfg)

    # ------------------------------------------------------------
    # 3️⃣ Save FINAL config.yaml into the baseline folder
    # ------------------------------------------------------------
    print("\n💾 Saving baseline config.yaml into experiment directory...\n")

    # Build paths using the updated config
    paths = get_paths(baseline_cfg, cfg_path)

    # Path where the baseline model was stored
    baseline_dir = paths.base_dir / "baseline"

    # Create folder if needed
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Write config file
    config_save_path = baseline_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(baseline_cfg, f)

    print(f"📄 Saved baseline config to: {config_save_path}")

    print("\n✅ BASELINE pipeline complete!\n")


if __name__ == "__main__":
    run_baseline_pipeline()

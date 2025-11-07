"""
Pipeline for training and evaluating the baseline UNet model.
This version keeps all configuration changes in memory (does not modify config.yaml on disk).
"""

import os
from copy import deepcopy

from src.training.train import train_model
from src.training.eval import evaluate   # note: consistent import path
from src.utils.config import load_config


def run_baseline_pipeline():
    print("\n🚀 Starting BASELINE training + evaluation pipeline...\n")

    # ============================================================
    # --- LOAD CONFIG ---
    # ============================================================
    cfg, _ = load_config(return_path=True)
    baseline_cfg = deepcopy(cfg)

    # ------------------------------------------------------------
    # 1️⃣ Train baseline model
    # ------------------------------------------------------------
    print("\n🏋️ Training baseline model...\n")

    # modify config in memory (not on disk)
    baseline_cfg["train"]["phase"] = "training"
    baseline_cfg["train"]["paths"]["subfolder"] = "baseline"

    # run training directly with config dict
    train_model(cfg=baseline_cfg)

    # ------------------------------------------------------------
    # 2️⃣ Evaluate baseline model
    # ------------------------------------------------------------
    baseline_cfg["evaluation"]["phase"] = "baseline_evaluation"

    # run evaluation directly with config dict
    evaluate(cfg=baseline_cfg)

    print("\n✅ BASELINE pipeline complete!\n")


if __name__ == "__main__":
    run_baseline_pipeline()

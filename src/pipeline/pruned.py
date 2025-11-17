# """
# Pipeline for pruning, evaluating, retraining, and re-evaluating the pruned UNet.
# This version keeps all configuration changes in memory (does not modify config.yaml on disk).
# """

# import os
# from copy import deepcopy

# from src.pruning.l1_pruning import run_pruning
# from src.training.train import train_model
# from src.training.eval import evaluate
# from src.utils.config import load_config


# def run_pruned_pipeline():
#     print("\n✂️ Starting PRUNED model pipeline...\n")

#     # ============================================================
#     # --- LOAD CONFIG ---
#     # ============================================================
#     cfg, _ = load_config(return_path=True)
#     pruned_cfg = deepcopy(cfg)

#     # ------------------------------------------------------------
#     # 1️⃣ Prune baseline model
#     # ------------------------------------------------------------
#     print("\n✂️ Running pruning step...\n")

#     # run pruning directly with in-memory config
#     run_pruning(cfg=pruned_cfg)

#     # ------------------------------------------------------------
#     # 2️⃣ Evaluate pruned model before retraining
#     # ------------------------------------------------------------
#     print("\n🔍 Evaluating pruned model...\n")

#     pruned_cfg["evaluation"]["phase"] = "pruned_evaluation"

#     evaluate(cfg=pruned_cfg)

#     # ------------------------------------------------------------
#     # 3️⃣ Retrain pruned model
#     # ------------------------------------------------------------
#     print("\n🏋️ Retraining pruned model...\n")

#     pruned_cfg["train"]["phase"] = "retraining"
#     pruned_cfg["train"]["paths"]["subfolder"] = "pruned"

#     train_model(cfg=pruned_cfg)

#     # ------------------------------------------------------------
#     # 4️⃣ Evaluate retrained pruned model
#     # ------------------------------------------------------------
#     print("\n🔍 Evaluating retrained pruned model...\n")

#     pruned_cfg["evaluation"]["phase"] = "retrained_pruned_evaluation"

#     evaluate(cfg=pruned_cfg)

#     print("\n✅ PRUNED pipeline complete!\n")


# if __name__ == "__main__":
#     run_pruned_pipeline()

"""
Pipeline for pruning, evaluating, retraining, and re-evaluating the pruned UNet.
This version keeps all configuration changes in memory (does not modify config.yaml on disk).
"""

import os
import yaml
from copy import deepcopy

from src.pruning.l1_pruning import run_pruning
from src.training.train import train_model
from src.training.eval import evaluate
from src.utils.config import load_config
from src.utils.paths import get_paths


def run_pruned_pipeline():
    print("\n✂️ Starting PRUNED model pipeline...\n")

    # ============================================================
    # --- LOAD CONFIG ---
    # ============================================================
    cfg, cfg_path = load_config(return_path=True)
    pruned_cfg = deepcopy(cfg)

    # ------------------------------------------------------------
    # 1️⃣ Prune baseline model
    # ------------------------------------------------------------
    print("\n✂️ Running pruning step...\n")
    run_pruning(cfg=pruned_cfg)

    # ------------------------------------------------------------
    # 2️⃣ Evaluate pruned model before retraining
    # ------------------------------------------------------------
    print("\n🔍 Evaluating pruned model...\n")
    pruned_cfg["evaluation"]["phase"] = "pruned_evaluation"
    evaluate(cfg=pruned_cfg)

    # ------------------------------------------------------------
    # 3️⃣ Retrain pruned model
    # ------------------------------------------------------------
    print("\n🏋️ Retraining pruned model...\n")
    pruned_cfg["train"]["phase"] = "retraining"
    pruned_cfg["train"]["paths"]["subfolder"] = "pruned"
    train_model(cfg=pruned_cfg)

    # ------------------------------------------------------------
    # 4️⃣ Evaluate retrained pruned model
    # ------------------------------------------------------------
    print("\n🔍 Evaluating retrained pruned model...\n")
    pruned_cfg["evaluation"]["phase"] = "retrained_pruned_evaluation"
    evaluate(cfg=pruned_cfg)

    # ------------------------------------------------------------
    # 5️⃣ SAVE FINAL CONFIG TO PRUNED DIRECTORY  🔥
    # ------------------------------------------------------------
    print("\n💾 Saving pruned config.yaml into experiment directory...\n")

    # Build paths using final in-memory config
    paths = get_paths(pruned_cfg, cfg_path)

    # pruned model base folder (e.g. .../exp24/pruned/...)
    pruned_dir = paths.pruned_model.parent  # pruned/xxx/pruned_model/
    exp_root = pruned_dir.parent            # the folder right above "pruned_model"

    # Ensure folder exists
    exp_root.mkdir(parents=True, exist_ok=True)

    # Save config
    config_save_path = exp_root / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(pruned_cfg, f)

    print(f"📄 Saved pruned config to: {config_save_path}")

    print("\n✅ PRUNED pipeline complete!\n")


if __name__ == "__main__":
    run_pruned_pipeline()

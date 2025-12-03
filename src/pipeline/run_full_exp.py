# import subprocess
# import yaml
# import shutil
# import os

# CONFIG_PATH = "/media/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# def run(cmd):
#     print(f"\n🚀 Running: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"❌ Command failed: {cmd}")


# def load_config(path=CONFIG_PATH):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_config(cfg, path=CONFIG_PATH):
#     with open(path, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)


# def run_baseline():
#     print("\n===============================")
#     print("   🧱 RUNNING BASELINE")
#     print("===============================\n")

#     # Run baseline WITHOUT touching config.yaml
#     run("python -m src.pipeline.baseline")


# def run_pruned(mode):
#     print("\n===============================")
#     print(f"   ✂️ RUNNING PRUNED ({mode})")
#     print("===============================\n")

#     # --- Load ORIGINAL config from backup, not modified one ---
#     cfg = load_config(BACKUP_PATH)

#     # --- Modify in memory ---
#     cfg["pruning"]["reinitialize_weights"] = (None if mode == "none" else mode)
#     cfg["train"]["parameters"]["learning_rate"] = 2e-3
#     cfg["train"]["parameters"]["num_epochs"] = 15

#     # --- Save modified config temporarily ---
#     save_config(cfg)

#     # --- Run pruning with modified config ---
#     run("python -m src.pipeline.pruned")


# def main():

#     # --- Backup original config EXACTLY ---
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     # -------------------------------------
#     # 1) BASELINE RUN (config untouched)
#     # -------------------------------------
#     run_baseline()

#     # -------------------------------------
#     # 2) PRUNED RUNS (config modified TEMPORARILY)
#     # -------------------------------------
#     run_pruned("none")
#     run_pruned("random")
#     run_pruned("rewind")

#     # --- Restore EXACT original config (byte-for-byte) ---
#     shutil.copy(BACKUP_PATH, CONFIG_PATH)
#     print("\n🔄 Restored original config.yaml")

#     print("\n🎉 FULL EXPERIMENT COMPLETED\n")


# if __name__ == "__main__":
#     main()


# import subprocess
# import yaml
# import shutil
# import os

# CONFIG_PATH = "/media/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# # The different pruning ratios to sweep for encoders.4
# PRUNING_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50]

# # The weight initialization modes for pruning
# PRUNE_MODES = ["none", "random", "rewind"]


# def run(cmd):
#     """Utility to run shell commands."""
#     print(f"\n🚀 Running: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"❌ Command failed: {cmd}")


# def load_config(path=CONFIG_PATH):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_config(cfg, path=CONFIG_PATH):
#     with open(path, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)


# def run_baseline():
#     print("\n===============================")
#     print("   🧱 RUNNING BASELINE")
#     print("===============================\n")

#     # Run baseline WITHOUT touching config.yaml
#     run("python -m src.pipeline.baseline")


# def run_pruned(mode, enc4_ratio):
#     """
#     mode:  "none", "random", or "rewind"
#     enc4_ratio: pruning ratio for encoders.4 block
#     """
#     print("\n===============================")
#     print(f"   ✂️ RUNNING PRUNED: mode={mode}, encoders.4={enc4_ratio}")
#     print("===============================\n")

#     # --- Load ORIGINAL config from backup ---
#     cfg = load_config(BACKUP_PATH)

#     # --- Modify pruning settings ---
#     cfg["pruning"]["reinitialize_weights"] = (None if mode == "none" else mode)

#     # Override only encoders.4 ratio — keep others untouched
#     if "block_ratios" not in cfg["pruning"]:
#         cfg["pruning"]["block_ratios"] = {}

#     cfg["pruning"]["block_ratios"]["encoders.4"] = float(enc4_ratio)

#     # --- Training hyperparameters for pruning ---
#     cfg["train"]["parameters"]["num_epochs"] = 30

#     # --- Save modified config temporarily ---
#     save_config(cfg)

#     # --- Run pruning with modified config ---
#     run("python -m src.pipeline.pruned")


# def main():

#     # --- Backup original config EXACTLY ---
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     # -------------------------------------
#     # 1) BASELINE RUN (only once)
#     # -------------------------------------
#     run_baseline()

#     # -------------------------------------
#     # 2) SWEEP OVER PRUNING RATIOS
#     # -------------------------------------
#     for ratio in PRUNING_RATIOS:
#         print(f"\n🔄 Starting pruning-ratio sweep: encoders.4 = {ratio}\n")

#         for mode in PRUNE_MODES:
#             run_pruned(mode, ratio)

#     # --- Restore EXACT original config ---
#     shutil.copy(BACKUP_PATH, CONFIG_PATH)
#     print("\n🔄 Restored original config.yaml")

#     print("\n🎉 FULL EXPERIMENT COMPLETED\n")


# if __name__ == "__main__":
#     main()


import subprocess
import yaml
import shutil
import os

CONFIG_PATH = "/media/ttoxopeus/basic_UNet/src/config.yaml"
BACKUP_PATH = CONFIG_PATH + ".backup"

# Only sweep this block
TARGET_BLOCK = "decoders.1"

# Full list of blocks (must stay for resetting)
ALL_BLOCKS = [
    "encoders.0",
    "encoders.1",
    "encoders.2",
    "encoders.3",
    "encoders.4",
    "bottleneck",
    "decoders.1",
    "decoders.3",
    "decoders.5",
    "decoders.7",
    "decoders.9",
]

# Ratios to test
RATIOS = [0.01, 0.05, 0.1, 0,2]

# Weight initialization modes
INIT_MODES = ["none", "random", "rewind"]


def run(cmd):
    print(f"\n🚀 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Command failed: {cmd}")


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def sweep_pruned(block_name, ratio, init_mode):
    """
    Runs the ENTIRE pruned pipeline (prune → eval → retrain → eval)
    using python -m src.pipeline.pruned, with updated config.
    """

    print("\n===============================")
    print(f"🔥 FULL PRUNED PIPELINE | block={block_name} | ratio={ratio} | init={init_mode}")
    print("===============================\n")

    # Load fresh original config
    cfg = load_config(BACKUP_PATH)

    # Reset ALL blocks to 0.0
    for blk in ALL_BLOCKS:
        cfg["pruning"]["ratios"]["block_ratios"][blk] = 0.0

    # Apply only the selected pruning on the target block
    cfg["pruning"]["ratios"]["block_ratios"][block_name] = float(ratio)

    # Apply reinitialization mode
    cfg["pruning"]["reinitialize_weights"] = (
        None if init_mode == "none" else init_mode
    )

    # Save config.yaml
    save_config(cfg)

    # Run pruning pipeline
    run("python -m src.pipeline.pruned")


def main():

    # Backup original config
    shutil.copy(CONFIG_PATH, BACKUP_PATH)
    print("📂 Backed up original config.yaml → config.yaml.backup")

    # Sweep ONLY decoders.1
    for ratio in RATIOS:
        for mode in INIT_MODES:
            sweep_pruned(TARGET_BLOCK, ratio, mode)

    # Restore the original config at the end
    shutil.copy(BACKUP_PATH, CONFIG_PATH)
    print("\n🔄 Restored original config.yaml")

    print("\n🎉 FULL PRUNING SWEEP COMPLETED\n")


if __name__ == "__main__":
    main()

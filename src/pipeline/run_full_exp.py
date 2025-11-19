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


import subprocess
import yaml
import shutil
import os

CONFIG_PATH = "/media/ttoxopeus/basic_UNet/src/config.yaml"
BACKUP_PATH = CONFIG_PATH + ".backup"

# The different pruning ratios to sweep for encoders.4
PRUNING_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50]

# The weight initialization modes for pruning
PRUNE_MODES = ["none", "random", "rewind"]


def run(cmd):
    """Utility to run shell commands."""
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


def run_baseline():
    print("\n===============================")
    print("   🧱 RUNNING BASELINE")
    print("===============================\n")

    # Run baseline WITHOUT touching config.yaml
    run("python -m src.pipeline.baseline")


def run_pruned(mode, enc4_ratio):
    """
    mode:  "none", "random", or "rewind"
    enc4_ratio: pruning ratio for encoders.4 block
    """
    print("\n===============================")
    print(f"   ✂️ RUNNING PRUNED: mode={mode}, encoders.4={enc4_ratio}")
    print("===============================\n")

    # --- Load ORIGINAL config from backup ---
    cfg = load_config(BACKUP_PATH)

    # --- Modify pruning settings ---
    cfg["pruning"]["reinitialize_weights"] = (None if mode == "none" else mode)

    # Override only encoders.4 ratio — keep others untouched
    if "block_ratios" not in cfg["pruning"]:
        cfg["pruning"]["block_ratios"] = {}

    cfg["pruning"]["block_ratios"]["encoders.4"] = float(enc4_ratio)

    # --- Training hyperparameters for pruning ---
    cfg["train"]["parameters"]["num_epochs"] = 30

    # --- Save modified config temporarily ---
    save_config(cfg)

    # --- Run pruning with modified config ---
    run("python -m src.pipeline.pruned")


def main():

    # --- Backup original config EXACTLY ---
    shutil.copy(CONFIG_PATH, BACKUP_PATH)
    print("📂 Backed up original config.yaml → config.yaml.backup")

    # -------------------------------------
    # 1) BASELINE RUN (only once)
    # -------------------------------------
    run_baseline()

    # -------------------------------------
    # 2) SWEEP OVER PRUNING RATIOS
    # -------------------------------------
    for ratio in PRUNING_RATIOS:
        print(f"\n🔄 Starting pruning-ratio sweep: encoders.4 = {ratio}\n")

        for mode in PRUNE_MODES:
            run_pruned(mode, ratio)

    # --- Restore EXACT original config ---
    shutil.copy(BACKUP_PATH, CONFIG_PATH)
    print("\n🔄 Restored original config.yaml")

    print("\n🎉 FULL EXPERIMENT COMPLETED\n")


if __name__ == "__main__":
    main()

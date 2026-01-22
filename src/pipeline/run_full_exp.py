## different ratios and reinit sweep for one layer

# import subprocess
# import yaml
# import shutil
# import os

# CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# # Only sweep this block
# TARGET_BLOCK = "decoders.1"

# # Full list of blocks (must stay for resetting)
# ALL_BLOCKS = [
#     "encoders.0",
#     "encoders.1",
#     "encoders.2",
#     "encoders.3",
#     "encoders.4",
#     "bottleneck",
#     "decoders.1",
#     "decoders.3",
#     "decoders.5",
#     "decoders.7",
#     "decoders.9",
# ]

# # Ratios to test
# RATIOS = [0.01, 0.05, 0.1, 0.2]

# # Weight initialization modes
# INIT_MODES = ["none", "random", "rewind"]


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


# def sweep_pruned(block_name, ratio, init_mode):
#     """
#     Runs the ENTIRE pruned pipeline (prune → eval → retrain → eval)
#     using python -m src.pipeline.pruned, with updated config.
#     """

#     print("\n===============================")
#     print(f"🔥 FULL PRUNED PIPELINE | block={block_name} | ratio={ratio} | init={init_mode}")
#     print("===============================\n")

#     # Load fresh original config
#     cfg = load_config(BACKUP_PATH)

#     # Reset ALL blocks to 0.0
#     for blk in ALL_BLOCKS:
#         cfg["pruning"]["ratios"]["block_ratios"][blk] = 0.0

#     # Apply only the selected pruning on the target block
#     cfg["pruning"]["ratios"]["block_ratios"][block_name] = float(ratio)

#     # Apply reinitialization mode
#     cfg["pruning"]["reinitialize_weights"] = (
#         None if init_mode == "none" else init_mode
#     )

#     # Save config.yaml
#     save_config(cfg)

#     # Run pruning pipeline
#     run("python -m src.pipeline.pruned")


# def main():

#     # Backup original config
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     # Sweep ONLY decoders.1
#     for ratio in RATIOS:
#         for mode in INIT_MODES:
#             sweep_pruned(TARGET_BLOCK, ratio, mode)

#     # Restore the original config at the end
#     shutil.copy(BACKUP_PATH, CONFIG_PATH)
#     print("\n🔄 Restored original config.yaml")

#     print("\n🎉 FULL PRUNING SWEEP COMPLETED\n")


# if __name__ == "__main__":
#     main()


# import subprocess
# import yaml
# import shutil
# import os

# CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# # Layers to sweep
# ALL_BLOCKS = [
#     "encoders.0",
#     "encoders.1",
#     "encoders.2",
#     "encoders.3",
#     "encoders.4",
#     "bottleneck",
#     "decoders.1",
#     "decoders.3",
#     "decoders.5",
#     "decoders.7",
#     "decoders.9",
# ]

# # Pruning ratios to test
# RATIOS = [0.2, 0.4, 0.6, 0.8, 0.9]   # you can adjust this


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


# def run_pruning_experiment(layer, ratio):
#     """
#     Run the pruned model pipeline for *one layer and one ratio*.
#     """

#     print("\n===============================")
#     print(f"🔥 Pruning Layer = {layer} | Ratio = {ratio}")
#     print("===============================\n")

#     # Load clean config
#     cfg = load_config(BACKUP_PATH)

#     # Reset ALL pruning ratios to 0
#     for blk in ALL_BLOCKS:
#         cfg["pruning"]["ratios"]["block_ratios"][blk] = 0.0

#     # Apply pruning only to selected layer
#     cfg["pruning"]["ratios"]["block_ratios"][layer] = float(ratio)

#     # Disable weight reinitialization for stability
#     cfg["pruning"]["reinitialize_weights"] = None

#     # Save modified config
#     save_config(cfg)

#     # Run pipeline
#     run("python -m src.pipeline.pruned")

#     print(f"✅ Completed: {layer} @ {ratio}\n")


# def main():

#     # Backup original config
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     # Sweep all layers × ratios
#     for layer in ALL_BLOCKS:
#         for ratio in RATIOS:

#             # Optionally skip ratio=0 for plots (kept here for baseline)
#             run_pruning_experiment(layer, ratio)

#     # Restore final config
#     shutil.copy(BACKUP_PATH, CONFIG_PATH)
#     print("\n🔄 Restored original config.yaml")

#     print("\n🎉 FULL LAYER × RATIO PRUNING SWEEP COMPLETED\n")


# if __name__ == "__main__":
#     main()



# ### GLOBAL PRUNING SWEEP

# import subprocess
# import yaml
# import shutil
# import os


# CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# # Blocks to prune (all get same ratio)
# ALL_BLOCKS = [
#     "encoders.0",
#     "encoders.1",
#     "encoders.2",
#     "encoders.3",
#     "encoders.4",
#     "bottleneck",
#     "decoders.1",
#     "decoders.3",
#     "decoders.5",
#     "decoders.7",
#     "decoders.9",
# ]

# # Global pruning ratios to test
# RATIOS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


# def run(cmd):
#     print(f"\n🚀 Running: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"❌ Command failed: {cmd}")


# def load_config(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_config(cfg, path=CONFIG_PATH):
#     with open(path, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)


# def run_pruning_experiment(ratio):
#     """
#     Run the pruned model pipeline with the SAME prune ratio
#     applied to ALL blocks.
#     """

#     print("\n===============================")
#     print(f"🔥 Global pruning | Ratio = {ratio}")
#     print("===============================\n")

#     # Load clean config
#     cfg = load_config(BACKUP_PATH)

#     # Apply same pruning ratio to all blocks
#     for blk in ALL_BLOCKS:
#         cfg["pruning"]["ratios"]["block_ratios"][blk] = float(ratio)

#     # Disable weight reinitialization
#     cfg["pruning"]["reinitialize_weights"] = None

#     # Save modified config
#     save_config(cfg)

#     # Run pipeline
#     run("python -m src.pipeline.pruned")

#     print(f"✅ Completed global pruning @ ratio {ratio}\n")


# def main():

#     # Backup original config
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     # Sweep over global pruning ratios
#     for ratio in RATIOS:
#         run_pruning_experiment(ratio)

#     # Restore original config
#     shutil.copy(BACKUP_PATH, CONFIG_PATH)
#     print("\n🔄 Restored original config.yaml")

#     print("\n🎉 GLOBAL PRUNING RATIO SWEEP COMPLETED\n")


# if __name__ == "__main__":
#     main()




## PRUNING ENCODER AND DECODER PARTS SEPERATELY

import subprocess
import yaml
import shutil
import os

CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
BACKUP_PATH = CONFIG_PATH + ".backup"

# -------------------------------
# Block definitions
# -------------------------------

ENCODER_BLOCKS = [
    "encoders.0",
    "encoders.1",
    "encoders.2",
    "encoders.3",
    "encoders.4",
]

DEEP_ENCODERS = [
    "encoders.3",
    "encoders.4",
]

BOTTLENECK = ["bottleneck"]

DEEP_DECODERS = [
    "decoders.1",
    "decoders.3",
]

DECODER_BLOCKS = [
    "decoders.1",
    "decoders.3",
    "decoders.5",
    "decoders.7",
    "decoders.9",
]

ALL_BLOCKS = (
    ENCODER_BLOCKS
    + BOTTLENECK
    + DECODER_BLOCKS
)

# Global pruning ratios
RATIOS = [
    0.00, 0.05, 0.1, 0.15, 0.2,
    0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7,
    0.75, 0.8, 0.85, 0.9, 0.95, 0.99
]


# -------------------------------
# Helpers
# -------------------------------

def run(cmd):
    print(f"\n🚀 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Command failed: {cmd}")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def set_block_ratios(cfg, active_blocks, ratio):
    """
    Set selected blocks to `ratio`, all others to 0.0
    """
    for blk in ALL_BLOCKS:
        cfg["pruning"]["ratios"]["block_ratios"][blk] = (
            float(ratio) if blk in active_blocks else 0.0
        )


def run_sweep(name, active_blocks):
    print("\n" + "=" * 60)
    print(f"🔥 STARTING SWEEP: {name}")
    print("=" * 60 + "\n")

    for ratio in RATIOS:
        print(f"\n➡️  {name} | Ratio = {ratio}")

        cfg = load_config(BACKUP_PATH)

        set_block_ratios(cfg, active_blocks, ratio)

        # Disable weight reinitialization
        cfg["pruning"]["reinitialize_weights"] = None

        save_config(cfg)

        run("python -m src.pipeline.pruned")

        print(f"✅ Completed {name} @ ratio {ratio}\n")


# -------------------------------
# Main
# -------------------------------

def main():
    shutil.copy(CONFIG_PATH, BACKUP_PATH)
    print("📂 Backed up original config.yaml → config.yaml.backup")

    # 1️⃣ Encoder-only pruning
    run_sweep(
        name="ENCODER_ONLY",
        active_blocks=ENCODER_BLOCKS,
    )

    # 2️⃣ Deep core pruning (deep encoders + bottleneck + deep decoders)
    run_sweep(
        name="DEEP_CORE",
        active_blocks=DEEP_ENCODERS + BOTTLENECK + DEEP_DECODERS,
    )

    # 3️⃣ Decoder-only pruning
    run_sweep(
        name="DECODER_ONLY",
        active_blocks=DECODER_BLOCKS,
    )

    shutil.copy(BACKUP_PATH, CONFIG_PATH)
    print("\n🔄 Restored original config.yaml")
    print("\n🎉 ALL PRUNING SWEEPS COMPLETED\n")


if __name__ == "__main__":
    main()


# ### CORRELATION PRUNING FOR DIFFERENT TRESHOLDS USES BLOCK RATIOS DEFINED IN THE CONFIG

# ### SIMILAR-FEATURE (CORRELATION) THRESHOLD SWEEP
# # Keeps your block_ratios EXACTLY as in config (only middle layers pruned)

# import subprocess
# import yaml
# import shutil

# CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# THRESHOLDS = [.99, .98, .96, .94, 0.92, 0.88, 0.84, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2]

# # Your similarity-based method name (based on your config)
# SIM_METHOD = "correlation"


# def run(cmd: str):
#     print(f"\n🚀 Running: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"❌ Command failed: {cmd}")


# def load_config(path: str):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_config(cfg, path: str = CONFIG_PATH):
#     with open(path, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)


# def run_similarity_experiment(threshold: float):
#     print("\n===============================")
#     print(f"🔥 Similar-feature pruning | method={SIM_METHOD} | threshold={threshold}")
#     print("===============================\n")

#     # Load clean config (keeps your middle-layer block_ratios unchanged)
#     cfg = load_config(BACKUP_PATH)

#     # Set similarity pruning method + threshold
#     cfg["pruning"]["method"] = SIM_METHOD
#     cfg["pruning"]["threshold"] = float(threshold)

#     # Ensure no re-init (optional, but explicit)
#     cfg["pruning"]["reinitialize_weights"] = None

#     # (IMPORTANT) Do NOT touch block_ratios here.
#     # Your backup config already contains the correct "middle layers = 0.99" setup.

#     save_config(cfg)

#     run("python -m src.pipeline.pruned")
#     print(f"✅ Completed similarity pruning @ threshold {threshold}\n")


# def main():
#     # Backup original config once
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     for thr in THRESHOLDS:
#         run_similarity_experiment(thr)

#     # Restore
#     shutil.copy(BACKUP_PATH, CONFIG_PATH)
#     print("\n🔄 Restored original config.yaml")
#     print("\n🎉 THRESHOLD SWEEP COMPLETED\n")


# if __name__ == "__main__":
#     main()



# ### L1NORM PRUNING, TEST DIFFERENT REINIT MODES, USES CURRENT BLOCK RATIOS

# import subprocess
# import yaml
# import shutil

# CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# # Experiment settings
# BASELINE_EPOCHS = 120
# PRUNE_METHOD = "l1_norm"

# # Try each weight reinit mode
# # None  -> keep weights
# # "random" -> random reinit after rebuild
# # "rewind" -> rebuild using rewind checkpoint weights
# REINIT_MODES = [None, "random", "rewind"]


# def run(cmd: str):
#     print(f"\n🚀 Running: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"❌ Command failed: {cmd}")


# def load_config(path: str):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_config(cfg, path: str = CONFIG_PATH):
#     with open(path, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)


# def run_baseline_training():
#     print("\n===============================")
#     print(f"🏋️  Baseline training | epochs={BASELINE_EPOCHS}")
#     print("===============================\n")

#     cfg = load_config(BACKUP_PATH)

#     # Train baseline for BASELINE_EPOCHS
#     cfg["train"]["phase"] = "training"
#     cfg["train"]["parameters"]["num_epochs"] = int(BASELINE_EPOCHS)

#     # Make sure evaluation phase won't break anything (optional)
#     cfg.setdefault("evaluation", {})
#     cfg["evaluation"]["phase"] = "baseline_evaluation"

#     save_config(cfg)
#     run("python -m src.pipeline.baseline")
#     print("✅ Completed baseline training\n")


# def run_pruned_experiment(reinit_mode):
#     print("\n===============================")
#     print(f"✂️  L1 pruning | method={PRUNE_METHOD} | reinit={reinit_mode}")
#     print("===============================\n")

#     # Always start from clean backup config (keeps your block_ratios unchanged)
#     cfg = load_config(BACKUP_PATH)

#     # Ensure pruning is L1 norm
#     cfg["pruning"]["method"] = PRUNE_METHOD

#     # Set reinit mode
#     cfg["pruning"]["reinitialize_weights"] = reinit_mode

#     # IMPORTANT: do not touch cfg["pruning"]["ratios"]["block_ratios"]
#     # Your backup config already contains the ratios you want.

#     # For rewind: ensure the training pipeline actually saved a rewind checkpoint.
#     # Your paths logic auto-detects epoch*.pth if rewind_ckpt is not set.
#     # So make sure your baseline training produced at least one epoch_*.pth.
#     # If your save_interval is > BASELINE_EPOCHS, rewind won't exist.
#     # (You can set cfg["train"]["parameters"]["save_interval"] accordingly in the backup config.)

#     # Run full pruned pipeline (prune -> eval -> retrain -> eval)
#     cfg["train"]["phase"] = "retraining"
#     cfg["train"]["parameters"]["num_epochs"] = 25
#     cfg["evaluation"]["phase"] = "pruned_evaluation"

#     save_config(cfg)
#     run("python -m src.pipeline.pruned")
#     print(f"✅ Completed pruning run | reinit={reinit_mode}\n")


# def main():
#     # Backup original config once
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     try:
#         # 1) Train baseline once
#         run_baseline_training()

#         # 2) Sweep reinit modes for L1 pruning
#         for mode in REINIT_MODES:
#             run_pruned_experiment(mode)

#     finally:
#         # Always restore original config
#         shutil.copy(BACKUP_PATH, CONFIG_PATH)
#         print("\n🔄 Restored original config.yaml")
#         print("\n🎉 REINIT SWEEP COMPLETED\n")


# if __name__ == "__main__":
#     main()



# ### L1NORM PRUNING, TEST DIFFERENT REINIT MODES, USES CURRENT BLOCK RATIOS
# ### (NO BASELINE TRAINING — assumes baseline artifacts already exist)

# import subprocess
# import yaml
# import shutil

# CONFIG_PATH = "/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml"
# BACKUP_PATH = CONFIG_PATH + ".backup"

# # Experiment settings
# PRUNE_METHOD = "l1_norm"

# # Try each weight reinit mode
# # None     -> keep weights
# # "random" -> random reinit after rebuild
# # "rewind" -> rebuild using rewind checkpoint weights
# REINIT_MODES = [None, "random", "rewind"]


# def run(cmd: str):
#     print(f"\n🚀 Running: {cmd}\n")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"❌ Command failed: {cmd}")


# def load_config(path: str):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def save_config(cfg, path: str = CONFIG_PATH):
#     with open(path, "w") as f:
#         yaml.dump(cfg, f, sort_keys=False)


# def run_pruned_experiment(reinit_mode):
#     print("\n===============================")
#     print(f"✂️  L1 pruning | method={PRUNE_METHOD} | reinit={reinit_mode}")
#     print("===============================\n")

#     # Always start from clean backup config (keeps your block_ratios unchanged)
#     cfg = load_config(BACKUP_PATH)

#     # Ensure pruning is L1 norm
#     cfg["pruning"]["method"] = PRUNE_METHOD

#     # Set reinit mode
#     cfg["pruning"]["reinitialize_weights"] = reinit_mode

#     # IMPORTANT: do not touch cfg["pruning"]["ratios"]["block_ratios"]
#     # The backup config already contains the block ratios to use.

#     # Run full pruned pipeline (prune -> eval -> retrain -> eval)
#     cfg["train"]["phase"] = "retraining"
#     cfg["train"]["parameters"]["num_epochs"] = 25
#     cfg.setdefault("evaluation", {})
#     cfg["evaluation"]["phase"] = "pruned_evaluation"

#     save_config(cfg)
#     run("python -m src.pipeline.pruned")
#     print(f"✅ Completed pruning run | reinit={reinit_mode}\n")


# def main():
#     # Backup original config once
#     shutil.copy(CONFIG_PATH, BACKUP_PATH)
#     print("📂 Backed up original config.yaml → config.yaml.backup")

#     try:
#         # Sweep reinit modes for L1 pruning
#         for mode in REINIT_MODES:
#             run_pruned_experiment(mode)

#     finally:
#         # Always restore original config
#         shutil.copy(BACKUP_PATH, CONFIG_PATH)
#         print("\n🔄 Restored original config.yaml")
#         print("\n🎉 REINIT SWEEP COMPLETED\n")


# if __name__ == "__main__":
#     main()

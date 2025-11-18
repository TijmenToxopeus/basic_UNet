import subprocess
import yaml
import shutil
import os

CONFIG_PATH = "/media/ttoxopeus/basic_UNet/src/config.yaml"


def run(cmd):
    """Utility to run shell commands."""
    print(f"\n🚀 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Command failed: {cmd}")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def run_baseline():
    print("\n===============================")
    print("   🧱 RUNNING BASELINE")
    print("===============================\n")
    run("python -m src.pipeline.baseline")    # calls train + eval (if your baseline pipeline does both)


def run_pruned(mode):
    """
    mode ∈ ["none", "random", "rewind"]
    """
    print("\n===============================")
    print(f"   ✂️ RUNNING PRUNED ({mode})")
    print("===============================\n")

    cfg = load_config()

    # --- 1. Set pruning mode ---
    if mode == "none":
        cfg["pruning"]["reinitialize_weights"] = None
    else:
        cfg["pruning"]["reinitialize_weights"] = mode

    # --- 2. Override learning rate + epochs ---
    cfg["train"]["parameters"]["learning_rate"] = 3e-3
    cfg["train"]["parameters"]["num_epochs"] = 30

    # save modified config
    save_config(cfg)

    # --- 3. Run the pruning pipeline ---
    run("python -m src.pipeline.pruned")


def main():

    # --- backup original config ---
    shutil.copy(CONFIG_PATH, CONFIG_PATH + ".backup")
    print("📂 Backed up original config.yaml → config.yaml.backup")

    # ----------------------------
    # 1) BASELINE RUN
    # ----------------------------
    run_baseline()

    # ----------------------------
    # 2) PRUNED RUNS
    # ----------------------------
    run_pruned("none")      # keep weights
    run_pruned("random")    # random init
    run_pruned("rewind")    # rewind init

    # --- restore original config ---
    shutil.copy(CONFIG_PATH + ".backup", CONFIG_PATH)
    print("\n🔄 Restored original config.yaml")

    print("\n🎉 FULL EXPERIMENT COMPLETED\n")


if __name__ == "__main__":
    main()

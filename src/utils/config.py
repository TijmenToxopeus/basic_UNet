import yaml
from pathlib import Path

def load_config(config_path="/mnt/hdd/ttoxopeus/basic_UNet/src/config.yaml", return_path=False):
    """
    Load a single YAML configuration file.
    Always uses the provided absolute path (no defaults or merging).
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if return_path:
        return cfg, str(config_path)
    return cfg

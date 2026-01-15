# src/pipeline/experiments.py
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import yaml

from src.pipeline.baseline import run_baseline_pipeline
from src.pruning.run_pruning import run_pruning
from src.training.train import train_model
from src.training.eval import evaluate
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.reproducibility import seed_everything


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively updates dict d with u (in-place) and returns d."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def save_cfg_variant(cfg: Dict[str, Any], cfg_path, save_dir) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)


def run_full_experiment():
    cfg, cfg_path = load_config(return_path=True)
    base_cfg = deepcopy(cfg)

    exp = base_cfg["experiment"]
    seed = exp.get("seed", 42)
    deterministic = exp.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    # ------------------------------------------------------------
    # 0) Ensure baseline exists (train + baseline eval once)
    # ------------------------------------------------------------
    # You can do a simple existence check:
    paths = get_paths(base_cfg, cfg_path)
    baseline_ckpt = paths.base_dir / "baseline" / "training" / "final_model.pth"

    if not baseline_ckpt.exists():
        run_baseline_pipeline()
    else:
        print(f"✅ Baseline already exists: {baseline_ckpt}")

    # ------------------------------------------------------------
    # 1) Define variants (grid)
    # ------------------------------------------------------------
    variants: List[Dict[str, Any]] = [
        {
            "tag": "l1_strong",
            "patch": {
                "pruning": {
                    "method": "l1_norm",
                    "reinitialize_weights": None,
                    "ratios": {"block_ratios": {"encoders.3": 0.9, "encoders.4": 0.9, "bottleneck": 0.9}}
                },
                "train": {"phase": "retraining"},
                "evaluation": {"phase": "pruned_evaluation"},
            },
        },
        {
            "tag": "sf_t90",
            "patch": {
                "pruning": {
                    "method": "similar_feature",
                    "threshold": 0.90,
                    "reinitialize_weights": None,
                },
                "train": {"phase": "retraining"},
            },
        },
        {
            "tag": "l1_rewind",
            "patch": {
                "pruning": {
                    "method": "l1_norm",
                    "reinitialize_weights": "rewind",
                },
                "train": {"phase": "retraining"},
            },
        },
    ]

    # ------------------------------------------------------------
    # 2) Run variants
    # ------------------------------------------------------------
    for v in variants:
        tag = v["tag"]
        cfg_v = deepcopy(base_cfg)
        deep_update(cfg_v, v["patch"])

        # helpful: store tag in cfg for logging/summaries
        cfg_v.setdefault("experiment", {})
        cfg_v["experiment"]["variant_tag"] = tag

        # --- pruning
        print(f"\n===== VARIANT: {tag} =====")
        run_pruning(cfg=cfg_v)

        # --- eval pruned
        cfg_v["evaluation"]["phase"] = "pruned_evaluation"
        evaluate(cfg=cfg_v)

        # --- retrain pruned
        cfg_v["train"]["phase"] = "retraining"
        train_model(cfg=cfg_v)

        # --- eval retrained
        cfg_v["evaluation"]["phase"] = "retrained_pruned_evaluation"
        evaluate(cfg=cfg_v)

        # --- save cfg into variant folder (use your suffix-based pathing)
        paths_v = get_paths(cfg_v, cfg_path)
        # exp_root = folder above pruned_model/ (same logic you use)
        pruned_dir = paths_v.pruned_model.parent
        exp_root = pruned_dir.parent
        save_cfg_variant(cfg_v, cfg_path, exp_root)
        print(f"📄 Saved variant config to: {exp_root / 'config.yaml'}")


if __name__ == "__main__":
    run_full_experiment()

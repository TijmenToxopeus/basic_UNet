from __future__ import annotations

from copy import deepcopy

from src.pruning.run_pruning import run_pruning
from src.training.train import train_model
from src.training.eval import evaluate
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.reproducibility import seed_everything


REDUNDANCY_METHODS = ["pearson_correlation", "cosine_similarity"]
UNIFORM_RATIOS = [0.0, 0.20, 0.40, 0.60, 0.80, 0.99]


def _build_uniform_block_ratios(cfg: dict, ratio: float) -> dict:
    base_block_ratios = cfg.get("pruning", {}).get("ratios", {}).get("block_ratios", {})
    if not base_block_ratios:
        raise ValueError("Missing pruning.ratios.block_ratios in config.")
    return {block: float(ratio) for block in base_block_ratios.keys()}


def run_redundancy_threshold_experiment() -> None:
    """
    Compare redundancy pruning methods over a ratio sweep.

    Flow per variant:
      prune -> evaluate pruned -> retrain -> evaluate retrained

    Only modifies:
      - pruning.method
      - pruning.ratios.default
      - pruning.ratios.block_ratios

    Assumes baseline checkpoint already exists for the configured experiment.
    """
    cfg, cfg_path = load_config(return_path=True)
    base_cfg = deepcopy(cfg)

    exp_cfg = base_cfg["experiment"]
    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    base_paths = get_paths(base_cfg, cfg_path)
    if not base_paths.baseline_ckpt.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found. Expected existing baseline at: {base_paths.baseline_ckpt}"
        )

    print("\n🧪 Starting redundancy threshold experiment...")
    print(f"📦 Baseline checkpoint: {base_paths.baseline_ckpt}")
    print(f"📊 Methods: {REDUNDANCY_METHODS}")
    print(f"🎚️ Threshold from config: {base_cfg['pruning']['threshold']}\n")
    print(f"📉 Uniform ratios: {UNIFORM_RATIOS}\n")

    for method in REDUNDANCY_METHODS:
        for ratio in UNIFORM_RATIOS:
            run_cfg = deepcopy(base_cfg)

            run_cfg["pruning"]["method"] = method
            run_cfg["pruning"]["ratios"]["default"] = float(ratio)
            run_cfg["pruning"]["ratios"]["block_ratios"] = _build_uniform_block_ratios(run_cfg, ratio)

            variant_paths = get_paths(run_cfg, cfg_path)
            print("=" * 88)
            print(
                f"▶ Variant: method={method} | threshold={run_cfg['pruning']['threshold']:.2f} | "
                f"uniform_ratio={ratio:.2f}"
            )
            print(f"📁 Variant suffix: {variant_paths.suffix}")
            print("=" * 88)

            # 1) Prune
            run_pruning(cfg=run_cfg)

            # 2) Evaluate pruned
            run_cfg["evaluation"]["phase"] = "pruned_evaluation"
            evaluate(cfg=run_cfg)

            # 3) Retrain pruned
            run_cfg["train"]["phase"] = "retraining"
            train_model(cfg=run_cfg)

            # 4) Evaluate retrained pruned
            run_cfg["evaluation"]["phase"] = "retrained_pruned_evaluation"
            evaluate(cfg=run_cfg)

    print("\n✅ Redundancy threshold experiment complete.\n")


if __name__ == "__main__":
    run_redundancy_threshold_experiment()

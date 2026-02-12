"""
Evaluate all existing pruned model variants for one experiment (no pruning/retraining).

This script:
- loads config.yaml
- finds all subfolders under results/<model>/<experiment>/pruned/
- parses each folder suffix back into pruning config fields
- runs evaluation for available checkpoints
"""

from __future__ import annotations

import re
import shutil
from copy import deepcopy
from pathlib import Path

from src.training.eval import evaluate
from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.reproducibility import seed_everything


def _parse_suffix(suffix: str, block_order: list[str]) -> dict:
    """
    Parse a pruned folder suffix created by ExperimentPaths._get_suffix_from_ratios().

    Expected examples:
    - l1_norm_50_50_50_50_50_50_50_50_50_50_50
    - corr_t90_50_50_50_50_50_50_50_50_50_50_50
    - l1_norm_50_..._random
    - l1_norm_50_..._rewind
    """
    mode = None
    if suffix.endswith("_random"):
        mode = "random"
        core = suffix[: -len("_random")]
    elif suffix.endswith("_rewind"):
        mode = "rewind"
        core = suffix[: -len("_rewind")]
    else:
        core = suffix

    m = re.match(r"^(l1_norm|corr_t\d+)_(.+)$", core)
    if not m:
        raise ValueError(f"Unrecognized suffix format: {suffix}")

    method_token, ratios_token = m.groups()

    if method_token == "l1_norm":
        method = "l1_norm"
        threshold = None
    else:
        method = "correlation"
        thr_match = re.match(r"corr_t(\d+)", method_token)
        if not thr_match:
            raise ValueError(f"Could not parse threshold from: {method_token}")
        threshold = int(thr_match.group(1)) / 100.0

    ratio_vals = [int(x) / 100.0 for x in ratios_token.split("_")]
    if len(ratio_vals) != len(block_order):
        raise ValueError(
            f"Ratio count mismatch for {suffix}: "
            f"{len(ratio_vals)} values, expected {len(block_order)}."
        )

    block_ratios = {blk: ratio_vals[i] for i, blk in enumerate(block_order)}

    return {
        "method": method,
        "threshold": threshold,
        "reinitialize_weights": mode,
        "block_ratios": block_ratios,
    }


def _run_eval_if_ckpt_exists(cfg: dict, phase: str, ckpt: Path) -> bool:
    """Run evaluate(cfg) for phase if checkpoint exists, otherwise skip."""
    if not ckpt.exists():
        print(f"⚠️ Skipping {phase}: checkpoint not found: {ckpt}")
        return False

    cfg["evaluation"]["phase"] = phase
    paths = get_paths(cfg)
    pred_dir = paths.eval_save_dir / "predictions"
    if pred_dir.exists():
        for item in pred_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"🧹 Cleared old predictions: {pred_dir}")

    print(f"\n🔍 Evaluating phase={phase}")
    evaluate(cfg=cfg)
    return True


def run_ood_eval_pipeline():
    print("\n🔎 Starting OOD evaluation sweep over pruned models...\n")

    cfg, cfg_path = load_config(return_path=True)
    exp_cfg = cfg["experiment"]

    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    base_paths = get_paths(cfg, cfg_path)
    pruned_root = base_paths.base_dir / "pruned"

    if not pruned_root.exists():
        raise FileNotFoundError(f"Pruned directory not found: {pruned_root}")

    base_block_ratios = cfg.get("pruning", {}).get("ratios", {}).get("block_ratios", {})
    if not base_block_ratios:
        raise ValueError("No pruning.ratios.block_ratios found in config.yaml")
    block_order = list(base_block_ratios.keys())

    variant_dirs = sorted([p for p in pruned_root.iterdir() if p.is_dir()])
    if not variant_dirs:
        print(f"⚠️ No pruned variants found under: {pruned_root}")
        return

    print(f"📂 Found {len(variant_dirs)} pruned variants in: {pruned_root}")

    completed = []
    failed = []

    for idx, vdir in enumerate(variant_dirs, start=1):
        suffix = vdir.name
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(variant_dirs)}] Variant: {suffix}")
        print("=" * 80)

        try:
            parsed = _parse_suffix(suffix, block_order=block_order)
            run_cfg = deepcopy(cfg)

            run_cfg["pruning"]["method"] = parsed["method"]
            if parsed["threshold"] is not None:
                run_cfg["pruning"]["threshold"] = float(parsed["threshold"])
            run_cfg["pruning"]["reinitialize_weights"] = parsed["reinitialize_weights"]
            run_cfg["pruning"]["ratios"]["block_ratios"] = parsed["block_ratios"]

            paths = get_paths(run_cfg, cfg_path)

            ran_pruned = _run_eval_if_ckpt_exists(
                run_cfg,
                phase="pruned_evaluation",
                ckpt=paths.pruned_model,
            )
            ran_retrained = _run_eval_if_ckpt_exists(
                run_cfg,
                phase="retrained_pruned_evaluation",
                ckpt=paths.retrain_pruned_dir / "final_model.pth",
            )

            completed.append(
                {
                    "suffix": suffix,
                    "pruned_eval": ran_pruned,
                    "retrained_eval": ran_retrained,
                }
            )

        except Exception as e:
            failed.append((suffix, str(e)))
            print(f"❌ Failed variant {suffix}: {e}")

    print("\n✅ OOD evaluation sweep finished.")
    print(f"✔️ Completed variants: {len(completed)}")
    print(f"❌ Failed variants: {len(failed)}")

    if failed:
        print("\nFailed list:")
        for suffix, err in failed:
            print(f"- {suffix}: {err}")


if __name__ == "__main__":
    run_ood_eval_pipeline()

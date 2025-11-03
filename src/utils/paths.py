from pathlib import Path
import yaml
import shutil
from datetime import datetime


class ExperimentPaths:
    """
    Central path manager for UNet pruning experiments.

    - Builds consistent folder structures for training, evaluation, pruning.
    - Saves config snapshot once per experiment.
    - Avoids pre-creating folders for unused phases (e.g., pruned).
    """

    def __init__(self, cfg, config_file_path=None):
        self.cfg = cfg
        self.config_file_path = config_file_path
        exp_cfg = cfg["experiment"]

        # --- Base experiment info ---
        self.model_name = exp_cfg["model_name"]
        self.exp_name = exp_cfg["experiment_name"]
        self.device = exp_cfg.get("device", "cuda")

        # --- Root results directory ---
        self.results_root = Path(cfg["train"]["paths"]["save_root"]).resolve()
        self.base_dir = self.results_root / self.model_name / self.exp_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # --- Generate pruning suffix (e.g., 0_0_10_20_30_30_30_20_10) ---
        self.suffix = self._get_suffix_from_ratios(cfg)

        # ============================================================
        # TRAINING PATHS
        # ============================================================
        train_cfg = cfg.get("train", {})
        train_paths = train_cfg.get("paths", {})

        self.train_dir = Path(train_paths.get("train_dir", ""))
        self.label_dir = Path(train_paths.get("label_dir", ""))

        train_subfolder = train_paths.get("subfolder", "baseline")
        if "retrain" in train_cfg.get("phase", "").lower():
            train_subfolder = f"{train_subfolder}/{self.suffix}"

        # Include training phase directory
        self.train_save_dir = self.base_dir / train_subfolder / "training"
        self.train_save_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # EVALUATION PATHS
        # ============================================================
        eval_cfg = cfg.get("evaluation", {})
        eval_paths = eval_cfg.get("paths", {})

        self.eval_dir = Path(eval_paths.get("eval_dir", ""))
        self.eval_label_dir = Path(eval_paths.get("label_dir", ""))

        eval_subfolder = eval_paths.get("subfolder", "baseline")
        if "pruned" in eval_subfolder or "retrain" in eval_cfg.get("phase", "").lower():
            eval_subfolder = f"{eval_subfolder}/{self.suffix}"

        # Include evaluation phase directory
        self.eval_save_dir = self.base_dir / eval_subfolder / "evaluation"
        self.eval_save_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # PRUNING PATHS
        # ============================================================
        prune_cfg = cfg.get("pruning", {})
        ckpt_cfg = prune_cfg.get("ckpt_path", {})
        save_cfg = prune_cfg.get("save_path", {})

        # Baseline checkpoint before pruning
        self.baseline_ckpt = (
            self.base_dir / ckpt_cfg.get("subfolder", "baseline") /
            "training" / ckpt_cfg.get("ckpt_name", "final_model.pth")
        )

        # Pruned model path (folder created later by prune.py)
        self.pruned_dir = (
            self.base_dir / save_cfg.get("subfolder", "pruned") / self.suffix
        )
        self.pruned_model = self.pruned_dir / save_cfg.get("filename", "pruned_model.pth")

        # Logs directory
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    def _get_suffix_from_ratios(self, cfg):
        """Generate suffix like 0_0_10_20_30_30_30_20_10 if ratios exist."""
        prune_cfg = cfg.get("pruning", {})
        ratios = prune_cfg.get("ratios", {}).get("block_ratios", None)
        if ratios:
            return "_".join(str(int(v * 100)) for v in ratios.values())
        return "no_prune"

    # ------------------------------------------------------------
    def save_config_snapshot(self):
        """
        Save one copy of the experiment config to:
          results/<model>/<experiment>/config.yaml
        Only saves if not already present.
        """
        if not self.config_file_path:
            return

        config_dst = self.base_dir / "config.yaml"
        if not config_dst.exists():
            shutil.copy2(self.config_file_path, config_dst)
            print(f"💾 Saved config snapshot to {config_dst}")

    # ------------------------------------------------------------
    def __repr__(self):
        return (
            f"Experiment: {self.exp_name} ({self.model_name})\n"
            f"  🧠 Baseline checkpoint: {self.baseline_ckpt}\n"
            f"  ✂️  Pruned model: {self.pruned_model}\n"
            f"  💾 Training dir: {self.train_save_dir}\n"
            f"  🔍 Eval dir: {self.eval_save_dir}\n"
            f"  📂 Logs dir: {self.logs_dir}\n"
        )


# ------------------------------------------------------------
def get_paths(cfg, config_file_path=None):
    """Convenience wrapper for ExperimentPaths."""
    return ExperimentPaths(cfg, config_file_path)

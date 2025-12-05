from pathlib import Path
import shutil


class ExperimentPaths:
    """
    Central path manager for UNet pruning experiments.

    - Builds consistent folder structures for training, evaluation, pruning.
    - Folders are only created when explicitly needed (lazy creation).
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

        # --- Pruning ratio suffix (e.g. "0_0_10_20_30_30_30_20_10") ---
        self.suffix = self._get_suffix_from_ratios(cfg)

        # ============================================================
        # TRAINING PATHS
        # ============================================================
        train_cfg = cfg.get("train", {})
        train_paths = train_cfg.get("paths", {})
        self.train_dir = Path(train_paths.get("train_dir", ""))
        self.label_dir = Path(train_paths.get("label_dir", ""))

        phase = train_cfg.get("phase", "training").lower()
        if phase == "training":
            self.train_save_dir = self.base_dir / "baseline" / "training"
        elif phase == "retraining":
            self.train_save_dir = self.base_dir / "pruned" / self.suffix / "retraining_pruned"
        else:
            raise ValueError(f"❌ Unknown training phase: {phase}")

        # (No mkdir yet — created lazily)

        # ============================================================
        # EVALUATION PATHS
        # ============================================================
        eval_cfg = cfg.get("evaluation", {})
        eval_phase = eval_cfg.get("phase", "baseline_evaluation").lower()
        eval_paths = eval_cfg.get("paths", {})
        self.eval_dir = Path(eval_paths.get("eval_dir", ""))
        self.eval_label_dir = Path(eval_paths.get("label_dir", ""))

        if eval_phase == "baseline_evaluation":
            self.eval_save_dir = self.base_dir / "baseline" / "evaluation"
        elif eval_phase == "pruned_evaluation":
            self.eval_save_dir = self.base_dir / "pruned" / self.suffix / "pruned_evaluation"
        elif eval_phase == "retrained_pruned_evaluation":
            self.eval_save_dir = self.base_dir / "pruned" / self.suffix / "retrained_pruned_evaluation"
        else:
            raise ValueError(f"❌ Unknown evaluation phase: {eval_phase}")

        # ============================================================
        # PRUNING PATHS
        # ============================================================
        prune_cfg = cfg.get("pruning", {})
        ckpt_cfg = prune_cfg.get("ckpt", {})
        save_cfg = prune_cfg.get("save", {})

        # Baseline checkpoint
        self.baseline_ckpt = (
            self.base_dir
            / ckpt_cfg.get("subfolder", "baseline")
            / "training"
            / ckpt_cfg.get("ckpt_name", "final_model.pth")
        )

        rewind_folder = (
            self.base_dir
            / ckpt_cfg.get("subfolder", "baseline")
            / "training"
        )

        # ============================================================
        # REWIND CHECKPOINT (used ONLY during pruning with rewind)
        # ============================================================
        self.rewind_ckpt = None   # <--- SAFE DEFAULT

        use_rewind = (
            prune_cfg.get("reinitialize_weights", None) == "rewind"
        )

        if use_rewind:
            ckpt_name = ckpt_cfg.get("rewind_ckpt", None)

            if ckpt_name:
                # User explicitly provided filename
                self.rewind_ckpt = rewind_folder / ckpt_name

            else:
                # Auto-detect first epoch_*.pth file
                epoch_files = sorted(rewind_folder.glob("epoch*.pth"))

                if len(epoch_files) == 0:
                    self.rewind_ckpt = None
                else:
                    self.rewind_ckpt = epoch_files[0]

        # Main pruning directories (not created until pruning)
        self.pruned_dir = self.base_dir / save_cfg.get("subfolder", "pruned") / self.suffix
        self.pruned_model_dir = self.pruned_dir / "pruned_model"
        self.pruned_eval_dir = self.pruned_dir / "pruned_evaluation"
        self.retrain_pruned_dir = self.pruned_dir / "retraining_pruned"
        self.retrained_pruned_eval_dir = self.pruned_dir / "retrained_pruned_evaluation"

        self.pruned_model = self.pruned_model_dir / save_cfg.get("filename", "pruned_model.pth")

        # Logs directory (always needed)
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # def _get_suffix_from_ratios(self, cfg):
    #     """Generate suffix like 0_0_10_20_30_30_30_20_10 if ratios exist."""
    #     prune_cfg = cfg.get("pruning", {})
    #     ratios = prune_cfg.get("ratios", {}).get("block_ratios", None)
    #     if ratios:
    #         return "_".join(str(int(v * 100)) for v in ratios.values())
    #     return "no_prune"

    def _get_suffix_from_ratios(self, cfg):
        """
        Generate descriptive suffix for folder/model naming.
        Includes:
        - pruning method (l1 or corr_tXX)
        - block ratios
        - weight reinit mode
        """
        
        prune_cfg = cfg.get("pruning", {})
        ratios = prune_cfg.get("ratios", {}).get("block_ratios", None)
        method = prune_cfg.get("method", "l1_norm") 

        # ------------------------------
        # 1) METHOD PREFIX
        # ------------------------------
        if method == "correlation":
            threshold = prune_cfg.get("threshold", 0.90)   # default
            t_int = int(threshold * 100)
            method_prefix = f"corr_t{t_int}"
        else:
            method_prefix = "l1_norm"

        # ------------------------------
        # 2) RATIOS SUFFIX (old behavior)
        # ------------------------------
        if ratios:
            ratios_suffix = "_".join(str(int(v * 100)) for v in ratios.values())
        else:
            ratios_suffix = "no_prune"

        suffix = f"{method_prefix}_{ratios_suffix}"

        # ------------------------------
        # 3) REINIT MODE SUFFIX
        # ------------------------------
        mode = prune_cfg.get("reinitialize_weights", None)
        if mode == "random":
            suffix += "_random"
        elif mode == "rewind":
            suffix += "_rewind"

        return suffix

    # ------------------------------------------------------------
    def ensure_dir(self, path):
        """Create a directory only when explicitly needed."""
        Path(path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    def save_config_snapshot(self):
        """Save one copy of the experiment config to base folder."""
        if not self.config_file_path:
            return
        config_dst = self.base_dir / "config.yaml"
        if not config_dst.exists():
            shutil.copy2(self.config_file_path, config_dst)
            #print(f"💾 Saved config snapshot to {config_dst}")

    # ------------------------------------------------------------
    # def __repr__(self):
    #     return (
    #         f"Experiment: {self.exp_name} ({self.model_name})\n"
    #         f"  🧠 Baseline checkpoint: {self.baseline_ckpt}\n"
    #         f"  ✂️  Pruned model: {self.pruned_model}\n"
    #         f"  💾 Training dir: {self.train_save_dir}\n"
    #         f"  🔍 Eval dir: {self.eval_save_dir}\n"
    #         f"  📂 Logs dir: {self.logs_dir}\n"
    #     )


# ------------------------------------------------------------
def get_paths(cfg, config_file_path=None):
    """Convenience wrapper for ExperimentPaths."""
    return ExperimentPaths(cfg, config_file_path)

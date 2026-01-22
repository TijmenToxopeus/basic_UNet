import os
import wandb

def setup_wandb(cfg, job_type="training"):
    """
    Initialize a W&B run for each pipeline stage.
    Can be fully disabled via config or environment variable.
    """

    # -------------------------------
    # Global / config-based disable
    # -------------------------------
    exp_cfg = cfg.get("experiment", {})
    use_wandb = exp_cfg.get("use_wandb", False)  # <-- default OFF

    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None  # 🔴 IMPORTANT: no wandb run object

    # -------------------------------
    # Normal W&B setup
    # -------------------------------
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "online"  # never prompt

    exp_name   = exp_cfg.get("experiment_name", "unnamed_exp")
    model_name = exp_cfg.get("model_name", "UNet")
    entity     = exp_cfg.get("wandb_entity", "tijmen-toxo-tu-delft")

    run = wandb.init(
        project="unet-pruning",
        entity=entity,
        group=exp_name,
        name=f"{exp_name}_{job_type}",
        job_type=job_type,
        config=cfg,
        dir=os.path.join("results", model_name, exp_name),
        reinit=True,
        resume=False,
        notes=f"Stage: {job_type} for {exp_name}",
        tags=[model_name, job_type],
    )

    return run

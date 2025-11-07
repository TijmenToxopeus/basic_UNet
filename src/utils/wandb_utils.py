import wandb
import os

def setup_wandb(cfg, job_type="training"):
    """
    Initialize a W&B run for each pipeline stage.
    All runs from the same experiment are grouped together.
    """
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("experiment_name", "unnamed_exp")
    model_name = exp_cfg.get("model_name", "UNet")
    entity = exp_cfg.get("wandb_entity", "tijmen-toxo-tu-delft")

    run_name = f"{exp_name}_{job_type}"

    run = wandb.init(
        project="unet-pruning",
        entity=entity,
        group=exp_name,                    # ✅ this makes them fall under one experiment
        name=run_name,                     # unique run name
        job_type=job_type,
        config=cfg,
        dir=os.path.join("results", model_name, exp_name),
        reinit=True,
        resume=False,
        notes=f"Stage: {job_type} for {exp_name}",
        tags=[model_name, job_type],
    )

    print(f"🧭 W&B initialized for '{job_type}' under experiment '{exp_name}' ({entity})")
    print(f"📂 Logs stored in: {wandb.run.dir}")

    return run


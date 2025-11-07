# import wandb
# import os

# def setup_wandb(cfg, job_type="training"):
#     """Initialize W&B run with grouping and family linking."""
#     project = "unet-pruning"
#     entity = "tijmen-toxo-tu-delft"
#     exp_name = cfg["experiment"]["experiment_name"]
#     model_name = cfg["experiment"]["model_name"]
#     phase = cfg["train"].get("phase", "none")

#     # If no key, go offline
#     if os.getenv("WANDB_API_KEY") is None:
#         print("⚠️ W&B API key not found. Running in offline mode.")
#         os.environ["WANDB_MODE"] = "offline"

#     # Initialize the run
#     run = wandb.init(
#         project=project,
#         entity=entity,
#         group=exp_name,                  # groups all related runs together
#         job_type=job_type,               # "training", "evaluation", "pruning"
#         name=f"{exp_name}_{phase}",      # unique run name
#         config=cfg,
#         tags=[model_name, job_type],
#         notes=f"{model_name} - {job_type} ({phase})",
#         reinit=True,                     # allow multiple init in one session
#         resume="allow",                  # links runs from same group/family
#     )
#     return run


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


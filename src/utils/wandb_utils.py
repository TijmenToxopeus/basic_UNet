import wandb
import os

def setup_wandb(cfg, job_type="training"):
    """Initialize W&B run with grouping and family linking."""
    project = "unet-pruning"
    entity = "tijmen-toxo-tu-delft"
    exp_name = cfg["experiment"]["experiment_name"]
    model_name = cfg["experiment"]["model_name"]
    phase = cfg["train"].get("phase", "none")

    # If no key, go offline
    if os.getenv("WANDB_API_KEY") is None:
        print("⚠️ W&B API key not found. Running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"

    # Initialize the run
    run = wandb.init(
        project=project,
        entity=entity,
        group=exp_name,                  # groups all related runs together
        job_type=job_type,               # "training", "evaluation", "pruning"
        name=f"{exp_name}_{phase}",      # unique run name
        config=cfg,
        tags=[model_name, job_type],
        notes=f"{model_name} - {job_type} ({phase})",
        reinit=True,                     # allow multiple init in one session
        resume="allow",                  # links runs from same group/family
    )
    return run

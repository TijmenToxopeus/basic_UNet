# ------------------------------------------------------------
# weight_initialization.py
# ------------------------------------------------------------

import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict

from src.pruning.rebuild import prune_conv_weights, find_prev_conv_name


# ============================================================
# Utility: Reset parameters for RANDOM mode
# ============================================================
def reset_parameters(m):
    """
    Resets parameters of a layer if it implements `.reset_parameters()`.
    Used for full random initialization of the pruned model.
    """
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


# ============================================================
# Utility: Find earliest epoch_*.pth checkpoint
# ============================================================
def get_earliest_checkpoint(training_dir):
    """
    Returns the earliest epoch_*.pth checkpoint inside the training directory.
    Example filenames: epoch_1.pth, epoch_10.pth, epoch_50.pth

    Returns:
        str or None
    """
    training_dir = Path(training_dir)
    if not training_dir.exists():
        return None

    ckpts = list(training_dir.glob("epoch_*.pth"))
    if not ckpts:
        return None

    # Sort numerically by epoch number
    def epoch_number(path):
        num = path.stem.replace("epoch_", "")
        return int(num) if num.isdigit() else 999999

    ckpts = sorted(ckpts, key=epoch_number)
    return ckpts[0]


# ============================================================
# Core: Slice a rewind checkpoint using pruning masks
# ============================================================
def prune_rewind_checkpoint(full_state, masks, original_model, pruned_model):
    """
    Slice the FULL checkpoint (from early epoch) to match pruned architecture.

    Args:
        full_state: full unpruned model state_dict
        masks: pruning masks {layer_name: BoolTensor}
        original_model: UNet baseline (full channels)
        pruned_model: UNet pruned (reduced channels)

    Returns:
        OrderedDict compatible with pruned_model
    """

    pruned_state = OrderedDict()
    pruned_modules = dict(pruned_model.named_modules())

    # --------------------------------------------------------
    # Conv Layers (using your existing functions)
    # --------------------------------------------------------
    for name, module in original_model.named_modules():

        if not isinstance(module, nn.Conv2d):
            continue

        if f"{name}.weight" not in full_state:
            continue

        if name not in pruned_modules:
            continue

        pruned_mod = pruned_modules[name]

        # Case 1: This layer was pruned → slice with mask
        if name in masks:
            w_new, b_new = prune_conv_weights(module, masks, name)

            pruned_state[f"{name}.weight"] = w_new.clone()

            if b_new is not None and pruned_mod.bias is not None:
                pruned_state[f"{name}.bias"] = b_new.clone()

        # Case 2: Not pruned → copy full layer (if shape matches)
        else:
            full_w = full_state[f"{name}.weight"]
            if full_w.shape == pruned_mod.weight.shape:
                pruned_state[f"{name}.weight"] = full_w.clone()

            if pruned_mod.bias is not None:
                full_b = full_state.get(f"{name}.bias")
                if full_b is not None and full_b.shape == pruned_mod.bias.shape:
                    pruned_state[f"{name}.bias"] = full_b.clone()

    # --------------------------------------------------------
    # BatchNorm layers (must also be sliced!)
    # --------------------------------------------------------
    copy_pruned_batchnorm_weights(original_model, pruned_model, masks, full_state, pruned_state)

    # --------------------------------------------------------
    # Final: copy remaining matching keys (e.g., final_conv)
    # --------------------------------------------------------
    target_sd = pruned_model.state_dict()

    for k, v in full_state.items():
        if k not in pruned_state and k in target_sd:
            if v.shape == target_sd[k].shape:
                pruned_state[k] = v.clone()

    return pruned_state


# ============================================================
# Slice BatchNorm weights (used in rewind)
# ============================================================
def copy_pruned_batchnorm_weights(original, pruned, masks, full_state, pruned_state):
    """
    Copies BatchNorm2d weights & stats, slicing them according to conv masks.
    """
    for name, module in original.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue

        if name not in pruned.named_modules():
            continue

        pruned_bn = pruned.named_modules()[name]

        # Find which conv mask applies
        mask_name = find_prev_conv_name(name, masks)

        # Case: BN is unpruned → copy full if shapes match
        if mask_name is None:
            for key in ["weight", "bias", "running_mean", "running_var"]:
                full_key = f"{name}.{key}"
                if full_key in full_state and full_state[full_key].shape == pruned_bn.state_dict()[key].shape:
                    pruned_state[f"{name}.{key}"] = full_state[full_key].clone()
            continue

        # Case: BN is pruned → slice using conv mask
        mask = masks[mask_name]
        keep = torch.where(mask)[0]

        for key in ["weight", "bias", "running_mean", "running_var"]:
            full_key = f"{name}.{key}"
            if full_key in full_state:
                pruned_state[f"{name}.{key}"] = full_state[full_key][keep].clone()


# ============================================================
# Apply fine-tuning initialization mode
# ============================================================
def apply_finetune_mode(pruned_model, finetune_cfg, masks, baseline_model, device, paths):
    """
    Applies initialization mode AFTER pruning:
      - current → keep pruned baseline weights
      - random → reset all weights
      - rewind → load earliest baseline checkpoint sliced with masks
    """

    mode = finetune_cfg["mode"]
    print(f"🔧 Applying finetune mode: {mode}")

    # ------------------------------------------------------------
    # MODE 1 — CURRENT: do nothing
    # ------------------------------------------------------------
    if mode == "current":
        print("🔄 Keeping pruned baseline weights.")
        return pruned_model

    # ------------------------------------------------------------
    # MODE 2 — RANDOM: reset all parameters
    # ------------------------------------------------------------
    if mode == "random":
        print("🎲 Randomly reinitializing pruned model...")
        pruned_model.apply(reset_parameters)

        for m in pruned_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

        print("✅ Random initialization complete.")
        return pruned_model

    # ------------------------------------------------------------
    # MODE 3 — REWIND: prune an early model checkpoint
    # ------------------------------------------------------------
    if mode == "rewind":

        baseline_dir = paths.baseline_training_dir
        print(f"📂 Searching for rewind checkpoints in: {baseline_dir}")

        rewind_ckpt = get_earliest_checkpoint(baseline_dir)

        if rewind_ckpt is None:
            raise FileNotFoundError(f"❌ No epoch_*.pth found in: {baseline_dir}")

        rewind_ckpt = Path(rewind_ckpt)
        print(f"⏮ Using earliest checkpoint: {rewind_ckpt.name}")

        full_state = torch.load(rewind_ckpt, map_location=device)

        print("✂️ Slicing rewind checkpoint using pruning masks...")
        pruned_state = prune_rewind_checkpoint(
            full_state=full_state,
            masks=masks,
            original_model=baseline_model,
            pruned_model=pruned_model,
        )

        print("⏳ Loading sliced rewind weights...")
        pruned_model.load_state_dict(pruned_state, strict=False)

        # Reset BN stats because early checkpoint stats don't match pruned shape
        for m in pruned_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

        print("✅ Rewind initialization complete.")
        return pruned_model

    # ------------------------------------------------------------
    # Unknown mode
    # ------------------------------------------------------------
    raise ValueError(f"❌ Unknown finetune mode: {mode}")

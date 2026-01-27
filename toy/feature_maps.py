# from __future__ import annotations

# from pathlib import Path
# from typing import Iterable, Dict
# import math

# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# from toy.models import build_tiny_unet
# from src.models.unet import UNet  # allowlist for torch.load


# def _load_baseline(ckpt: Path, in_ch: int, out_ch: int, features: list[int], device: torch.device):
#     model = build_tiny_unet(in_ch=in_ch, out_ch=out_ch, features=features).to(device)
#     state = torch.load(ckpt, map_location=device)
#     model.load_state_dict(state)
#     model.eval()
#     return model


# def _load_full_model(ckpt: Path, device: torch.device):
#     torch.serialization.add_safe_globals([UNet])
#     obj = torch.load(ckpt, map_location=device, weights_only=False)
#     print(f"[DEBUG] loaded {ckpt.name} type: {type(obj)}")
#     if isinstance(obj, dict):
#         raise RuntimeError(f"Checkpoint {ckpt} is a state_dict; rerun experiments to save full models.")
#     model = obj.to(device).eval()
#     return model


# def _capture(model: torch.nn.Module, layers: Iterable[str], x: torch.Tensor) -> Dict[str, torch.Tensor]:
#     acts: Dict[str, torch.Tensor] = {}
#     hooks = []
#     for name, module in model.named_modules():
#         if name in layers:
#             hooks.append(module.register_forward_hook(lambda _m, _i, o, n=name: acts.setdefault(n, o.detach().cpu())))
#     with torch.no_grad():
#         _ = model(x)
#     for h in hooks:
#         h.remove()
#     return acts


# def _plot_grid(
#     t: torch.Tensor,
#     title: str,
#     out_path: Path,
#     max_channels: int = 16,
#     cmap: str = "gray",
#     prune_mask: list[bool] | None = None,
#     weight_norms: list[float] | None = None,
# ):
#     if t.dim() == 4:
#         t = t[0]
#     c = min(t.shape[0], max_channels)
#     ncols = int(math.ceil(math.sqrt(c)))
#     nrows = int(math.ceil(c / ncols))
#     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
#     axes = axes.flatten()
#     for i, ax in enumerate(axes):
#         ax.axis("off")
#         if i < c:
#             img = t[i]
#             img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#             ax.imshow(img, cmap=cmap)
#             # Add channel number and weight L1 norm
#             text_label = f"ch {i}"
#             if weight_norms is not None and i < len(weight_norms):
#                 text_label += f"\nW={weight_norms[i]:.2f}"
#             ax.text(0.02, 0.98, text_label, color="white", fontsize=8, ha="left", va="top",
#                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
#             # Red border for pruned channels
#             if prune_mask is not None and i < len(prune_mask) and not prune_mask[i]:
#                 ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
#                             color="red", linewidth=2))
#     fig.suptitle(title)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)


# def compare_feature_maps(
#     *,
#     baseline_ckpt: Path,
#     pruned_ckpt: Path,
#     retrained_ckpt: Path | None,
#     sample: torch.Tensor,
#     layers: list[str],
#     features: list[int],
#     out_ch: int,
#     device: torch.device,
#     out_dir: Path,
#     max_channels: int = 16,
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     baseline_model = _load_baseline(baseline_ckpt, sample.shape[1], out_ch, features, device)
#     pruned_model = _load_full_model(pruned_ckpt, device)
#     retrained_model = _load_full_model(retrained_ckpt, device) if (retrained_ckpt and retrained_ckpt.exists()) else None

#     # Compute weight L1 norms per layer
#     weight_norms_dict: Dict[str, list[float]] = {}
#     for lname in layers:
#         for name, module in baseline_model.named_modules():
#             if name == lname and isinstance(module, torch.nn.Conv2d):
#                 w = module.weight  # [out_ch, in_ch, H, W]
#                 norms = w.reshape(w.shape[0], -1).abs().sum(dim=1).tolist()
#                 weight_norms_dict[lname] = norms
#                 break

#     sample_device = sample.to(device)
#     acts_baseline = _capture(baseline_model, layers, sample_device)
#     acts_pruned = _capture(pruned_model, layers, sample_device)

#     # Build prune masks
#     print("\n[DEBUG] L1 norms per channel (baseline):")
#     prune_masks: Dict[str, list[bool]] = {}

#     for lname, act_base in acts_baseline.items():
#         base_c = act_base.shape[1] if act_base.dim() == 4 else act_base.shape[0]

#         pruned_c = acts_pruned.get(lname)
#         pruned_c = pruned_c.shape[1] if (pruned_c is not None and pruned_c.dim() == 4) else (pruned_c.shape[0] if pruned_c is not None else 0)
#         mask = [True] * min(base_c, pruned_c) + [False] * max(0, base_c - pruned_c)
#         prune_masks[lname] = mask

#     # Plot baseline with weight norms and pruned channels marked
#     for lname, act in acts_baseline.items():
#         _plot_grid(
#             act,
#             f"baseline - {lname}",
#             out_dir / f"baseline_marked_{lname.replace('.', '_')}.png",
#             max_channels=max_channels,
#             prune_mask=prune_masks.get(lname),
#             weight_norms=weight_norms_dict.get(lname),
#         )

#     # Plot pruned/retrained normally with weight norms
#     for label, acts in [("pruned", acts_pruned), ("retrained", _capture(retrained_model, layers, sample_device) if retrained_model else None)]:
#         if acts is None:
#             continue
#         for lname, act in acts.items():
#             _plot_grid(act, f"{label} - {lname}", out_dir / f"{label}_{lname.replace('.', '_')}.png",
#                       max_channels=max_channels, weight_norms=weight_norms_dict.get(lname))


from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict
import math

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from toy.models import build_tiny_unet
from src.models.unet import UNet  # allowlist for torch.load

# ✅ use the same utilities as your L1NormPruning method
from src.pruning.model_inspect import compute_l1_norms, get_pruning_masks_blockwise


def _load_baseline(ckpt: Path, in_ch: int, out_ch: int, features: list[int], device: torch.device):
    # Load state_dict baseline (tiny UNet)
    model = build_tiny_unet(in_ch=in_ch, out_ch=out_ch, features=features)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_full_model(ckpt: Path, device: torch.device):
    # Load full model objects (pruned / retrained) saved via torch.save(model)
    torch.serialization.add_safe_globals([UNet])
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    print(f"[DEBUG] loaded {ckpt.name} type: {type(obj)}")
    if isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint {ckpt} is a state_dict; rerun experiments to save full models.")
    model = obj.to(device).eval()
    return model


def _capture(model: torch.nn.Module, layers: Iterable[str], x: torch.Tensor) -> Dict[str, torch.Tensor]:
    acts: Dict[str, torch.Tensor] = {}
    hooks = []

    # ✅ force x to the model device (safe)
    dev = next(model.parameters()).device
    x = x.to(dev)

    for name, module in model.named_modules():
        if name in layers:
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _i, o, n=name: acts.setdefault(n, o.detach().cpu())
                )
            )
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    return acts


def _plot_grid(
    t: torch.Tensor,
    title: str,
    out_path: Path,
    max_channels: int = 16,
    cmap: str = "gray",
    prune_mask: list[bool] | None = None,       # mask over ORIGINAL baseline channels
    weight_norms: list[float] | None = None,    # norms over ORIGINAL baseline channels
    channel_ids: list[int] | None = None,       # NEW: mapping plotted index -> original channel id
):
    if t.dim() == 4:
        t = t[0]  # [C,H,W]

    c = min(t.shape[0], max_channels)
    ncols = int(math.ceil(math.sqrt(c)))
    nrows = int(math.ceil(c / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= c:
            continue

        img = t[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img, cmap=cmap)

        # Which original channel is this?
        ch_id = channel_ids[i] if (channel_ids is not None and i < len(channel_ids)) else i

        # Label
        text_label = f"ch {ch_id}"
        if weight_norms is not None and ch_id < len(weight_norms):
            text_label += f"\nW={weight_norms[ch_id]:.2f}"

        ax.text(
            0.02, 0.98, text_label,
            color="white", fontsize=8, ha="left", va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

        # Red border if this original channel was pruned (mask False)
        if prune_mask is not None and ch_id < len(prune_mask) and not prune_mask[ch_id]:
            ax.add_patch(
                patches.Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    fill=False,
                    color="red",
                    linewidth=2,
                )
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare_feature_maps(
    *,
    baseline_ckpt: Path,
    pruned_ckpt: Path,
    retrained_ckpt: Path | None,
    sample: torch.Tensor,
    layers: list[str],          # IMPORTANT: should be Conv2d module names for L1 masks to match
    features: list[int],
    out_ch: int,
    device: torch.device,
    out_dir: Path,
    max_channels: int = 16,

    # ✅ pruning settings used in your actual L1NormPruning experiment
    block_ratios: dict[str, float] | None = None,
    default_ratio: float = 0.25,
    seed: int = 0,
    deterministic: bool = True,
):
    """
    Capture and save activation grids for baseline, pruned, and optional retrained models.
    Baseline plots get red borders on channels that the REAL L1 blockwise pruning would remove.
    Pruned/retrained plots are labeled with the ORIGINAL baseline channel indices (so "pruned channels"
    are not falsely shown as "the last ones").
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load models ----
    baseline_model = _load_baseline(baseline_ckpt, sample.shape[1], out_ch, features, device)
    pruned_model = _load_full_model(pruned_ckpt, device)
    retrained_model = _load_full_model(retrained_ckpt, device) if (retrained_ckpt and retrained_ckpt.exists()) else None

    # ---- Compute REAL L1 norms + masks (same as L1NormPruning) ----
    norms_dict = compute_l1_norms(baseline_model)  # {conv_name: [float per out_channel]}

    real_masks = get_pruning_masks_blockwise(
        baseline_model,
        norms_dict,
        block_ratios=block_ratios or {},
        default_ratio=default_ratio,
        seed=seed,
        deterministic=deterministic,
    )  # {conv_name: [bool per out_channel]}

    # Helper: mapping from pruned channel index -> original channel id
    def kept_indices(mask: list[bool]) -> list[int]:
        return [i for i, keep in enumerate(mask) if keep]

    # ---- Capture activations ----
    sample_device = sample.to(device)
    acts_baseline = _capture(baseline_model, layers, sample_device)
    acts_pruned = _capture(pruned_model, layers, sample_device)
    acts_retrained = _capture(retrained_model, layers, sample_device) if retrained_model else None

    # ---- Plot baseline: true pruning mask + true norms ----
    for lname, act in acts_baseline.items():
        mask = real_masks.get(lname)
        norms = norms_dict.get(lname)

        if mask is None:
            print(f"[WARN] No pruning mask found for layer '{lname}'. "
                  f"Make sure 'layers' contains Conv2d module names used by compute_l1_norms().")

        _plot_grid(
            act,
            f"baseline - {lname}",
            out_dir / f"baseline_marked_{lname.replace('.', '_')}.png",
            max_channels=max_channels,
            prune_mask=mask,
            weight_norms=norms,
            channel_ids=None,  # baseline already uses original channel ids
        )

    # ---- Plot pruned/retrained: label channels with ORIGINAL indices ----
    for label, acts in [("pruned", acts_pruned), ("retrained", acts_retrained)]:
        if acts is None:
            continue

        for lname, act in acts.items():
            mask = real_masks.get(lname)
            norms = norms_dict.get(lname)

            ch_ids = kept_indices(mask) if mask is not None else None

            _plot_grid(
                act,
                f"{label} - {lname}",
                out_dir / f"{label}_{lname.replace('.', '_')}.png",
                max_channels=max_channels,
                prune_mask=mask,        # mask over original channels
                weight_norms=norms,     # norms over original channels
                channel_ids=ch_ids,     # map plotted channels -> original ids
            )

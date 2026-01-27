from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict
import math

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from toy.models import build_tiny_unet
from src.models.unet import UNet  # allowlist for torch.load


def _load_baseline(ckpt: Path, in_ch: int, out_ch: int, features: list[int], device: torch.device):
    model = build_tiny_unet(in_ch=in_ch, out_ch=out_ch, features=features).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_full_model(ckpt: Path, device: torch.device):
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
    for name, module in model.named_modules():
        if name in layers:
            hooks.append(module.register_forward_hook(lambda _m, _i, o, n=name: acts.setdefault(n, o.detach().cpu())))
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
    prune_mask: list[bool] | None = None,
    l1_norms: list[float] | None = None,
):
    if t.dim() == 4:
        t = t[0]
    c = min(t.shape[0], max_channels)
    ncols = int(math.ceil(math.sqrt(c)))
    nrows = int(math.ceil(c / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < c:
            img = t[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap=cmap)
            # Add channel number and L1 norm as text
            text_label = f"ch {i}"
            if l1_norms is not None and i < len(l1_norms):
                text_label += f"\nL1={l1_norms[i]:.3f}"
            ax.text(0.02, 0.98, text_label, color="white", fontsize=8, ha="left", va="top",
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
            # Red border for pruned channels
            if prune_mask is not None and i < len(prune_mask) and not prune_mask[i]:
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                            color="red", linewidth=2))
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
    layers: list[str],
    features: list[int],
    out_ch: int,
    device: torch.device,
    out_dir: Path,
    max_channels: int = 16,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = _load_baseline(baseline_ckpt, sample.shape[1], out_ch, features, device)
    pruned_model = _load_full_model(pruned_ckpt, device)
    retrained_model = _load_full_model(retrained_ckpt, device) if (retrained_ckpt and retrained_ckpt.exists()) else None

    # Debug: print WEIGHT L1 norms (what pruning actually uses)
    print("\n[DEBUG] Weight L1 norms per output channel (baseline):")
    for name, module in baseline_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            w = module.weight  # [out_ch, in_ch, H, W]
            norms = w.reshape(w.shape[0], -1).abs().sum(dim=1).tolist()  # per output channel
            print(f"{name}: {[f'{n:.3f}' for n in norms]}")

    sample_device = sample.to(device)
    acts_baseline = _capture(baseline_model, layers, sample_device)
    acts_pruned = _capture(pruned_model, layers, sample_device)

    # Compute L1 norms and prune masks
    print("\n[DEBUG] L1 norms per channel (baseline):")
    l1_norms_dict: Dict[str, list[float]] = {}
    prune_masks: Dict[str, list[bool]] = {}

    for lname, act_base in acts_baseline.items():
        base_c = act_base.shape[1] if act_base.dim() == 4 else act_base.shape[0]
        norms = act_base.reshape(base_c, -1).abs().sum(dim=1).tolist()
        l1_norms_dict[lname] = norms
        print(f"{lname}: {[f'{n:.3f}' for n in norms]}")

        pruned_c = acts_pruned.get(lname)
        pruned_c = pruned_c.shape[1] if (pruned_c is not None and pruned_c.dim() == 4) else (pruned_c.shape[0] if pruned_c is not None else 0)
        mask = [True] * min(base_c, pruned_c) + [False] * max(0, base_c - pruned_c)
        prune_masks[lname] = mask

    # Plot baseline with L1 norms and pruned channels marked
    for lname, act in acts_baseline.items():
        _plot_grid(
            act,
            f"baseline - {lname}",
            out_dir / f"baseline_marked_{lname.replace('.', '_')}.png",
            max_channels=max_channels,
            prune_mask=prune_masks.get(lname),
            l1_norms=l1_norms_dict.get(lname),
        )

    # Plot pruned/retrained normally with L1 norms
    for label, acts in [("pruned", acts_pruned), ("retrained", _capture(retrained_model, layers, sample_device) if retrained_model else None)]:
        if acts is None:
            continue
        for lname, act in acts.items():
            c = act.shape[1] if act.dim() == 4 else act.shape[0]
            norms = act.reshape(c, -1).abs().sum(dim=1).tolist()
            _plot_grid(act, f"{label} - {lname}", out_dir / f"{label}_{lname.replace('.', '_')}.png",
                      max_channels=max_channels, l1_norms=norms)

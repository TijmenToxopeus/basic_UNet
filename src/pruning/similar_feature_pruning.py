# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import torchvision.transforms as T

# # --- NEW: reproducibility (central utility) ---
# from src.utils.reproducibility import seed_everything


# ############################################################
# # 1. COLLECT FEATURE MAPS OVER MANY SLICES (WITH BATCHING)
# ############################################################
# def get_feature_maps_batched(model, xs, batch_size=4):
#     """
#     xs: list of tensors shaped [C, H, W] (e.g., [1, 256, 256]).
#     Returns:
#         activations[layer_name] = list of fm ∈ [C, H, W]
#     """
#     device = next(model.parameters()).device
#     model.eval()

#     activations = {}

#     def hook(name):
#         def fn(m, inp, out):
#             # out: [B, C, H, W]
#             out_cpu = out.detach().cpu()
#             if name not in activations:
#                 activations[name] = [out_cpu[b] for b in range(out_cpu.size(0))]
#             else:
#                 activations[name].extend([out_cpu[b] for b in range(out_cpu.size(0))])
#         return fn

#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             handles.append(module.register_forward_hook(hook(name)))

#     for i in range(0, len(xs), batch_size):
#         batch = torch.stack(xs[i:i + batch_size]).to(device)
#         with torch.no_grad():
#             _ = model(batch)

#     for h in handles:
#         h.remove()

#     return activations


# ############################################################
# # 2. SIMILARITY MATRIX (AVERAGED ACROSS SLICES)
# ############################################################
# def compute_similarity_matrix_multi(fmaps, device=None):
#     """
#     fmaps : list of [C, H, W] tensors (typically on CPU)
#     Returns:
#         averaged similarity matrix of shape [C, C] as numpy array
#     """
#     sims = []

#     for fmap in fmaps:
#         C = fmap.size(0)
#         X = fmap.reshape(C, -1).float()

#         # Use provided device (model device) for speed/repro consistency
#         if device is not None:
#             X = X.to(device)

#         # Normalize per-channel
#         X = X - X.mean(dim=1, keepdim=True)
#         X = X / (X.std(dim=1, keepdim=True) + 1e-8)

#         sim = (X @ X.T) / X.size(1)
#         sim = torch.clamp(sim, -1, 1)

#         sims.append(sim.detach().cpu())

#     sims = torch.stack(sims, dim=0)  # [N, C, C]
#     return sims.mean(dim=0).numpy()


# ############################################################
# # 3. DETECT REDUNDANT PAIRS
# ############################################################
# def find_redundant_pairs(sim_matrix, threshold):
#     C = sim_matrix.shape[0]
#     pairs = []
#     for i in range(C):
#         for j in range(i + 1, C):
#             if sim_matrix[i, j] > threshold:
#                 pairs.append((i, j))
#     return pairs


# ############################################################
# # 4. GROUP REDUNDANT CHANNELS
# ############################################################
# def group_channels(redundant_pairs):
#     groups = {}
#     gid = 0

#     for (i, j) in redundant_pairs:
#         found = False

#         for g, members in groups.items():
#             if i in members or j in members:
#                 members.add(i)
#                 members.add(j)
#                 found = True
#                 break

#         if not found:
#             groups[gid] = {i, j}
#             gid += 1

#     return groups


# ############################################################
# # 5. COMPUTE LAYER PRUNING MASK
# ############################################################
# def compute_layer_mask(C, groups, sim_matrix, prune_ratio):
#     keep_mask = torch.ones(C, dtype=torch.bool)
#     scores = {}

#     for _, chs in groups.items():
#         chs = sorted(chs)
#         rep = chs[0]
#         others = chs[1:]
#         for j in others:
#             scores[j] = scores.get(j, 0) + sim_matrix[rep, j]

#     if len(scores) == 0:
#         return keep_mask, []

#     sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     candidate_idxs = [idx for idx, _ in sorted_candidates]

#     max_prune = int(C * prune_ratio)
#     to_prune = candidate_idxs[:max_prune]

#     for idx in to_prune:
#         keep_mask[idx] = False

#     return keep_mask, to_prune


# ############################################################
# # 6. VISUALIZE REDUNDANCY GROUPS (OPTIONAL)
# ############################################################
# def plot_redundancy_groups(fmaps, groups, mask=None):
#     for gid, chans in groups.items():
#         chans = sorted(chans)
#         n = len(chans)

#         fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.6))
#         if n == 1:
#             axes = [axes]

#         for ax, ch in zip(axes, chans):
#             img = fmaps[0][ch]
#             img = (img - img.min()) / (img.max() - img.min() + 1e-5)

#             ax.imshow(img, cmap="gray")
#             ax.set_title(f"ch {ch}", fontsize=8)
#             ax.axis("off")

#             if mask is not None and not mask[ch]:
#                 rect = patches.Rectangle(
#                     (0, 0), 1, 1,
#                     edgecolor="red", facecolor="none", linewidth=2,
#                     transform=ax.transAxes
#                 )
#                 ax.add_patch(rect)

#         plt.tight_layout()
#         plt.show()


# ############################################################
# # LAYER NAME → PRUNING RATIO LOOKUP
# ############################################################
# def get_ratio_for_layer(layer_name, block_ratios):
#     for block, ratio in block_ratios.items():
#         if layer_name.startswith(block):
#             return ratio
#     return 0.0


# ############################################################
# # LOAD RANDOM ACDC SLICES (DETERMINISTIC)
# ############################################################
# def load_random_slices_acdc(img_dir, num_slices=20, seed=None, deterministic=False):
#     """
#     Returns a list of tensors shaped [1, 256, 256].

#     Uses a *local* numpy RNG so it doesn't affect global numpy randomness.
#     If seed is provided, selection is deterministic.
#     """
#     if seed is not None:
#         # This seeds python/np/torch consistently (optional but safe)
#         seed_everything(seed, deterministic=deterministic)
#         rng = np.random.default_rng(seed)
#     else:
#         rng = np.random.default_rng()

#     transform = T.Compose([
#         T.ToTensor(),            # HxW -> 1xHxW (float in [0,1] if input is float)
#         T.Resize((256, 256)),
#     ])

#     nii_files = [
#         os.path.join(img_dir, f)
#         for f in os.listdir(img_dir)
#         if f.endswith(".nii.gz")
#     ]
#     if len(nii_files) == 0:
#         raise FileNotFoundError(f"No .nii.gz files found in: {img_dir}")

#     example_slices = []

#     for _ in range(num_slices):
#         # Pick random volume
#         nii_path = rng.choice(nii_files)
#         volume = nib.load(nii_path).get_fdata()

#         # Pick random slice index
#         idx = rng.integers(0, volume.shape[-1])
#         img2d = volume[:, :, idx]

#         # Normalize robustly
#         img2d = img2d.astype(np.float32)
#         img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-8)

#         # Transform to tensor [1, 256, 256]
#         img_tensor = transform(img2d).float()
#         example_slices.append(img_tensor)

#     return example_slices


# ############################################################
# # 7. FULL REDUNDANCY-BASED PRUNING WITH BATCHING
# ############################################################
# def get_redundancy_masks(
#     model,
#     example_slices,
#     block_ratios,
#     threshold=0.9,
#     batch_size=4,
#     plot=False
# ):
#     """
#     example_slices : list of [C, H, W] tensors (many slices)
#     Returns:
#         masks[layer_name] = boolean tensor [C_out] indicating channels kept
#     """

#     print("\n🔥 Running correlation-based pruning with batching...")
#     print(f"  Using {len(example_slices)} slices")
#     print(f"  Batch size = {batch_size}")
#     print(f"  threshold  = {threshold}\n")

#     device = next(model.parameters()).device

#     # 1 — Collect feature maps over many slices
#     activations = get_feature_maps_batched(model, example_slices, batch_size=batch_size)

#     masks = {}
#     total_before = 0
#     total_after = 0

#     for layer_name, fmaps in activations.items():
#         C = fmaps[0].shape[0]
#         prune_ratio = get_ratio_for_layer(layer_name, block_ratios)

#         if prune_ratio == 0.0:
#             masks[layer_name] = torch.ones(C, dtype=torch.bool)
#             continue

#         print(f"\n=== Layer: {layer_name} (ratio={prune_ratio}) ===")

#         # 2 — Compute averaged similarity matrix (on model device)
#         sim = compute_similarity_matrix_multi(fmaps, device=device)

#         # 3 — Detect redundancy
#         pairs = find_redundant_pairs(sim, threshold)
#         groups = group_channels(pairs)

#         # 4 — Compute mask
#         mask, pruned_idxs = compute_layer_mask(C, groups, sim, prune_ratio)

#         if plot and len(groups) > 0:
#             plot_redundancy_groups(fmaps, groups, mask)

#         keep = mask.sum().item()
#         total_before += C
#         total_after += keep

#         print(f"  Channels: {C}, kept: {keep}, pruned: {C - keep}")
#         masks[layer_name] = mask.clone()

#     print("\n================ SUMMARY ================")
#     print(f"Total channels before: {total_before}")
#     print(f"Total channels after : {total_after}")
#     if total_before > 0:
#         print(f"Total pruning: {100 * (total_before - total_after) / total_before:.2f}%")
#     print("=========================================\n")

#     return masks



from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T


############################################################
# 1. COLLECT FEATURE MAPS OVER MANY SLICES (WITH BATCHING)
############################################################
def get_feature_maps_batched(model: nn.Module, xs: List[torch.Tensor], batch_size: int = 4) -> Dict[str, List[torch.Tensor]]:
    """
    xs: list of tensors shaped [C, H, W] (e.g., [1, 256, 256]).
    Returns:
        activations[layer_name] = list of feature maps shaped [C, H, W] (one per slice)
    """
    device = next(model.parameters()).device

    # IMPORTANT: preserve the incoming mode
    was_training = model.training
    model.eval()

    activations: Dict[str, List[torch.Tensor]] = {}

    def hook(name: str):
        def fn(m, inp, out):
            # out: [B, C, H, W]
            out_cpu = out.detach().cpu()
            if name not in activations:
                activations[name] = [out_cpu[b] for b in range(out_cpu.size(0))]
            else:
                activations[name].extend([out_cpu[b] for b in range(out_cpu.size(0))])
        return fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))

    for i in range(0, len(xs), batch_size):
        batch = torch.stack(xs[i:i + batch_size]).to(device)
        with torch.no_grad():
            _ = model(batch)

    for h in handles:
        h.remove()

    # restore mode
    model.train(was_training)

    return activations


############################################################
# 2. SIMILARITY MATRIX (AVERAGED ACROSS SLICES)
############################################################
def compute_similarity_matrix_multi(fmaps: List[torch.Tensor], device: Optional[torch.device] = None) -> np.ndarray:
    """
    fmaps : list of [C, H, W] tensors (typically on CPU)
    Returns:
        averaged similarity matrix of shape [C, C] as numpy array
    """
    sims = []

    for fmap in fmaps:
        C = fmap.size(0)
        X = fmap.reshape(C, -1).float()

        if device is not None:
            X = X.to(device)

        # Normalize per-channel (zero mean, unit std)
        X = X - X.mean(dim=1, keepdim=True)
        X = X / (X.std(dim=1, keepdim=True) + 1e-8)

        sim = (X @ X.T) / X.size(1)
        sim = torch.clamp(sim, -1, 1)

        sims.append(sim.detach().cpu())

    sims = torch.stack(sims, dim=0)  # [N, C, C]
    return sims.mean(dim=0).numpy()


############################################################
# 3. DETECT REDUNDANT PAIRS
############################################################
def find_redundant_pairs(sim_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    C = sim_matrix.shape[0]
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            if sim_matrix[i, j] > threshold:
                pairs.append((i, j))
    return pairs


############################################################
# 4. GROUP REDUNDANT CHANNELS
############################################################
def group_channels(redundant_pairs: List[Tuple[int, int]]) -> Dict[int, set]:
    groups: Dict[int, set] = {}
    gid = 0

    for (i, j) in redundant_pairs:
        found = False
        for _, members in groups.items():
            if i in members or j in members:
                members.add(i)
                members.add(j)
                found = True
                break

        if not found:
            groups[gid] = {i, j}
            gid += 1

    return groups


############################################################
# 5. COMPUTE LAYER PRUNING MASK
############################################################
def compute_layer_mask(
    conv: torch.nn.Conv2d,
    groups: dict,
    sim_matrix: np.ndarray,
    prune_ratio: float,
) -> tuple[torch.Tensor, list[int]]:
    """
    Compute a pruning mask for a Conv2d layer using correlation groups.

    Behavior:
      - For each redundancy group, the representative is the channel with
        the highest L1-norm (importance).
      - Only non-representative channels are candidates for pruning.
      - If there are more candidates than allowed by prune_ratio,
        prune the most redundant ones first.
      - Enforces pruning of at most floor(C * prune_ratio) channels.

    Args:
        conv: nn.Conv2d layer (used to compute L1 importance internally)
        groups: {gid: set(channel_indices)} from group_channels()
        sim_matrix: [C, C] numpy similarity matrix
        prune_ratio: fraction of channels to prune (0..1)

    Returns:
        keep_mask: torch.BoolTensor [C], True = keep
        to_prune: list of pruned channel indices
    """
    C = conv.out_channels
    keep_mask = torch.ones(C, dtype=torch.bool)

    # How many channels may we prune?
    max_prune = int(np.floor(C * prune_ratio))
    max_prune = max(0, min(C, max_prune))
    if max_prune == 0 or len(groups) == 0:
        return keep_mask, []

    # --- Compute L1 importance per output channel (same definition as L1 pruning) ---
    with torch.no_grad():
        w = conv.weight.detach()
        importance = w.abs().view(C, -1).sum(dim=1).cpu()  # [C]

    # --- Accumulate redundancy scores for pruning candidates ---
    scores: dict[int, float] = {}

    for _, chs in groups.items():
        chs = sorted(chs)
        if len(chs) <= 1:
            continue

        # Representative = most important channel in this group
        rep = max(chs, key=lambda k: float(importance[k]))

        # All others are pruning candidates
        for j in chs:
            if j == rep:
                continue
            sim_value = float(sim_matrix[rep, j])
            scores[j] = scores.get(j, 0.0) + sim_value

    if len(scores) == 0:
        return keep_mask, []

    # --- Rank candidates by redundancy (highest first) ---
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidate_idxs = [idx for idx, _ in sorted_candidates]

    # --- Prune top-K ---
    to_prune = candidate_idxs[:max_prune]
    for idx in to_prune:
        keep_mask[idx] = False

    return keep_mask, to_prune



############################################################
# 6. VISUALIZE REDUNDANCY GROUPS (OPTIONAL)
############################################################
def plot_redundancy_groups(fmaps: List[torch.Tensor], groups: Dict[int, set], mask: Optional[torch.Tensor] = None) -> None:
    for gid, chans in groups.items():
        chans = sorted(chans)
        n = len(chans)

        fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.6))
        if n == 1:
            axes = [axes]

        for ax, ch in zip(axes, chans):
            img = fmaps[0][ch]
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)

            ax.imshow(img, cmap="gray")
            ax.set_title(f"ch {ch}", fontsize=8)
            ax.axis("off")

            if mask is not None and not bool(mask[ch]):
                rect = patches.Rectangle(
                    (0, 0), 1, 1,
                    edgecolor="red", facecolor="none", linewidth=2,
                    transform=ax.transAxes
                )
                ax.add_patch(rect)

        plt.tight_layout()
        plt.show()


############################################################
# LAYER NAME HELPERS (match L1 block logic)
############################################################
def _infer_block_from_layer_name(name: str) -> str:
    """
    Map:
      encoders.0.net.0 -> encoders.0
      encoders.0.net.3 -> encoders.0
      decoders.1.net.3 -> decoders.1
      bottleneck.net.3 -> bottleneck
      final_conv        -> final_conv
    """
    parts = name.split(".")
    if parts[0] == "bottleneck":
        return "bottleneck"
    if parts[0] == "final_conv":
        return "final_conv"
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def _is_doubleconv_conv(name: str) -> bool:
    # the conv layers inside your DoubleConv blocks are net.0 and net.3
    return name.endswith(".net.0") or name.endswith(".net.3")


def get_ratio_for_layer(layer_name: str, block_ratios: dict, default_ratio: float) -> float:
    """
    L1-like behavior:
      - match prefix ratios
      - fallback to default_ratio
    """
    for block, ratio in (block_ratios or {}).items():
        if layer_name.startswith(block):
            return float(ratio)
    return float(default_ratio)


############################################################
# LOAD RANDOM ACDC SLICES (DETERMINISTIC WITHOUT GLOBAL SEEDING)
############################################################
def load_random_slices_acdc(img_dir: str, num_slices: int = 20, seed: Optional[int] = None) -> List[torch.Tensor]:
    """
    Returns a list of tensors shaped [1, 256, 256].

    Uses a local numpy RNG so it does NOT change global RNG state.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    transform = T.Compose([
        T.ToTensor(),            # HxW -> 1xHxW
        T.Resize((256, 256)),
    ])

    nii_files = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith(".nii.gz")
    ]
    if len(nii_files) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in: {img_dir}")

    example_slices: List[torch.Tensor] = []

    for _ in range(num_slices):
        nii_path = rng.choice(nii_files)
        volume = nib.load(nii_path).get_fdata()

        idx = int(rng.integers(0, volume.shape[-1]))
        img2d = volume[:, :, idx].astype(np.float32)

        # normalize
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-8)

        img_tensor = transform(img2d).float()
        example_slices.append(img_tensor)

    return example_slices


############################################################
# 7. FULL REDUNDANCY-BASED PRUNING WITH BLOCK-CONSISTENT MASKS
############################################################
def get_redundancy_masks(
    model: nn.Module,
    example_slices: List[torch.Tensor],
    block_ratios: dict,
    *,
    default_ratio: float = 0.25,
    threshold: float = 0.9,
    batch_size: int = 4,
    plot: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
        masks[layer_name] = boolean tensor [C_out] indicating channels kept

    IMPORTANT CHANGES vs old version:
      - L1-like default_ratio fallback
      - skip final_conv always
      - block-consistent masks: compute one mask per block (prefer net.3) and apply to both net.0 and net.3
    """
    print("\n🔥 Running correlation-based pruning (L1-compatible mode)...")
    print(f"  Using {len(example_slices)} slices")
    print(f"  Batch size     = {batch_size}")
    print(f"  threshold      = {threshold}")
    print(f"  default_ratio  = {default_ratio}\n")

    device = next(model.parameters()).device

    # Collect feature maps
    activations = get_feature_maps_batched(model, example_slices, batch_size=batch_size)

    # Group relevant conv layers by block
    block_to_layers: Dict[str, List[str]] = {}
    for layer_name in activations.keys():
        if "final_conv" in layer_name:
            continue
        if not _is_doubleconv_conv(layer_name):
            continue
        block = _infer_block_from_layer_name(layer_name)
        block_to_layers.setdefault(block, []).append(layer_name)

    masks: Dict[str, torch.Tensor] = {}

    total_before = 0
    total_after = 0

    # Compute one mask per block, based on the representative layer (prefer net.3)
    for block, layer_names in block_to_layers.items():
        # choose representative
        rep_layer = None
        for ln in layer_names:
            if ln.endswith(".net.3"):
                rep_layer = ln
                break
        if rep_layer is None:
            rep_layer = layer_names[0]

        fmaps = activations[rep_layer]
        C = int(fmaps[0].shape[0])

        # determine prune ratio (L1-like)
        prune_ratio = get_ratio_for_layer(rep_layer, block_ratios, default_ratio)

        # hard safety: never prune to 0 channels
        prune_ratio = float(np.clip(prune_ratio, 0.0, 0.99))

        if prune_ratio <= 0.0:
            block_mask = torch.ones(C, dtype=torch.bool)
            pruned_idxs: List[int] = []
            groups = {}
        else:
            sim = compute_similarity_matrix_multi(fmaps, device=device)
            pairs = find_redundant_pairs(sim, threshold)
            groups = group_channels(pairs)
            conv = dict(model.named_modules())[rep_layer]
            block_mask, pruned_idxs = compute_layer_mask(
            conv=conv,
            groups=groups,
            sim_matrix=sim,
            prune_ratio=prune_ratio,
        )

        keep = int(block_mask.sum().item())
        total_before += C
        total_after += keep

        print(f"=== Block: {block} | rep={rep_layer} | C={C} | ratio={prune_ratio:.2f} | kept={keep} | pruned={C-keep}")
        if plot and len(groups) > 0:
            plot_redundancy_groups(fmaps, groups, block_mask)

        # Apply same mask to both convs in this block
        for ln in layer_names:
            masks[ln] = block_mask.clone()

    # Ensure final conv is never pruned (L1 behavior)
    if hasattr(model, "final_conv") and isinstance(model.final_conv, nn.Conv2d):
        masks["final_conv"] = torch.ones(model.final_conv.out_channels, dtype=torch.bool)

    print("\n================ SUMMARY ================")
    print(f"Total channels before (counted blocks): {total_before}")
    print(f"Total channels after  (counted blocks): {total_after}")
    if total_before > 0:
        print(f"Total pruning: {100 * (total_before - total_after) / total_before:.2f}%")
    print("=========================================\n")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(masks, save_path / "correlation_masks.pt")
        meta = {
            "threshold": threshold,
            "default_ratio": default_ratio,
            "block_ratios": block_ratios,
            "layers": {
                name: {"kept": int(mask.sum().item()), "total": int(mask.numel())}
                for name, mask in masks.items()
            },
        }
        (save_path / "correlation_masks_meta.json").write_text(json.dumps(meta, indent=2))

    return masks

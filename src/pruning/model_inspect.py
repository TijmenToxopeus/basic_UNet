# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np


# # -----------------------------------------------------------------------------
# # 1️⃣ Compute per-layer L1 norms
# # -----------------------------------------------------------------------------
# def compute_l1_norms(model: nn.Module) -> dict:
#     """
#     Compute per-filter L1 norms for all Conv2d and ConvTranspose2d layers in a model.

#     Returns:
#         dict: {layer_name: torch.Tensor of shape [out_channels]}
#     """
#     norms = {}
#     for name, layer in model.named_modules():
#         if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
#             w = layer.weight.data.abs().view(layer.weight.size(0), -1)
#             l1_vals = w.sum(dim=1)
#             norms[name] = l1_vals
#     return norms


# # -----------------------------------------------------------------------------
# # 2️⃣ Compute summary statistics from L1 norms
# # -----------------------------------------------------------------------------
# def compute_l1_stats(norms: dict) -> dict:
#     """
#     Compute summary statistics from per-filter L1 norms.

#     Args:
#         norms (dict): {layer_name: torch.Tensor of L1 values}

#     Returns:
#         dict: {
#             layer_name: {
#                 'Mean L1', 'Min L1', 'Max L1', 'L1 Std'
#             }
#         }
#     """
#     stats = {}
#     for name, l1_vals in norms.items():
#         stats[name] = {
#             "Mean L1": l1_vals.mean().item(),
#             "Min L1": l1_vals.min().item(),
#             "Max L1": l1_vals.max().item(),
#             "L1 Std": l1_vals.std().item(),
#         }
#     return stats


# # -----------------------------------------------------------------------------
# # 3️⃣ Generate pruning masks (structured, block-wise)
# # -----------------------------------------------------------------------------
# def get_pruning_masks_blockwise(model, norms, block_ratios=None, default_ratio=0.3):
#     """
#     Generate structured pruning masks based on L1 norms and block ratios.

#     Args:
#         model (nn.Module): Model to inspect.
#         norms (dict): {layer_name: L1 tensor per filter}.
#         block_ratios (dict, optional): {block_name: prune_ratio}.
#         default_ratio (float): Default prune ratio for unspecified blocks.

#     Returns:
#         dict: {layer_name: torch.BoolTensor mask}
#     """
#     masks = {}
#     print("🔧 Generating pruning masks...\n")

#     for name, l1_vals in norms.items():
#         num_out = len(l1_vals)

#         # Identify block (e.g. encoders.0, bottleneck, decoders.1)
#         parts = name.split(".")
#         block = ".".join(parts[:2]) if parts[0] != "bottleneck" else "bottleneck"
#         block = block.replace(".net", "")

#         # 🧠 Skip ConvTranspose2d upsampling layers and final conv
#         if (
#             ("decoders." in block and not ".net" in name and not "bottleneck" in block)
#             or "final_conv" in name
#         ):
#             #print(f"⏭️  Skipping layer {block} (upsampling or final conv).")
#             mask = torch.ones(num_out, dtype=torch.bool)
#             masks[name] = mask
#             continue

#         # Determine pruning ratio
#         ratio = block_ratios.get(block, default_ratio) if block_ratios else default_ratio

#         if ratio <= 0.0:
#             mask = torch.ones(num_out, dtype=torch.bool)
#             print(f"Block {block:15s} | ratio=0.00 → keeping all {num_out} filters.")
#         elif ratio >= 1.0:
#             mask = torch.zeros(num_out, dtype=torch.bool)
#             print(f"Block {block:15s} | ratio=1.00 → pruning all {num_out} filters.")
#         else:
#             # Threshold pruning
#             thresh = torch.quantile(l1_vals, ratio)
#             mask = l1_vals > thresh
#             print(f"Block {block:15s} | Layer {name:25s} | ratio={ratio:.2f} | "
#                   f"thresh={thresh:.4f} | kept {mask.sum().item()}/{num_out}")

#         masks[name] = mask

#     #print(f"\n✅ Generated pruning masks for {len(masks)} layers.\n")
#     return masks


# # -----------------------------------------------------------------------------
# # 4️⃣ Compute actual prune ratios (unchanged)
# # -----------------------------------------------------------------------------
# def compute_actual_prune_ratios(original_model, pruned_model):
#     """
#     Compare two models (same architecture) and compute actual pruning ratios
#     for Conv2d layers (based on output channels).

#     Returns:
#         dict: {layer_name: prune_ratio}
#     """
#     ratios = {}
#     for (name_orig, layer_orig), (name_pruned, layer_pruned) in zip(
#         original_model.named_modules(), pruned_model.named_modules()
#     ):
#         if isinstance(layer_orig, nn.Conv2d) and isinstance(layer_pruned, nn.Conv2d):
#             if layer_orig.out_channels > 0:
#                 ratio = 1 - (layer_pruned.out_channels / layer_orig.out_channels)
#                 ratios[name_orig] = round(float(ratio), 4)
#     return ratios


# # -----------------------------------------------------------------------------
# # 5️⃣ Build model summary DataFrame
# # -----------------------------------------------------------------------------
# def model_to_dataframe_with_l1(
#     model: nn.Module,
#     l1_stats: dict,
#     block_ratios: dict = None,
#     post_prune_ratios: dict = None,
#     remove_nan_layers: bool = False
# ) -> pd.DataFrame:
#     """
#     Build a DataFrame summarizing all convolutional + batchnorm layers
#     in a UNet-like model.
#     """

#     layers = []

#     for name, layer in model.named_modules():
#         layer_type = layer.__class__.__name__

#         # Include Conv2d, ConvTranspose2d, BatchNorm2d
#         if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):

#             # Extract shape
#             shape = None
#             if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
#                 shape = tuple(layer.weight.shape)

#             num_params = sum(p.numel() for p in layer.parameters())

#             # Channels (BN uses num_features)
#             if isinstance(layer, nn.BatchNorm2d):
#                 in_ch = out_ch = layer.num_features
#             else:
#                 in_ch = layer.in_channels
#                 out_ch = layer.out_channels

#             stats = l1_stats.get(name, {})

#             layers.append({
#                 "Layer": name,
#                 "Type": layer_type,
#                 "Shape": shape,
#                 "In Ch": in_ch,
#                 "Out Ch": out_ch,
#                 "Num Params": num_params,
#                 **stats
#             })

#     df = pd.DataFrame(layers)

#     # Sort in UNet structure:
#     enc = df[df["Layer"].str.startswith("encoders")].copy()
#     bott = df[df["Layer"].str.startswith("bottleneck")].copy()
#     dec = df[df["Layer"].str.startswith("decoders") | df["Layer"].str.startswith("final_conv")].copy()

#     df_sorted = pd.concat([enc, bott, dec], ignore_index=True)

#     # Drop or keep NaN rows
#     if remove_nan_layers:
#         df_sorted = df_sorted.dropna(subset=["Mean L1"]).reset_index(drop=True)
#     else:
#         df_sorted = df_sorted.reset_index(drop=True)

#     # Add block ratios
#     df_sorted["Block Ratio"] = None
#     df_sorted["Post-Prune Ratio"] = None

#     if block_ratios:
#         for block_name, ratio in block_ratios.items():
#             mask = df_sorted["Layer"].str.startswith(block_name)
#             df_sorted.loc[mask, "Block Ratio"] = ratio

#     if post_prune_ratios:
#         for layer_name, ratio in post_prune_ratios.items():
#             df_sorted.loc[df_sorted["Layer"] == layer_name, "Post-Prune Ratio"] = ratio

#     return df_sorted



# # -----------------------------------------------------------------------------
# # 6️⃣ Visualization utilities for L1 distributions
# # -----------------------------------------------------------------------------
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# def compute_global_hist_max(norms: dict, bins=40, logx=False):
#     """
#     Compute the maximum histogram bin count across all layers,
#     so all plots can share the same y-axis scale.
#     """
#     global_max = 0

#     for vals in norms.values():
#         vals = vals.cpu().numpy()

#         # If logx, avoid issues with zeros/negatives in binning
#         if logx:
#             vals = vals[vals > 0]

#         counts, _ = np.histogram(vals, bins=bins)
#         if counts.size > 0:
#             global_max = max(global_max, int(counts.max()))

#     # Avoid ylim(0, 0) if something weird happens
#     return max(global_max, 1)


# def plot_l1_histograms(norms: dict, save_dir=None, bins=40, logx=False):
#     """
#     Plot per-layer L1 magnitude histograms with summary statistics.
#     Y-axes are shared across layers using the largest histogram bin count.
#     """
#     global_ymax = compute_global_hist_max(norms, bins=bins, logx=logx)

#     for name, vals in norms.items():
#         vals = vals.cpu().numpy()

#         mean = np.mean(vals)
#         median = np.median(vals)
#         p25, p75 = np.percentile(vals, [25, 75])
#         std = np.std(vals)
#         minv, maxv = np.min(vals), np.max(vals)

#         plt.figure(figsize=(6, 3))
#         sns.histplot(vals, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')

#         if logx:
#             plt.xscale('log')

#         # --- Shared y-axis scale across all layers ---
#         plt.ylim(0, global_ymax)

#         # --- Vertical lines for key stats ---
#         plt.axvline(mean, color='red', linestyle='--', linewidth=1, label=f"Mean = {mean:.2f}")
#         plt.axvline(median, color='orange', linestyle='-', linewidth=1.2, label=f"Median = {median:.2f}")
#         plt.axvline(p25, color='gray', linestyle=':', linewidth=1)
#         plt.axvline(p75, color='gray', linestyle=':', linewidth=1)

#         # --- Annotate stats box ---
#         textstr = (
#             f"min = {minv:.2f}\n"
#             f"max = {maxv:.2f}\n"
#         )
#         plt.gca().text(
#             0.98, 0.95, textstr, transform=plt.gca().transAxes,
#             fontsize=8, verticalalignment='top', horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6)
#         )

#         plt.title(name)
#         plt.xlabel("Per-filter L1 norm")
#         plt.ylabel("Count")
#         plt.legend(fontsize=8)
#         plt.tight_layout()

#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             path = os.path.join(save_dir, f"{name.replace('.', '_')}_hist.png")
#             plt.savefig(path, dpi=150)
#             plt.close()
#         else:
#             plt.show()


# def plot_l1_summary(df: pd.DataFrame, save_path=None):
#     """
#     Plot global layer-wise L1 mean and spread (e.g., barplot or boxplot).
#     """
#     plt.figure(figsize=(10, 4))
#     sns.barplot(data=df, x="Layer", y="Mean L1")
#     plt.xticks(rotation=90, fontsize=6)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()


# def inspect_model_l1(model, save_dir=None):
#     """
#     Compute, summarize, and visualize L1 distributions for a model.
#     """
#     norms = compute_l1_norms(model)
#     stats = compute_l1_stats(norms)
#     df = model_to_dataframe_with_l1(model, stats)

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         df.to_csv(os.path.join(save_dir, "l1_summary.csv"), index=False)
#         plot_l1_histograms(norms, save_dir=save_dir)
#         plot_l1_summary(df, save_path=os.path.join(save_dir, "l1_means.png"))
#     else:
#         plot_l1_histograms(norms)
#         plot_l1_summary(df)

#     return df

# # def plot_l1_histograms(norms: dict, save_dir=None, bins=40, logx=False):
# #     """
# #     Plot per-layer L1 magnitude histograms with summary statistics.
# #     """
# #     for name, vals in norms.items():
# #         vals = vals.cpu().numpy()
# #         mean = np.mean(vals)
# #         median = np.median(vals)
# #         p25, p75 = np.percentile(vals, [25, 75])
# #         std = np.std(vals)
# #         minv, maxv = np.min(vals), np.max(vals)

# #         plt.figure(figsize=(6,3))
# #         sns.histplot(vals, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
# #         if logx:
# #             plt.xscale('log')
        
# #         # --- Vertical lines for key stats ---
# #         plt.axvline(mean, color='red', linestyle='--', linewidth=1, label=f"Mean = {mean:.2f}")
# #         plt.axvline(median, color='orange', linestyle='-', linewidth=1.2, label=f"Median = {median:.2f}")
# #         plt.axvline(p25, color='gray', linestyle=':', linewidth=1)
# #         plt.axvline(p75, color='gray', linestyle=':', linewidth=1)

# #         # --- Annotate stats box ---
# #         textstr = (
# #             f"min = {minv:.2f}\n"
# #             f"max = {maxv:.2f}\n"
# #         )
# #         plt.gca().text(
# #             0.98, 0.95, textstr, transform=plt.gca().transAxes,
# #             fontsize=8, verticalalignment='top', horizontalalignment='right',
# #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6)
# #         )

# #         plt.title(name)
# #         plt.xlabel("Per-filter L1 norm")
# #         plt.ylabel("Count")
# #         plt.legend(fontsize=8)
# #         plt.tight_layout()

# #         if save_dir:
# #             import os
# #             os.makedirs(save_dir, exist_ok=True)
# #             path = os.path.join(save_dir, f"{name.replace('.', '_')}_hist.png")
# #             plt.savefig(path, dpi=150)
# #             plt.close()
# #         else:
# #             plt.show()


# # def plot_l1_summary(df: pd.DataFrame, save_path=None):
# #     """
# #     Plot global layer-wise L1 mean and spread (e.g., barplot or boxplot).
# #     """
# #     plt.figure(figsize=(10,4))
# #     sns.barplot(data=df, x="Layer", y="Mean L1")
# #     plt.xticks(rotation=90, fontsize=6)
# #     plt.tight_layout()
# #     if save_path:
# #         plt.savefig(save_path)
# #         plt.close()
# #     else:
# #         plt.show()


# # def inspect_model_l1(model, save_dir=None):
# #     """
# #     Compute, summarize, and visualize L1 distributions for a model.
# #     """
# #     norms = compute_l1_norms(model)
# #     stats = compute_l1_stats(norms)
# #     df = model_to_dataframe_with_l1(model, stats)

# #     if save_dir:
# #         os.makedirs(save_dir, exist_ok=True)
# #         df.to_csv(os.path.join(save_dir, "l1_summary.csv"), index=False)
# #         plot_l1_histograms(norms, save_dir=save_dir)
# #         plot_l1_summary(df, save_path=os.path.join(save_dir, "l1_means.png"))
# #     else:
# #         plot_l1_histograms(norms)
# #         plot_l1_summary(df)

# #     return df


import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# --- NEW: optional reproducibility (your helper file) ---
from src.utils.reproducibility import seed_everything


# -----------------------------------------------------------------------------
# 1️⃣ Compute per-layer L1 norms
# -----------------------------------------------------------------------------
def compute_l1_norms(model: nn.Module) -> dict:
    """
    Compute per-filter L1 norms for all Conv2d and ConvTranspose2d layers in a model.

    Returns:
        dict: {layer_name: torch.Tensor of shape [out_channels]}
    """
    norms = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            w = layer.weight.data.abs().view(layer.weight.size(0), -1)
            l1_vals = w.sum(dim=1)
            norms[name] = l1_vals
    return norms


# -----------------------------------------------------------------------------
# 2️⃣ Compute summary statistics from L1 norms
# -----------------------------------------------------------------------------
def compute_l1_stats(norms: dict) -> dict:
    """
    Compute summary statistics from per-filter L1 norms.

    Args:
        norms (dict): {layer_name: torch.Tensor of L1 values}

    Returns:
        dict: {layer_name: {'Mean L1','Min L1','Max L1','L1 Std'}}
    """
    stats = {}
    for name, l1_vals in norms.items():
        stats[name] = {
            "Mean L1": l1_vals.mean().item(),
            "Min L1": l1_vals.min().item(),
            "Max L1": l1_vals.max().item(),
            "L1 Std": l1_vals.std().item(),
        }
    return stats


def _infer_block_from_layer_name(name: str) -> str:
    """
    Map a module name like 'encoders.0.net.0' -> 'encoders.0'
    and 'bottleneck.net.0' -> 'bottleneck'
    """
    parts = name.split(".")
    if parts[0] == "bottleneck":
        return "bottleneck"
    if len(parts) >= 2:
        return ".".join(parts[:2])  # e.g. encoders.0, decoders.3
    return parts[0]


def _should_skip_for_pruning(layer_name: str, layer: nn.Module) -> bool:
    """
    Skip layers that should never be structurally pruned:
    - final classifier conv
    - ConvTranspose2d upsampling layers (changing them breaks shapes unless handled carefully)
    """
    if "final_conv" in layer_name:
        return True
    if isinstance(layer, nn.ConvTranspose2d):
        return True
    return False


# -----------------------------------------------------------------------------
# 3️⃣ Generate pruning masks (structured, block-wise)
# -----------------------------------------------------------------------------
def get_pruning_masks_blockwise(
    model: nn.Module,
    norms: dict,
    block_ratios: dict = None,
    default_ratio: float = 0.3,
    seed: int | None = None,
    deterministic: bool = False,
):
    """
    Generate structured pruning masks based on L1 norms and block ratios.

    Key improvement vs thresholding:
    - Uses top-k smallest L1 filters (tie-safe) so you prune exactly k filters.

    Args:
        model: Model to inspect.
        norms: {layer_name: L1 tensor per filter}
        block_ratios: {block_name: prune_ratio}
        default_ratio: default prune ratio if block not specified
        seed: optional global seed for consistency
        deterministic: if True, enable deterministic torch settings

    Returns:
        masks: {layer_name: torch.BoolTensor mask of kept channels}
    """
    if seed is not None:
        seed_everything(seed, deterministic=deterministic)

    masks = {}
    block_ratios = block_ratios or {}

    print("🔧 Generating pruning masks...\n")

    # For correct skip behavior, we need access to the layer objects too
    layer_lookup = dict(model.named_modules())

    for name, l1_vals in norms.items():
        num_out = int(l1_vals.numel())
        layer = layer_lookup.get(name, None)

        block = _infer_block_from_layer_name(name)
        ratio = float(block_ratios.get(block, default_ratio))

        # Skip layers that shouldn't be pruned structurally
        if layer is not None and _should_skip_for_pruning(name, layer):
            masks[name] = torch.ones(num_out, dtype=torch.bool)
            continue

        # Clamp ratio into [0,1]
        ratio = max(0.0, min(1.0, ratio))

        if ratio <= 0.0:
            mask = torch.ones(num_out, dtype=torch.bool)
            print(f"Block {block:15s} | ratio=0.00 → keeping all {num_out} filters.")
            masks[name] = mask
            continue

        if ratio >= 1.0:
            mask = torch.zeros(num_out, dtype=torch.bool)
            print(f"Block {block:15s} | ratio=1.00 → pruning all {num_out} filters.")
            masks[name] = mask
            continue

        # --- Tie-safe top-k pruning ---
        k_prune = int(np.floor(num_out * ratio))
        k_prune = max(0, min(num_out, k_prune))

        if k_prune == 0:
            mask = torch.ones(num_out, dtype=torch.bool)
            print(f"Block {block:15s} | Layer {name:25s} | ratio={ratio:.2f} | prune=0 → kept {num_out}/{num_out}")
            masks[name] = mask
            continue

        # Sort ascending (smallest L1 = least important)
        # argsort is deterministic given the same tensor values on CPU
        # (and still stable enough for our purposes)
        sorted_idx = torch.argsort(l1_vals, descending=False)

        prune_idx = sorted_idx[:k_prune]
        mask = torch.ones(num_out, dtype=torch.bool)
        mask[prune_idx] = False

        kept = int(mask.sum().item())
        print(
            f"Block {block:15s} | Layer {name:25s} | ratio={ratio:.2f} | "
            f"pruned {k_prune}/{num_out} | kept {kept}/{num_out}"
        )

        masks[name] = mask

    return masks


# -----------------------------------------------------------------------------
# 4️⃣ Compute actual prune ratios (unchanged)
# -----------------------------------------------------------------------------
def compute_actual_prune_ratios(original_model, pruned_model):
    """
    Compare two models (same architecture) and compute actual pruning ratios
    for Conv2d layers (based on output channels).
    """
    ratios = {}
    for (name_orig, layer_orig), (name_pruned, layer_pruned) in zip(
        original_model.named_modules(), pruned_model.named_modules()
    ):
        if isinstance(layer_orig, nn.Conv2d) and isinstance(layer_pruned, nn.Conv2d):
            if layer_orig.out_channels > 0:
                ratio = 1 - (layer_pruned.out_channels / layer_orig.out_channels)
                ratios[name_orig] = round(float(ratio), 4)
    return ratios


# -----------------------------------------------------------------------------
# 5️⃣ Build model summary DataFrame
# -----------------------------------------------------------------------------
def model_to_dataframe_with_l1(
    model: nn.Module,
    l1_stats: dict,
    block_ratios: dict = None,
    post_prune_ratios: dict = None,
    remove_nan_layers: bool = False
) -> pd.DataFrame:
    """
    Build a DataFrame summarizing all convolutional + batchnorm layers
    in a UNet-like model.
    """
    layers = []

    for name, layer in model.named_modules():
        layer_type = layer.__class__.__name__

        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            shape = None
            if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
                shape = tuple(layer.weight.shape)

            num_params = sum(p.numel() for p in layer.parameters())

            if isinstance(layer, nn.BatchNorm2d):
                in_ch = out_ch = layer.num_features
            else:
                in_ch = layer.in_channels
                out_ch = layer.out_channels

            stats = l1_stats.get(name, {})

            layers.append({
                "Layer": name,
                "Type": layer_type,
                "Shape": shape,
                "In Ch": in_ch,
                "Out Ch": out_ch,
                "Num Params": num_params,
                **stats
            })

    df = pd.DataFrame(layers)

    enc = df[df["Layer"].str.startswith("encoders")].copy()
    bott = df[df["Layer"].str.startswith("bottleneck")].copy()
    dec = df[df["Layer"].str.startswith("decoders") | df["Layer"].str.startswith("final_conv")].copy()

    df_sorted = pd.concat([enc, bott, dec], ignore_index=True)

    if remove_nan_layers:
        df_sorted = df_sorted.dropna(subset=["Mean L1"]).reset_index(drop=True)
    else:
        df_sorted = df_sorted.reset_index(drop=True)

    df_sorted["Block Ratio"] = None
    df_sorted["Post-Prune Ratio"] = None

    if block_ratios:
        for block_name, ratio in block_ratios.items():
            mask = df_sorted["Layer"].str.startswith(block_name)
            df_sorted.loc[mask, "Block Ratio"] = ratio

    if post_prune_ratios:
        for layer_name, ratio in post_prune_ratios.items():
            df_sorted.loc[df_sorted["Layer"] == layer_name, "Post-Prune Ratio"] = ratio

    return df_sorted


# -----------------------------------------------------------------------------
# 6️⃣ Visualization utilities for L1 distributions
# -----------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt


def compute_global_hist_max(norms: dict, bins=40, logx=False):
    global_max = 0
    for vals in norms.values():
        vals = vals.cpu().numpy()
        if logx:
            vals = vals[vals > 0]
        counts, _ = np.histogram(vals, bins=bins)
        if counts.size > 0:
            global_max = max(global_max, int(counts.max()))
    return max(global_max, 1)


def plot_l1_histograms(norms: dict, save_dir=None, bins=40, logx=False):
    global_ymax = compute_global_hist_max(norms, bins=bins, logx=logx)

    for name, vals in norms.items():
        vals = vals.cpu().numpy()

        mean = np.mean(vals)
        median = np.median(vals)
        p25, p75 = np.percentile(vals, [25, 75])
        minv, maxv = np.min(vals), np.max(vals)

        plt.figure(figsize=(6, 3))
        sns.histplot(vals, bins=bins, alpha=0.7, edgecolor="black")

        if logx:
            plt.xscale("log")

        plt.ylim(0, global_ymax)

        plt.axvline(mean, linestyle="--", linewidth=1, label=f"Mean = {mean:.2f}")
        plt.axvline(median, linestyle="-", linewidth=1.2, label=f"Median = {median:.2f}")
        plt.axvline(p25, linestyle=":", linewidth=1)
        plt.axvline(p75, linestyle=":", linewidth=1)

        textstr = f"min = {minv:.2f}\nmax = {maxv:.2f}\n"
        plt.gca().text(
            0.98, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
        )

        plt.title(name)
        plt.xlabel("Per-filter L1 norm")
        plt.ylabel("Count")
        plt.legend(fontsize=8)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{name.replace('.', '_')}_hist.png")
            plt.savefig(path, dpi=150)
            plt.close()
        else:
            plt.show()


def plot_l1_summary(df: pd.DataFrame, save_path=None):
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="Layer", y="Mean L1")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def inspect_model_l1(model, save_dir=None):
    norms = compute_l1_norms(model)
    stats = compute_l1_stats(norms)
    df = model_to_dataframe_with_l1(model, stats)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, "l1_summary.csv"), index=False)
        plot_l1_histograms(norms, save_dir=save_dir)
        plot_l1_summary(df, save_path=os.path.join(save_dir, "l1_means.png"))
    else:
        plot_l1_histograms(norms)
        plot_l1_summary(df)

    return df

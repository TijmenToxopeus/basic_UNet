# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np 


# def model_to_dataframe_with_l1(
#     model: nn.Module,
#     remove_nan_layers: bool = False,
#     block_ratios: dict = None,
#     post_prune_ratios: dict = None
# ):
#     """
#     Build a DataFrame summarizing all convolutional layers in a UNet-like model,
#     including L1 statistics, parameter counts, and optionally pruning ratios.

#     Args:
#         model (nn.Module): Model to inspect.
#         remove_nan_layers (bool): If True, drops layers without L1 stats (e.g. container modules).
#         block_ratios (dict, optional): Mapping {block_name: prune_ratio} to include in table.
#         post_prune_ratios (dict, optional): Mapping {layer_name: actual_prune_ratio} after pruning.

#     Returns:
#         pd.DataFrame: Layer summary with columns:
#         ['Layer', 'Type', 'Shape', 'In Ch', 'Out Ch',
#          'Num Params', 'Mean L1', 'Min L1', 'Max L1', 'L1 Std',
#          'Block Ratio', 'Post-Prune Ratio']
#     """
#     layers = []

#     def recurse(module, prefix=""):
#         for name, layer in module.named_children():
#             layer_name = f"{prefix}{name}"
#             layer_type = layer.__class__.__name__

#             # Recurse into container layers
#             if isinstance(layer, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
#                 recurse(layer, prefix + name + ".")
#                 continue

#             # Extract general attributes
#             shape = getattr(layer, "weight", None)
#             shape = tuple(shape.shape) if shape is not None and hasattr(shape, "shape") else None
#             num_params = sum(p.numel() for p in layer.parameters())
#             in_ch = getattr(layer, "in_channels", None)
#             out_ch = getattr(layer, "out_channels", None)

#             # Compute L1 stats for Conv layers
#             mean_l1 = min_l1 = max_l1 = std_l1 = None
#             if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
#                 w = layer.weight.data.abs().view(layer.weight.size(0), -1)
#                 l1_vals = w.sum(dim=1)
#                 mean_l1 = l1_vals.mean().item()
#                 min_l1 = l1_vals.min().item()
#                 max_l1 = l1_vals.max().item()
#                 std_l1 = l1_vals.std().item()

#             layers.append({
#                 "Layer": layer_name,
#                 "Type": layer_type,
#                 "Shape": shape,
#                 "In Ch": in_ch,
#                 "Out Ch": out_ch,
#                 "Num Params": num_params,
#                 "Mean L1": mean_l1,
#                 "Min L1": min_l1,
#                 "Max L1": max_l1,
#                 "L1 Std": std_l1,
#             })

#             recurse(layer, prefix + name + ".")

#     recurse(model)
#     df = pd.DataFrame(layers)

#     # Sort UNet layers (encoders → bottleneck → decoders)
#     enc = df[df["Layer"].str.startswith("encoders")].copy()
#     bott = df[df["Layer"].str.startswith("bottleneck")].copy()
#     dec = df[df["Layer"].str.startswith("decoders") | df["Layer"].str.startswith("final_conv")].copy()
#     df_sorted = pd.concat([enc, bott, dec], ignore_index=True)

#     if remove_nan_layers:
#         df_sorted = df_sorted.dropna(subset=["Mean L1"]).reset_index(drop=True)

#     # --- Add optional ratio columns ---
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


# def get_pruning_masks_blockwise(df, block_ratios=None, default_ratio=0.3):
#     """
#     Generate structured pruning masks for Conv2d layers, grouped by U-Net blocks.

#     Args:
#         df (pd.DataFrame): model summary with L1 stats.
#         block_ratios (dict): optional {block_name: prune_ratio} mapping.
#                              e.g. {"encoders.0": 0.2, "encoders.1": 0.3, "bottleneck": 0.4}
#         default_ratio (float): used for blocks not listed in block_ratios.

#     Returns:
#         dict: {layer_name: torch.BoolTensor mask}
#     """
#     masks = {}

#     # --- Filter only Conv2d layers (ignore final output conv) ---
#     conv_df = (
#         df[
#             (df["Type"].str.contains("Conv2d")) &
#             (~df["Layer"].str.contains("final_conv"))
#         ].dropna(subset=["Mean L1"])
#     )

#     print("🔧 Generating pruning masks...\n")

#     for _, row in conv_df.iterrows():
#         name = row["Layer"]
#         num_out = int(row["Out Ch"])

#         # Identify block name — normalize to avoid "bottleneck.net"
#         block = ".".join(name.split(".")[:2])
#         block = block.replace(".net", "")  # normalize

#         ratio = block_ratios.get(block, default_ratio) if block_ratios else default_ratio

#         # --- Handle edge cases explicitly ---
#         if ratio <= 0.0:
#             mask = torch.ones(num_out, dtype=torch.bool)
#             print(f"Block {block:15s} | ratio=0.00 → keeping all {num_out} filters.")
#         elif ratio >= 1.0:
#             mask = torch.zeros(num_out, dtype=torch.bool)
#             print(f"Block {block:15s} | ratio=1.00 → pruning all {num_out} filters.")
#         else:
#             # Reconstruct approximate per-filter L1 values
#             l1_vals = np.linspace(row["Min L1"], row["Max L1"], num_out)

#             # Compute threshold for this block
#             local_thresh = np.percentile(l1_vals, ratio * 100)
#             mask = torch.tensor(l1_vals > local_thresh)

#             num_kept = mask.sum().item()
#             print(f"Block {block:15s} | Layer {name:25s} | ratio={ratio:.2f} | "
#                   f"threshold={local_thresh:.4f} | kept {num_kept}/{num_out}")

#         masks[name] = mask

#     print(f"\n✅ Generated pruning masks for {len(masks)} Conv2d layers.\n")
#     return masks


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



import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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
        dict: {
            layer_name: {
                'Mean L1', 'Min L1', 'Max L1', 'L1 Std'
            }
        }
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


# -----------------------------------------------------------------------------
# 3️⃣ Generate pruning masks (structured, block-wise)
# -----------------------------------------------------------------------------
def get_pruning_masks_blockwise(model, norms, block_ratios=None, default_ratio=0.3):
    """
    Generate structured pruning masks based on L1 norms and block ratios.

    Args:
        model (nn.Module): Model to inspect.
        norms (dict): {layer_name: L1 tensor per filter}.
        block_ratios (dict, optional): {block_name: prune_ratio}.
        default_ratio (float): Default prune ratio for unspecified blocks.

    Returns:
        dict: {layer_name: torch.BoolTensor mask}
    """
    masks = {}
    print("🔧 Generating pruning masks...\n")

    for name, l1_vals in norms.items():
        num_out = len(l1_vals)

        # Identify block (e.g. encoders.0, bottleneck, decoders.1)
        parts = name.split(".")
        block = ".".join(parts[:2]) if parts[0] != "bottleneck" else "bottleneck"
        block = block.replace(".net", "")

        ratio = block_ratios.get(block, default_ratio) if block_ratios else default_ratio

        if ratio <= 0.0:
            mask = torch.ones(num_out, dtype=torch.bool)
            print(f"Block {block:15s} | ratio=0.00 → keeping all {num_out} filters.")
        elif ratio >= 1.0:
            mask = torch.zeros(num_out, dtype=torch.bool)
            print(f"Block {block:15s} | ratio=1.00 → pruning all {num_out} filters.")
        else:
            # Threshold pruning
            thresh = torch.quantile(l1_vals, ratio)
            mask = l1_vals > thresh
            print(f"Block {block:15s} | Layer {name:25s} | ratio={ratio:.2f} | "
                  f"thresh={thresh:.4f} | kept {mask.sum().item()}/{num_out}")

        masks[name] = mask

    print(f"\n✅ Generated pruning masks for {len(masks)} layers.\n")
    return masks


# -----------------------------------------------------------------------------
# 4️⃣ Compute actual prune ratios (unchanged)
# -----------------------------------------------------------------------------
def compute_actual_prune_ratios(original_model, pruned_model):
    """
    Compare two models (same architecture) and compute actual pruning ratios
    for Conv2d layers (based on output channels).

    Returns:
        dict: {layer_name: prune_ratio}
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
    Build a DataFrame summarizing all convolutional layers in a UNet-like model.

    Args:
        model (nn.Module): model to inspect
        l1_stats (dict): per-layer L1 statistics
        block_ratios (dict, optional): {block_name: prune_ratio}
        post_prune_ratios (dict, optional): {layer_name: actual_prune_ratio}
        remove_nan_layers (bool): remove non-conv layers

    Returns:
        pd.DataFrame
    """
    layers = []
    for name, layer in model.named_modules():
        layer_type = layer.__class__.__name__

        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            shape = tuple(layer.weight.shape)
            num_params = sum(p.numel() for p in layer.parameters())
            in_ch, out_ch = layer.in_channels, layer.out_channels
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

    # Sort: encoders → bottleneck → decoders
    enc = df[df["Layer"].str.startswith("encoders")].copy()
    bott = df[df["Layer"].str.startswith("bottleneck")].copy()
    dec = df[df["Layer"].str.startswith("decoders") | df["Layer"].str.startswith("final_conv")].copy()
    df_sorted = pd.concat([enc, bott, dec], ignore_index=True)

    if remove_nan_layers:
        df_sorted = df_sorted.dropna(subset=["Mean L1"]).reset_index(drop=True)

    # Add ratios
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

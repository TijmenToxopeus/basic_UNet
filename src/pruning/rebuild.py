# import torch
# import torch.nn as nn
# import os
# import json
# from pathlib import Path
# from src.models.unet import UNet
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# def find_prev_conv_name(current_name, masks):
#     """
#     Find the most recent Conv2d layer (with a mask) that precedes the given one.
#     Used to align input channels during pruning.
#     """
#     all_layers = list(masks.keys())
#     try:
#         idx = all_layers.index(current_name)
#         if idx > 0:
#             return all_layers[idx - 1]
#     except ValueError:
#         pass
#     return None



# def get_pruned_feature_sizes(model, masks):
#     """
#     Extract pruned encoder, bottleneck, and decoder feature sizes
#     from a U-Net given pruning masks.

#     Args:
#         model: U-Net model (before pruning)
#         masks: dict {layer_name: BoolTensor of kept filters}

#     Returns:
#         enc_features (list[int]): output channels per encoder block
#         bottleneck_out (int): bottleneck output channels
#         dec_features (list[int]): output channels per decoder DoubleConv
#     """
#     enc_features = []
#     dec_features = []

#     # --- Encoder blocks ---
#     for i in range(len(model.encoders)):
#         layer_name = f"encoders.{i}.net.3"
#         if layer_name in masks:
#             out_ch = int(masks[layer_name].sum())
#         else:
#             out_ch = model.encoders[i].net[3].out_channels
#         enc_features.append(out_ch)

#     # --- Bottleneck ---
#     bottleneck_name = "bottleneck.net.3"
#     if bottleneck_name in masks:
#         bottleneck_out = int(masks[bottleneck_name].sum())
#     else:
#         bottleneck_out = model.bottleneck.net[3].out_channels

#     # --- Decoder blocks (skip ConvTranspose2d layers) ---
#     num_decoders = len(model.decoders)
#     for i in range(1, num_decoders, 2):  # 1, 3, 5, 7, ...
#         layer_name = f"decoders.{i}.net.3"
#         if layer_name in masks:
#             out_ch = int(masks[layer_name].sum())
#         else:
#             # fallback: read from model
#             out_ch = model.decoders[i].net[3].out_channels
#         dec_features.append(out_ch)

#     # print(f"Encoder features: {enc_features}")
#     # print(f"Bottleneck out_channels: {bottleneck_out}")
#     # print(f"Decoder features: {dec_features}")

#     return enc_features, bottleneck_out, dec_features


# def build_pruned_unet(model, enc_features, dec_features=None, bottleneck_out=None):
#     """
#     Build a fresh UNet with reduced encoder and decoder features.
#     Allows asymmetric pruning.

#     Args:
#         model: Original UNet model (for reference)
#         enc_features (list[int]): encoder output channels per block
#         dec_features (list[int], optional): decoder output channels per block
#         bottleneck_out (int, optional): bottleneck output channels (if not same as enc_features[-1]*2)
#     """
#     device = next(model.parameters()).device

#     # If no decoder features provided, assume symmetric
#     if dec_features is None:
#         dec_features = list(reversed(enc_features))
#         print("⚠️ No dec_features provided — assuming symmetric decoder.")

#     # If bottleneck_out not provided, use model default
#     if bottleneck_out is None:
#         bottleneck_out = model.bottleneck.net[3].out_channels

#     # --- Build the pruned UNet ---
#     pruned_model = UNet(
#         in_ch=model.encoders[0].net[0].in_channels,
#         out_ch=model.final_conv.out_channels,
#         enc_features=enc_features,
#         dec_features=dec_features,
#         bottleneck_out=bottleneck_out
#     ).to(device)

#     print(f"✅ Built pruned UNet | enc: {enc_features}, dec: {dec_features}, bottleneck: {bottleneck_out}")
#     return pruned_model



# def prune_conv_weights(module, masks, name):
#     """Return pruned weights and biases for a single Conv2d."""
#     w, b = module.weight.data, module.bias.data if module.bias is not None else None
#     mask_out = masks[name]
#     keep_out = torch.where(mask_out)[0]
#     w_new, b_new = w[keep_out, :, :, :], b[keep_out] if b is not None else None

#     prev = find_prev_conv_name(name, masks)
#     if prev:
#         mask_in = masks[prev]
#         keep_in = torch.where(mask_in)[0]
#         w_new = w_new[:, keep_in, :, :]
#     return w_new, b_new


# def copy_pruned_weights(original, pruned, masks, verbose=True):
#     """
#     Copy pruned Conv2d weights from the original model to the new (pruned) model.
#     Includes shape consistency checks and detailed logging.
#     """
#     for name, mod in original.named_modules():
#         print(f"name: {name} , mod: {mod}")
#         # Skip non-conv layers, unmasked layers, or final conv
#         if not isinstance(mod, nn.Conv2d) or name not in masks or "final_conv" in name:
#             continue

#         try:
#             new_mod = dict(pruned.named_modules())[name]
#         except KeyError:
#             if verbose:
#                 print(f"⚠️  Layer {name} not found in pruned model — skipping.")
#             continue

#         # Generate pruned weights
#         w_new, b_new = prune_conv_weights(mod, masks, name)

#         # --- Shape checks ---
#         expected_shape = tuple(new_mod.weight.shape)
#         actual_shape = tuple(w_new.shape)

#         if expected_shape != actual_shape:
#             # print(f"❌ Shape mismatch in layer '{name}':")
#             # print(f"   Expected: {expected_shape}, Got: {actual_shape}")
#             # print(f"   (Hint: mismatch likely from previous layer pruning or wrong feature propagation)")
#             # # Optional: skip assignment to avoid crash
#             continue

#         # --- Assign weights if shapes match ---
#         new_mod.weight = nn.Parameter(w_new.clone())
#         if b_new is not None:
#             if new_mod.bias is not None and b_new.shape == new_mod.bias.shape:
#                 new_mod.bias = nn.Parameter(b_new.clone())
#             else:
#                 print(f"⚠️  Bias shape mismatch in {name}: skipping bias copy.")

#         if verbose:
#             print(f"Copied weights for {name} | shape: {w_new.shape}")

#     # print("🔧 Weight copying completed.")
#     return pruned




# def copy_all_weights_with_pruning(original, pruned, masks, prunable_layers, verbose=False):
#     """
#     Copies weights from the original UNet to the newly pruned UNet.
    
#     Behavior:
#     - If prunable_layers[name] == True:
#           Apply pruning logic (slice output channels using masks[name])
#     - If prunable_layers[name] == False:
#           Copy weights from original → pruned
#           But auto-correct shapes if input/output channels changed
#     """

#     orig_dict = original.state_dict()
#     pruned_dict = pruned.state_dict()
#     new_state = {}

#     for key, pruned_tensor in pruned_dict.items():

#         if key not in orig_dict:
#             if verbose:
#                 print(f"⚠️ {key} not found in original, keeping default")
#             new_state[key] = pruned_tensor
#             continue

#         orig_tensor = orig_dict[key]

#         # Module name = everything before ".weight" or ".bias"
#         module_name = key.rsplit(".", 1)[0]

#         # 1️⃣ CASE A — LAYER IS PRUNABLE
#         if module_name in prunable_layers and prunable_layers[module_name]:
#             mask = masks[module_name]

#             # Slice output channels (always dimension 0)
#             sliced = orig_tensor[mask]

#             # Slice input channels (dimension 1) if there is a parent mask
#             parent = find_parent_layer(module_name)
#             if parent in masks and orig_tensor.ndim >= 3:
#                 parent_mask = masks[parent]
#                 sliced = sliced[:, parent_mask, ...]

#             # Final safety: match pruned model shape
#             if sliced.shape != pruned_tensor.shape:
#                 if verbose:
#                     print(f"⚠️ Shape mismatch in {key}, adjusting…")
#                 # Resize by slicing or padding
#                 sliced = resize_tensor(sliced, pruned_tensor.shape)

#             new_state[key] = sliced.clone()

#             if verbose:
#                 print(f"✂️ Pruned copy → {key}: {tuple(sliced.shape)}")

#         # 2️⃣ CASE B — LAYER IS NON-PRUNABLE
#         else:
#             # Copy directly, but shapes might differ
#             tensor = orig_tensor.clone()

#             if tensor.shape != pruned_tensor.shape:

#                 if verbose:
#                     print(f"🔧 Correcting shape for {key}: {tensor.shape} → {pruned_tensor.shape}")

#                 tensor = resize_tensor(tensor, pruned_tensor.shape)

#             new_state[key] = tensor

#             if verbose:
#                 print(f"📥 Copied non-pruned layer → {key}: {tuple(new_state[key].shape)}")


#     # Load new full state
#     pruned.load_state_dict(new_state, strict=False)

#     if verbose:
#         print("\n✅ Weight reconstruction completed.\n")

#     return pruned


# def find_parent_layer(name):
#     """Finds the previous block in UNet based on naming pattern."""
    
#     parts = name.split(".")
    
#     # example: encoders.2.double_conv.1 → parent is encoders.2.double_conv.0
#     if parts[-1].isdigit():
#         # inside double_conv block
#         idx = int(parts[-1])
#         if idx > 0:
#             parts[-1] = str(idx - 1)
#             return ".".join(parts)

#     # fallback: no parent found
#     return None


# def resize_tensor(t, target_shape):
#     """
#     Resize a tensor to target_shape by:
#     - slicing if too large
#     - zero-padding if too small
#     Works for Conv2d, ConvTranspose2d, BatchNorm, Linear, etc.
#     """
#     import numpy as np
#     out = torch.zeros(target_shape, dtype=t.dtype, device=t.device)

#     slices = tuple(slice(0, min(s, ts)) for s, ts in zip(t.shape, target_shape))
#     out[slices] = t[slices]
#     return out



# def plot_unet_schematic(enc_features, dec_features, bottleneck_out, 
#                         in_ch=1, out_ch=1, figsize=(10, 6), title="U-Net Structure"):
#     """
#     Draws a simple schematic of the U-Net structure with channel counts.

#     Args:
#         enc_features (list[int]): encoder block output channels
#         dec_features (list[int]): decoder block output channels
#         bottleneck_out (int): bottleneck output channels
#         in_ch (int): input channels
#         out_ch (int): output channels
#         figsize (tuple): figure size
#         title (str): plot title
#     """

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 6)
#     ax.axis("off")

#     # --- layout constants ---
#     n = len(enc_features)
#     x_enc = 1
#     x_dec = 9
#     y_start = 5
#     y_step = 1

#     # --- draw encoder blocks ---
#     encoder_positions = []
#     for i, ch in enumerate(enc_features):
#         y = y_start - i * y_step
#         rect = patches.Rectangle((x_enc, y), 1.5, 0.6, facecolor="#66b3ff", edgecolor="k", lw=1.2)
#         ax.add_patch(rect)
#         ax.text(x_enc + 0.75, y + 0.3, f"{in_ch if i==0 else enc_features[i-1]} → {ch}", 
#                 ha="center", va="center", fontsize=9)
#         encoder_positions.append((x_enc + 1.5, y + 0.3))  # right edge center

#     # --- bottleneck ---
#     bottleneck_y = y_start - n * y_step
#     rect = patches.Rectangle((x_enc + 3.5, bottleneck_y), 1.5, 0.6, facecolor="#ffcc99", edgecolor="k", lw=1.2)
#     ax.add_patch(rect)
#     ax.text(x_enc + 4.25, bottleneck_y + 0.3, f"{enc_features[-1]} → {bottleneck_out}",
#             ha="center", va="center", fontsize=9)

#     # --- draw decoder blocks ---
#     decoder_positions = []
#     for i, ch in enumerate(dec_features):
#         y = bottleneck_y + (i + 1) * y_step
#         rect = patches.Rectangle((x_dec - 2.5, y), 1.5, 0.6, facecolor="#99ff99", edgecolor="k", lw=1.2)
#         ax.add_patch(rect)

#         # find skip connection source (mirror of encoder)
#         skip_src = encoder_positions[-(i + 1)]
#         ax.plot([skip_src[0], x_dec - 2.5], [skip_src[1], y + 0.3], 'k--', lw=0.8)

#         # text with channels
#         # in_ch_str = bottleneck_out if i == 0 else dec_features[i - 1]
#         in_ch_str = dec_features[i]
#         ax.text(x_dec - 1.75, y + 0.3, f"{enc_features[-(i + 1)]}+{in_ch_str} → {ch}",
#                 ha="center", va="center", fontsize=9)
#         decoder_positions.append((x_dec - 2.5, y + 0.3))

#     # --- final output ---
#     ax.arrow(x_dec - 1, y_start + 0.3, 1.0, 0, head_width=0.15, head_length=0.2, fc="k", ec="k")
#     ax.text(x_dec + 0.2, y_start + 0.3, f"{dec_features[-1]} → {out_ch}", va="center", fontsize=9)
#     ax.text(5, 5.7, title, fontsize=13, ha="center", fontweight="bold")

#     plt.show()

# # def inspect_batchnorm_stats(model, tag=""):
# #     """
# #     Print running_mean and running_var statistics for the first BatchNorm2d layer.
# #     Useful for diagnosing BN issues after pruning.
# #     """
# #     print(f"\n🔍 BatchNorm stats {tag}")

# #     for m in model.modules():
# #         if isinstance(m, nn.BatchNorm2d):
# #             mean_abs = m.running_mean.abs().mean().item()
# #             var_mean = m.running_var.mean().item()
# #             num_tracked = (
# #                 m.num_batches_tracked.item()
# #                 if hasattr(m, "num_batches_tracked")
# #                 else None
# #             )

# #             print(
# #                 f"BN |mean(running_mean)| = {mean_abs:.3e}, "
# #                 f"mean(running_var) = {var_mean:.3e}, "
# #                 f"num_batches_tracked = {num_tracked}"
# #             )
# #             return

# #     print("⚠️ No BatchNorm2d layers found in model.")

# # def recalibrate_batchnorm(
# #     model,
# #     dataloader,
# #     device,
# #     num_batches=10,
# # ):
# #     """
# #     Recompute BatchNorm running statistics after pruning.

# #     Args:
# #         model: pruned model
# #         dataloader: training dataloader
# #         device: torch device
# #         num_batches: number of batches used for recalibration
# #     """
# #     model.train()

# #     with torch.no_grad():
# #         for i, batch in enumerate(dataloader):
# #             if i >= num_batches:
# #                 break

# #             # Adjust key if your dataset uses a different naming
# #             x = batch["image"].to(device, non_blocking=True)
# #             model(x)



# def rebuild_pruned_unet(model, masks, save_path=None):
#     """Main orchestrator."""

#     print("🔧 Rebuilding pruned UNet architecture...")

#     def is_prunable(mod):
#         return isinstance(mod, nn.Conv2d)

#     prunable_layers = {
#         name: is_prunable(module)
#         for name, module in model.named_modules()
#     }

#     enc_features, bottleneck_out, dec_features = get_pruned_feature_sizes(model, masks)

#     pruned_model = build_pruned_unet(model, enc_features, dec_features=dec_features, bottleneck_out=bottleneck_out)
#     #pruned_model = copy_pruned_weights(model, pruned_model, masks)
#     pruned_model = copy_all_weights_with_pruning(model, pruned_model, masks, prunable_layers, verbose=False)

#     # inspect_batchnorm_stats(pruned_model, tag="before BN recalibration")

#     # recalibrate_batchnorm(
#     #     pruned_model,
#     #     train_loader,
#     #     device,
#     #     num_batches=10
#     # )

#     # inspect_batchnorm_stats(pruned_model, tag="after BN recalibration")

#     plot_unet_schematic(enc_features, dec_features, bottleneck_out, 
#                         in_ch=1, out_ch=4, figsize=(10, 6), title="U-Net Structure")


#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         torch.save(pruned_model.state_dict(), save_path)
#         print(f"💾 Saved pruned model to {save_path}")

#         meta = {
#             "enc_features": enc_features,
#             "dec_features": dec_features,
#             "bottleneck_out": bottleneck_out,
#         }

#         meta_path = save_path.with_name(save_path.stem + "_meta.json")
#         with open(meta_path, "w") as f:
#             json.dump(meta, f, indent=4)
#         #print(f"🧾 Saved metadata to {meta_path}")

#     print("✅ UNet successfully rebuilt.")
#     return pruned_model


# def load_full_pruned_model(meta, ckpt_path, in_ch, out_ch, device):
#     enc_features = meta["enc_features"]
#     dec_features = meta["dec_features"]
#     bottleneck_out = meta["bottleneck_out"]

#     base = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
#     model = build_pruned_unet(base, enc_features, dec_features, bottleneck_out).to(device)

#     state = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(state)
#     return model



import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from src.models.unet import UNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- NEW: optional reproducibility ---
from src.utils.reproducibility import seed_everything


def find_prev_conv_name(current_name, masks):
    """
    Find the most recent Conv2d layer (with a mask) that precedes the given one.
    Used to align input channels during pruning.
    """
    all_layers = list(masks.keys())
    try:
        idx = all_layers.index(current_name)
        if idx > 0:
            return all_layers[idx - 1]
    except ValueError:
        pass
    return None


def get_pruned_feature_sizes(model, masks):
    """
    Extract pruned encoder, bottleneck, and decoder feature sizes
    from a U-Net given pruning masks.

    Args:
        model: U-Net model (before pruning)
        masks: dict {layer_name: BoolTensor of kept filters}

    Returns:
        enc_features (list[int]): output channels per encoder block
        bottleneck_out (int): bottleneck output channels
        dec_features (list[int]): output channels per decoder DoubleConv
    """
    enc_features = []
    dec_features = []

    # --- Encoder blocks ---
    for i in range(len(model.encoders)):
        layer_name = f"encoders.{i}.net.3"
        if layer_name in masks:
            out_ch = int(masks[layer_name].sum())
        else:
            out_ch = model.encoders[i].net[3].out_channels
        enc_features.append(out_ch)

    # --- Bottleneck ---
    bottleneck_name = "bottleneck.net.3"
    if bottleneck_name in masks:
        bottleneck_out = int(masks[bottleneck_name].sum())
    else:
        bottleneck_out = model.bottleneck.net[3].out_channels

    # --- Decoder blocks (skip ConvTranspose2d layers) ---
    num_decoders = len(model.decoders)
    for i in range(1, num_decoders, 2):  # 1, 3, 5, 7, ...
        layer_name = f"decoders.{i}.net.3"
        if layer_name in masks:
            out_ch = int(masks[layer_name].sum())
        else:
            out_ch = model.decoders[i].net[3].out_channels
        dec_features.append(out_ch)

    return enc_features, bottleneck_out, dec_features


def build_pruned_unet(model, enc_features, dec_features=None, bottleneck_out=None):
    """
    Build a fresh UNet with reduced encoder and decoder features.
    Allows asymmetric pruning.

    Args:
        model: Original UNet model (for reference)
        enc_features (list[int]): encoder output channels per block
        dec_features (list[int], optional): decoder output channels per block
        bottleneck_out (int, optional): bottleneck output channels
    """
    device = next(model.parameters()).device

    if dec_features is None:
        dec_features = list(reversed(enc_features))
        print("⚠️ No dec_features provided — assuming symmetric decoder.")

    if bottleneck_out is None:
        bottleneck_out = model.bottleneck.net[3].out_channels

    pruned_model = UNet(
        in_ch=model.encoders[0].net[0].in_channels,
        out_ch=model.final_conv.out_channels,
        enc_features=enc_features,
        dec_features=dec_features,
        bottleneck_out=bottleneck_out
    ).to(device)

    print(f"✅ Built pruned UNet | enc: {enc_features}, dec: {dec_features}, bottleneck: {bottleneck_out}")
    return pruned_model


def prune_conv_weights(module, masks, name):
    """Return pruned weights and biases for a single Conv2d."""
    w = module.weight.data
    b = module.bias.data if module.bias is not None else None

    mask_out = masks[name]
    keep_out = torch.where(mask_out)[0]

    w_new = w[keep_out, :, :, :]
    b_new = b[keep_out] if b is not None else None

    prev = find_prev_conv_name(name, masks)
    if prev:
        mask_in = masks[prev]
        keep_in = torch.where(mask_in)[0]
        w_new = w_new[:, keep_in, :, :]

    return w_new, b_new


def copy_pruned_weights(original, pruned, masks, verbose=True):
    """
    Copy pruned Conv2d weights from the original model to the new (pruned) model.
    Includes shape consistency checks and detailed logging.
    """
    pruned_modules = dict(pruned.named_modules())

    for name, mod in original.named_modules():
        if not isinstance(mod, nn.Conv2d) or name not in masks or "final_conv" in name:
            continue

        if name not in pruned_modules:
            if verbose:
                print(f"⚠️  Layer {name} not found in pruned model — skipping.")
            continue

        new_mod = pruned_modules[name]

        w_new, b_new = prune_conv_weights(mod, masks, name)

        expected_shape = tuple(new_mod.weight.shape)
        actual_shape = tuple(w_new.shape)

        if expected_shape != actual_shape:
            if verbose:
                print(f"⚠️ Shape mismatch in {name}: expected {expected_shape}, got {actual_shape} — skipping.")
            continue

        new_mod.weight = nn.Parameter(w_new.clone())
        if b_new is not None:
            if new_mod.bias is not None and b_new.shape == new_mod.bias.shape:
                new_mod.bias = nn.Parameter(b_new.clone())
            elif verbose:
                print(f"⚠️ Bias shape mismatch in {name}: skipping bias copy.")

        if verbose:
            print(f"Copied weights for {name} | shape: {w_new.shape}")

    return pruned


def find_parent_layer(name):
    """Finds the previous block in UNet based on naming pattern."""
    parts = name.split(".")
    if parts[-1].isdigit():
        idx = int(parts[-1])
        if idx > 0:
            parts[-1] = str(idx - 1)
            return ".".join(parts)
    return None


def resize_tensor(t, target_shape):
    """
    Resize a tensor to target_shape by slicing/padding with zeros.
    Works for Conv2d, ConvTranspose2d, BatchNorm, Linear, etc.
    """
    out = torch.zeros(target_shape, dtype=t.dtype, device=t.device)
    slices = tuple(slice(0, min(s, ts)) for s, ts in zip(t.shape, target_shape))
    out[slices] = t[slices]
    return out


def copy_all_weights_with_pruning(
    original,
    pruned,
    masks,
    prunable_layers,
    verbose: bool = False,
    resize_log: list | None = None,
):
    """
    Copies weights from the original UNet to the newly pruned UNet.

    - If prunable_layers[module_name] == True:
          Apply pruning logic using masks[module_name]
    - Else:
          Copy weights, resizing if needed
    """
    orig_dict = original.state_dict()
    pruned_dict = pruned.state_dict()
    new_state = {}

    for key, pruned_tensor in pruned_dict.items():
        if key not in orig_dict:
            if verbose:
                print(f"⚠️ {key} not found in original, keeping default")
            new_state[key] = pruned_tensor
            continue

        orig_tensor = orig_dict[key]
        module_name = key.rsplit(".", 1)[0]

        if module_name in prunable_layers and prunable_layers[module_name]:
            mask = masks[module_name]
            sliced = orig_tensor[mask]

            parent = find_parent_layer(module_name)
            if parent in masks and orig_tensor.ndim >= 3:
                parent_mask = masks[parent]
                sliced = sliced[:, parent_mask, ...]

            if sliced.shape != pruned_tensor.shape:
                if verbose:
                    print(f"⚠️ Shape mismatch in {key}, adjusting…")
                if resize_log is not None:
                    resize_log.append({
                        "key": key,
                        "from": list(sliced.shape),
                        "to": list(pruned_tensor.shape),
                        "reason": "pruned_layer_resize",
                    })
                sliced = resize_tensor(sliced, pruned_tensor.shape)

            new_state[key] = sliced.clone()

            if verbose:
                print(f"✂️ Pruned copy → {key}: {tuple(sliced.shape)}")
        else:
            tensor = orig_tensor.clone()
            if tensor.shape != pruned_tensor.shape:
                if verbose:
                    print(f"🔧 Correcting shape for {key}: {tensor.shape} → {pruned_tensor.shape}")
                if resize_log is not None:
                    resize_log.append({
                        "key": key,
                        "from": list(tensor.shape),
                        "to": list(pruned_tensor.shape),
                        "reason": "non_pruned_layer_resize",
                    })
                tensor = resize_tensor(tensor, pruned_tensor.shape)

            new_state[key] = tensor
            if verbose:
                print(f"📥 Copied non-pruned layer → {key}: {tuple(new_state[key].shape)}")

    pruned.load_state_dict(new_state, strict=False)

    if verbose:
        print("\n✅ Weight reconstruction completed.\n")

    return pruned


def plot_unet_schematic(enc_features, dec_features, bottleneck_out,
                        in_ch=1, out_ch=1, figsize=(10, 6), title="U-Net Structure"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    n = len(enc_features)
    x_enc = 1
    x_dec = 9
    y_start = 5
    y_step = 1

    encoder_positions = []
    for i, ch in enumerate(enc_features):
        y = y_start - i * y_step
        rect = patches.Rectangle((x_enc, y), 1.5, 0.6, facecolor="#66b3ff", edgecolor="k", lw=1.2)
        ax.add_patch(rect)
        ax.text(x_enc + 0.75, y + 0.3, f"{in_ch if i==0 else enc_features[i-1]} → {ch}",
                ha="center", va="center", fontsize=9)
        encoder_positions.append((x_enc + 1.5, y + 0.3))

    bottleneck_y = y_start - n * y_step
    rect = patches.Rectangle((x_enc + 3.5, bottleneck_y), 1.5, 0.6, facecolor="#ffcc99", edgecolor="k", lw=1.2)
    ax.add_patch(rect)
    ax.text(x_enc + 4.25, bottleneck_y + 0.3, f"{enc_features[-1]} → {bottleneck_out}",
            ha="center", va="center", fontsize=9)

    for i, ch in enumerate(dec_features):
        y = bottleneck_y + (i + 1) * y_step
        rect = patches.Rectangle((x_dec - 2.5, y), 1.5, 0.6, facecolor="#99ff99", edgecolor="k", lw=1.2)
        ax.add_patch(rect)

        skip_src = encoder_positions[-(i + 1)]
        ax.plot([skip_src[0], x_dec - 2.5], [skip_src[1], y + 0.3], "k--", lw=0.8)

        in_ch_str = dec_features[i]
        ax.text(x_dec - 1.75, y + 0.3, f"{enc_features[-(i + 1)]}+{in_ch_str} → {ch}",
                ha="center", va="center", fontsize=9)

    ax.arrow(x_dec - 1, y_start + 0.3, 1.0, 0, head_width=0.15, head_length=0.2, fc="k", ec="k")
    ax.text(x_dec + 0.2, y_start + 0.3, f"{dec_features[-1]} → {out_ch}", va="center", fontsize=9)
    ax.text(5, 5.7, title, fontsize=13, ha="center", fontweight="bold")

    plt.show()


def rebuild_pruned_unet(model, masks, save_path=None, seed=None, deterministic=False):
    """
    Main orchestrator.

    Seeding is optional here. It only matters if you later enable anything
    that is stochastic (e.g., BN recalibration dataloader, random debug sampling).
    """
    if seed is not None:
        seed_everything(seed, deterministic=deterministic)

    print("🔧 Rebuilding pruned UNet architecture...")

    def is_prunable(mod):
        return isinstance(mod, nn.Conv2d)

    prunable_layers = {
        name: is_prunable(module)
        for name, module in model.named_modules()
    }

    enc_features, bottleneck_out, dec_features = get_pruned_feature_sizes(model, masks)

    pruned_model = build_pruned_unet(
        model,
        enc_features,
        dec_features=dec_features,
        bottleneck_out=bottleneck_out
    )

    resize_log = []
    pruned_model = copy_all_weights_with_pruning(
        model,
        pruned_model,
        masks,
        prunable_layers,
        verbose=False,
        resize_log=resize_log,
    )

    plot_unet_schematic(
        enc_features, dec_features, bottleneck_out,
        in_ch=1, out_ch=4, figsize=(10, 6),
        title="U-Net Structure"
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(pruned_model.state_dict(), save_path)
        print(f"💾 Saved pruned model to {save_path}")

        meta = {
            "enc_features": enc_features,
            "dec_features": dec_features,
            "bottleneck_out": bottleneck_out,
        }

        meta_path = save_path.with_name(save_path.stem + "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        if resize_log:
            resize_log_path = save_path.with_name(save_path.stem + "_resize_log.json")
            with open(resize_log_path, "w") as f:
                json.dump(resize_log, f, indent=2)
            print(f"🧾 Saved resize log: {resize_log_path}")

    print("✅ UNet successfully rebuilt.")
    return pruned_model


def load_full_pruned_model(meta, ckpt_path, in_ch, out_ch, device):
    enc_features = meta["enc_features"]
    dec_features = meta["dec_features"]
    bottleneck_out = meta["bottleneck_out"]

    base = UNet(in_ch=in_ch, out_ch=out_ch, enc_features=enc_features).to(device)
    model = build_pruned_unet(base, enc_features, dec_features, bottleneck_out).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model

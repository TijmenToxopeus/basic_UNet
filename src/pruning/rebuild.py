import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from src.models.unet import UNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
            # fallback: read from model
            out_ch = model.decoders[i].net[3].out_channels
        dec_features.append(out_ch)

    # print(f"Encoder features: {enc_features}")
    # print(f"Bottleneck out_channels: {bottleneck_out}")
    # print(f"Decoder features: {dec_features}")

    return enc_features, bottleneck_out, dec_features


def build_pruned_unet(model, enc_features, dec_features=None, bottleneck_out=None):
    """
    Build a fresh UNet with reduced encoder and decoder features.
    Allows asymmetric pruning.

    Args:
        model: Original UNet model (for reference)
        enc_features (list[int]): encoder output channels per block
        dec_features (list[int], optional): decoder output channels per block
        bottleneck_out (int, optional): bottleneck output channels (if not same as enc_features[-1]*2)
    """
    device = next(model.parameters()).device

    # If no decoder features provided, assume symmetric
    if dec_features is None:
        dec_features = list(reversed(enc_features))
        print("⚠️ No dec_features provided — assuming symmetric decoder.")

    # If bottleneck_out not provided, use model default
    if bottleneck_out is None:
        bottleneck_out = model.bottleneck.net[3].out_channels

    # --- Build the pruned UNet ---
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
    w, b = module.weight.data, module.bias.data if module.bias is not None else None
    mask_out = masks[name]
    keep_out = torch.where(mask_out)[0]
    w_new, b_new = w[keep_out, :, :, :], b[keep_out] if b is not None else None

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
    for name, mod in original.named_modules():
        # Skip non-conv layers, unmasked layers, or final conv
        if not isinstance(mod, nn.Conv2d) or name not in masks or "final_conv" in name:
            continue

        try:
            new_mod = dict(pruned.named_modules())[name]
        except KeyError:
            if verbose:
                print(f"⚠️  Layer {name} not found in pruned model — skipping.")
            continue

        # Generate pruned weights
        w_new, b_new = prune_conv_weights(mod, masks, name)

        # --- Shape checks ---
        expected_shape = tuple(new_mod.weight.shape)
        actual_shape = tuple(w_new.shape)

        if expected_shape != actual_shape:
            # print(f"❌ Shape mismatch in layer '{name}':")
            # print(f"   Expected: {expected_shape}, Got: {actual_shape}")
            # print(f"   (Hint: mismatch likely from previous layer pruning or wrong feature propagation)")
            # # Optional: skip assignment to avoid crash
            continue

        # --- Assign weights if shapes match ---
        new_mod.weight = nn.Parameter(w_new.clone())
        if b_new is not None:
            if new_mod.bias is not None and b_new.shape == new_mod.bias.shape:
                new_mod.bias = nn.Parameter(b_new.clone())
            else:
                print(f"⚠️  Bias shape mismatch in {name}: skipping bias copy.")

        if verbose:
            print(f"Copied weights for {name} | shape: {w_new.shape}")

    # print("🔧 Weight copying completed.")
    return pruned


def plot_unet_schematic(enc_features, dec_features, bottleneck_out, 
                        in_ch=1, out_ch=1, figsize=(10, 6), title="U-Net Structure"):
    """
    Draws a simple schematic of the U-Net structure with channel counts.

    Args:
        enc_features (list[int]): encoder block output channels
        dec_features (list[int]): decoder block output channels
        bottleneck_out (int): bottleneck output channels
        in_ch (int): input channels
        out_ch (int): output channels
        figsize (tuple): figure size
        title (str): plot title
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # --- layout constants ---
    n = len(enc_features)
    x_enc = 1
    x_dec = 9
    y_start = 5
    y_step = 1

    # --- draw encoder blocks ---
    encoder_positions = []
    for i, ch in enumerate(enc_features):
        y = y_start - i * y_step
        rect = patches.Rectangle((x_enc, y), 1.5, 0.6, facecolor="#66b3ff", edgecolor="k", lw=1.2)
        ax.add_patch(rect)
        ax.text(x_enc + 0.75, y + 0.3, f"{in_ch if i==0 else enc_features[i-1]} → {ch}", 
                ha="center", va="center", fontsize=9)
        encoder_positions.append((x_enc + 1.5, y + 0.3))  # right edge center

    # --- bottleneck ---
    bottleneck_y = y_start - n * y_step
    rect = patches.Rectangle((x_enc + 3.5, bottleneck_y), 1.5, 0.6, facecolor="#ffcc99", edgecolor="k", lw=1.2)
    ax.add_patch(rect)
    ax.text(x_enc + 4.25, bottleneck_y + 0.3, f"{enc_features[-1]} → {bottleneck_out}",
            ha="center", va="center", fontsize=9)

    # --- draw decoder blocks ---
    decoder_positions = []
    for i, ch in enumerate(dec_features):
        y = bottleneck_y + (i + 1) * y_step
        rect = patches.Rectangle((x_dec - 2.5, y), 1.5, 0.6, facecolor="#99ff99", edgecolor="k", lw=1.2)
        ax.add_patch(rect)

        # find skip connection source (mirror of encoder)
        skip_src = encoder_positions[-(i + 1)]
        ax.plot([skip_src[0], x_dec - 2.5], [skip_src[1], y + 0.3], 'k--', lw=0.8)

        # text with channels
        # in_ch_str = bottleneck_out if i == 0 else dec_features[i - 1]
        in_ch_str = dec_features[i]
        ax.text(x_dec - 1.75, y + 0.3, f"{enc_features[-(i + 1)]}+{in_ch_str} → {ch}",
                ha="center", va="center", fontsize=9)
        decoder_positions.append((x_dec - 2.5, y + 0.3))

    # --- final output ---
    ax.arrow(x_dec - 1, y_start + 0.3, 1.0, 0, head_width=0.15, head_length=0.2, fc="k", ec="k")
    ax.text(x_dec + 0.2, y_start + 0.3, f"{dec_features[-1]} → {out_ch}", va="center", fontsize=9)
    ax.text(5, 5.7, title, fontsize=13, ha="center", fontweight="bold")

    plt.show()



def rebuild_pruned_unet(model, masks, save_path=None):
    """Main orchestrator."""

    print("🔧 Rebuilding pruned UNet architecture...")

    enc_features, bottleneck_out, dec_features = get_pruned_feature_sizes(model, masks)

    pruned_model = build_pruned_unet(model, enc_features, dec_features=dec_features, bottleneck_out=bottleneck_out)
    pruned_model = copy_pruned_weights(model, pruned_model, masks)

    plot_unet_schematic(enc_features, dec_features, bottleneck_out, 
                        in_ch=1, out_ch=4, figsize=(10, 6), title="U-Net Structure")


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        #print(f"🧾 Saved metadata to {meta_path}")

    print("✅ UNet successfully rebuilt.")
    return pruned_model

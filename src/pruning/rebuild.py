import torch
import torch.nn as nn
import os
from src.models.unet import UNet

def rebuild_pruned_unet(model, masks, save_path=None):
    """
    Rebuild a U-Net model after structured pruning using L1 masks.

    Args:
        model (UNet): trained UNet model (before pruning)
        masks (dict): {layer_name: BoolTensor of kept filters}
        save_path (str, optional): where to save the new model

    Returns:
        pruned_model (UNet): rebuilt UNet with reduced feature maps
    """
    device = next(model.parameters()).device
    print("🔧 Rebuilding pruned UNet architecture...")

    # --- Determine encoder feature sizes after pruning ---
    enc_features = []
    for i in range(len(model.encoders)):
        layer_name = f"encoders.{i}.net.3"  # second Conv2d in each DoubleConv
        if layer_name in masks:
            out_ch = masks[layer_name].sum().item()
        else:
            out_ch = model.encoders[i].net[3].out_channels
        enc_features.append(int(out_ch))

    # dec_features = [358, 204, 114, 57]  # hardcoded for pruned model

    # --- Bottleneck output channels ---
    bottleneck_layer = "bottleneck.net.3"
    if bottleneck_layer in masks:
        bottleneck_out = int(masks[bottleneck_layer].sum().item())
    else:
        bottleneck_out = model.bottleneck.net[3].out_channels

    print(f"Encoder features (after pruning): {enc_features}")
    print(f"Bottleneck out_channels: {bottleneck_out}")

    # --- Build new UNet with pruned feature sizes ---
    pruned_model = UNet(
        in_ch=model.encoders[0].net[0].in_channels,
        out_ch=model.final_conv.out_channels,
        features=enc_features
    ).to(device)

    # pruned_model = UNet(
    #     in_ch=model.encoders[0].net[0].in_channels,
    #     out_ch=model.final_conv.out_channels,
    #     encoder_features=enc_features,
    #     decoder_features=dec_features,
    #     bottleneck_features=bottleneck_out
    # ).to(device)

    # --- Copy and prune weights for each Conv2d layer ---
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if name not in masks:
            continue

        old_weight = module.weight.data
        old_bias = module.bias.data if module.bias is not None else None
        mask_out = masks[name]

        # prune output filters
        keep_out = torch.where(mask_out)[0]
        new_weight = old_weight[keep_out, :, :, :]
        new_bias = old_bias[keep_out] if old_bias is not None else None

        # prune input channels if previous layer was pruned
        prev_name = find_prev_conv_name(name, masks)
        if prev_name:
            mask_in = masks[prev_name]
            keep_in = torch.where(mask_in)[0]
            new_weight = new_weight[:, keep_in, :, :]

        # assign pruned weights to new model layer
        try:
            new_module = dict(pruned_model.named_modules())[name]
            new_module.weight = nn.Parameter(new_weight.clone())
            if new_bias is not None:
                new_module.bias = nn.Parameter(new_bias.clone())
        except KeyError:
            # skip layers that no longer exist (due to channel mismatch)
            continue

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(pruned_model.state_dict(), save_path)
        print(f"💾 Saved pruned model to: {save_path}")

    print("✅ UNet successfully rebuilt.")
    return pruned_model


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


def extract_architecture_from_masks(masks):
    """Return encoder_features, bottleneck_features, decoder_features lists."""

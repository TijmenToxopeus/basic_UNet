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


def _mask_or_all(mask: torch.Tensor | None, out_ch: int) -> torch.Tensor:
    if mask is None:
        return torch.ones(out_ch, dtype=torch.bool)
    if mask.numel() != out_ch:
        raise ValueError(f"Mask length {mask.numel()} != out_ch {out_ch}")
    return mask.bool()


def _collect_unet_masks(model: UNet, masks: dict) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    enc_masks: list[torch.Tensor] = []
    for i in range(len(model.encoders)):
        out_ch = model.encoders[i].net[3].out_channels
        m = masks.get(f"encoders.{i}.net.3", None)
        if m is None:
            m = masks.get(f"encoders.{i}.net.0", None)
        enc_masks.append(_mask_or_all(m, out_ch))

    bottleneck_out = model.bottleneck.net[3].out_channels
    m = masks.get("bottleneck.net.3", None)
    if m is None:
        m = masks.get("bottleneck.net.0", None)
    bottleneck_mask = _mask_or_all(m, bottleneck_out)

    dec_masks: list[torch.Tensor] = []
    for j in range(0, len(model.decoders), 2):
        out_ch = model.decoders[j + 1].net[3].out_channels
        name = f"decoders.{j+1}.net.3"
        m = masks.get(name, None)
        if m is None:
            m = masks.get(f"decoders.{j+1}.net.0", None)
        dec_masks.append(_mask_or_all(m, out_ch))

    if any(int(m.sum()) == 0 for m in enc_masks):
        raise ValueError("One or more encoder blocks pruned to zero channels.")
    if int(bottleneck_mask.sum()) == 0:
        raise ValueError("Bottleneck pruned to zero channels.")
    if any(int(m.sum()) == 0 for m in dec_masks):
        raise ValueError("One or more decoder blocks pruned to zero channels.")

    return enc_masks, bottleneck_mask, dec_masks


def _idx(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.where(mask.to(device))[0]


@torch.no_grad()
def copy_unet_weights_strict(
    original: UNet,
    pruned: UNet,
    enc_masks: list[torch.Tensor],
    bottleneck_mask: torch.Tensor,
    dec_masks: list[torch.Tensor],
) -> None:
    """
    Strict UNet weight copy using mask-consistent channel slicing.
    Raises if any shape mismatch occurs.
    """
    dev = next(pruned.parameters()).device

    # ---------- Encoder ----------
    for i in range(len(original.encoders)):
        enc_src = original.encoders[i]
        enc_dst = pruned.encoders[i]

        m_out = enc_masks[i]
        m_in = enc_masks[i - 1] if i > 0 else torch.ones(enc_src.net[0].in_channels, dtype=torch.bool)

        out_idx = _idx(m_out, dev)
        in_idx = _idx(m_in, dev)

        # conv0
        w0 = enc_src.net[0].weight.data[out_idx][:, in_idx, :, :]
        if w0.shape != enc_dst.net[0].weight.shape:
            raise ValueError(f"Encoder {i} conv0 shape mismatch: {w0.shape} vs {enc_dst.net[0].weight.shape}")
        enc_dst.net[0].weight.data.copy_(w0)

        # bn0
        for attr in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(enc_src.net[1], attr)
            d = getattr(enc_dst.net[1], attr)
            v = t.data[out_idx]
            if v.shape != d.shape:
                raise ValueError(f"Encoder {i} bn0 {attr} mismatch: {v.shape} vs {d.shape}")
            d.data.copy_(v)

        # conv1
        w1 = enc_src.net[3].weight.data[out_idx][:, out_idx, :, :]
        if w1.shape != enc_dst.net[3].weight.shape:
            raise ValueError(f"Encoder {i} conv1 shape mismatch: {w1.shape} vs {enc_dst.net[3].weight.shape}")
        enc_dst.net[3].weight.data.copy_(w1)

        # bn1
        for attr in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(enc_src.net[4], attr)
            d = getattr(enc_dst.net[4], attr)
            v = t.data[out_idx]
            if v.shape != d.shape:
                raise ValueError(f"Encoder {i} bn1 {attr} mismatch: {v.shape} vs {d.shape}")
            d.data.copy_(v)

    # ---------- Bottleneck ----------
    m_out = bottleneck_mask
    m_in = enc_masks[-1]
    out_idx = _idx(m_out, dev)
    in_idx = _idx(m_in, dev)

    w0 = original.bottleneck.net[0].weight.data[out_idx][:, in_idx, :, :]
    if w0.shape != pruned.bottleneck.net[0].weight.shape:
        raise ValueError(f"Bottleneck conv0 mismatch: {w0.shape} vs {pruned.bottleneck.net[0].weight.shape}")
    pruned.bottleneck.net[0].weight.data.copy_(w0)

    for attr in ("weight", "bias", "running_mean", "running_var"):
        t = getattr(original.bottleneck.net[1], attr)
        d = getattr(pruned.bottleneck.net[1], attr)
        v = t.data[out_idx]
        if v.shape != d.shape:
            raise ValueError(f"Bottleneck bn0 {attr} mismatch: {v.shape} vs {d.shape}")
        d.data.copy_(v)

    w1 = original.bottleneck.net[3].weight.data[out_idx][:, out_idx, :, :]
    if w1.shape != pruned.bottleneck.net[3].weight.shape:
        raise ValueError(f"Bottleneck conv1 mismatch: {w1.shape} vs {pruned.bottleneck.net[3].weight.shape}")
    pruned.bottleneck.net[3].weight.data.copy_(w1)

    for attr in ("weight", "bias", "running_mean", "running_var"):
        t = getattr(original.bottleneck.net[4], attr)
        d = getattr(pruned.bottleneck.net[4], attr)
        v = t.data[out_idx]
        if v.shape != d.shape:
            raise ValueError(f"Bottleneck bn1 {attr} mismatch: {v.shape} vs {d.shape}")
        d.data.copy_(v)

    # ---------- Decoder ----------
    num_blocks = len(dec_masks)
    for j in range(num_blocks):
        up_src = original.decoders[2 * j]
        up_dst = pruned.decoders[2 * j]
        dec_src = original.decoders[2 * j + 1]
        dec_dst = pruned.decoders[2 * j + 1]

        up_in_mask = bottleneck_mask if j == 0 else dec_masks[j - 1]
        up_out_mask = dec_masks[j]
        up_in_idx = _idx(up_in_mask, dev)
        up_out_idx = _idx(up_out_mask, dev)

        # ConvTranspose2d weight: [in, out, k, k]
        w_up = up_src.weight.data[up_in_idx][:, up_out_idx, :, :]
        if w_up.shape != up_dst.weight.shape:
            raise ValueError(f"Decoder {j} upconv mismatch: {w_up.shape} vs {up_dst.weight.shape}")
        up_dst.weight.data.copy_(w_up)
        if up_src.bias is not None and up_dst.bias is not None:
            b = up_src.bias.data[up_out_idx]
            if b.shape != up_dst.bias.shape:
                raise ValueError(f"Decoder {j} upconv bias mismatch: {b.shape} vs {up_dst.bias.shape}")
            up_dst.bias.data.copy_(b)

        # DoubleConv input is concat(skip, up)
        skip_mask = enc_masks[-(j + 1)]
        skip_idx = _idx(skip_mask, dev)
        up_idx = up_out_idx

        orig_skip_ch = original.encoders[-(j + 1)].net[3].out_channels
        input_idx = torch.cat([skip_idx, orig_skip_ch + up_idx], dim=0)

        out_idx = up_out_idx
        w0 = dec_src.net[0].weight.data[out_idx][:, input_idx, :, :]
        if w0.shape != dec_dst.net[0].weight.shape:
            raise ValueError(f"Decoder {j} conv0 mismatch: {w0.shape} vs {dec_dst.net[0].weight.shape}")
        dec_dst.net[0].weight.data.copy_(w0)

        for attr in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(dec_src.net[1], attr)
            d = getattr(dec_dst.net[1], attr)
            v = t.data[out_idx]
            if v.shape != d.shape:
                raise ValueError(f"Decoder {j} bn0 {attr} mismatch: {v.shape} vs {d.shape}")
            d.data.copy_(v)

        w1 = dec_src.net[3].weight.data[out_idx][:, out_idx, :, :]
        if w1.shape != dec_dst.net[3].weight.shape:
            raise ValueError(f"Decoder {j} conv1 mismatch: {w1.shape} vs {dec_dst.net[3].weight.shape}")
        dec_dst.net[3].weight.data.copy_(w1)

        for attr in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(dec_src.net[4], attr)
            d = getattr(dec_dst.net[4], attr)
            v = t.data[out_idx]
            if v.shape != d.shape:
                raise ValueError(f"Decoder {j} bn1 {attr} mismatch: {v.shape} vs {d.shape}")
            d.data.copy_(v)

    # ---------- Final conv ----------
    final_in_mask = dec_masks[-1]
    in_idx = _idx(final_in_mask, dev)
    w_final = original.final_conv.weight.data[:, in_idx, :, :]
    if w_final.shape != pruned.final_conv.weight.shape:
        raise ValueError(f"Final conv mismatch: {w_final.shape} vs {pruned.final_conv.weight.shape}")
    pruned.final_conv.weight.data.copy_(w_final)
    if original.final_conv.bias is not None and pruned.final_conv.bias is not None:
        pruned.final_conv.bias.data.copy_(original.final_conv.bias.data)

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
                    print(f"⚠️ Non-pruned shape mismatch for {key}: {tensor.shape} vs {pruned_tensor.shape} — keeping init")
                if resize_log is not None:
                    resize_log.append({
                        "key": key,
                        "from": list(tensor.shape),
                        "to": list(pruned_tensor.shape),
                        "reason": "non_pruned_mismatch_kept_init",
                    })
                new_state[key] = pruned_tensor
            else:
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

    enc_masks, bottleneck_mask, dec_masks = _collect_unet_masks(model, masks)
    enc_features = [int(m.sum().item()) for m in enc_masks]
    bottleneck_out = int(bottleneck_mask.sum().item())
    dec_features = [int(m.sum().item()) for m in dec_masks]

    pruned_model = build_pruned_unet(
        model,
        enc_features,
        dec_features=dec_features,
        bottleneck_out=bottleneck_out
    )

    copy_unet_weights_strict(
        original=model,
        pruned=pruned_model,
        enc_masks=enc_masks,
        bottleneck_mask=bottleneck_mask,
        dec_masks=dec_masks,
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

        resize_log_path = save_path.with_name(save_path.stem + "_resize_log.json")
        if resize_log_path.exists():
            print(f"🧾 Resize log exists: {resize_log_path}")

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

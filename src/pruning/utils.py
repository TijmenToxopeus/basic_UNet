import os
import torch
import torch.nn as nn
import copy
import yaml
import numpy as np

from src.models.unet import UNet

# def compute_block_l1_norms(model):
#     """Return {block_name: tensor of L1 norms per filter (from first conv)}"""

# def get_pruning_masks(norms, prune_ratio=0.3, per_block=True, global_prune=False):
#     """Generate binary masks (1 = keep, 0 = prune) based on norms."""

# def count_parameters(model):
#     """Return number of trainable parameters."""

# def apply_pruning_masks(model, masks):
#     """Zero out pruned filters (optional) before rebuild."""

# def save_pruning_stats(norms, masks, save_dir):
#     """Dump per-layer pruning statistics to JSON."""





def load_prune_config(path="configs/prune.yaml"):
    """Load pruning configuration from YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["prune"]



def compute_block_l1_norms(model: nn.Module, include_transpose=True):
    """
    Compute L1 norms of each filter in all Conv2d (and optionally ConvTranspose2d) layers.
    
    Returns:
        dict: {layer_name: tensor of L1 norms per output channel}
    """
    norms = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or (include_transpose and isinstance(module, nn.ConvTranspose2d)):
            # Flatten each filter and sum absolute values
            weight = module.weight.data.abs()
            l1_per_filter = weight.view(weight.size(0), -1).sum(dim=1)
            norms[name] = l1_per_filter

    return norms



# ---------------------------------------------------------
# 2️⃣ Get pruning masks per block
# ---------------------------------------------------------
def get_pruning_masks(norms, ratios, default_ratio=0.3, global_prune=False):
    """
    Compute boolean masks for each block based on L1 norms.
    Supports per-block pruning ratios and global thresholding.
    """
    masks = {}

    if global_prune:
        # Combine all norms for a single threshold
        all_l1 = torch.cat(list(norms.values()))
        num_prune = int(default_ratio * len(all_l1))
        global_thresh = torch.topk(all_l1, num_prune, largest=False).values.max()

    for name, l1 in norms.items():
        # Determine block-specific ratio
        ratio = ratios.get(name, default_ratio)
        num_filters = len(l1)
        num_prune = int(ratio * num_filters)

        if global_prune:
            mask = l1 > global_thresh
        else:
            if num_prune == 0:
                mask = torch.ones_like(l1, dtype=torch.bool)
            else:
                threshold = torch.topk(l1, num_prune, largest=False).values.max()
                mask = l1 > threshold

        masks[name] = mask

    return masks




# def rebuild_pruned_unet(model, masks, save_path="/media/ttoxopeus/basic_UNet/results/pruned_models/model_pruned_structured.pth"):
#     """
#     Build a new smaller UNet with pruned filters physically removed.
#     Ensures encoder-decoder symmetry and saves the model.
#     """
#     # --- Phase 1: Determine kept filters per block ---
#     enc_keepers = []
#     dec_keepers = []
#     bottleneck_keepers = None

#     for name, mask in masks.items():
#         if "encoders" in name:
#             enc_keepers.append(mask.sum().item())
#         elif "bottleneck" in name:
#             bottleneck_keepers = mask.sum().item()
#         elif "decoders" in name:
#             dec_keepers.append(mask.sum().item())

#     dec_keepers = dec_keepers[::-1]  # reverse decoder order

#     print("\n📊 Kept filters per block:")
#     print(f"Encoders: {enc_keepers}")
#     print(f"Bottleneck: {bottleneck_keepers}")
#     print(f"Decoders: {dec_keepers}")

#     # --- Phase 2: Create new UNet with reduced channels ---
#     in_ch = model.encoders[0].net[0].in_channels
#     new_features = enc_keepers
#     new_model = model.__class__(in_ch=in_ch, out_ch=model.final_conv.out_channels,
#                                 features=new_features)

#     # --- Phase 3: Copy surviving weights ---
#     old_state = model.state_dict()
#     new_state = new_model.state_dict()

#     for name, param in old_state.items():
#         block_name = ".".join(name.split(".")[:2])
#         if block_name not in masks:
#             continue

#         mask = masks[block_name]
#         if "weight" in name and param.ndim == 4:
#             kept_idx = mask.nonzero(as_tuple=False).squeeze(1)
#             pruned_weight = param[kept_idx, ...]
#             if pruned_weight.size(1) > new_state[name].size(1):
#                 pruned_weight = pruned_weight[:, :new_state[name].size(1), :, :]
#             new_state[name].copy_(pruned_weight)
#         elif "bias" in name and name in new_state:
#             kept_idx = mask.nonzero(as_tuple=False).squeeze(1)
#             pruned_bias = param[kept_idx]
#             new_state[name].copy_(pruned_bias)

#     new_model.load_state_dict(new_state)
#     print("✅ Rebuilt new UNet with physically pruned filters.")

#     # --- Save pruned model ---
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     torch.save(new_model.state_dict(), save_path)
#     print(f"💾 Saved pruned model to: {save_path}")

#     return new_model

def rebuild_pruned_unet(model, masks, save_path=None):
    """
    Rebuild a smaller UNet with physically removed filters.
    Computes decoder channel sizes from encoder + bottleneck masks to
    maintain skip-connection symmetry.
    """
    # ------------------------------
    # 1️⃣ Collect kept filters per block
    # ------------------------------
    enc_keepers = []
    dec_keepers = []
    bottleneck_keepers = None

    for name, mask in masks.items():
        if "encoders" in name:
            enc_keepers.append(int(mask.sum().item()))
        elif "bottleneck" in name:
            bottleneck_keepers = int(mask.sum().item())
        elif "decoders" in name:
            dec_keepers.append(int(mask.sum().item()))

    # sort by depth
    dec_keepers = dec_keepers[::-1]

    print("\n📊 Kept filters per block:")
    print(f"Encoders: {enc_keepers}")
    print(f"Bottleneck: {bottleneck_keepers}")
    print(f"Decoders (raw): {dec_keepers}")

    # ------------------------------
    # 2️⃣ Compute decoder input/output channels
    # ------------------------------
    dec_channels = []  # (up_in, up_out, doubleconv_in, doubleconv_out)
    up_in = bottleneck_keepers
    for i, skip_ch in enumerate(reversed(enc_keepers)):
        # output channels after DoubleConv = skip_ch
        doubleconv_out = skip_ch
        # input channels = skip + upsampled
        doubleconv_in = skip_ch + up_in
        dec_channels.append((up_in, skip_ch, doubleconv_in, doubleconv_out))
        up_in = skip_ch
    dec_channels = dec_channels[::-1]  # shallow → deep order

    print("Decoder channel plan:")
    for i, (up_in, up_out, d_in, d_out) in enumerate(dec_channels):
        print(f"  Block {i}: upconv {up_in}->{up_out}, doubleconv {d_in}->{d_out}")

    # ------------------------------
    # 3️⃣ Rebuild new UNet skeleton
    # ------------------------------
    in_ch = model.encoders[0].net[0].in_channels
    out_ch = model.final_conv.out_channels

    # build encoder
    new_encoders = nn.ModuleList()
    last_in = in_ch
    for kept in enc_keepers:
        new_encoders.append(_make_doubleconv(model, last_in, kept))
        last_in = kept

    # build bottleneck
    new_bottleneck = _make_doubleconv(model, enc_keepers[-1], bottleneck_keepers)

    # build decoder
    new_decoders = nn.ModuleList()
    for up_in, up_out, d_in, d_out in dec_channels:
        new_decoders.append(nn.ConvTranspose2d(up_in, up_out, 2, 2))
        new_decoders.append(_make_doubleconv(model, d_in, d_out))

    # final conv
    new_final = nn.Conv2d(enc_keepers[0], out_ch, 1)

    # rebuild UNet
    new_model = model.__class__(in_ch, out_ch, features=enc_keepers)
    new_model.encoders = new_encoders
    new_model.bottleneck = new_bottleneck
    new_model.decoders = new_decoders
    new_model.final_conv = new_final
    new_model.pool = model.pool

    # ------------------------------
    # 4️⃣ Copy surviving weights safely
    # ------------------------------
    old_state = model.state_dict()
    new_state = new_model.state_dict()
    copied, skipped = 0, 0

    for name, param in old_state.items():
        if name not in new_state:
            skipped += 1
            continue
        if new_state[name].shape == param.shape:
            new_state[name].copy_(param)
            copied += 1
        else:
            skipped += 1
            print(f"⚠️  Skipped {name}: {param.shape} → {new_state[name].shape}")

    new_model.load_state_dict(new_state)
    print(f"✅ Weights copied: {copied}  |  Skipped: {skipped}")

    # ------------------------------
    # 5️⃣ Save model
    # ------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(new_model.state_dict(), save_path)
        print(f"💾 Saved pruned model to: {save_path}")

    print("✅ Rebuilt new UNet with physically pruned filters and consistent skips.")
    return new_model


# --- helper to reuse activation/layout from existing model ---
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


def _make_doubleconv(model, in_ch, out_ch):
    """Rebuild a DoubleConv with consistent '.net.' naming."""
    block = next(m for m in model.modules() if isinstance(m, nn.Sequential))
    layers = list(block.children())
    conv1, bn1, relu1, conv2, bn2, relu2 = layers

    new_block = DoubleConv(in_ch, out_ch)
    # copy init parameters (optional)
    return new_block




# # ---------------------------------------------------------
# # 3️⃣ Prune a DoubleConv block coherently
# # ---------------------------------------------------------
# def prune_doubleconv_block(block, mask_in=None, mask_out=None):
#     """
#     block: nn.Sequential (DoubleConv)
#     mask_in: BoolTensor of input channels to keep
#     mask_out: BoolTensor for filters to keep (determined by first conv)
#     """
#     conv1, bn1, relu1, conv2, bn2, relu2 = block

#     # indices to keep
#     out_idx = torch.nonzero(mask_out, as_tuple=False).squeeze(1)
#     if mask_in is not None:
#         in_idx = torch.nonzero(mask_in, as_tuple=False).squeeze(1)
#     else:
#         in_idx = torch.arange(conv1.in_channels)

#     # rebuild conv1
#     new_conv1 = nn.Conv2d(
#         in_channels=len(in_idx),
#         out_channels=len(out_idx),
#         kernel_size=conv1.kernel_size,
#         stride=conv1.stride,
#         padding=conv1.padding,
#         bias=(conv1.bias is not None)
#     )
#     new_conv1.weight.data = conv1.weight.data[out_idx][:, in_idx]
#     if conv1.bias is not None:
#         new_conv1.bias.data = conv1.bias.data[out_idx]

#     # rebuild bn1 accordingly
#     new_bn1 = nn.BatchNorm2d(len(out_idx))
#     new_bn1.weight.data = bn1.weight.data[out_idx]
#     new_bn1.bias.data = bn1.bias.data[out_idx]
#     new_bn1.running_mean = bn1.running_mean[out_idx]
#     new_bn1.running_var = bn1.running_var[out_idx]

#     # rebuild conv2 (same in/out channels as conv1 output)
#     new_conv2 = nn.Conv2d(
#         in_channels=len(out_idx),
#         out_channels=len(out_idx),
#         kernel_size=conv2.kernel_size,
#         stride=conv2.stride,
#         padding=conv2.padding,
#         bias=(conv2.bias is not None)
#     )
#     new_conv2.weight.data = conv2.weight.data[out_idx][:, out_idx]
#     if conv2.bias is not None:
#         new_conv2.bias.data = conv2.bias.data[out_idx]

#     # rebuild bn2
#     new_bn2 = nn.BatchNorm2d(len(out_idx))
#     new_bn2.weight.data = bn2.weight.data[out_idx]
#     new_bn2.bias.data = bn2.bias.data[out_idx]
#     new_bn2.running_mean = bn2.running_mean[out_idx]
#     new_bn2.running_var = bn2.running_var[out_idx]

#     new_block = nn.Sequential(
#         new_conv1, new_bn1, relu1,
#         new_conv2, new_bn2, relu2
#     )
#     return new_block, mask_out


# # ---------------------------------------------------------
# # 4️⃣ Apply pruning block-wise through encoder/decoder
# # ---------------------------------------------------------
# def apply_skip_aware_pruning(model, prune_ratio=0.3):
#     norms = compute_block_l1_norms(model)
#     masks = get_pruning_masks(norms, prune_ratio)
#     new_model = UNet_like_copy(model)

#     encoder_blocks = list(model.encoders)
#     decoder_blocks = list(model.decoders[1::2])  # skip upconvs

#     for i, (enc_block, dec_block) in enumerate(zip(encoder_blocks, reversed(decoder_blocks))):
#         mask = masks.get(f"encoders.{i}.net", None)
#         if mask is None:
#             continue
#         new_enc, mask_out = prune_doubleconv_block(enc_block.net, None, mask)
#         new_dec, _ = prune_doubleconv_block(dec_block.net, mask_out, mask_out)
#         new_model.encoders[i].net = new_enc
#         new_model.decoders[-(2*i+1)].net = new_dec  # maintain symmetry

#     return new_model, masks



# # --- Helper to assign nested modules ---
# def set_by_name(model, name, new_module):
#     parts = name.split('.')
#     m = model
#     for p in parts[:-1]:
#         m = getattr(m, p)
#     setattr(m, parts[-1], new_module)


# # # --- Simple deep copy of UNet structure (without weights) ---
# # def UNet_like_copy(model):
# #     cls = model.__class__
# #     kwargs = model.__dict__['_modules']  # fallback: use same init args
# #     new_model = cls(in_ch=1, out_ch=4)
# #     return new_model

# def UNet_like_copy(model):
#     # Extract init parameters dynamically
#     cfg = {
#         "in_ch": model.encoders[0].net[0].in_channels,
#         "out_ch": model.final_conv.out_channels,
#         "features": [blk.net[0].out_channels for blk in model.encoders]
#     }
#     return UNet(**cfg)



# # ---------------------------------------------------------
# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total, trainable
